/*
 * Mamba-3 SSM selective scan -- CUDA kernels.
 *
 * Forward: Chunked SSD -- each block processes one 32-step chunk for one (batch, head).
 *   Grid: batch * n_heads * n_chunks blocks (counter-based inter-chunk sync).
 * Backward: Sequential scan with chunked recomputation (unchanged).
 *   Grid: batch * n_heads blocks.
 *
 * Features:
 *   - Data-dependent A: a_bar = exp(A_val * dt_pos)
 *   - Data-dependent lambda: per-head trapezoidal mixing
 *   - DD-RoPE: tanh-bounded rotary embeddings on B/C in state-space
 *   - Fused Z-gate: SiLU gating folded into scan output write
 *   - Learned h_init: per-head initial hidden state
 */

#include "ssm_scan.h"
#include <cmath>
#include <cstdio>

#define LOG2E 1.44269504089f

__device__ __forceinline__ float cos_approx(float x) {
    float r;
    asm("cos.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ float sin_approx(float x) {
    float r;
    asm("sin.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + exp2f(x * LOG2E));
}

/* Block-level sum reduction: warp shuffle + shared memory inter-warp reduce.
 * Returns the sum in thread 0 only; other threads get undefined value.
 * smem must have at least (blockDim.x / 32) floats available. */
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int n_warps = blockDim.x / 32;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int w = 0; w < n_warps; w++) sum += smem[w];
        val = sum;
    }
    __syncthreads();
    return val;
}

/* --- Chunked SSD Forward Kernel ----------------------------------------- */
/*
 * Each block processes ONE chunk of SSD_FWD_CHUNK steps for one (batch, head).
 * Grid: batch * n_heads * n_chunks blocks.
 * Chunks within the same (batch, head) synchronize via atomic counters --
 * the GPU scheduler pipelines independent chains across SMs.
 *
 * Warp specialization: 8 warps x 32 threads, each warp owns state_size/8
 * elements. y reduction uses warp shuffle + cross-warp shared memory.
 */

#define SSD_FWD_CHUNK 32

__global__ void ssm_ssd_fwd_kernel(
    const float* __restrict__ x,         // [batch, seq, n_heads, head_dim]
    const float* __restrict__ dt,        // [batch, seq, n_heads]
    const float* __restrict__ b,         // [batch, seq, n_groups, d_state]
    const float* __restrict__ c,         // [batch, seq, n_groups, d_state]
    const float* __restrict__ D,         // [n_heads]
    const float* __restrict__ dt_bias,   // [n_heads]
    const float* __restrict__ h_init,    // [n_heads, d_state] (or NULL)
    const float* __restrict__ lambda_in, // [batch, seq, n_heads] (NULL = fixed 0.5)
    const float* __restrict__ theta,     // [batch, seq, n_heads, d_state/2] (NULL = disabled)
    const float* __restrict__ A_vals,    // [batch, seq, n_heads]
    const float* __restrict__ z_in,      // [batch, seq, n_heads, head_dim] (NULL = no gating)
    float* __restrict__ y,               // [batch, seq, n_heads, head_dim]
    float* __restrict__ y_gated,         // [batch, seq, n_heads, head_dim] (NULL = skip)
    int seq, int n_heads, int head_dim, int d_state, int n_groups, int n_chunks
) {
    /* One block per (batch, head). Sequentially scans all timesteps.
     * No inter-block synchronization — avoids deadlock on large grids. */
    int block_id = blockIdx.x;
    int batch_idx = block_id / n_heads;
    int head_idx = block_id % n_heads;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    int state_size = head_dim * d_state;
    int half_d_state = d_state / 2;
    float D_val = D[head_idx];
    float dt_bias_val = dt_bias[head_idx];
    int group = head_idx / (n_heads / n_groups);

    // Per-thread state
    float h[16], prev_bx[16];
    int n_my_elems = 0;
    for (int idx = tid; idx < state_size; idx += n_threads) {
        int s = idx % d_state;
        h[n_my_elems] = (h_init != nullptr) ? h_init[head_idx * d_state + s] : 0.0f;
        prev_bx[n_my_elems] = 0.0f;
        n_my_elems++;
    }

    extern __shared__ float shared[];
    float* x_cache = shared;
    float* bc_cache = shared + head_dim;
    float* cum_angle_sh = shared + head_dim + d_state;

    // DD-RoPE: init cumulative angles
    if (theta != nullptr) {
        for (int k = tid; k < half_d_state; k += n_threads) {
            cum_angle_sh[k] = 0.0f;
        }
    }

    int b_batch_stride = seq * n_groups * d_state;
    int b_seq_stride = n_groups * d_state;
    int b_group_offset = group * d_state;
    int dt_batch_stride = seq * n_heads;
    int theta_batch_stride = seq * n_heads * half_d_state;
    int theta_seq_stride = n_heads * half_d_state;
    int theta_head_offset = head_idx * half_d_state;
    const float TWO_PI = 6.28318530f;

    int x_base = batch_idx * seq * n_heads * head_dim + head_idx * head_dim;
    int x_stride = n_heads * head_dim;

    __syncthreads();

    for (int t = 0; t < seq; t++) {
        float dt_raw = dt[batch_idx * dt_batch_stride + t * n_heads + head_idx];
        float dt_pos = softplus(dt_raw + dt_bias_val);

        float A_val = A_vals[batch_idx * dt_batch_stride + t * n_heads + head_idx];
        float a_bar = exp2f(A_val * dt_pos * LOG2E);

        float lam = (lambda_in != nullptr)
            ? lambda_in[batch_idx * dt_batch_stride + t * n_heads + head_idx]
            : 0.5f;
        float beta = (1.0f - lam) * a_bar;
        float gamma_val = lam;

        // DD-RoPE: update cumulative angles
        if (theta != nullptr) {
            for (int k = tid; k < half_d_state; k += n_threads) {
                float tv = tanhf(theta[batch_idx * theta_batch_stride + t * theta_seq_stride + theta_head_offset + k]) * 3.14159265f;
                cum_angle_sh[k] = fmodf(cum_angle_sh[k] + dt_pos * tv, TWO_PI);
            }
            __syncthreads();
        }

        // Cache x and B
        for (int p = tid; p < head_dim; p += n_threads)
            x_cache[p] = x[x_base + t * x_stride + p];
        for (int s = tid; s < d_state; s += n_threads)
            bc_cache[s] = b[batch_idx * b_batch_stride + t * b_seq_stride + b_group_offset + s];
        __syncthreads();

        // DD-RoPE: rotate B
        if (theta != nullptr) {
            for (int k = tid; k < half_d_state; k += n_threads) {
                float ca = cos_approx(cum_angle_sh[k]), sa = sin_approx(cum_angle_sh[k]);
                float b0 = bc_cache[2*k], b1 = bc_cache[2*k+1];
                bc_cache[2*k]   = ca*b0 - sa*b1;
                bc_cache[2*k+1] = sa*b0 + ca*b1;
            }
            __syncthreads();
        }

        // State update
        int local_i = 0;
        for (int idx = tid; idx < state_size; idx += n_threads) {
            int p = idx / d_state, s = idx % d_state;
            float bx = bc_cache[s] * x_cache[p];
            h[local_i] = a_bar * h[local_i] + beta * prev_bx[local_i] + gamma_val * bx;
            prev_bx[local_i] = bx;
            local_i++;
        }

        // Load C, zero x_cache for y reduction
        for (int s = tid; s < d_state; s += n_threads)
            bc_cache[s] = c[batch_idx * b_batch_stride + t * b_seq_stride + b_group_offset + s];
        for (int p = tid; p < head_dim; p += n_threads)
            x_cache[p] = 0.0f;
        __syncthreads();

        // DD-RoPE: rotate C
        if (theta != nullptr) {
            for (int k = tid; k < half_d_state; k += n_threads) {
                float ca = cos_approx(cum_angle_sh[k]), sa = sin_approx(cum_angle_sh[k]);
                float c0 = bc_cache[2*k], c1 = bc_cache[2*k+1];
                bc_cache[2*k]   = ca*c0 - sa*c1;
                bc_cache[2*k+1] = sa*c0 + ca*c1;
            }
            __syncthreads();
        }

        // y = C @ h (warp-level reduction via atomicAdd on shared)
        local_i = 0;
        for (int idx = tid; idx < state_size; idx += n_threads) {
            int p = idx / d_state, s = idx % d_state;
            atomicAdd(&x_cache[p], bc_cache[s] * h[local_i]);
            local_i++;
        }
        __syncthreads();

        // Output + optional SiLU gate
        for (int p = tid; p < head_dim; p += n_threads) {
            int out_idx = x_base + t * x_stride + p;
            float x_val = x[out_idx];
            float y_val = x_cache[p] + D_val * x_val;
            y[out_idx] = y_val;
            if (z_in != nullptr && y_gated != nullptr) {
                float z_val = z_in[out_idx];
                float sig_z = 1.0f / (1.0f + exp2f(-z_val * LOG2E));
                y_gated[out_idx] = y_val * z_val * sig_z;
            }
        }
        __syncthreads();
    }

}
/* ─── Backward Kernel (fused — matches forward signature) ────── */
/*
 * Fused backward for Mamba-3 SSM.
 * Indexes B/C by n_groups (not n_heads) with group mapping.
 *
 * Trapezoidal discretization:
 *   h[t] = a_bar * h[t-1] + (1-λ) * a_bar * prev_bx + λ * bx[t]
 *
 * Data-dependent A: a_bar = exp(A_val * dt_pos) where A_val < 0 (from -softplus(dd_A))
 *
 * DD-RoPE: B and C are rotated by cumulative angles before use.
 * Theta gradients are written per-position via reverse-cumsum of d_angle.
 *
 * Produces gradients for ALL SSM parameters:
 *   dx, d_dt (raw), db (grouped), dc (grouped), dD, d_dt_bias,
 *   d_lambda, d_h_init, d_theta (per-position), d_A_vals (per-position)
 */

__global__ void ssm_scan_bwd_kernel(
    const float* __restrict__ x,        // [batch, seq, n_heads, head_dim]
    const float* __restrict__ dt,       // [batch, seq, n_heads]
    const float* __restrict__ b,        // [batch, seq, n_groups, d_state]
    const float* __restrict__ c,        // [batch, seq, n_groups, d_state]
    const float* __restrict__ D,        // [n_heads]
    const float* __restrict__ dt_bias,  // [n_heads]
    const float* __restrict__ dy,       // [batch, seq, n_heads, head_dim]
    const float* __restrict__ lambda_in, // [batch, seq, n_heads] — data-dependent λ
    const float* __restrict__ h_init,   // [n_heads, d_state] — learned initial state (NULL = zeros)
    float* __restrict__ h_checkpoints,          // [batch*n_heads, n_chunks, n_threads, max_elems] — chunk boundary
    float* __restrict__ pbx_checkpoints,        // same
    float* __restrict__ h_saved_buf,             // [batch*n_heads, CHUNK_SIZE, n_threads, max_elems] — within-chunk saved
    float* __restrict__ pbx_saved_buf,           // same
    const float* __restrict__ theta,    // [batch, seq, n_heads, d_state/2] — DD-RoPE (always enabled)
    const float* __restrict__ A_vals,   // [batch, seq, n_heads] — data-dependent A
    float* __restrict__ dx,             // [batch, seq, n_heads, head_dim]
    float* __restrict__ d_dt,           // [batch, seq, n_heads]
    float* __restrict__ db,             // [batch, seq, n_groups, d_state]
    float* __restrict__ dc,             // [batch, seq, n_groups, d_state]
    float* __restrict__ d_lambda,       // [batch, seq, n_heads] — gradient for lambda (NULL if not needed)
    float* __restrict__ d_h_init,       // [n_heads, d_state] — gradient for h_init (NULL if not needed)
    float* __restrict__ d_theta_out,    // [batch, seq, n_heads, d_state/2] — per-position DD-RoPE theta gradient
    float* __restrict__ d_A_vals,       // [batch, seq, n_heads] — gradient for A_vals
    float* __restrict__ dD_buf,         // [batch * n_heads] partial sums
    float* __restrict__ d_dt_bias_buf,  // [batch * n_heads] partial sums
    int seq, int n_heads, int head_dim, int d_state, int n_groups
) {
    int block_id = blockIdx.x;
    int batch_idx = block_id / n_heads;
    int head_idx = block_id % n_heads;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    int state_size = head_dim * d_state;
    const int max_elems = (state_size + n_threads - 1) / n_threads;

    float D_val = D[head_idx];
    float dt_bias_val = dt_bias[head_idx];
    int group = head_idx / (n_heads / n_groups);
    const float TWO_PI = 6.28318530f;

    // Strides for x/y: [batch, seq, n_heads, head_dim]
    int x_batch_stride = seq * n_heads * head_dim;
    int x_seq_stride = n_heads * head_dim;
    // Strides for dt: [batch, seq, n_heads]
    int dt_batch_stride = seq * n_heads;
    int dt_seq_stride = n_heads;
    // Strides for B/C: [batch, seq, n_groups, d_state]
    int b_batch_stride = seq * n_groups * d_state;
    int b_seq_stride = n_groups * d_state;
    int b_group_offset = group * d_state;

    int x_base = batch_idx * x_batch_stride + head_idx * head_dim;
    int dt_base = batch_idx * dt_batch_stride + head_idx;

    extern __shared__ float shared[];

    // Shared memory layout:
    //   [0..head_dim)                                  dx accumulation (zeroed per timestep)
    //   [head_dim..+d_state)                           bwd_bc_b: rotated B cache
    //   [+d_state..+d_state)                           bwd_bc_c: rotated C cache
    //   [+d_state/2)                                   bwd_cum_angle: DD-RoPE cumulative angles
    //   [+d_state)                                     bwd_db_rot_accum
    //   [+d_state)                                     bwd_dc_rot_accum
    //   [+head_dim)                                    bwd_x_cache: x values for backward inner loop
    //   [+head_dim)                                    bwd_dy_cache: dy values for backward inner loop
    int half_d_state = d_state / 2;
    float* bwd_bc_b = shared + head_dim;
    float* bwd_bc_c = shared + head_dim + d_state;
    float* bwd_cum_angle = shared + head_dim + 2 * d_state;
    float* bwd_db_rot_accum = shared + head_dim + 2 * d_state + half_d_state;
    float* bwd_dc_rot_accum = shared + head_dim + 3 * d_state + half_d_state;
    float* bwd_x_cache = shared + head_dim + 4 * d_state + half_d_state;
    float* bwd_dy_cache = bwd_x_cache + head_dim;

    // DD-RoPE: init cumulative angle
    for (int k = tid; k < half_d_state; k += n_threads) {
        bwd_cum_angle[k] = 0.0f;
    }
    int theta_batch_stride_bwd = seq * n_heads * half_d_state;
    int theta_seq_stride_bwd = n_heads * half_d_state;
    int theta_head_offset_bwd = head_idx * half_d_state;
    __syncthreads();

    // Per-thread DD-RoPE state (tid < half_d_state only)
    float my_cum_angle_saved[8];   // cumulative rotation angle at each timestep within a chunk
    float my_dt_pos_saved[8];      // per-timestep dt_pos for theta gradient chain rule
    float my_theta_val = 0.0f;     // tanh-bounded theta * pi
    float d_angle_accum = 0.0f;    // reverse-cumsum of d_angle for backprop

    const int CHUNK_SIZE = 8;

    // Per-thread state arrays (must match forward kernel's h[16])
    #define MAX_ELEMS 16
    float h[MAX_ELEMS], pbx[MAX_ELEMS];
    float dh[MAX_ELEMS], d_bx_carry[MAX_ELEMS];
    if (max_elems > MAX_ELEMS) return; // safety: config exceeds compile-time limit

    for (int i = 0; i < max_elems; i++) {
        int idx = tid + i * n_threads;
        int s = idx % d_state;
        h[i] = (h_init != nullptr) ? h_init[head_idx * d_state + s] : 0.0f;
        pbx[i] = 0.0f;
        dh[i] = 0.0f; d_bx_carry[i] = 0.0f;
    }

    float dD_local = 0.0f;
    float d_dt_bias_local = 0.0f;
    int n_chunks = (seq + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // External checkpoint buffer indexing: [block_id][chunk][tid][elem]
    int ckpt_chunk_stride = n_threads * max_elems;
    int ckpt_block_base = block_id * n_chunks * ckpt_chunk_stride;

    // Within-chunk saved state buffer indexing: [block_id][local_t][tid][elem]
    int saved_step_stride = n_threads * max_elems;
    int saved_block_base = block_id * CHUNK_SIZE * saved_step_stride;

    // cum_angle_boundary: 1 float per chunk, only tid < half_d_state
    // max_seq_len=2048, CHUNK_SIZE=8 → max 256 chunks
    float cum_angle_boundary[256];

    // ── Forward pass: save h/prev_bx at chunk boundaries to external buffers ──
    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int ckpt_idx = ckpt_block_base + chunk * ckpt_chunk_stride + tid * max_elems;
        for (int i = 0; i < max_elems; i++) {
            h_checkpoints[ckpt_idx + i] = h[i];
            pbx_checkpoints[ckpt_idx + i] = pbx[i];
        }
        if (tid < half_d_state) {
            cum_angle_boundary[chunk] = bwd_cum_angle[tid];
        }
        int t_start = chunk * CHUNK_SIZE;
        int t_end = min(t_start + CHUNK_SIZE, seq);
        for (int t = t_start; t < t_end; t++) {
            float dt_raw = dt[dt_base + t * dt_seq_stride];
            float dt_pos = softplus(dt_raw + dt_bias_val);
            // DD-RoPE: update cumulative angles
            __syncthreads();
            for (int k = tid; k < half_d_state; k += n_threads) {
                float tv = tanhf(theta[batch_idx * theta_batch_stride_bwd + t * theta_seq_stride_bwd + theta_head_offset_bwd + k]) * 3.14159265f;
                bwd_cum_angle[k] = fmodf(bwd_cum_angle[k] + dt_pos * tv, TWO_PI);
            }
            // Load B into shared and rotate
            int bc_base_t = batch_idx * b_batch_stride + t * b_seq_stride + b_group_offset;
            for (int s = tid; s < d_state; s += n_threads) {
                bwd_bc_b[s] = b[bc_base_t + s];
            }
            __syncthreads();
            for (int k = tid; k < half_d_state; k += n_threads) {
                float ca = cos_approx(bwd_cum_angle[k]);
                float sa = sin_approx(bwd_cum_angle[k]);
                float b0 = bwd_bc_b[2*k], b1 = bwd_bc_b[2*k+1];
                bwd_bc_b[2*k]   = ca*b0 - sa*b1;
                bwd_bc_b[2*k+1] = sa*b0 + ca*b1;
            }
            __syncthreads();
            float A_val = A_vals[batch_idx * dt_batch_stride + t * n_heads + head_idx];
            float a_bar = exp2f(A_val * dt_pos * LOG2E);
            float lam = lambda_in[dt_base + t * dt_seq_stride];
            float beta = (1.0f - lam) * a_bar;
            float gamma_val = lam;
            for (int elem_idx = tid; elem_idx < state_size; elem_idx += n_threads) {
                int p = elem_idx / d_state;
                int s = elem_idx % d_state;
                int li = (elem_idx - tid) / n_threads;
                if (li < max_elems) {
                    float x_val = x[x_base + t * x_seq_stride + p];
                    float bx = bwd_bc_b[s] * x_val;
                    h[li] = a_bar * h[li] + beta * pbx[li] + gamma_val * bx;
                    pbx[li] = bx;
                }
            }
        }
    }
    // ── Backward pass: process chunks in reverse ──
    for (int i = 0; i < max_elems; i++) { dh[i] = 0.0f; d_bx_carry[i] = 0.0f; }

    for (int chunk = n_chunks - 1; chunk >= 0; chunk--) {
        int t_start = chunk * CHUNK_SIZE;
        int t_end = min(t_start + CHUNK_SIZE, seq);

        // Restore from external checkpoint buffers
        int ckpt_idx = ckpt_block_base + chunk * ckpt_chunk_stride + tid * max_elems;
        for (int i = 0; i < max_elems; i++) {
            h[i] = h_checkpoints[ckpt_idx + i];
            pbx[i] = pbx_checkpoints[ckpt_idx + i];
        }
        // Restore cumulative angle for this chunk
        float my_chunk_cum_angle = 0.0f;
        if (tid < half_d_state) {
            my_chunk_cum_angle = cum_angle_boundary[chunk];
            bwd_cum_angle[tid] = cum_angle_boundary[chunk];
        }
        __syncthreads();

        // Forward recompute within chunk, saving state
        for (int t = t_start; t < t_end; t++) {
            int local_t = t - t_start;
            float dt_raw = dt[dt_base + t * dt_seq_stride];
            float dt_pos = softplus(dt_raw + dt_bias_val);
            // DD-RoPE: update cumulative angles and save per-timestep values
            __syncthreads();
            for (int k = tid; k < half_d_state; k += n_threads) {
                float tv = tanhf(theta[batch_idx * theta_batch_stride_bwd + t * theta_seq_stride_bwd + theta_head_offset_bwd + k]) * 3.14159265f;
                bwd_cum_angle[k] = fmodf(bwd_cum_angle[k] + dt_pos * tv, TWO_PI);
            }
            if (tid < half_d_state) {
                my_theta_val = tanhf(theta[batch_idx * theta_batch_stride_bwd + t * theta_seq_stride_bwd + theta_head_offset_bwd + tid]) * 3.14159265f;
                my_chunk_cum_angle = fmodf(my_chunk_cum_angle + dt_pos * my_theta_val, TWO_PI);
                my_cum_angle_saved[local_t] = my_chunk_cum_angle;
                my_dt_pos_saved[local_t] = dt_pos;
            }
            // Load B into shared and rotate
            int bc_base_t = batch_idx * b_batch_stride + t * b_seq_stride + b_group_offset;
            for (int s = tid; s < d_state; s += n_threads) {
                bwd_bc_b[s] = b[bc_base_t + s];
            }
            __syncthreads();
            for (int k = tid; k < half_d_state; k += n_threads) {
                float ca = cos_approx(bwd_cum_angle[k]);
                float sa = sin_approx(bwd_cum_angle[k]);
                float b0 = bwd_bc_b[2*k], b1 = bwd_bc_b[2*k+1];
                bwd_bc_b[2*k]   = ca*b0 - sa*b1;
                bwd_bc_b[2*k+1] = sa*b0 + ca*b1;
            }
            __syncthreads();
            float A_val = A_vals[batch_idx * dt_batch_stride + t * n_heads + head_idx];
            float a_bar = exp2f(A_val * dt_pos * LOG2E);
            float lam = lambda_in[dt_base + t * dt_seq_stride];
            float beta = (1.0f - lam) * a_bar;
            float gamma_val = lam;

            // Save state to global buffer (coalesced: adjacent tids write adjacent addresses)
            {
                int si = saved_block_base + local_t * saved_step_stride + tid * max_elems;
                for (int i = 0; i < max_elems; i++) {
                    h_saved_buf[si + i] = h[i];
                    pbx_saved_buf[si + i] = pbx[i];
                }
            }

            for (int elem_idx = tid; elem_idx < state_size; elem_idx += n_threads) {
                int p = elem_idx / d_state;
                int s = elem_idx % d_state;
                int li = (elem_idx - tid) / n_threads;
                if (li < max_elems) {
                    float x_val = x[x_base + t * x_seq_stride + p];
                    float bx = bwd_bc_b[s] * x_val;
                    h[li] = a_bar * h[li] + beta * pbx[li] + gamma_val * bx;
                    pbx[li] = bx;
                }
            }
        }

        // Backward through chunk (reverse time)
        for (int t = t_end - 1; t >= t_start; t--) {
            int local_t = t - t_start;
            float dt_raw = dt[dt_base + t * dt_seq_stride];
            float dt_pos = softplus(dt_raw + dt_bias_val);
            float A_val_t = A_vals[batch_idx * dt_batch_stride + t * n_heads + head_idx];
            float a_bar = exp2f(A_val_t * dt_pos * LOG2E);
            float lam = lambda_in[dt_base + t * dt_seq_stride];
            float beta = (1.0f - lam) * a_bar;
            float gamma_val = lam;

            // sigmoid(dt + dt_bias) for chain rule
            float sig = 1.0f / (1.0f + exp2f(-(dt_raw + dt_bias_val) * LOG2E));

            // Load B/C into shared memory
            int bc_base = batch_idx * b_batch_stride + t * b_seq_stride + b_group_offset;
            for (int s = tid; s < d_state; s += n_threads) bwd_bc_b[s] = b[bc_base + s];
            for (int s = tid; s < d_state; s += n_threads) bwd_bc_c[s] = c[bc_base + s];
            __syncthreads();
            // DD-RoPE: set cumulative angles and rotate B/C
            if (tid < half_d_state) {
                bwd_cum_angle[tid] = my_cum_angle_saved[local_t];
            }
            __syncthreads();
            for (int k = tid; k < half_d_state; k += n_threads) {
                float ca = cos_approx(bwd_cum_angle[k]);
                float sa = sin_approx(bwd_cum_angle[k]);
                float b0 = bwd_bc_b[2*k], b1 = bwd_bc_b[2*k+1];
                bwd_bc_b[2*k]   = ca*b0 - sa*b1;
                bwd_bc_b[2*k+1] = sa*b0 + ca*b1;
                float c0 = bwd_bc_c[2*k], c1 = bwd_bc_c[2*k+1];
                bwd_bc_c[2*k]   = ca*c0 - sa*c1;
                bwd_bc_c[2*k+1] = sa*c0 + ca*c1;
            }
            // Zero db/dc rot accum for d_theta computation
            for (int s = tid; s < d_state; s += n_threads) {
                bwd_db_rot_accum[s] = 0.0f;
                bwd_dc_rot_accum[s] = 0.0f;
            }
            __syncthreads();

            // Cache x and dy in shared memory (each read once, used d_state times)
            for (int p = tid; p < head_dim; p += n_threads) {
                bwd_x_cache[p] = x[x_base + t * x_seq_stride + p];
                bwd_dy_cache[p] = dy[x_base + t * x_seq_stride + p];
                shared[p] = 0.0f; // zero dx accumulation
            }
            __syncthreads();

            float da_bar_local = 0.0f;
            float d_lam_local = 0.0f;

            for (int elem_idx = tid; elem_idx < state_size; elem_idx += n_threads) {
                int p = elem_idx / d_state;
                int s = elem_idx % d_state;
                int li = (elem_idx - tid) / n_threads;
                if (li >= max_elems) continue;

                float dy_val = bwd_dy_cache[p];
                float x_val = bwd_x_cache[p];

                // Read B/C from shared memory (rotated if DD-RoPE, raw otherwise)
                float b_val = bwd_bc_b[s];
                float c_val = bwd_bc_c[s];

                int si = saved_block_base + local_t * saved_step_stride + tid * max_elems + li;
                float h_prev = h_saved_buf[si];
                float prev_bx_val = pbx_saved_buf[si];
                float bx_cur = b_val * x_val;
                float h_cur = a_bar * h_prev + beta * prev_bx_val + gamma_val * bx_cur;

                dh[li] += c_val * dy_val;

                // dc and db: un-rotate per-thread if DD-RoPE
                float dc_contrib = dy_val * h_cur;
                float db_contrib = (gamma_val * dh[li] + d_bx_carry[li]) * x_val;

                {
                    int k = s / 2;
                    float ca = cos_approx(bwd_cum_angle[k]);
                    float sa = sin_approx(bwd_cum_angle[k]);
                    // Un-rotate: split contribution across both pair elements
                    if (s % 2 == 0) {
                        atomicAdd(&db[bc_base + s],     ca * db_contrib);
                        atomicAdd(&db[bc_base + s + 1], -sa * db_contrib);
                        atomicAdd(&dc[bc_base + s],     ca * dc_contrib);
                        atomicAdd(&dc[bc_base + s + 1], -sa * dc_contrib);
                    } else {
                        atomicAdd(&db[bc_base + s - 1], sa * db_contrib);
                        atomicAdd(&db[bc_base + s],     ca * db_contrib);
                        atomicAdd(&dc[bc_base + s - 1], sa * dc_contrib);
                        atomicAdd(&dc[bc_base + s],     ca * dc_contrib);
                    }
                    atomicAdd(&bwd_db_rot_accum[s], db_contrib);
                    atomicAdd(&bwd_dc_rot_accum[s], dc_contrib);
                }

                da_bar_local += dh[li] * (h_prev + (1.0f - lam) * prev_bx_val);
                d_lam_local += dh[li] * (-a_bar * prev_bx_val + bx_cur);

                // dx via shared (use rotated b_val)
                float d_bx_total = gamma_val * dh[li] + d_bx_carry[li];
                atomicAdd(&shared[p], d_bx_total * b_val);

                d_bx_carry[li] = beta * dh[li];
                dh[li] = a_bar * dh[li];
            }
            __syncthreads();

            // Write dx and accumulate dD (x/dy already in shared cache)
            for (int p = tid; p < head_dim; p += n_threads) {
                float dy_val = bwd_dy_cache[p];
                dx[x_base + t * x_seq_stride + p] = shared[p] + D_val * dy_val;
                dD_local += dy_val * bwd_x_cache[p];
            }
            __syncthreads();

            // d_lambda: warp-reduce then single atomicAdd
            {
                float d_lam_sum = block_reduce_sum(d_lam_local, shared);
                if (tid == 0) {
                    atomicAdd(&d_lambda[dt_base + t * dt_seq_stride], d_lam_sum);
                }
            }

            // DD-RoPE: compute per-position d_theta via reverse-cumsum of d_angle
            {
                __syncthreads(); // ensures db/dc_rot_accum writes are visible
                if (tid < half_d_state) {
                    int k = tid;
                    float ca = cos_approx(bwd_cum_angle[k]);
                    float sa = sin_approx(bwd_cum_angle[k]);
                    // d_angle from B rotation: d(cum_angle) contribution
                    float b0_raw = b[bc_base + 2*k], b1_raw = b[bc_base + 2*k + 1];
                    float db0 = bwd_db_rot_accum[2*k], db1 = bwd_db_rot_accum[2*k + 1];
                    float d_angle_b = (-sa*b0_raw - ca*b1_raw)*db0 + (ca*b0_raw - sa*b1_raw)*db1;
                    // d_angle from C rotation
                    float c0_raw = c[bc_base + 2*k], c1_raw = c[bc_base + 2*k + 1];
                    float dc0 = bwd_dc_rot_accum[2*k], dc1 = bwd_dc_rot_accum[2*k + 1];
                    float d_angle_c = (-sa*c0_raw - ca*c1_raw)*dc0 + (ca*c0_raw - sa*c1_raw)*dc1;

                    // Accumulate d_angle from future timesteps (backward iterates in reverse)
                    // theta[t] affects cum_angle[s] for all s >= t, so we need sum of d_angle from t..T
                    d_angle_accum += d_angle_b + d_angle_c;

                    // Chain rule: d_theta[t] = d_angle_accum * dt_pos[t] * d(tanh(theta_raw[t])*pi)/d(theta_raw[t])
                    // where d(tanh(x)*pi)/dx = pi * (1 - tanh^2(x))
                    float dt_pos_t = my_dt_pos_saved[local_t];
                    float theta_raw_val = theta[batch_idx * theta_batch_stride_bwd + t * theta_seq_stride_bwd + theta_head_offset_bwd + k];
                    float tanh_val = tanhf(theta_raw_val);
                    float d_theta_t = d_angle_accum * dt_pos_t * (1.0f - tanh_val * tanh_val) * 3.14159265f;

                    // Write per-position gradient: [batch, seq, n_heads, d_state/2]
                    d_theta_out[batch_idx * theta_batch_stride_bwd + t * theta_seq_stride_bwd + theta_head_offset_bwd + k] = d_theta_t;

                    // DD-RoPE contribution to d_dt_pos: cum_angle[k] += dt_pos * theta_val[k]
                    // so d_dt_pos += sum_k(d_cum_angle[k] * theta_val[k])
                    // d_angle_accum IS d_cum_angle[k], my_theta_val IS tanh(theta_raw)*pi
                    // Warp-shuffle reduce across tid 0..half_d_state-1 (fits in one warp)
                    float d_dt_from_rope = d_angle_accum * my_theta_val;
                    for (int offset = 16; offset > 0; offset >>= 1)
                        d_dt_from_rope += __shfl_xor_sync(0xffffffff, d_dt_from_rope, offset);
                    if (tid == 0) {
                        float d_dt_rope_val = d_dt_from_rope * sig;
                        atomicAdd(&d_dt[dt_base + t * dt_seq_stride], d_dt_rope_val);
                        d_dt_bias_local += d_dt_rope_val;
                    }
                }
            }

            // Chain rule for data-dependent A: a_bar = exp(A_val * dt_pos)
            // d_a_bar/d_A_val = a_bar * dt_pos
            // d_a_bar/d_dt_pos = a_bar * A_val
            // d_dt_pos/d_dt_raw = sigmoid(dt_raw + dt_bias)
            {
                float d_A_val_local = da_bar_local * a_bar * dt_pos;
                float d_dt_local = da_bar_local * a_bar * A_val_t * sig;
                d_dt_bias_local += d_dt_local;

                // Fused warp-reduce for d_A_vals and d_dt (was 256-way contention each)
                int warp_id = tid / 32;
                int lane_id = tid % 32;
                int n_warps = n_threads / 32;
                // Intra-warp reduce both values
                for (int offset = 16; offset > 0; offset >>= 1) {
                    d_A_val_local += __shfl_xor_sync(0xffffffff, d_A_val_local, offset);
                    d_dt_local += __shfl_xor_sync(0xffffffff, d_dt_local, offset);
                }
                if (lane_id == 0) {
                    shared[warp_id] = d_A_val_local;
                    shared[n_warps + warp_id] = d_dt_local;
                }
                __syncthreads();
                if (tid == 0) {
                    float sum_dA = 0.0f, sum_ddt = 0.0f;
                    for (int w = 0; w < n_warps; w++) {
                        sum_dA += shared[w];
                        sum_ddt += shared[n_warps + w];
                    }
                    atomicAdd(&d_A_vals[batch_idx * dt_batch_stride + t * n_heads + head_idx], sum_dA);
                    atomicAdd(&d_dt[dt_base + t * dt_seq_stride], sum_ddt);
                }
                __syncthreads();
            }
        }
    }

    // Write d_h_init gradient (dh holds gradient w.r.t. initial state)
    if (d_h_init != nullptr) {
        for (int i = 0; i < max_elems; i++) {
            int idx = tid + i * n_threads;
            if (idx < state_size) {
                int s = idx % d_state;
                atomicAdd(&d_h_init[head_idx * d_state + s], dh[i]);
            }
        }
    }

    // Reduce dD_local, d_dt_bias_local across all threads
    // Warp shuffle for intra-warp, shared memory for inter-warp
    float* reduce_buf = shared;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int n_warps = n_threads / 32; // 256/32 = 8

    // Helper: warp-level reduction then inter-warp via shared
    unsigned int warp_reduce_mask = (1u << n_warps) - 1;  // only n_warps threads participate
    #define WARP_REDUCE_AND_STORE(val, out_ptr) do { \
        for (int offset = 16; offset > 0; offset >>= 1) \
            val += __shfl_xor_sync(0xffffffff, val, offset); \
        if (lane_id == 0) reduce_buf[warp_id] = val; \
        __syncthreads(); \
        if (tid < n_warps) { \
            val = reduce_buf[tid]; \
            for (int offset = n_warps / 2; offset > 0; offset >>= 1) \
                val += __shfl_xor_sync(warp_reduce_mask, val, offset); \
            if (tid == 0) *(out_ptr) = val; \
        } \
        __syncthreads(); \
    } while(0)

    WARP_REDUCE_AND_STORE(dD_local, &dD_buf[block_id]);
    WARP_REDUCE_AND_STORE(d_dt_bias_local, &d_dt_bias_buf[block_id]);
    #undef WARP_REDUCE_AND_STORE
}

/* ─── Reduction Kernel for per-head parameter gradients ──────── */

__global__ void reduce_param_grads_kernel(
    const float* __restrict__ dD_buf,         // [batch * n_heads]
    const float* __restrict__ d_dt_bias_buf,  // [batch * n_heads]
    float* __restrict__ dD,                   // [n_heads]
    float* __restrict__ d_dt_bias,            // [n_heads]
    int batch, int n_heads
) {
    int head_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (head_idx >= n_heads) return;

    float sum_D = 0.0f, sum_dtb = 0.0f;
    for (int b = 0; b < batch; b++) {
        sum_D += dD_buf[b * n_heads + head_idx];
        sum_dtb += d_dt_bias_buf[b * n_heads + head_idx];
    }
    atomicAdd(&dD[head_idx], sum_D);
    atomicAdd(&d_dt_bias[head_idx], sum_dtb);
}

/* --- Host API: Chunked SSD Forward -------------------------------------- */

extern "C"
void ssm_scan_fwd_gpu(
    const float* d_x, const float* d_dt,
    const float* d_b, const float* d_c,
    const float* d_D, const float* d_dt_bias,
    const float* d_h_init,
    const float* d_lambda,
    const float* d_theta,
    const float* d_A_vals,
    const float* d_z,
    float* d_y,
    float* d_y_gated,
    int batch, int seq, int n_heads, int head_dim, int d_state, int n_groups
) {
    int n_chunks = (seq + SSD_FWD_CHUNK - 1) / SSD_FWD_CHUNK;
    int n_blocks = batch * n_heads;
    int n_threads = 256;

    int shared_mem = (d_theta != nullptr)
        ? (head_dim + d_state + d_state / 2) * sizeof(float)
        : (head_dim + d_state) * sizeof(float);

    ssm_ssd_fwd_kernel<<<n_blocks, n_threads, shared_mem>>>(
        d_x, d_dt, d_b, d_c, d_D, d_dt_bias, d_h_init, d_lambda, d_theta, d_A_vals, d_z,
        d_y, d_y_gated,
        seq, n_heads, head_dim, d_state, n_groups, n_chunks
    );
}

/* --- Host API: Backward (unchanged) ------------------------------------- */

extern "C"
void ssm_scan_bwd_gpu_v2(
    const float* d_x, const float* d_dt,
    const float* d_b, const float* d_c,
    const float* d_D, const float* d_dt_bias,
    const float* d_dy,
    const float* d_lambda_in,
    const float* d_h_init,
    float* d_h_checkpoints, float* d_pbx_checkpoints,
    float* d_h_saved_buf, float* d_pbx_saved_buf,
    const float* d_theta_in,
    const float* d_A_vals_in,
    float* d_dx, float* d_ddt, float* d_db, float* d_dc,
    float* d_d_lambda,
    float* d_d_h_init,
    float* d_d_theta,
    float* d_d_A_vals,
    float* d_dD, float* d_d_dt_bias,
    float* ws_dD_buf, float* ws_d_dtb_buf,
    int batch, int seq, int n_heads, int head_dim, int d_state, int n_groups
) {
    int n_blocks = batch * n_heads;
    int n_threads = 256;
    // DD-RoPE always on: head_dim (dx accum) + 4*d_state + d_state/2 + 2*head_dim (x/dy cache)
    int shared_needed = 3 * head_dim + 4 * d_state + d_state / 2;
    int shared_mem = ((shared_needed > n_threads) ? shared_needed : n_threads) * sizeof(float);

    size_t dt_size = (size_t)batch * seq * n_heads * sizeof(float);
    size_t bc_size = (size_t)batch * seq * n_groups * d_state * sizeof(float);
    cudaMemset(d_ddt, 0, dt_size);
    cudaMemset(d_db, 0, bc_size);
    cudaMemset(d_dc, 0, bc_size);
    cudaMemset(d_d_lambda, 0, dt_size);
    cudaMemset(d_d_A_vals, 0, dt_size);

    ssm_scan_bwd_kernel<<<n_blocks, n_threads, shared_mem>>>(
        d_x, d_dt, d_b, d_c, d_D, d_dt_bias, d_dy, d_lambda_in,
        d_h_init, d_h_checkpoints, d_pbx_checkpoints,
        d_h_saved_buf, d_pbx_saved_buf,
        d_theta_in, d_A_vals_in,
        d_dx, d_ddt, d_db, d_dc, d_d_lambda, d_d_h_init, d_d_theta, d_d_A_vals,
        ws_dD_buf, ws_d_dtb_buf,
        seq, n_heads, head_dim, d_state, n_groups
    );

    int reduce_threads = min(n_heads, 256);
    int reduce_blocks = (n_heads + reduce_threads - 1) / reduce_threads;
    reduce_param_grads_kernel<<<reduce_blocks, reduce_threads>>>(
        ws_dD_buf, ws_d_dtb_buf,
        d_dD, d_d_dt_bias,
        batch, n_heads
    );
}
