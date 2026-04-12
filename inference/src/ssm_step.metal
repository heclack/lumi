#include <metal_stdlib>
using namespace metal;

// ─── Fused RMSNorm ───────────────────────────────────────────────────────────
// Replaces 4 separate Candle dispatches (sqr, mean, rsqrt, mul) with 1.
// Uses simd_sum for cross-lane reduction — 1 threadgroup of 32 threads.

struct RmsNormParams {
    int dim;
    float eps;
};

kernel void rms_norm_fused(
    device const float* x       [[buffer(0)]],  // [dim]
    device const float* gamma   [[buffer(1)]],  // [dim]
    device float* out           [[buffer(2)]],  // [dim]
    constant RmsNormParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    const int dim = params.dim;
    const float eps = params.eps;

    // Phase 1: each thread accumulates sum-of-squares for its strided elements
    float local_ss = 0.0f;
    for (int i = tid; i < dim; i += 32) {
        float v = x[i];
        local_ss += v * v;
    }

    // Cross-lane reduction via simd_sum (all 32 threads in one SIMD group)
    float ss = simd_sum(local_ss);
    float scale = rsqrt(ss / float(dim) + eps);

    // Phase 2: normalize and scale
    for (int i = tid; i < dim; i += 32) {
        out[i] = x[i] * scale * gamma[i];
    }
}

// ─── Fused Pre-SSM Kernel ─────────────────────────────────────────────────────
// Replaces ~35 separate Candle dispatches with 1. Takes the flat in_proj output
// and produces all SSM inputs: x_heads (with SiLU), b_expanded, c_expanded
// (with BCNorm + group expansion), a_bar, lambda_vals, dt_pos, theta, z.
//
// 1 threadgroup of 1024 threads. Work is phased with barriers.

struct PreSsmParams {
    int d_inner;        // expand * d_model (e.g. 2048)
    int bc_size;        // n_groups * d_state (e.g. 512)
    int n_heads;        // e.g. 64
    int head_dim;       // d_inner / n_heads (e.g. 32)
    int d_state;        // e.g. 64
    int n_groups;       // e.g. 8
    int heads_per_group;// n_heads / n_groups (e.g. 8)
    int half_ds;        // d_state / 2 (e.g. 32)
    int theta_proj;     // n_heads * half_ds (e.g. 2048)
    float norm_eps;
};

kernel void pre_ssm_fused(
    // Input: flat in_proj output
    device const float* projected      [[buffer(0)]],   // [in_proj_out]
    // Norm weights + biases
    device const float* b_norm_gamma   [[buffer(1)]],   // [bc_size]
    device const float* c_norm_gamma   [[buffer(2)]],   // [bc_size]
    device const float* b_bias_vec     [[buffer(3)]],   // [bc_size]
    device const float* c_bias_vec     [[buffer(4)]],   // [bc_size]
    device const float* dt_bias_vec    [[buffer(5)]],   // [n_heads]
    // Outputs (all pre-allocated)
    device float* x_heads_out          [[buffer(6)]],   // [n_heads * head_dim] = [d_inner]
    device float* z_out                [[buffer(7)]],   // [d_inner]
    device float* b_expanded_out       [[buffer(8)]],   // [n_heads * d_state]
    device float* c_expanded_out       [[buffer(9)]],   // [n_heads * d_state]
    device float* a_bar_out            [[buffer(10)]],  // [n_heads]
    device float* lambda_out           [[buffer(11)]],  // [n_heads]
    device float* dt_pos_out           [[buffer(12)]],  // [n_heads]
    device float* theta_out            [[buffer(13)]],  // [theta_proj]
    constant PreSsmParams& params      [[buffer(14)]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]]
) {
    const int d_inner = params.d_inner;
    const int bc_size = params.bc_size;
    const int n_heads = params.n_heads;
    const int d_state = params.d_state;
    const int n_groups = params.n_groups;
    const int heads_per_group = params.heads_per_group;
    const int half_ds = params.half_ds;
    const int theta_proj = params.theta_proj;
    const float eps = params.norm_eps;

    // ── Projected layout offsets ──
    // [x_ssm: d_inner][z: d_inner][b: bc_size][c: bc_size][dt: n_heads][lambda: n_heads][dd_a: n_heads][theta: theta_proj]
    device const float* x_ssm_ptr  = projected;
    device const float* z_ptr      = projected + d_inner;
    device const float* b_proj_ptr = projected + 2 * d_inner;
    device const float* c_proj_ptr = projected + 2 * d_inner + bc_size;
    device const float* dt_ptr     = projected + 2 * d_inner + 2 * bc_size;
    device const float* lam_ptr    = projected + 2 * d_inner + 2 * bc_size + n_heads;
    device const float* dda_ptr    = projected + 2 * d_inner + 2 * bc_size + 2 * n_heads;
    device const float* theta_ptr  = projected + 2 * d_inner + 2 * bc_size + 3 * n_heads;

    // ── Phase 1: SiLU(x_ssm) → x_heads_out, copy z → z_out ──
    // 1024 threads, each handles ~2 elements of d_inner=2048
    for (int i = tid; i < d_inner; i += 1024) {
        float xv = x_ssm_ptr[i];
        x_heads_out[i] = xv / (1.0f + fast::exp(-xv));  // SiLU
        z_out[i] = z_ptr[i];
    }

    // ── Phase 2: BCNorm for B and C in parallel ──
    // We have 32 SIMD groups (1024/32). Groups 0..n_groups-1 handle B, groups n_groups..2*n_groups-1 handle C.
    // Each SIMD group (32 threads) computes RMSNorm for one group of d_state elements.
    //
    // Shared memory for per-group normalized B/C (needed for expansion later)
    threadgroup float b_normed_shared[512]; // bc_size max
    threadgroup float c_normed_shared[512];

    if (simd_gid < (uint)n_groups) {
        // BCNorm for B: group simd_gid
        int group = simd_gid;
        int base = group * d_state;
        float local_ss = 0.0f;
        for (int s = simd_lane; s < d_state; s += 32) {
            float v = b_proj_ptr[base + s];
            local_ss += v * v;
        }
        float ss = simd_sum(local_ss);
        float scale = rsqrt(ss / float(d_state) + eps);
        for (int s = simd_lane; s < d_state; s += 32) {
            b_normed_shared[base + s] = b_proj_ptr[base + s] * scale * b_norm_gamma[base + s] + b_bias_vec[base + s];
        }
    } else if (simd_gid < (uint)(2 * n_groups)) {
        // BCNorm for C: group (simd_gid - n_groups)
        int group = simd_gid - n_groups;
        int base = group * d_state;
        float local_ss = 0.0f;
        for (int s = simd_lane; s < d_state; s += 32) {
            float v = c_proj_ptr[base + s];
            local_ss += v * v;
        }
        float ss = simd_sum(local_ss);
        float scale = rsqrt(ss / float(d_state) + eps);
        for (int s = simd_lane; s < d_state; s += 32) {
            c_normed_shared[base + s] = c_proj_ptr[base + s] * scale * c_norm_gamma[base + s] + c_bias_vec[base + s];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Per-head scalars (threads 0..n_heads-1) ──
    if (tid < (uint)n_heads) {
        int h = tid;
        // sigmoid(lambda)
        float lam_raw = lam_ptr[h];
        lambda_out[h] = 1.0f / (1.0f + fast::exp(-lam_raw));

        // dt + bias → softplus → dt_pos
        float dt_val = dt_ptr[h] + dt_bias_vec[h];
        float dt_sp = log(1.0f + fast::exp(dt_val));
        dt_pos_out[h] = dt_sp;

        // -softplus(dd_A) clamped to [-1e6, -1e-4]
        float dda = dda_ptr[h];
        float a_val = -log(1.0f + fast::exp(dda));
        a_val = clamp(a_val, -1e6f, -1e-4f);

        // a_bar = exp(A * dt_pos)
        a_bar_out[h] = fast::exp(a_val * dt_sp);
    }

    // ── Phase 4: B/C group→head expansion + theta passthrough ──
    // b_expanded[head * d_state + s] = b_normed[group * d_state + s] where group = head / heads_per_group
    int total_expand = n_heads * d_state;  // e.g. 4096
    for (int i = tid; i < total_expand; i += 1024) {
        int head = i / d_state;
        int s = i % d_state;
        int group = head / heads_per_group;
        b_expanded_out[i] = b_normed_shared[group * d_state + s];
        c_expanded_out[i] = c_normed_shared[group * d_state + s];
    }

    // Copy theta (just pass through — avoids a Candle narrow dispatch)
    for (int i = tid; i < theta_proj; i += 1024) {
        theta_out[i] = theta_ptr[i];
    }
}


/// Fused SSM step kernel for single-token Mamba-3 inference.
///
/// One threadgroup per head, head_dim threads per group.
/// Each thread handles one (head, p) pair, looping over d_state.
///
/// Fuses: DD-RoPE rotation, trapezoidal SSM update, output contraction.
/// Eliminates ~40 separate Metal kernel dispatches per layer per token.

struct SsmStepParams {
    int n_heads;
    int head_dim;
    int d_state;
    int half_ds;      // d_state / 2
};

kernel void ssm_step_fused(
    // Inputs (read-only)
    device const float* x_heads      [[buffer(0)]],  // [n_heads, head_dim]
    device const float* b_expanded   [[buffer(1)]],  // [n_heads, d_state]
    device const float* c_expanded   [[buffer(2)]],  // [n_heads, d_state]
    device const float* a_bar        [[buffer(3)]],  // [n_heads]
    device const float* lambda_vals  [[buffer(4)]],  // [n_heads]
    device const float* d_skip       [[buffer(5)]],  // [n_heads]
    device const float* theta_raw    [[buffer(6)]],  // [n_heads, half_ds] (DD-RoPE, may be unused)
    device const float* dt_pos       [[buffer(7)]],  // [n_heads] (DD-RoPE)
    // State (read-write)
    device float* h                  [[buffer(8)]],   // [n_heads, head_dim, d_state]
    device float* prev_bx            [[buffer(9)]],   // [n_heads, head_dim, d_state]
    device float* cum_angle          [[buffer(10)]],  // [n_heads, half_ds]
    // Output
    device float* y_heads            [[buffer(11)]],  // [n_heads, head_dim]
    // Params
    device const SsmStepParams& params [[buffer(12)]],
    // Z gate (SiLU gating folded into output write)
    device const float* z_gate       [[buffer(13)]],  // [n_heads, head_dim]
    // Thread IDs
    uint head_id [[threadgroup_position_in_grid]],
    uint p       [[thread_position_in_threadgroup]]
) {
    const int n_heads  = params.n_heads;
    const int head_dim = params.head_dim;
    const int d_state  = params.d_state;
    const int half_ds  = params.half_ds;

    if (head_id >= (uint)n_heads || p >= (uint)head_dim) return;

    // ── 1. Load per-head scalars ──
    float ab    = a_bar[head_id];
    float lam   = lambda_vals[head_id];
    float D_val = d_skip[head_id];
    float x_val = x_heads[head_id * head_dim + p];

    // ── 2. DD-RoPE: rotate B and C in-place (threadgroup-shared) ──
    // We need rotated B[head, s] and C[head, s] for all d_state values.
    // Only thread p=0 updates cum_angle (once per head), all threads read rotated B/C.
    // Threadgroup memory layout: [b_rot: d_state floats][c_rot: d_state floats]
    threadgroup float bc_shared[128]; // 2 * d_state (max d_state=64)
    threadgroup float* b_rot = bc_shared;
    threadgroup float* c_rot = bc_shared + d_state;

    // Thread 0 per head: update cumulative angles and compute rotated B/C
    if (p == 0) {
        float dtp = dt_pos[head_id];
        for (int k = 0; k < half_ds; k++) {
            float angle = fast::tanh(theta_raw[head_id * half_ds + k]) * M_PI_F;
            float ca_val = cum_angle[head_id * half_ds + k] + dtp * angle;
            ca_val = fmod(ca_val, 2.0f * M_PI_F);
            cum_angle[head_id * half_ds + k] = ca_val;

            float cs = fast::cos(ca_val);
            float sn = fast::sin(ca_val);

            int idx = head_id * d_state;
            float b0 = b_expanded[idx + 2*k];
            float b1 = b_expanded[idx + 2*k + 1];
            b_rot[2*k]     = cs * b0 - sn * b1;
            b_rot[2*k + 1] = sn * b0 + cs * b1;

            float c0 = c_expanded[idx + 2*k];
            float c1 = c_expanded[idx + 2*k + 1];
            c_rot[2*k]     = cs * c0 - sn * c1;
            c_rot[2*k + 1] = sn * c0 + cs * c1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 3. SSM state update + output contraction ──
    // h_base = head_id * head_dim * d_state + p * d_state
    int h_base = head_id * head_dim * d_state + p * d_state;
    float y_val = D_val * x_val; // skip connection

    float one_minus_lam = 1.0f - lam;

    for (int s = 0; s < d_state; s++) {
        float b_val = b_rot[s];
        float c_val = c_rot[s];
        float bx_val = b_val * x_val;
        int idx = h_base + s;

        // Trapezoidal: h = a_bar * h + (1-lambda) * a_bar * prev_bx + lambda * bx
        float h_new = ab * h[idx] + one_minus_lam * ab * prev_bx[idx] + lam * bx_val;
        h[idx] = h_new;
        prev_bx[idx] = bx_val;

        // Output contraction: y += C * h
        y_val += c_val * h_new;
    }

    // Fused SiLU gating: y = y_val * silu(z) = y_val * z * sigmoid(z)
    float z_val = z_gate[head_id * head_dim + p];
    float z_sig = 1.0f / (1.0f + fast::exp(-z_val));
    y_heads[head_id * head_dim + p] = y_val * z_val * z_sig;
}


/// Windowed SSM kernel for batched eval (perplexity, MC scoring).
///
/// Same parallelization as ssm_step_fused: one threadgroup per head, head_dim
/// threads per group. Each thread loops over (timestep, d_state) for its
/// owned (head, p) pair. h[head, p, :] and prev_bx[head, p, :] live in
/// per-thread private memory across the entire window — loaded once at the
/// start, written back once at the end. Eliminates per-token kernel launch
/// overhead and global state traffic.
struct SsmWindowParams {
    int n_heads;
    int head_dim;
    int d_state;
    int half_ds;      // d_state / 2
    int seq_len;
};

kernel void ssm_window_fused(
    device const float* x_seq        [[buffer(0)]],   // [seq, n_heads, head_dim]
    device const float* b_seq        [[buffer(1)]],   // [seq, n_heads, d_state]
    device const float* c_seq        [[buffer(2)]],   // [seq, n_heads, d_state]
    device const float* a_bar_seq    [[buffer(3)]],   // [seq, n_heads]
    device const float* lambda_seq   [[buffer(4)]],   // [seq, n_heads]
    device const float* d_skip       [[buffer(5)]],   // [n_heads]
    device const float* theta_seq    [[buffer(6)]],   // [seq, n_heads, half_ds] (may be unused)
    device const float* dt_pos_seq   [[buffer(7)]],   // [seq, n_heads]
    device float*       h            [[buffer(8)]],   // [n_heads, head_dim, d_state]   state
    device float*       prev_bx      [[buffer(9)]],   // [n_heads, head_dim, d_state]   state
    device float*       cum_angle    [[buffer(10)]],  // [n_heads, half_ds]              state
    device float*       y_seq        [[buffer(11)]],  // [seq, n_heads, head_dim]   output
    constant SsmWindowParams& params [[buffer(12)]],
    device const float* z_seq        [[buffer(13)]],  // [seq, n_heads, head_dim]  z gate
    uint head_id [[threadgroup_position_in_grid]],
    uint p       [[thread_position_in_threadgroup]]
) {
    const int n_heads  = params.n_heads;
    const int head_dim = params.head_dim;
    const int d_state  = params.d_state;
    const int half_ds  = params.half_ds;
    const int seq_len  = params.seq_len;

    if (head_id >= (uint)n_heads || p >= (uint)head_dim) return;

    // ── Load private state for this (head, p) pair ──
    // d_state is bounded at compile time; we statically allocate the max.
    float h_local[64];        // d_state ≤ 64 in current configs
    float prev_local[64];
    int h_base = head_id * head_dim * d_state + p * d_state;
    for (int s = 0; s < d_state; s++) {
        h_local[s]    = h[h_base + s];
        prev_local[s] = prev_bx[h_base + s];
    }

    float D_val = d_skip[head_id];

    // Threadgroup-shared rotated B/C for the *current* timestep.
    threadgroup float bc_shared[128]; // 2 * d_state (max d_state = 64)
    threadgroup float* b_rot = bc_shared;
    threadgroup float* c_rot = bc_shared + d_state;

    for (int t = 0; t < seq_len; t++) {
        // ── Per-timestep B/C rotation (thread 0 per head, others wait) ──
        if (p == 0) {
            float dtp = dt_pos_seq[t * n_heads + head_id];
            int theta_off = (t * n_heads + head_id) * half_ds;
            int bc_off    = (t * n_heads + head_id) * d_state;
            for (int k = 0; k < half_ds; k++) {
                float angle = fast::tanh(theta_seq[theta_off + k]) * M_PI_F;
                float ca_val = cum_angle[head_id * half_ds + k] + dtp * angle;
                ca_val = fmod(ca_val, 2.0f * M_PI_F);
                cum_angle[head_id * half_ds + k] = ca_val;

                float cs = fast::cos(ca_val);
                float sn = fast::sin(ca_val);

                float b0 = b_seq[bc_off + 2*k];
                float b1 = b_seq[bc_off + 2*k + 1];
                b_rot[2*k]     = cs * b0 - sn * b1;
                b_rot[2*k + 1] = sn * b0 + cs * b1;

                float c0 = c_seq[bc_off + 2*k];
                float c1 = c_seq[bc_off + 2*k + 1];
                c_rot[2*k]     = cs * c0 - sn * c1;
                c_rot[2*k + 1] = sn * c0 + cs * c1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── SSM step on private state ──
        float ab  = a_bar_seq[t * n_heads + head_id];
        float lam = lambda_seq[t * n_heads + head_id];
        float one_minus_lam = 1.0f - lam;
        float x_val = x_seq[(t * n_heads + head_id) * head_dim + p];
        float y_val = D_val * x_val;

        for (int s = 0; s < d_state; s++) {
            float bx_val = b_rot[s] * x_val;
            float h_new  = ab * h_local[s] + one_minus_lam * ab * prev_local[s] + lam * bx_val;
            h_local[s]    = h_new;
            prev_local[s] = bx_val;
            y_val += c_rot[s] * h_new;
        }

        // Fused SiLU gating: y = y_val * silu(z)
        int yz_idx = (t * n_heads + head_id) * head_dim + p;
        float z_val = z_seq[yz_idx];
        float z_sig = 1.0f / (1.0f + fast::exp(-z_val));
        y_seq[yz_idx] = y_val * z_val * z_sig;

        // Make sure all threads finished reading b_rot/c_rot before thread 0
        // overwrites them at the next timestep.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write private state back to global so the next window can resume ──
    for (int s = 0; s < d_state; s++) {
        h[h_base + s]       = h_local[s];
        prev_bx[h_base + s] = prev_local[s];
    }
}


// ─── Fused GEMV ──────────────────────────────────────────────────────────────
// Matrix-vector multiply: y = W @ x, where W is [out_dim, in_dim] row-major.
//
// Optimized for single-token inference on M4 Pro:
// - x loaded into threadgroup shared memory (read once, used by all rows)
// - Each SIMD group (32 threads) computes one output row via simd_sum
// - 8 SIMD groups per threadgroup → 8 output rows per threadgroup
// - W streamed from device memory with coalesced access

struct GemvParams {
    int out_dim;
    int in_dim;
};

kernel void gemv_fused(
    device const float* W       [[buffer(0)]],  // [out_dim, in_dim] row-major
    device const float* x       [[buffer(1)]],  // [in_dim]
    device float* y             [[buffer(2)]],  // [out_dim]
    constant GemvParams& params [[buffer(3)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]]
) {
    const int out_dim = params.out_dim;
    const int in_dim  = params.in_dim;

    // Each SIMD group handles one output row. 8 rows per threadgroup.
    int row = tg_id * 8 + simd_gid;
    if (row >= out_dim) return;

    // Dot product: W[row, :] · x — read x directly from device memory
    // (unified memory means x is likely in L2 after first threadgroup reads it)
    device const float* w_row = W + row * in_dim;
    float acc = 0.0f;
    for (int j = simd_lane; j < in_dim; j += 32) {
        acc += w_row[j] * x[j];
    }
    float dot = simd_sum(acc);

    if (simd_lane == 0) {
        y[row] = dot;
    }
}

// ─── Fused GEMV + Residual Add ───────────────────────────────────────────────
// y = W @ x + residual. Used for out_proj + residual skip connection.

kernel void gemv_residual_fused(
    device const float* W       [[buffer(0)]],  // [out_dim, in_dim] row-major
    device const float* x       [[buffer(1)]],  // [in_dim]
    device const float* residual[[buffer(2)]],  // [out_dim]
    device float* y             [[buffer(3)]],  // [out_dim]
    constant GemvParams& params [[buffer(4)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]]
) {
    const int out_dim = params.out_dim;
    const int in_dim  = params.in_dim;

    int row = tg_id * 8 + simd_gid;
    if (row >= out_dim) return;

    device const float* w_row = W + row * in_dim;
    float acc = 0.0f;
    for (int j = simd_lane; j < in_dim; j += 32) {
        acc += w_row[j] * x[j];
    }
    float dot = simd_sum(acc);

    if (simd_lane == 0) {
        y[row] = dot + residual[row];
    }
}
