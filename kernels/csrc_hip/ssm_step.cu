#include "hip/hip_runtime.h"
/*
 * Mamba-3 SSM single-token step -- HIP kernel (gfx1201/RDNA4, wave32).
 *
 * One timestep of the same recurrence `ssm_ssd_fwd_kernel` (ssm_scan.cu)
 * computes over a whole sequence, but with persistent state (h, prev_bx,
 * cum_angle) carried across calls instead of looped internally. Used for
 * autoregressive decoding: one call per generated token.
 *
 * Numerics are kept bit-for-bit identical to ssm_scan.cu's helpers (softplus,
 * cos_approx/sin_approx, exp2f-based a_bar/sigmoid, tanhf-bounded DD-RoPE) so
 * that stepping this kernel T times reproduces ssm_scan_fwd_gpu's output over
 * a T-length sequence -- see kernels/tests/ssm_step_equiv.rs.
 *
 * Grid: n_heads blocks. Block: head_dim threads (launcher-asserted
 * head_dim <= 1024, d_state == 64). Every thread reaches every barrier --
 * no early-return before a __syncthreads(), per METAL_TO_HIP.md 5.1: the
 * Metal original's early-return-then-barrier pattern is UB in HIP.
 */

#include <cmath>
#include <cstdio>

#define LOG2E 1.44269504089f

/* Hardcodes a 32-lane warp, matching ssm_scan.cu. RDNA (gfx10+) defaults to
 * wave32 for compute. Guarded to the device pass; the macro reads 64 during
 * host compilation. */
#ifdef __HIP_DEVICE_COMPILE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-pragma"
static_assert(__AMDGCN_WAVEFRONT_SIZE__ == 32,
              "These kernels assume a 32-lane wavefront; build with wave32 (RDNA default).");
#pragma clang diagnostic pop
#endif

/* --- Device helpers, copied verbatim from ssm_scan.cu for identical numerics --- */

__device__ __forceinline__ float cos_approx(float x) {
    return __cosf(x);
}
__device__ __forceinline__ float sin_approx(float x) {
    return __sinf(x);
}

__device__ __forceinline__ float softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + exp2f(x * LOG2E));
}

/* --- Single-timestep SSM step kernel ------------------------------------- */

__global__ void ssm_step_kernel(
    const float* __restrict__ x,        // [n_heads, head_dim]
    const float* __restrict__ dt,       // [n_heads] (raw)
    const float* __restrict__ b,        // [n_groups, d_state] (post-BCNorm)
    const float* __restrict__ c,        // [n_groups, d_state] (post-BCNorm)
    const float* __restrict__ d_skip,   // [n_heads]
    const float* __restrict__ dt_bias,  // [n_heads]
    const float* __restrict__ lambda,   // [n_heads] (post-sigmoid)
    const float* __restrict__ theta,    // [n_heads, d_state/2] (raw)
    const float* __restrict__ a_vals,   // [n_heads] (post neg-softplus-clamp)
    const float* __restrict__ z,        // [n_heads, head_dim] or nullptr
    float* __restrict__ h,              // [n_heads, head_dim, d_state] (state, in place)
    float* __restrict__ prev_bx,        // [n_heads, head_dim, d_state] (state, in place)
    float* __restrict__ cum_angle,      // [n_heads, d_state/2] (state, in place)
    float* __restrict__ y,              // [n_heads, head_dim]
    int n_heads, int head_dim, int d_state, int n_groups
) {
    int head_id = blockIdx.x;
    int tid = threadIdx.x;
    int half_d_state = d_state / 2;
    int group = head_id / (n_heads / n_groups);

    // Per-head scalars, computed redundantly by every thread from the raw
    // inputs -- cheaper than a broadcast, matches the training kernel.
    float dt_raw = dt[head_id];
    float dt_bias_val = dt_bias[head_id];
    float dt_pos = softplus(dt_raw + dt_bias_val);

    float A_val = a_vals[head_id];
    float a_bar = exp2f(A_val * dt_pos * LOG2E);

    float lam = lambda[head_id];
    float beta = (1.0f - lam) * a_bar;
    float gamma_val = lam;

    float D_val = d_skip[head_id];

    // d_state == 64 is enforced by the launcher; these are sized for the max.
    __shared__ float b_rot[64];
    __shared__ float c_rot[64];

    int theta_head_offset = head_id * half_d_state;
    int b_group_offset = group * d_state;
    const float TWO_PI = 6.28318530f;

    // DD-RoPE: update cum_angle[k] BEFORE rotating B/C for this timestep
    // (matches ssm_ssd_fwd_kernel's ordering exactly). Each k is owned by
    // exactly one thread (strided loop, robust for any head_dim), so the
    // read-modify-write to global cum_angle has no race. B and C use the
    // same just-updated angle, so fusing their rotation into one loop
    // (rather than ssm_scan.cu's two separate passes) is equivalent.
    for (int k = tid; k < half_d_state; k += blockDim.x) {
        float tv = tanhf(theta[theta_head_offset + k]) * 3.14159265f;
        float ca_val = fmodf(cum_angle[theta_head_offset + k] + dt_pos * tv, TWO_PI);
        cum_angle[theta_head_offset + k] = ca_val;

        float ca = cos_approx(ca_val), sa = sin_approx(ca_val);

        float b0 = b[b_group_offset + 2 * k], b1 = b[b_group_offset + 2 * k + 1];
        b_rot[2 * k]     = ca * b0 - sa * b1;
        b_rot[2 * k + 1] = sa * b0 + ca * b1;

        float c0 = c[b_group_offset + 2 * k], c1 = c[b_group_offset + 2 * k + 1];
        c_rot[2 * k]     = ca * c0 - sa * c1;
        c_rot[2 * k + 1] = sa * c0 + ca * c1;
    }
    // Every thread of the block reaches this barrier -- no thread has
    // returned early (deliberate: block size == head_dim exactly).
    __syncthreads();

    // Main loop: thread p owns (head_id, p), loops over d_state.
    int p = tid;
    float x_val = x[head_id * head_dim + p];
    int h_base = head_id * head_dim * d_state + p * d_state;
    float y_val = D_val * x_val;

    for (int s = 0; s < d_state; s++) {
        float bx = b_rot[s] * x_val;
        int idx = h_base + s;
        float h_new = a_bar * h[idx] + beta * prev_bx[idx] + gamma_val * bx;
        h[idx] = h_new;
        prev_bx[idx] = bx;
        y_val += c_rot[s] * h_new;
    }

    int out_idx = head_id * head_dim + p;
    if (z != nullptr) {
        // Fused SiLU gate: y = y_val * z * sigmoid(z). Same exp2f/LOG2E form
        // ssm_ssd_fwd_kernel uses for y_gated, so numerics match exactly.
        float z_val = z[out_idx];
        float sig_z = 1.0f / (1.0f + exp2f(-z_val * LOG2E));
        y[out_idx] = y_val * z_val * sig_z;
    } else {
        y[out_idx] = y_val;
    }
}

/* --- Host API ------------------------------------------------------------- */

extern "C"
void ssm_step_gpu(
    const float* x, const float* dt, const float* b, const float* c,
    const float* d_skip, const float* dt_bias,
    const float* lambda, const float* theta, const float* a_vals,
    const float* z,
    float* h, float* prev_bx, float* cum_angle,
    float* y,
    int n_heads, int head_dim, int d_state, int n_groups
) {
    // d_state == 64 is load-bearing: b_rot/c_rot are __shared__ float[64],
    // and the DD-RoPE loop assumes d_state/2 == 32 pairs. Fail loudly rather
    // than silently corrupt (METAL_TO_HIP.md 5.6).
    if (d_state != 64) {
        fprintf(stderr,
            "ERROR: ssm_step_gpu requires d_state == 64 (got %d); the DD-RoPE "
            "shared rotation buffers are sized for 64.\n", d_state);
        return;
    }
    if (head_dim <= 0 || head_dim > 1024) {
        fprintf(stderr,
            "ERROR: ssm_step_gpu requires 0 < head_dim <= 1024 (got %d); "
            "exceeds max threads per block.\n", head_dim);
        return;
    }

    ssm_step_kernel<<<n_heads, head_dim>>>(
        x, dt, b, c, d_skip, dt_bias, lambda, theta, a_vals, z,
        h, prev_bx, cum_angle, y,
        n_heads, head_dim, d_state, n_groups
    );
}
