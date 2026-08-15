#ifndef SSM_SCAN_H
#define SSM_SCAN_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * SSD forward: one block per (batch, head), sequential over timesteps.
 */
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
);

/*
 * Backward pass (sequential scan with chunked recomputation).
 * Grid: batch * n_heads blocks.
 */
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
    float* d_d_lambda, float* d_d_h_init,
    float* d_d_theta, float* d_d_A_vals,
    float* d_dD, float* d_d_dt_bias,
    float* ws_dD_buf, float* ws_d_dtb_buf,
    int batch, int seq, int n_heads, int head_dim, int d_state, int n_groups,
    int chunk_size
);

#ifdef __cplusplus
}
#endif

/*
 * Shared device math helpers, used by both ssm_scan.cu and ssm_step.cu.
 * Living here (single definition) is what keeps the two kernels' numerics
 * bit-for-bit identical -- the property kernels/tests/ssm_step_equiv.rs
 * verifies. Device-only: guarded so a plain host compiler including this
 * header for the extern "C" declarations above is unaffected.
 */
#ifdef __HIPCC__

#define LOG2E 1.44269504089f

/* These kernels hardcode a 32-lane warp. RDNA (gfx10+) defaults to wave32 for
 * compute, so the shuffle reductions are correct as written -- but would break
 * silently if ever built for wave64. Guarded to the device pass; the macro
 * reads 64 during host compilation. */
#ifdef __HIP_DEVICE_COMPILE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-pragma"
static_assert(__AMDGCN_WAVEFRONT_SIZE__ == 32,
              "These kernels assume a 32-lane wavefront; build with wave32 (RDNA default).");
#pragma clang diagnostic pop
#endif

/* Was PTX cos.approx.f32 / sin.approx.f32 in the CUDA original. __cosf/__sinf
 * are the HIP fast-math equivalents, lowering to the v_cos_f32/v_sin_f32
 * hardware instructions. */
__device__ __forceinline__ float cos_approx(float x) {
    return __cosf(x);
}
__device__ __forceinline__ float sin_approx(float x) {
    return __sinf(x);
}

__device__ __forceinline__ float softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + exp2f(x * LOG2E));
}

#endif /* __HIPCC__ */

#endif /* SSM_SCAN_H */
