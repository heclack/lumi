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

#endif /* SSM_SCAN_H */
