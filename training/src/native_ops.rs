// FFI bindings for all native CUDA kernels: cuBLAS, element-wise, AdamW.
// These bypass Burn entirely — raw GPU pointer operations.

#[cfg(feature = "cuda")]
extern "C" {
    // ─── CUDA runtime ───
    pub fn cudaDeviceSynchronize() -> i32;

    // ─── cuBLAS ───
    pub fn cublas_init();
    pub fn cublas_destroy();

    /// C = A @ B (row-major: A[m,k] × B[k,n] → C[m,n])
    pub fn matmul_f32(A: *const f32, B: *const f32, C: *mut f32, m: i32, n: i32, k: i32);

    /// C = A @ B^T (row-major: A[m,k] × B[n,k]^T → C[m,n])
    pub fn matmul_f32_bt(A: *const f32, B: *const f32, C: *mut f32, m: i32, n: i32, k: i32);

    /// C = A^T @ B (row-major: A[k,m]^T × B[k,n] → C[m,n])
    pub fn matmul_f32_at(A: *const f32, B: *const f32, C: *mut f32, m: i32, n: i32, k: i32);

    /// C += A^T @ B (accumulate weight gradients across micro-batches)
    pub fn matmul_f32_at_accum(A: *const f32, B: *const f32, C: *mut f32, m: i32, n: i32, k: i32);

    /// C += A @ B (accumulate)
    pub fn matmul_f32_accum(A: *const f32, B: *const f32, C: *mut f32, m: i32, n: i32, k: i32);

    /// Batched: C[i] = A[i] @ B[i], strides = m*k, k*n, m*n
    pub fn matmul_f32_batched(A: *const f32, B: *const f32, C: *mut f32,
                               m: i32, n: i32, k: i32, batch_count: i32);
    /// Batched: C[i] = A[i] @ B[i]^T
    pub fn matmul_f32_bt_batched(A: *const f32, B: *const f32, C: *mut f32,
                                  m: i32, n: i32, k: i32, batch_count: i32);
    /// Batched: C[i] = A[i]^T @ B[i]
    pub fn matmul_f32_at_batched(A: *const f32, B: *const f32, C: *mut f32,
                                  m: i32, n: i32, k: i32, batch_count: i32);

    // ─── RMSNorm ───
    pub fn rmsnorm_fwd(x: *const f32, out: *mut f32, gamma: *const f32,
                       eps: f32, rows: i32, dim: i32);
    pub fn rmsnorm_bwd(x: *const f32, dy: *const f32, gamma: *const f32,
                       dx: *mut f32, dgamma: *mut f32, eps: f32, rows: i32, dim: i32);

    // ─── Fused RMSNorm + Bias ───
    pub fn rmsnorm_bias_fwd(x: *const f32, out: *mut f32, gamma: *const f32, bias: *const f32,
                            eps: f32, rows: i32, dim: i32);
    pub fn rmsnorm_bias_bwd(x: *const f32, dy: *const f32, gamma: *const f32,
                            dx: *mut f32, dgamma: *mut f32, dbias: *mut f32,
                            eps: f32, rows: i32, dim: i32);

    // ─── SiLU ───
    pub fn silu_fwd(x: *const f32, y: *mut f32, n: i32);
    pub fn silu_bwd(x: *const f32, dy: *const f32, dx: *mut f32, n: i32);

    // ─── Sigmoid ───
    pub fn sigmoid_fwd(x: *const f32, y: *mut f32, n: i32);
    pub fn sigmoid_bwd(x: *const f32, dy: *const f32, dx: *mut f32, n: i32);

    // ─── Fused SiLU-Gate (y = ssm_out * SiLU(z)) ───
    pub fn fused_silu_gate_fwd(ssm_out: *const f32, z: *const f32, y: *mut f32, n: i32);
    pub fn fused_silu_gate_bwd(ssm_out: *const f32, z: *const f32, dy: *const f32,
                                d_ssm_out: *mut f32, d_z: *mut f32, n: i32);

    // ─── Causal Softmax (for attention) ───
    pub fn causal_softmax_fwd(scores: *const f32, output: *mut f32, batch_heads: i32, seq: i32, window_size: i32);
    pub fn causal_softmax_bwd(output: *const f32, d_output: *const f32, d_scores: *mut f32,
                               batch_heads: i32, seq: i32, window_size: i32);

    // ─── GQA expand/reduce ───
    pub fn gqa_expand(src: *const f32, dst: *mut f32,
                      batch: i32, kv_heads: i32, n_heads: i32, seq: i32, head_dim: i32);
    pub fn gqa_reduce(src: *const f32, dst: *mut f32,
                      batch: i32, kv_heads: i32, n_heads: i32, seq: i32, head_dim: i32);

    // ─── Scale tensor ───
    pub fn scale_tensor(x: *const f32, y: *mut f32, scale: f32, n: i32);

    // ─── Transpose [batch, d1, d2, d3] → [batch, d2, d1, d3] ───
    pub fn transpose_0213(src: *const f32, dst: *mut f32,
                          batch: i32, dim1: i32, dim2: i32, dim3: i32);

    // ─── Element-wise ───
    pub fn elemwise_mul(a: *const f32, b: *const f32, y: *mut f32, n: i32);
    pub fn elemwise_add(a: *const f32, b: *const f32, y: *mut f32, n: i32);
    pub fn elemwise_mul_bwd(a: *const f32, b: *const f32, dy: *const f32,
                            da: *mut f32, db: *mut f32, n: i32);

    // ─── Broadcast bias add ───
    pub fn bias_add_fwd(x: *const f32, bias: *const f32, y: *mut f32, rows: i32, dim: i32);
    pub fn bias_add_bwd(dy: *const f32, d_bias: *mut f32, rows: i32, dim: i32);

    // ─── Strided split/assemble (column range ops on row-major matrix) ───
    pub fn strided_split(src: *const f32, dst: *mut f32,
                         rows: i32, total_cols: i32, col_offset: i32, split_cols: i32);
    pub fn strided_assemble(src: *const f32, dst: *mut f32,
                            rows: i32, total_cols: i32, col_offset: i32, split_cols: i32);
    pub fn fused_split_5(
        src: *const f32,
        dst0: *mut f32, dst1: *mut f32, dst2: *mut f32, dst3: *mut f32, dst4: *mut f32,
        rows: i32, total_cols: i32,
        off0: i32, len0: i32, off1: i32, len1: i32,
        off2: i32, len2: i32, off3: i32, len3: i32, off4: i32, len4: i32,
    );

    // ─── Embedding ───
    pub fn embedding_lookup(embedding: *const f32, token_ids: *const i32,
                            out: *mut f32, batch_seq: i32, dim: i32);
    pub fn embedding_bwd(dy: *const f32, token_ids: *const i32,
                         d_embedding: *mut f32, batch_seq: i32, dim: i32, vocab: i32);

    // ─── Loss ───
    pub fn sparse_cross_entropy_fwd(logits: *const f32, targets: *const i32,
                                     loss: *mut f32, per_token_loss: *mut f32,
                                     batch_seq: i32, vocab: i32);
    pub fn sparse_cross_entropy_bwd(logits: *const f32, targets: *const i32,
                                     d_logits: *mut f32, batch_seq: i32, vocab: i32);

    // ─── AdamW ───
    pub fn adamw_update(w: *mut f32, grad: *const f32, m: *mut f32, v: *mut f32,
                        lr: f32, beta1: f32, beta2: f32, eps: f32,
                        weight_decay: f32, step: i32, n: i32);

    pub fn adamw_clipped_update(w: *mut f32, grad: *const f32, m: *mut f32, v: *mut f32,
                                total_norm_sq_ptr: *const f32, max_norm: f32,
                                lr: f32, beta1: f32, beta2: f32, eps: f32,
                                weight_decay: f32, step: i32, n: i32);

    // ─── Softplus ───
    pub fn softplus_fwd(x: *const f32, y: *mut f32, n: i32);
    pub fn softplus_bwd(x: *const f32, dy: *const f32, dx: *mut f32, n: i32);

    // ─── Negate ───
    pub fn negate(x: *const f32, y: *mut f32, n: i32);

    // ─── Fused neg-softplus-clamp (for data-dependent A) ───
    pub fn neg_softplus_clamp(x: *const f32, y: *mut f32, min_val: f32, max_val: f32, n: i32);

    // ─── Gradient clipping ───
    /// Compute sum of squares of buffer elements → output[0]
    pub fn grad_norm_squared(buf: *const f32, output: *mut f32, n: i32);
    /// Scale all elements: buf[i] *= scale
    pub fn grad_scale(buf: *mut f32, scale: f32, n: i32);
    /// Clamp values: buf[i] = clamp(buf[i], min_val, max_val)
    pub fn clamp_values(buf: *mut f32, min_val: f32, max_val: f32, n: i32);

    // ─── SSM Chunked SSD Forward ───
    pub fn ssm_scan_fwd_gpu(
        x: *const f32, dt: *const f32, b: *const f32, c: *const f32,
        d: *const f32, dt_bias: *const f32,
        h_init: *const f32,
        lambda_in: *const f32,
        theta: *const f32,
        a_vals: *const f32,
        z_in: *const f32,
        y: *mut f32,
        y_gated: *mut f32,
        batch: i32, seq: i32, n_heads: i32, head_dim: i32, d_state: i32, n_groups: i32,
    );
    /// GPU-direct backward with pre-allocated workspace (no cudaMalloc per call)
    pub fn ssm_scan_bwd_gpu_v2(
        x: *const f32, dt: *const f32, b: *const f32, c: *const f32,
        d: *const f32, dt_bias: *const f32,
        dy: *const f32,
        lambda_in: *const f32,
        h_init: *const f32,
        h_checkpoints: *mut f32, pbx_checkpoints: *mut f32,
        h_saved_buf: *mut f32, pbx_saved_buf: *mut f32,
        theta: *const f32,    // DD-RoPE: [batch, seq, n_heads, d_state/2]
        a_vals: *const f32,   // Data-dependent A: [batch, seq, n_heads] or null
        dx: *mut f32, ddt: *mut f32, db: *mut f32, dc: *mut f32,
        d_lambda: *mut f32, d_h_init: *mut f32, d_theta: *mut f32,
        d_a_vals: *mut f32,   // Gradient for A_vals: [batch, seq, n_heads] or null
        dD: *mut f32, d_dt_bias: *mut f32,
        ws_dD: *mut f32, ws_d_dtb: *mut f32,
        batch: i32, seq: i32, n_heads: i32, head_dim: i32, d_state: i32, n_groups: i32,
        chunk_size: i32,
    );
}

// Stubs for non-CUDA builds
#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub unsafe fn cublas_init() { panic!("CUDA not available") }
#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub unsafe fn cublas_destroy() { panic!("CUDA not available") }
