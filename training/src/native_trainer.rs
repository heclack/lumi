/// Native CUDA training loop — no Burn framework.
///
/// All operations run on GPU via cuBLAS + custom CUDA kernels.
/// Zero CPU-GPU transfers during training (except loss logging).

#[cfg(feature = "cuda")]
use crate::config::TrainingConfig;
#[cfg(feature = "cuda")]
use crate::gpu_memory::{TrainingBuffers, GpuBuf, LayerWeights};
#[cfg(feature = "cuda")]
use crate::native_ops;
#[cfg(feature = "cuda")]
use crate::data::TokenDataset;
#[cfg(feature = "cuda")]

#[cfg(feature = "cuda")]
fn chrono_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    format!("{}", secs)
}

/// Mixed-precision dispatch macros. When `mp` is true, uses BF16 tensor-core matmuls
/// (FP32 inputs → BF16 scratch → cublasGemmEx → FP32 output). When false, uses FP32 TF32.
#[cfg(feature = "cuda")]
macro_rules! mm {
    ($mp:expr, $buf:expr, $A:expr, $B:expr, $C:expr, $m:expr, $n:expr, $k:expr) => {
        if $mp {
            native_ops::matmul_bf16_from_f32($A, $B, $C, $buf.bf16_scratch_a.ptr, $buf.bf16_scratch_b.ptr, $m, $n, $k)
        } else {
            native_ops::matmul_f32($A, $B, $C, $m, $n, $k)
        }
    };
}
#[cfg(feature = "cuda")]
macro_rules! mm_bt {
    ($mp:expr, $buf:expr, $A:expr, $B:expr, $C:expr, $m:expr, $n:expr, $k:expr) => {
        if $mp {
            native_ops::matmul_bf16_from_f32_bt($A, $B, $C, $buf.bf16_scratch_a.ptr, $buf.bf16_scratch_b.ptr, $m, $n, $k)
        } else {
            native_ops::matmul_f32_bt($A, $B, $C, $m, $n, $k)
        }
    };
}
#[cfg(feature = "cuda")]
macro_rules! mm_at_accum {
    ($mp:expr, $buf:expr, $A:expr, $B:expr, $C:expr, $m:expr, $n:expr, $k:expr) => {
        if $mp {
            native_ops::matmul_bf16_from_f32_at_accum($A, $B, $C, $buf.bf16_scratch_a.ptr, $buf.bf16_scratch_b.ptr, $m, $n, $k)
        } else {
            native_ops::matmul_f32_at_accum($A, $B, $C, $m, $n, $k)
        }
    };
}
#[cfg(feature = "cuda")]
macro_rules! mm_batched {
    ($mp:expr, $buf:expr, $A:expr, $B:expr, $C:expr, $m:expr, $n:expr, $k:expr, $batch:expr) => {
        if $mp {
            native_ops::matmul_bf16_from_f32_batched($A, $B, $C, $buf.bf16_scratch_a.ptr, $buf.bf16_scratch_b.ptr, $m, $n, $k, $batch)
        } else {
            native_ops::matmul_f32_batched($A, $B, $C, $m, $n, $k, $batch)
        }
    };
}
#[cfg(feature = "cuda")]
macro_rules! mm_bt_batched {
    ($mp:expr, $buf:expr, $A:expr, $B:expr, $C:expr, $m:expr, $n:expr, $k:expr, $batch:expr) => {
        if $mp {
            native_ops::matmul_bf16_from_f32_bt_batched($A, $B, $C, $buf.bf16_scratch_a.ptr, $buf.bf16_scratch_b.ptr, $m, $n, $k, $batch)
        } else {
            native_ops::matmul_f32_bt_batched($A, $B, $C, $m, $n, $k, $batch)
        }
    };
}
#[cfg(feature = "cuda")]
macro_rules! mm_at_batched {
    ($mp:expr, $buf:expr, $A:expr, $B:expr, $C:expr, $m:expr, $n:expr, $k:expr, $batch:expr) => {
        if $mp {
            native_ops::matmul_bf16_from_f32_at_batched($A, $B, $C, $buf.bf16_scratch_a.ptr, $buf.bf16_scratch_b.ptr, $m, $n, $k, $batch)
        } else {
            native_ops::matmul_f32_at_batched($A, $B, $C, $m, $n, $k, $batch)
        }
    };
}

/// Destination pointers for Mamba block intermediate values.
/// Training points these to per-layer saved buffers; validation points to scratch.
#[cfg(feature = "cuda")]
struct MambaForwardDst {
    residual: *mut f32,
    x_norm: *mut f32,
    x_ssm_raw: *mut f32,
    x_act: *mut f32,
    z_buf: *mut f32,
    b_raw: *mut f32,
    c_raw: *mut f32,
    b_norm: *mut f32,
    c_norm: *mut f32,
    dt_buf: *mut f32,
    lambda_raw: *mut f32,
    dd_a_raw: *mut f32,
    theta_raw: *mut f32,
    ssm_out: *mut f32,
    y_gated: *mut f32,
}

/// Shared attention block forward pass. Used by both training and validation.
/// When `save` is Some, copies intermediates for backward. When None, inference-only.
#[cfg(feature = "cuda")]
unsafe fn attention_block_forward(
    x: *mut f32,
    buf: &TrainingBuffers,
    weights: &crate::gpu_memory::AttentionLayerWeights,
    save: Option<&crate::gpu_memory::AttentionPerLayerSaved>,
    bs: i32, d_model: i32, batch: i32, seq: i32,
    attn_n_heads: i32, attn_kv_heads: i32, attn_head_dim: i32,
    kv_dim: i32, mlp_dim: i32, window_size: i32, eps: f32,
    mp: bool,
) {
    let bnh = batch * attn_n_heads;
    let scale = 1.0f32 / (attn_head_dim as f32).sqrt();

    // Save residual
    let residual = if let Some(s) = save { s.residual.ptr } else { buf.block_out.ptr };
    cuda_memcpy_d2d(residual, x, (bs * d_model) as usize);

    // Pre-attention RMSNorm
    native_ops::rmsnorm_fwd(x, buf.x_norm.ptr, weights.attn_norm_gamma.ptr, eps, bs, d_model);
    if let Some(s) = save { cuda_memcpy_d2d(s.x_norm.ptr, buf.x_norm.ptr, (bs * d_model) as usize); }

    // Q/K/V projections
    mm!(mp, buf, buf.x_norm.ptr, weights.q_proj.ptr, buf.attn_q.ptr, bs, d_model, d_model);
    mm!(mp, buf, buf.x_norm.ptr, weights.k_proj.ptr, buf.attn_k.ptr, bs, kv_dim, d_model);
    mm!(mp, buf, buf.x_norm.ptr, weights.v_proj.ptr, buf.attn_v.ptr, bs, kv_dim, d_model);

    if let Some(s) = save {
        cuda_memcpy_d2d(s.q.ptr, buf.attn_q.ptr, (bs * d_model) as usize);
        cuda_memcpy_d2d(s.k.ptr, buf.attn_k.ptr, (bs * kv_dim) as usize);
        cuda_memcpy_d2d(s.v.ptr, buf.attn_v.ptr, (bs * kv_dim) as usize);
    }

    // GQA: transpose + expand K/V
    native_ops::transpose_0213(buf.attn_q.ptr, buf.attn_q_t.ptr, batch, seq, attn_n_heads, attn_head_dim);
    native_ops::transpose_0213(buf.attn_k.ptr, buf.attn_kv_temp.ptr, batch, seq, attn_kv_heads, attn_head_dim);
    native_ops::gqa_expand(buf.attn_kv_temp.ptr, buf.attn_k_t.ptr, batch, attn_kv_heads, attn_n_heads, seq, attn_head_dim);
    native_ops::transpose_0213(buf.attn_v.ptr, buf.attn_kv_temp.ptr, batch, seq, attn_kv_heads, attn_head_dim);
    native_ops::gqa_expand(buf.attn_kv_temp.ptr, buf.attn_v_t.ptr, batch, attn_kv_heads, attn_n_heads, seq, attn_head_dim);

    // Attention: scores → softmax → context
    mm_bt_batched!(mp, buf, buf.attn_q_t.ptr, buf.attn_k_t.ptr, buf.attn_scores.ptr,
        seq, seq, attn_head_dim, bnh);
    native_ops::scale_tensor(buf.attn_scores.ptr, buf.attn_scores.ptr,
        scale, batch * attn_n_heads * seq * seq);
    native_ops::causal_softmax_fwd(buf.attn_scores.ptr, buf.attn_weights.ptr, bnh, seq, window_size);
    mm_batched!(mp, buf, buf.attn_weights.ptr, buf.attn_v_t.ptr, buf.attn_context_t.ptr,
        seq, attn_head_dim, seq, bnh);
    native_ops::transpose_0213(buf.attn_context_t.ptr, buf.attn_context.ptr, batch, attn_n_heads, seq, attn_head_dim);

    // Output projection + residual
    mm!(mp, buf, buf.attn_context.ptr, weights.attn_out_proj.ptr, buf.block_out.ptr, bs, d_model, d_model);
    native_ops::elemwise_add(buf.block_out.ptr, residual, x, bs * d_model);

    if let Some(s) = save {
        cuda_memcpy_d2d(s.attn_weights.ptr, buf.attn_weights.ptr, (batch * attn_n_heads * seq * seq) as usize);
        cuda_memcpy_d2d(s.context.ptr, buf.attn_context.ptr, (bs * d_model) as usize);
    }

    // MLP: save residual
    let mlp_residual = if let Some(s) = save { s.mlp_input.ptr } else { buf.block_out.ptr };
    cuda_memcpy_d2d(mlp_residual, x, (bs * d_model) as usize);

    // MLP pre-norm
    native_ops::rmsnorm_fwd(x, buf.x_norm.ptr, weights.mlp_norm_gamma.ptr, eps, bs, d_model);
    if let Some(s) = save { cuda_memcpy_d2d(s.mlp_norm_out.ptr, buf.x_norm.ptr, (bs * d_model) as usize); }

    // MLP gate + up projections
    mm!(mp, buf, buf.x_norm.ptr, weights.mlp_gate.ptr, buf.attn_mlp_hidden.ptr, bs, mlp_dim, d_model);
    mm!(mp, buf, buf.x_norm.ptr, weights.mlp_up.ptr, buf.attn_mlp_gated.ptr, bs, mlp_dim, d_model);

    if let Some(s) = save {
        cuda_memcpy_d2d(s.gate_raw.ptr, buf.attn_mlp_hidden.ptr, (bs * mlp_dim) as usize);
        cuda_memcpy_d2d(s.up_out.ptr, buf.attn_mlp_gated.ptr, (bs * mlp_dim) as usize);
    }

    // SwiGLU: sigmoid(gate) * up
    native_ops::sigmoid_fwd(buf.attn_mlp_hidden.ptr, buf.attn_mlp_hidden.ptr, bs * mlp_dim);
    if let Some(s) = save { cuda_memcpy_d2d(s.gate_act.ptr, buf.attn_mlp_hidden.ptr, (bs * mlp_dim) as usize); }
    native_ops::elemwise_mul(buf.attn_mlp_hidden.ptr, buf.attn_mlp_gated.ptr, buf.attn_mlp_gated.ptr, bs * mlp_dim);
    if let Some(s) = save { cuda_memcpy_d2d(s.y_gated.ptr, buf.attn_mlp_gated.ptr, (bs * mlp_dim) as usize); }

    // MLP down + residual
    mm!(mp, buf, buf.attn_mlp_gated.ptr, weights.mlp_down.ptr, buf.block_out.ptr, bs, d_model, mlp_dim);
    native_ops::elemwise_add(buf.block_out.ptr, mlp_residual, x, bs * d_model);
}

/// Shared Mamba block forward pass. Used by both training and validation.
#[cfg(feature = "cuda")]
unsafe fn mamba_block_forward(
    x: *mut f32,
    projected: *mut f32,
    lambda_buf: *mut f32,
    a_vals_buf: *mut f32,
    block_out: *mut f32,
    dst: &MambaForwardDst,
    weights: &LayerWeights,
    buf: &TrainingBuffers,
    bs: i32, d_model: i32, d_inner: i32, in_proj_out: i32,
    n_heads: i32, n_groups: i32, d_state: i32,
    bc_size: i32, batch: i32, seq: i32, head_dim: i32,
    eps: f32,
    mp: bool,
) {
    // Save residual
    cuda_memcpy_d2d(dst.residual, x, (bs * d_model) as usize);

    // RMSNorm
    native_ops::rmsnorm_fwd(x, dst.x_norm, weights.norm_gamma.ptr, eps, bs, d_model);

    // in_proj
    mm!(mp, buf, dst.x_norm, weights.in_proj.ptr, projected, bs, in_proj_out, d_model);

    // Fused 5-way split
    native_ops::fused_split_5(
        projected,
        dst.x_ssm_raw, dst.z_buf, dst.b_raw, dst.c_raw, dst.dt_buf,
        bs, in_proj_out,
        0, d_inner, d_inner, d_inner, d_inner*2, bc_size, d_inner*2+bc_size, bc_size, d_inner*2+bc_size*2, n_heads,
    );

    // Lambda: split + sigmoid
    let lambda_offset = d_inner * 2 + bc_size * 2 + n_heads;
    native_ops::strided_split(projected, dst.lambda_raw, bs, in_proj_out, lambda_offset, n_heads);
    native_ops::sigmoid_fwd(dst.lambda_raw, lambda_buf, bs * n_heads);

    // dd_A: split + neg_softplus_clamp
    let dd_a_offset = lambda_offset + n_heads;
    native_ops::strided_split(projected, dst.dd_a_raw, bs, in_proj_out, dd_a_offset, n_heads);
    native_ops::neg_softplus_clamp(dst.dd_a_raw, a_vals_buf, -1e6, -1e-4, bs * n_heads);

    // DD-RoPE: theta split
    let theta_offset = dd_a_offset + n_heads;
    let theta_size = n_heads * d_state / 2;
    native_ops::strided_split(projected, dst.theta_raw, bs, in_proj_out, theta_offset, theta_size);

    // SiLU(x) — in-place OK when x_ssm_raw == x_act (validation)
    native_ops::silu_fwd(dst.x_ssm_raw, dst.x_act, bs * d_inner);

    // BCNorm + bias
    native_ops::rmsnorm_bias_fwd(dst.b_raw, dst.b_norm, weights.b_gamma.ptr, weights.b_bias.ptr, eps, bs, bc_size);
    native_ops::rmsnorm_bias_fwd(dst.c_raw, dst.c_norm, weights.c_gamma.ptr, weights.c_bias.ptr, eps, bs, bc_size);

    // SSM scan with fused Z-gate
    native_ops::ssm_scan_fwd_gpu(
        dst.x_act, dst.dt_buf, dst.b_norm, dst.c_norm,
        weights.d_skip.ptr, weights.dt_bias.ptr, weights.h_init.ptr,
        lambda_buf, dst.theta_raw, a_vals_buf, dst.z_buf,
        dst.ssm_out, dst.y_gated,
        batch, seq, n_heads, head_dim, d_state, n_groups,
    );

    // out_proj
    mm!(mp, buf, dst.y_gated, weights.out_proj.ptr, block_out, bs, d_model, d_inner);

    // Residual add
    native_ops::elemwise_add(block_out, dst.residual, x, bs * d_model);
}

/// Run native CUDA training.
#[cfg(feature = "cuda")]
pub fn train_native(config: &TrainingConfig, device_id: i32) {
    eprintln!("=== Lumi — Native CUDA Training ===");
    eprintln!("Pure cuBLAS + custom CUDA kernels");

    // Initialize CUDA device + cuBLAS
    unsafe {
        // Set CUDA device
        let err = cuda_set_device(device_id);
        if err != 0 {
            panic!("cudaSetDevice({}) failed: {}", device_id, err);
        }
        eprintln!("CUDA device {} initialized", device_id);
        native_ops::cublas_init();
        eprintln!("cuBLAS initialized");
    }

    let batch = config.batch_size;
    let seq = if config.seq_len > 0 { config.seq_len } else { config.model.max_seq_len };
    let bs = batch * seq;
    let d_model = config.model.d_model;
    let d_inner = config.model.d_inner();
    let n_heads = config.model.n_heads;
    let n_groups = config.model.n_groups;
    let d_state = config.model.d_state;
    let n_layers = config.model.n_layers;
    let vocab = config.model.vocab_size;
    let head_dim = d_inner / n_heads;
    let eps = config.model.norm_eps as f32;
    let bc_size = n_groups * d_state;
    let theta_proj_size = n_heads * d_state / 2; // DD-RoPE always enabled
    let in_proj_out = d_inner + d_inner + bc_size * 2 + n_heads + n_heads + n_heads + theta_proj_size; // +n_heads for dd_A

    // Attention dimensions (precomputed — used in forward, backward, and optimizer)
    let attn_n_heads_i = config.model.attn_n_heads as i32;
    let attn_kv_heads_i = config.model.attn_kv_heads as i32;
    let attn_head_dim_i = config.model.attn_head_dim() as i32;
    let kv_dim = config.model.attn_kv_dim();
    let mlp_dim = config.model.attn_mlp_dim();

    // Optimizer hyperparameters
    let max_grad_norm = 1.0f32;
    let beta1 = 0.9f32;
    let beta2 = 0.95f32;
    let adam_eps = 1e-8f32;
    let wd = config.weight_decay as f32;

    // Allocate all GPU buffers
    let mut buf = TrainingBuffers::allocate(
        &config.model, batch, seq, config.mixed_precision, config.bf16_activations,
    );
    let mp = config.mixed_precision;

    // Initialize weights: try to resume from checkpoint, else random init
    let mut resumed_epoch: usize = 0;
    let mut resumed_sample_pos: usize = 0;
    let mut resumed_tokens_seen: usize = 0;

    let start_step = if let Some(ckpt_dir) = crate::checkpoint::find_latest_checkpoint("checkpoints") {
        eprintln!("Resuming from checkpoint: {}", ckpt_dir);
        match crate::native_checkpoint::load_native_checkpoint(&mut buf, &ckpt_dir) {
            Ok(step) => {
                eprintln!("  Loaded weights from step {}", step);
                // Restore optimizer state (Adam m/v) if available
                match crate::native_checkpoint::load_optimizer_state(&mut buf, &ckpt_dir) {
                    Ok(true) => eprintln!("  Optimizer state restored"),
                    Ok(false) => {} // no optimizer file — already logged
                    Err(e) => eprintln!("  Failed to load optimizer state: {} — Adam restarts from zeros", e),
                }
                // Restore training position from meta.json
                let meta_path = format!("{}/meta.json", ckpt_dir);
                if let Ok(meta_str) = std::fs::read_to_string(&meta_path) {
                    if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta_str) {
                        resumed_epoch = meta["epoch"].as_u64().unwrap_or(0) as usize;
                        resumed_sample_pos = meta["sample_pos"].as_u64().unwrap_or(0) as usize;
                        resumed_tokens_seen = meta["tokens_seen"].as_u64().unwrap_or(0) as usize;
                        eprintln!("  Restored position: epoch={}, sample_pos={}, tokens_seen={}",
                            resumed_epoch, resumed_sample_pos, resumed_tokens_seen);
                    }
                }
                step
            }
            Err(e) => {
                eprintln!("  Failed to load checkpoint: {}. Using random init.", e);
                init_weights_random(&mut buf, config);
                0
            }
        }
    } else {
        init_weights_random(&mut buf, config);
        0
    };

    // Load dataset
    let dataset = if std::path::Path::new(&config.train_data).exists() {
        eprintln!("Loading dataset: {}", config.train_data);
        TokenDataset::from_binary(&config.train_data, seq)
    } else {
        panic!("{} not found — run preprocessing first", config.train_data);
    };
    eprintln!("Dataset: {} tokens, {} windows", dataset.tokens.len(), dataset.len());

    let start_time = std::time::Instant::now();

    eprintln!("Starting native training: {} steps, batch={}, seq={}",
        config.max_steps, batch, seq);
    eprintln!("Effective batch: {} tokens/step", bs);
    eprintln!();

    // Load validation dataset if available
    let val_dataset = if std::path::Path::new(&config.val_data).exists() {
        Some(TokenDataset::from_binary(&config.val_data, seq))
    } else {
        None
    };

    let grad_accum = config.gradient_accumulation.max(1);


    let mut best_val_loss = f32::MAX;
    let mut best_val_ckpt: Option<String> = None;
    let mut loss_val_last = 0.0f32;

    // Sequential sampling: coprime stride guarantees full coverage in one epoch
    let stride = crate::data::find_coprime_stride(dataset.len());
    let mut sample_pos: usize = resumed_sample_pos;
    let mut epoch: usize = resumed_epoch;
    let mut tokens_seen: usize = resumed_tokens_seen;
    eprintln!("Sequential sampling: stride={}, dataset={} windows", stride, dataset.len());

    let n_mamba = config.model.n_mamba_layers();
    let n_attn = config.model.n_attn_layers();

    eprintln!("Entering training loop...");
    for step in start_step..config.max_steps {
        if step == start_step { eprintln!("  First step starting (step {})...", step); }
        // Scale LR by 1/grad_accum (equivalent to averaging gradients)
        let lr = config.lr_at_step(step) as f32 / grad_accum as f32;

      // Zero all gradient buffers (one cudaMemset per layer thanks to contiguous allocation)
      for layer in 0..n_mamba { buf.d_layers[layer].zero_all(); }
      for aidx in 0..n_attn { buf.d_attn_layers[aidx].zero_all(); }
      buf.d_embedding.zero();
      buf.d_final_norm_gamma.zero();

      // is_log_step is per-step, not per-micro — hoist out of the accumulation loop.
      let is_log_step = step % 10 == 0 || step == config.max_steps - 1
          || (step > 0 && step % config.checkpoint_interval == 0)
          || (step > 0 && step % config.eval_interval == 0);

      // micro_loss accumulates per-micro losses across the grad-accum loop and is
      // averaged after the loop, so the reported value matches what the optimizer
      // actually descended on (previously this only held the LAST micro's loss).
      let mut micro_loss = 0.0f32;
      for _micro in 0..grad_accum {
        // ──── Load batch to GPU ────
        let (input_data, target_data) = get_batch_sequential(&dataset, batch, seq, stride, &mut sample_pos, &mut epoch);
        tokens_seen += batch * seq;
        upload_batch(&buf.input_ids, &buf.target_ids, &input_data, &target_data);

        if step == start_step { eprintln!("  Batch loaded, starting forward pass..."); }
        // ──── FORWARD PASS (all on GPU, zero debug syncs) ────

        // 1. Embedding lookup
        unsafe {
            native_ops::embedding_lookup(
                buf.embedding.ptr, buf.input_ids.ptr as *const i32,
                buf.x.ptr, bs as i32, d_model as i32,
            );
            if step == start_step { native_ops::cudaDeviceSynchronize(); eprintln!("  Embedding OK"); }
        }

        // 2. Process each block
        for layer in 0..n_layers {
            if step == start_step && (layer < 3 || layer == n_layers - 1) {
                unsafe { native_ops::cudaDeviceSynchronize(); }
                eprintln!("  Layer {}/{} starting...", layer, n_layers);
            }
            use crate::gpu_memory::LayerType;
            let layer_type = buf.layer_types[layer];

            match layer_type {
            LayerType::Attention(aidx) => {
                let ws = config.model.attn_window_size(aidx) as i32;
                unsafe {
                    attention_block_forward(
                        buf.x.ptr, &buf, &buf.attn_layers[aidx],
                        Some(&buf.attn_saved[aidx]),
                        bs as i32, d_model as i32, batch as i32, seq as i32,
                        attn_n_heads_i, attn_kv_heads_i, attn_head_dim_i,
                        kv_dim as i32, mlp_dim as i32, ws, eps,
                        mp,
                    );
                }
            }

            LayerType::Mamba(idx) => {
            if buf.bf16_activations {
                // BF16 save path: forward writes into shared FP32 scratch, then we
                // down-convert each live saved tensor into per-layer BF16 storage.
                let dst = MambaForwardDst {
                    residual: buf.residual.ptr,
                    x_norm: buf.x_norm.ptr,
                    x_ssm_raw: buf.x_ssm_raw.ptr,
                    x_act: buf.x_act.ptr,
                    z_buf: buf.z_buf.ptr,
                    b_raw: buf.b_raw.ptr,
                    c_raw: buf.c_raw.ptr,
                    b_norm: buf.b_norm_buf.ptr,
                    c_norm: buf.c_norm_buf.ptr,
                    dt_buf: buf.dt_buf.ptr,
                    lambda_raw: buf.lambda_raw.ptr,
                    dd_a_raw: buf.dd_a_raw.ptr,
                    theta_raw: buf.theta_raw.ptr,
                    ssm_out: buf.ssm_out.ptr,
                    y_gated: buf.y_gated.ptr,
                };
                unsafe {
                    mamba_block_forward(
                        buf.x.ptr, buf.projected.ptr, buf.lambda_buf.ptr, buf.a_vals_buf.ptr, buf.block_out.ptr,
                        &dst, &buf.layers[idx], &buf,
                        bs as i32, d_model as i32, d_inner as i32, in_proj_out as i32,
                        n_heads as i32, n_groups as i32, d_state as i32,
                        bc_size as i32, batch as i32, seq as i32, head_dim as i32, eps,
                        mp,
                    );
                    save_mamba_layer_bf16(
                        &buf, &buf.saved_bf16[idx],
                        bs, d_model, d_inner, bc_size, n_heads, theta_proj_size,
                    );
                }
            } else {
                let dst = MambaForwardDst {
                    residual: buf.saved[idx].residual.ptr,
                    x_norm: buf.saved[idx].x_norm.ptr,
                    x_ssm_raw: buf.saved[idx].x_ssm_raw.ptr,
                    x_act: buf.saved[idx].x_act.ptr,
                    z_buf: buf.saved[idx].z_buf.ptr,
                    b_raw: buf.saved[idx].b_raw.ptr,
                    c_raw: buf.saved[idx].c_raw.ptr,
                    b_norm: buf.saved[idx].b_norm.ptr,
                    c_norm: buf.saved[idx].c_norm.ptr,
                    dt_buf: buf.saved[idx].dt_buf.ptr,
                    lambda_raw: buf.saved[idx].lambda_raw.ptr,
                    dd_a_raw: buf.saved[idx].dd_a_raw.ptr,
                    theta_raw: buf.saved[idx].theta_raw.ptr,
                    ssm_out: buf.saved[idx].ssm_out.ptr,
                    y_gated: buf.saved[idx].y_gated.ptr,
                };
                unsafe {
                    mamba_block_forward(
                        buf.x.ptr, buf.projected.ptr, buf.lambda_buf.ptr, buf.a_vals_buf.ptr, buf.block_out.ptr,
                        &dst, &buf.layers[idx], &buf,
                        bs as i32, d_model as i32, d_inner as i32, in_proj_out as i32,
                        n_heads as i32, n_groups as i32, d_state as i32,
                        bc_size as i32, batch as i32, seq as i32, head_dim as i32, eps,
                        mp,
                    );
                }
            }
            } // end LayerType::Mamba
            } // end match layer_type
        } // end for layer

        // 3. Final norm
        unsafe {
            native_ops::rmsnorm_fwd(
                buf.x.ptr, buf.x_norm.ptr,
                buf.final_norm_gamma.ptr,
                eps, bs as i32, d_model as i32,
            );
        }

        // 4. Output projection (weight-tied with embedding): [bs, d_model] @ [d_model, vocab]
        // embedding is [vocab, d_model], so logits = x_norm @ embedding^T
        unsafe {
            native_ops::matmul_f32_bt(
                buf.x_norm.ptr, buf.embedding.ptr, buf.logits.ptr,
                bs as i32, vocab as i32, d_model as i32,
            );
        }

        // 5. Loss
        unsafe {
            native_ops::sparse_cross_entropy_fwd(
                buf.logits.ptr, buf.target_ids.ptr as *const i32,
                buf.loss.ptr, buf.per_token_loss.ptr,
                bs as i32, vocab as i32,
            );
        }

        // Read scalar loss on logging/checkpoint steps only (sync is expensive).
        if is_log_step {
            unsafe { native_ops::cudaDeviceSynchronize(); }
            micro_loss += buf.loss.to_host()[0];
        }

        if step == start_step {
            // Sync at the forward/backward boundary so any illegal-memory-access
            // error from a forward kernel is surfaced *here* instead of poisoning
            // the first cuBLAS call in backward. The return value is the first
            // error encountered since the last check — abort on non-zero.
            let sync_status = unsafe { native_ops::cudaDeviceSynchronize() };
            if sync_status != 0 {
                panic!("CUDA error at forward/backward boundary: {}", sync_status);
            }
            eprintln!("  Forward pass complete ({:.1}s)", start_time.elapsed().as_secs_f32());
        }
        // ──── BACKWARD PASS (all on GPU, reverse order) ────
        // 5b. Loss backward
        unsafe {
            native_ops::sparse_cross_entropy_bwd(
                buf.logits.ptr, buf.target_ids.ptr as *const i32,
                buf.d_logits.ptr, bs as i32, vocab as i32,
            );
        }

        // 4b. Output projection backward
        unsafe {
            native_ops::matmul_f32(
                buf.d_logits.ptr, buf.embedding.ptr, buf.d_x_norm.ptr,
                bs as i32, d_model as i32, vocab as i32,
            );
            native_ops::matmul_f32_at_accum(
                buf.x_norm.ptr, buf.d_logits.ptr, buf.d_embedding.ptr,
                d_model as i32, vocab as i32, bs as i32,
            );
        }

        // 3b. Final norm backward
        unsafe {
            native_ops::rmsnorm_bwd(
                buf.x.ptr, buf.d_x_norm.ptr,
                buf.final_norm_gamma.ptr,
                buf.d_x.ptr, buf.d_final_norm_gamma.ptr,
                eps, bs as i32, d_model as i32,
            );
        }


        // 2b. Backward through each block (reverse order)
        for layer in (0..n_layers).rev() {
            use crate::gpu_memory::LayerType;
            let layer_type = buf.layer_types[layer];

            match layer_type {
            LayerType::Attention(aidx) => {
                let aw = &buf.attn_layers[aidx];
                let daw = &buf.d_attn_layers[aidx];
                let attn_n_heads = config.model.attn_n_heads;
                let attn_kv_heads = config.model.attn_kv_heads;
                let attn_head_dim = config.model.attn_head_dim();
                let bnh = (batch * attn_n_heads) as i32;

                // ── MLP backward (reverse of forward) ──
                // d_x is the incoming gradient

                // MLP down backward: d_mlp_gated = d_x @ mlp_down^T, weight grad
                unsafe {
                    mm_bt!(mp, buf, buf.d_x.ptr, aw.mlp_down.ptr, buf.d_attn_mlp_gated.ptr,
                        bs as i32, mlp_dim as i32, d_model as i32);
                    mm_at_accum!(mp, buf, buf.attn_saved[aidx].y_gated.ptr, buf.d_x.ptr, daw.d_mlp_down,
                        mlp_dim as i32, d_model as i32, bs as i32);
                }

                // Gate backward: d_gate_act = d_gated * up, d_up = d_gated * gate_act
                unsafe {
                    native_ops::elemwise_mul_bwd(
                        buf.attn_saved[aidx].gate_act.ptr, buf.attn_saved[aidx].up_out.ptr,
                        buf.d_attn_mlp_gated.ptr,
                        buf.d_attn_mlp_hidden.ptr, buf.d_attn_mlp_gated.ptr, // reuse d_gated as d_up
                        (bs * mlp_dim) as i32);
                }

                // Sigmoid backward on gate
                unsafe {
                    native_ops::sigmoid_bwd(buf.attn_saved[aidx].gate_raw.ptr,
                        buf.d_attn_mlp_hidden.ptr, buf.d_attn_mlp_hidden.ptr,
                        (bs * mlp_dim) as i32);
                }

                // MLP gate/up weight grads + d_mlp_norm
                unsafe {
                    // d_mlp_norm from gate path
                    mm_bt!(mp, buf, buf.d_attn_mlp_hidden.ptr, aw.mlp_gate.ptr, buf.d_x_norm.ptr,
                        bs as i32, d_model as i32, mlp_dim as i32);
                    mm_at_accum!(mp, buf, buf.attn_saved[aidx].mlp_norm_out.ptr,
                        buf.d_attn_mlp_hidden.ptr, daw.d_mlp_gate,
                        d_model as i32, mlp_dim as i32, bs as i32);

                    // d_mlp_norm += from up path
                    mm_bt!(mp, buf, buf.d_attn_mlp_gated.ptr, aw.mlp_up.ptr, buf.d_block_out.ptr,
                        bs as i32, d_model as i32, mlp_dim as i32);
                    native_ops::elemwise_add(buf.d_x_norm.ptr, buf.d_block_out.ptr, buf.d_x_norm.ptr,
                        (bs * d_model) as i32);
                    mm_at_accum!(mp, buf, buf.attn_saved[aidx].mlp_norm_out.ptr,
                        buf.d_attn_mlp_gated.ptr, daw.d_mlp_up,
                        d_model as i32, mlp_dim as i32, bs as i32);
                }

                // MLP RMSNorm backward
                unsafe {
                    native_ops::rmsnorm_bwd(buf.attn_saved[aidx].mlp_input.ptr, buf.d_x_norm.ptr,
                        aw.mlp_norm_gamma.ptr, buf.d_block_out.ptr, daw.d_mlp_norm_gamma,
                        eps, bs as i32, d_model as i32);
                }

                // d_x = d_mlp_input (from MLP norm bwd) + d_x (from residual)
                unsafe {
                    native_ops::elemwise_add(buf.d_block_out.ptr, buf.d_x.ptr, buf.d_x.ptr,
                        (bs * d_model) as i32);
                }

                // ── Attention backward ──
                // d_x now has gradient flowing back through MLP residual

                // Output proj backward: d_context = d_x @ out_proj^T
                unsafe {
                    mm_bt!(mp, buf, buf.d_x.ptr, aw.attn_out_proj.ptr, buf.d_attn_context.ptr,
                        bs as i32, d_model as i32, d_model as i32);
                    mm_at_accum!(mp, buf, buf.attn_saved[aidx].context.ptr, buf.d_x.ptr, daw.d_attn_out_proj,
                        d_model as i32, d_model as i32, bs as i32);
                }

                // Transpose d_context: [batch, seq, n_heads, head_dim] → [batch, n_heads, seq, head_dim]
                unsafe {
                    native_ops::transpose_0213(buf.d_attn_context.ptr, buf.d_attn_context_t.ptr,
                        batch as i32, seq as i32, attn_n_heads as i32, attn_head_dim as i32);
                }

                // d_weights = d_context_t @ V_t^T: [bnh, seq, head_dim] @ [bnh, head_dim, seq] → [bnh, seq, seq]
                // d_V_t = weights^T @ d_context_t: [bnh, seq, seq]^T @ [bnh, seq, head_dim] → [bnh, seq, head_dim]
                unsafe {
                    // Reload saved attention weights and V_t for backward
                    // weights saved in attn_saved[aidx].attn_weights
                    // V_t: need to re-expand from saved V
                    native_ops::transpose_0213(buf.attn_saved[aidx].v.ptr, buf.attn_kv_temp.ptr,
                        batch as i32, seq as i32, attn_kv_heads as i32, attn_head_dim as i32);
                    native_ops::gqa_expand(buf.attn_kv_temp.ptr, buf.attn_v_t.ptr,
                        batch as i32, attn_kv_heads as i32, attn_n_heads as i32, seq as i32, attn_head_dim as i32);

                    // d_weights = d_context_t @ V_t^T
                    mm_bt_batched!(mp, buf,
                        buf.d_attn_context_t.ptr, buf.attn_v_t.ptr, buf.d_attn_scores.ptr,
                        seq as i32, seq as i32, attn_head_dim as i32, bnh);

                    // d_V_t = weights^T @ d_context_t
                    mm_at_batched!(mp, buf,
                        buf.attn_saved[aidx].attn_weights.ptr, buf.d_attn_context_t.ptr, buf.d_attn_v_t.ptr,
                        seq as i32, attn_head_dim as i32, seq as i32, bnh);
                }

                // Causal softmax backward (with optional sliding window)
                let ws = config.model.attn_window_size(aidx) as i32;
                unsafe {
                    native_ops::causal_softmax_bwd(
                        buf.attn_saved[aidx].attn_weights.ptr, buf.d_attn_scores.ptr,
                        buf.d_attn_scores.ptr, bnh, seq as i32, ws);

                    // Scale backward (same scale as forward)
                    let scale = 1.0f32 / (attn_head_dim as f32).sqrt();
                    native_ops::scale_tensor(buf.d_attn_scores.ptr, buf.d_attn_scores.ptr,
                        scale, (batch * attn_n_heads * seq * seq) as i32);
                }

                // d_Q_t = d_scores @ K_t: [bnh, seq, seq] @ [bnh, seq, head_dim] → [bnh, seq, head_dim]
                // d_K_t = d_scores^T @ Q_t: [bnh, seq, seq]^T @ [bnh, seq, head_dim] → [bnh, seq, head_dim]
                unsafe {
                    // Re-expand K_t from saved K
                    native_ops::transpose_0213(buf.attn_saved[aidx].k.ptr, buf.attn_kv_temp.ptr,
                        batch as i32, seq as i32, attn_kv_heads as i32, attn_head_dim as i32);
                    native_ops::gqa_expand(buf.attn_kv_temp.ptr, buf.attn_k_t.ptr,
                        batch as i32, attn_kv_heads as i32, attn_n_heads as i32, seq as i32, attn_head_dim as i32);

                    // Re-create Q_t from saved Q
                    native_ops::transpose_0213(buf.attn_saved[aidx].q.ptr, buf.attn_q_t.ptr,
                        batch as i32, seq as i32, attn_n_heads as i32, attn_head_dim as i32);

                    // d_Q_t = d_scores @ K_t
                    mm_batched!(mp, buf,
                        buf.d_attn_scores.ptr, buf.attn_k_t.ptr, buf.d_attn_q_t.ptr,
                        seq as i32, attn_head_dim as i32, seq as i32, bnh);

                    // d_K_t = d_scores^T @ Q_t
                    mm_at_batched!(mp, buf,
                        buf.d_attn_scores.ptr, buf.attn_q_t.ptr, buf.d_attn_k_t.ptr,
                        seq as i32, attn_head_dim as i32, seq as i32, bnh);
                }

                // GQA reduce: sum d_K_t and d_V_t from n_heads back to kv_heads
                // Transpose d_Q_t back: [batch, n_heads, seq, head_dim] → [batch, seq, n_heads, head_dim]
                unsafe {
                    native_ops::transpose_0213(buf.d_attn_q_t.ptr, buf.d_attn_q.ptr,
                        batch as i32, attn_n_heads as i32, seq as i32, attn_head_dim as i32);

                    // GQA reduce K: [batch, n_heads, seq, head_dim] → [batch, kv_heads, seq, head_dim]
                    native_ops::gqa_reduce(buf.d_attn_k_t.ptr, buf.d_attn_k_t.ptr,
                        batch as i32, attn_kv_heads as i32, attn_n_heads as i32, seq as i32, attn_head_dim as i32);
                    native_ops::transpose_0213(buf.d_attn_k_t.ptr, buf.d_attn_k.ptr,
                        batch as i32, attn_kv_heads as i32, seq as i32, attn_head_dim as i32);

                    // GQA reduce V
                    native_ops::gqa_reduce(buf.d_attn_v_t.ptr, buf.d_attn_v_t.ptr,
                        batch as i32, attn_kv_heads as i32, attn_n_heads as i32, seq as i32, attn_head_dim as i32);
                    native_ops::transpose_0213(buf.d_attn_v_t.ptr, buf.d_attn_v.ptr,
                        batch as i32, attn_kv_heads as i32, seq as i32, attn_head_dim as i32);
                }

                // Q/K/V projection weight grads + d_x_norm
                unsafe {
                    // d_x_norm from Q
                    mm_bt!(mp, buf, buf.d_attn_q.ptr, aw.q_proj.ptr, buf.d_x_norm.ptr,
                        bs as i32, d_model as i32, d_model as i32);
                    mm_at_accum!(mp, buf, buf.attn_saved[aidx].x_norm.ptr, buf.d_attn_q.ptr, daw.d_q_proj,
                        d_model as i32, d_model as i32, bs as i32);

                    // d_x_norm += from K
                    mm_bt!(mp, buf, buf.d_attn_k.ptr, aw.k_proj.ptr, buf.d_block_out.ptr,
                        bs as i32, d_model as i32, kv_dim as i32);
                    native_ops::elemwise_add(buf.d_x_norm.ptr, buf.d_block_out.ptr, buf.d_x_norm.ptr,
                        (bs * d_model) as i32);
                    mm_at_accum!(mp, buf, buf.attn_saved[aidx].x_norm.ptr, buf.d_attn_k.ptr, daw.d_k_proj,
                        d_model as i32, kv_dim as i32, bs as i32);

                    // d_x_norm += from V
                    mm_bt!(mp, buf, buf.d_attn_v.ptr, aw.v_proj.ptr, buf.d_block_out.ptr,
                        bs as i32, d_model as i32, kv_dim as i32);
                    native_ops::elemwise_add(buf.d_x_norm.ptr, buf.d_block_out.ptr, buf.d_x_norm.ptr,
                        (bs * d_model) as i32);
                    mm_at_accum!(mp, buf, buf.attn_saved[aidx].x_norm.ptr, buf.d_attn_v.ptr, daw.d_v_proj,
                        d_model as i32, kv_dim as i32, bs as i32);
                }

                // Attention RMSNorm backward
                unsafe {
                    native_ops::rmsnorm_bwd(buf.attn_saved[aidx].residual.ptr, buf.d_x_norm.ptr,
                        aw.attn_norm_gamma.ptr, buf.d_block_out.ptr, daw.d_attn_norm_gamma,
                        eps, bs as i32, d_model as i32);
                }

                // d_x = d_norm_bwd + d_x (from attention residual)
                unsafe {
                    native_ops::elemwise_add(buf.d_block_out.ptr, buf.d_x.ptr, buf.d_x.ptr,
                        (bs * d_model) as i32);
                }
            }

            LayerType::Mamba(idx) => {
            if buf.bf16_activations {
                let saved = &buf.saved_bf16[idx];

                // d_block_out = d_x
                unsafe {
                    cuda_memcpy_d2d(buf.d_block_out.ptr, buf.d_x.ptr, bs * d_model);
                }

                // out_proj backward: d_y_gated = d_block_out @ out_proj^T
                // y_gated is NOT saved — recompute as ssm_out * SiLU(z_buf) in stage_a.
                unsafe {
                    mm_bt!(mp, buf,
                        buf.d_block_out.ptr, buf.layers[idx].out_proj.ptr, buf.d_y_gated.ptr,
                        bs as i32, d_inner as i32, d_model as i32
                    );
                    cuda_convert_bf16_to_f32(buf.bwd_stage_a.ptr, saved.ssm_out.ptr, bs * d_inner);
                    cuda_convert_bf16_to_f32(buf.bwd_stage_b.ptr, saved.z_buf.ptr, bs * d_inner);
                    native_ops::silu_fwd(buf.bwd_stage_b.ptr, buf.bwd_stage_b.ptr, (bs * d_inner) as i32);
                    native_ops::elemwise_mul(buf.bwd_stage_a.ptr, buf.bwd_stage_b.ptr, buf.bwd_stage_a.ptr, (bs * d_inner) as i32);
                    mm_at_accum!(mp, buf,
                        buf.bwd_stage_a.ptr, buf.d_block_out.ptr, buf.d_layers[idx].d_out_proj,
                        d_inner as i32, d_model as i32, bs as i32
                    );
                }

                // Fused gate backward: needs ssm_out and z_buf concurrently
                unsafe {
                    cuda_convert_bf16_to_f32(buf.bwd_stage_a.ptr, saved.ssm_out.ptr, bs * d_inner);
                    cuda_convert_bf16_to_f32(buf.bwd_stage_b.ptr, saved.z_buf.ptr, bs * d_inner);
                    native_ops::fused_silu_gate_bwd(
                        buf.bwd_stage_a.ptr, buf.bwd_stage_b.ptr, buf.d_y_gated.ptr,
                        buf.d_ssm_out.ptr, buf.d_z.ptr,
                        (bs * d_inner) as i32,
                    );
                }

                // Recompute lambda_buf from saved lambda_raw (sequential use of stage_a OK)
                unsafe {
                    cuda_convert_bf16_to_f32(buf.bwd_stage_a.ptr, saved.lambda_raw.ptr, bs * n_heads);
                    native_ops::sigmoid_fwd(
                        buf.bwd_stage_a.ptr, buf.lambda_buf.ptr, (bs * n_heads) as i32,
                    );
                }

                // Recompute a_vals_buf from saved dd_a_raw
                unsafe {
                    cuda_convert_bf16_to_f32(buf.bwd_stage_a.ptr, saved.dd_a_raw.ptr, bs * n_heads);
                    native_ops::neg_softplus_clamp(
                        buf.bwd_stage_a.ptr, buf.a_vals_buf.ptr, -1e6, -1e-4, (bs * n_heads) as i32,
                    );
                }

                // SSM backward — needs 5 saved tensors concurrently (staged a..e).
                // x_act is NOT saved — recompute as SiLU(x_ssm_raw) into stage_a.
                unsafe {
                    cuda_convert_bf16_to_f32(buf.bwd_stage_a.ptr, saved.x_ssm_raw.ptr, bs * d_inner);
                    native_ops::silu_fwd(buf.bwd_stage_a.ptr, buf.bwd_stage_a.ptr, (bs * d_inner) as i32);
                    cuda_convert_bf16_to_f32(buf.bwd_stage_b.ptr, saved.dt_buf.ptr,    bs * n_heads);
                    cuda_convert_bf16_to_f32(buf.bwd_stage_c.ptr, saved.b_norm.ptr,    bs * bc_size);
                    cuda_convert_bf16_to_f32(buf.bwd_stage_d.ptr, saved.c_norm.ptr,    bs * bc_size);
                    cuda_convert_bf16_to_f32(buf.bwd_stage_e.ptr, saved.theta_raw.ptr, bs * theta_proj_size);
                    native_ops::ssm_scan_bwd_gpu_v2(
                        buf.bwd_stage_a.ptr,            // x_act
                        buf.bwd_stage_b.ptr,            // dt
                        buf.bwd_stage_c.ptr,            // b_norm
                        buf.bwd_stage_d.ptr,            // c_norm
                        buf.layers[idx].d_skip.ptr,
                        buf.layers[idx].dt_bias.ptr,
                        buf.d_ssm_out.ptr,
                        buf.lambda_buf.ptr,
                        buf.layers[idx].h_init.ptr,
                        buf.ssm_h_checkpoints.ptr, buf.ssm_pbx_checkpoints.ptr,
                        buf.ssm_h_saved.ptr, buf.ssm_pbx_saved.ptr,
                        buf.bwd_stage_e.ptr,            // theta_raw
                        buf.a_vals_buf.ptr,
                        buf.d_x_act.ptr,
                        buf.d_dt.ptr,
                        buf.d_b_norm.ptr,
                        buf.d_c_norm.ptr,
                        buf.d_lambda.ptr,
                        buf.d_layers[idx].d_h_init,
                        buf.d_theta_raw.ptr,
                        buf.d_a_vals.ptr,
                        buf.d_layers[idx].d_d_skip,
                        buf.d_layers[idx].d_dt_bias,
                        buf.ws_d_d_buf.ptr, buf.ws_d_dtb_buf.ptr,
                        batch as i32, seq as i32, n_heads as i32,
                        head_dim as i32, d_state as i32, n_groups as i32,
                        config.model.bwd_chunk_size as i32,
                    );
                }

                // DD-RoPE: assemble d_theta_raw back into d_projected
                {
                    let theta_offset = (d_inner * 2 + bc_size * 2 + n_heads + n_heads + n_heads) as i32;
                    let theta_size = (n_heads * d_state / 2) as i32;
                    unsafe {
                        native_ops::strided_assemble(
                            buf.d_theta_raw.ptr, buf.d_projected.ptr,
                            bs as i32, in_proj_out as i32, theta_offset, theta_size,
                        );
                    }
                }

                // BCNorm + bias backward (b_raw, then c_raw — sequential stage reuse)
                unsafe {
                    cuda_convert_bf16_to_f32(buf.bwd_stage_a.ptr, saved.b_raw.ptr, bs * bc_size);
                    native_ops::rmsnorm_bias_bwd(
                        buf.bwd_stage_a.ptr, buf.d_b_norm.ptr,
                        buf.layers[idx].b_gamma.ptr,
                        buf.d_b_norm.ptr,
                        buf.d_layers[idx].d_b_gamma, buf.d_layers[idx].d_b_bias,
                        eps, bs as i32, bc_size as i32,
                    );
                    cuda_convert_bf16_to_f32(buf.bwd_stage_a.ptr, saved.c_raw.ptr, bs * bc_size);
                    native_ops::rmsnorm_bias_bwd(
                        buf.bwd_stage_a.ptr, buf.d_c_norm.ptr,
                        buf.layers[idx].c_gamma.ptr,
                        buf.d_c_norm.ptr,
                        buf.d_layers[idx].d_c_gamma, buf.d_layers[idx].d_c_bias,
                        eps, bs as i32, bc_size as i32,
                    );
                }

                // SiLU backward on x_ssm
                unsafe {
                    cuda_convert_bf16_to_f32(buf.bwd_stage_a.ptr, saved.x_ssm_raw.ptr, bs * d_inner);
                    native_ops::silu_bwd(
                        buf.bwd_stage_a.ptr, buf.d_x_act.ptr, buf.d_x_act.ptr,
                        (bs * d_inner) as i32,
                    );
                }

                // Assemble d_projected from component gradients
                unsafe {
                    let mut col = 0i32;
                    native_ops::strided_assemble(
                        buf.d_x_act.ptr, buf.d_projected.ptr,
                        bs as i32, in_proj_out as i32, col, d_inner as i32,
                    );
                    col += d_inner as i32;
                    native_ops::strided_assemble(
                        buf.d_z.ptr, buf.d_projected.ptr,
                        bs as i32, in_proj_out as i32, col, d_inner as i32,
                    );
                    col += d_inner as i32;
                    native_ops::strided_assemble(
                        buf.d_b_norm.ptr, buf.d_projected.ptr,
                        bs as i32, in_proj_out as i32, col, bc_size as i32,
                    );
                    col += bc_size as i32;
                    native_ops::strided_assemble(
                        buf.d_c_norm.ptr, buf.d_projected.ptr,
                        bs as i32, in_proj_out as i32, col, bc_size as i32,
                    );
                    col += bc_size as i32;
                    native_ops::strided_assemble(
                        buf.d_dt.ptr, buf.d_projected.ptr,
                        bs as i32, in_proj_out as i32, col, n_heads as i32,
                    );
                    col += n_heads as i32;
                    // sigmoid backward on lambda
                    cuda_convert_bf16_to_f32(buf.bwd_stage_a.ptr, saved.lambda_raw.ptr, bs * n_heads);
                    native_ops::sigmoid_bwd(
                        buf.bwd_stage_a.ptr, buf.d_lambda.ptr, buf.d_lambda.ptr,
                        (bs * n_heads) as i32,
                    );
                    native_ops::strided_assemble(
                        buf.d_lambda.ptr, buf.d_projected.ptr,
                        bs as i32, in_proj_out as i32, col, n_heads as i32,
                    );
                    col += n_heads as i32;
                    // softplus backward on dd_A
                    cuda_convert_bf16_to_f32(buf.bwd_stage_a.ptr, saved.dd_a_raw.ptr, bs * n_heads);
                    native_ops::softplus_bwd(
                        buf.bwd_stage_a.ptr, buf.d_a_vals.ptr, buf.d_a_vals.ptr,
                        (bs * n_heads) as i32,
                    );
                    native_ops::negate(buf.d_a_vals.ptr, buf.d_a_vals.ptr, (bs * n_heads) as i32);
                    native_ops::strided_assemble(
                        buf.d_a_vals.ptr, buf.d_projected.ptr,
                        bs as i32, in_proj_out as i32, col, n_heads as i32,
                    );
                }

                // in_proj backward: x_norm is needed for weight grad
                unsafe {
                    mm_bt!(mp, buf,
                        buf.d_projected.ptr, buf.layers[idx].in_proj.ptr, buf.d_x_norm.ptr,
                        bs as i32, d_model as i32, in_proj_out as i32
                    );
                    cuda_convert_bf16_to_f32(buf.bwd_stage_a.ptr, saved.x_norm.ptr, bs * d_model);
                    mm_at_accum!(mp, buf,
                        buf.bwd_stage_a.ptr, buf.d_projected.ptr, buf.d_layers[idx].d_in_proj,
                        d_model as i32, in_proj_out as i32, bs as i32
                    );
                }

                // RMSNorm backward: residual is needed for x grad
                unsafe {
                    cuda_convert_bf16_to_f32(buf.bwd_stage_a.ptr, saved.residual.ptr, bs * d_model);
                    native_ops::rmsnorm_bwd(
                        buf.bwd_stage_a.ptr, buf.d_x_norm.ptr,
                        buf.layers[idx].norm_gamma.ptr,
                        buf.d_x.ptr, buf.d_layers[idx].d_norm_gamma,
                        eps, bs as i32, d_model as i32,
                    );
                }

                // Add residual gradient: d_x += d_block_out
                unsafe {
                    native_ops::elemwise_add(
                        buf.d_x.ptr, buf.d_block_out.ptr, buf.d_x.ptr, (bs * d_model) as i32,
                    );
                }
            } else {

            // d_block_out = d_x
            unsafe {
                cuda_memcpy_d2d(buf.d_block_out.ptr, buf.d_x.ptr, bs * d_model);
            }

            // out_proj backward: d_y_gated = d_block_out @ out_proj^T
            unsafe {
                mm_bt!(mp, buf,
                    buf.d_block_out.ptr, buf.layers[idx].out_proj.ptr, buf.d_y_gated.ptr,
                    bs as i32, d_inner as i32, d_model as i32
                );
                mm_at_accum!(mp, buf,
                    buf.saved[idx].y_gated.ptr, buf.d_block_out.ptr, buf.d_layers[idx].d_out_proj,
                    d_inner as i32, d_model as i32, bs as i32
                );
            }

            // Fused gate backward: uses saved ssm_out and raw z_buf
            unsafe {
                native_ops::fused_silu_gate_bwd(
                    buf.saved[idx].ssm_out.ptr, buf.saved[idx].z_buf.ptr, buf.d_y_gated.ptr,
                    buf.d_ssm_out.ptr, buf.d_z.ptr,
                    (bs * d_inner) as i32,
                );
            }

            // Recompute lambda for this layer (shared lambda_buf only has last layer's values)
            unsafe {
                native_ops::sigmoid_fwd(
                    buf.saved[idx].lambda_raw.ptr, buf.lambda_buf.ptr, (bs * n_heads) as i32,
                );
            }

            // Recompute A_vals for this layer (shared a_vals_buf only has last layer's values)
            unsafe {
                native_ops::neg_softplus_clamp(buf.saved[idx].dd_a_raw.ptr, buf.a_vals_buf.ptr, -1e6, -1e-4, (bs * n_heads) as i32);
            }

            // SSM backward (GPU-direct) — uses saved activations
            unsafe {
                native_ops::ssm_scan_bwd_gpu_v2(
                    buf.saved[idx].x_act.ptr,        // x input (post-SiLU)
                    buf.saved[idx].dt_buf.ptr,       // raw dt
                    buf.saved[idx].b_norm.ptr,       // normalized B (grouped)
                    buf.saved[idx].c_norm.ptr,       // normalized C (grouped)
                    buf.layers[idx].d_skip.ptr,      // D
                    buf.layers[idx].dt_bias.ptr,     // dt_bias
                    buf.d_ssm_out.ptr,               // dy (gradient from gate backward)
                    buf.lambda_buf.ptr,              // lambda_in: data-dependent λ
                    buf.layers[idx].h_init.ptr,
                    buf.ssm_h_checkpoints.ptr, buf.ssm_pbx_checkpoints.ptr,
                    buf.ssm_h_saved.ptr, buf.ssm_pbx_saved.ptr,
                    buf.saved[idx].theta_raw.ptr,
                    buf.a_vals_buf.ptr,              // data-dependent A values
                    buf.d_x_act.ptr,
                    buf.d_dt.ptr,
                    buf.d_b_norm.ptr,
                    buf.d_c_norm.ptr,
                    buf.d_lambda.ptr,
                    buf.d_layers[idx].d_h_init,
                    buf.d_theta_raw.ptr,
                    buf.d_a_vals.ptr,                // gradient for A_vals
                    buf.d_layers[idx].d_d_skip,
                    buf.d_layers[idx].d_dt_bias,
                    buf.ws_d_d_buf.ptr, buf.ws_d_dtb_buf.ptr,
                    batch as i32, seq as i32, n_heads as i32,
                    head_dim as i32, d_state as i32, n_groups as i32,
                    config.model.bwd_chunk_size as i32,
                );
            }

            // DD-RoPE: assemble d_theta_raw back into d_projected
            {
                let theta_offset = (d_inner * 2 + bc_size * 2 + n_heads + n_heads + n_heads) as i32;
                let theta_size = (n_heads * d_state / 2) as i32;
                unsafe {
                    native_ops::strided_assemble(
                        buf.d_theta_raw.ptr, buf.d_projected.ptr,
                        bs as i32, in_proj_out as i32, theta_offset, theta_size,
                    );
                }
            }

            // Fused BCNorm + bias backward on B and C
            unsafe {
                native_ops::rmsnorm_bias_bwd(
                    buf.saved[idx].b_raw.ptr, buf.d_b_norm.ptr,
                    buf.layers[idx].b_gamma.ptr,
                    buf.d_b_norm.ptr,
                    buf.d_layers[idx].d_b_gamma, buf.d_layers[idx].d_b_bias,
                    eps, bs as i32, bc_size as i32,
                );
                native_ops::rmsnorm_bias_bwd(
                    buf.saved[idx].c_raw.ptr, buf.d_c_norm.ptr,
                    buf.layers[idx].c_gamma.ptr,
                    buf.d_c_norm.ptr,
                    buf.d_layers[idx].d_c_gamma, buf.d_layers[idx].d_c_bias,
                    eps, bs as i32, bc_size as i32,
                );
            }

            // SiLU backward on x_ssm: uses saved x_ssm_raw
            unsafe {
                native_ops::silu_bwd(
                    buf.saved[idx].x_ssm_raw.ptr, buf.d_x_act.ptr, buf.d_x_act.ptr,
                    (bs * d_inner) as i32,
                );
            }

            // Assemble d_projected [bs, in_proj_out] from component gradients
            unsafe {
                let mut col = 0i32;
                native_ops::strided_assemble(
                    buf.d_x_act.ptr, buf.d_projected.ptr,
                    bs as i32, in_proj_out as i32, col, d_inner as i32,
                );
                col += d_inner as i32;
                native_ops::strided_assemble(
                    buf.d_z.ptr, buf.d_projected.ptr,
                    bs as i32, in_proj_out as i32, col, d_inner as i32,
                );
                col += d_inner as i32;
                native_ops::strided_assemble(
                    buf.d_b_norm.ptr, buf.d_projected.ptr,
                    bs as i32, in_proj_out as i32, col, bc_size as i32,
                );
                col += bc_size as i32;
                native_ops::strided_assemble(
                    buf.d_c_norm.ptr, buf.d_projected.ptr,
                    bs as i32, in_proj_out as i32, col, bc_size as i32,
                );
                col += bc_size as i32;
                native_ops::strided_assemble(
                    buf.d_dt.ptr, buf.d_projected.ptr,
                    bs as i32, in_proj_out as i32, col, n_heads as i32,
                );
                col += n_heads as i32;
                // sigmoid backward: d_lambda_raw = d_lambda * sig * (1 - sig)
                native_ops::sigmoid_bwd(
                    buf.saved[idx].lambda_raw.ptr, buf.d_lambda.ptr, buf.d_lambda.ptr,
                    (bs * n_heads) as i32,
                );
                native_ops::strided_assemble(
                    buf.d_lambda.ptr, buf.d_projected.ptr,
                    bs as i32, in_proj_out as i32, col, n_heads as i32,
                );
                col += n_heads as i32;
                // dd_A backward: d_dd_A = d_A_vals * (-sigmoid(dd_A)) * clamp_mask
                native_ops::softplus_bwd(
                    buf.saved[idx].dd_a_raw.ptr, buf.d_a_vals.ptr, buf.d_a_vals.ptr,
                    (bs * n_heads) as i32,
                );
                native_ops::negate(buf.d_a_vals.ptr, buf.d_a_vals.ptr, (bs * n_heads) as i32);
                native_ops::strided_assemble(
                    buf.d_a_vals.ptr, buf.d_projected.ptr,
                    bs as i32, in_proj_out as i32, col, n_heads as i32,
                );
            }

            // in_proj backward: uses saved x_norm for weight grad
            unsafe {
                mm_bt!(mp, buf,
                    buf.d_projected.ptr, buf.layers[idx].in_proj.ptr, buf.d_x_norm.ptr,
                    bs as i32, d_model as i32, in_proj_out as i32
                );
                mm_at_accum!(mp, buf,
                    buf.saved[idx].x_norm.ptr, buf.d_projected.ptr, buf.d_layers[idx].d_in_proj,
                    d_model as i32, in_proj_out as i32, bs as i32
                );
            }

            // RMSNorm backward: uses saved residual (correct per-layer input)
            unsafe {
                native_ops::rmsnorm_bwd(
                    buf.saved[idx].residual.ptr, buf.d_x_norm.ptr,
                    buf.layers[idx].norm_gamma.ptr,
                    buf.d_x.ptr, buf.d_layers[idx].d_norm_gamma,
                    eps, bs as i32, d_model as i32,
                );
            }

            // Add residual gradient: d_x += d_block_out
            unsafe {
                native_ops::elemwise_add(
                    buf.d_x.ptr, buf.d_block_out.ptr, buf.d_x.ptr, (bs * d_model) as i32,
                );
            }
            }
            } // end LayerType::Mamba
            } // end match layer_type
        } // end for layer (backward)

        // 1b. Embedding backward: scatter-add input gradient into d_embedding
        // This accumulates on top of the output projection gradient already in d_embedding
        unsafe {
            native_ops::embedding_bwd(
                buf.d_x.ptr, buf.input_ids.ptr as *const i32,
                buf.d_embedding.ptr, bs as i32, d_model as i32, vocab as i32,
            );
        }

      } // end grad_accum micro-batch loop

        // Average per-micro losses to get the true step loss (only meaningful on log steps).
        if is_log_step && grad_accum > 1 {
            micro_loss /= grad_accum as f32;
        }

        if step == start_step {
            unsafe { native_ops::cudaDeviceSynchronize(); }
            eprintln!("  Backward pass complete ({:.1}s)", start_time.elapsed().as_secs_f32());
        }
        // ──── GRADIENT CLIPPING + OPTIMIZER (fused — no D2H sync) ────
        // Compute global gradient norm on GPU, then each adamw_clipped_update reads it
        // from device memory. Zero CPU-GPU sync in this section.
        let adam_step = (step + 1) as i32;
        let norm_ptr = buf.grad_norm_scratch.ptr;

        unsafe {
            buf.grad_norm_scratch.zero();

            // Accumulate squared gradient norms across all parameters
            for layer in 0..n_mamba {
                native_ops::grad_norm_squared(buf.d_layers[layer].d_in_proj, norm_ptr, (d_model * in_proj_out) as i32);
                native_ops::grad_norm_squared(buf.d_layers[layer].d_out_proj, norm_ptr, (d_inner * d_model) as i32);
                native_ops::grad_norm_squared(buf.d_layers[layer].d_norm_gamma, norm_ptr, d_model as i32);
                native_ops::grad_norm_squared(buf.d_layers[layer].d_b_gamma, norm_ptr, bc_size as i32);
                native_ops::grad_norm_squared(buf.d_layers[layer].d_c_gamma, norm_ptr, bc_size as i32);
                native_ops::grad_norm_squared(buf.d_layers[layer].d_b_bias, norm_ptr, bc_size as i32);
                native_ops::grad_norm_squared(buf.d_layers[layer].d_c_bias, norm_ptr, bc_size as i32);
                native_ops::grad_norm_squared(buf.d_layers[layer].d_d_skip, norm_ptr, n_heads as i32);
                native_ops::grad_norm_squared(buf.d_layers[layer].d_dt_bias, norm_ptr, n_heads as i32);
                native_ops::grad_norm_squared(buf.d_layers[layer].d_h_init, norm_ptr, (n_heads * d_state) as i32);
            }
            native_ops::grad_norm_squared(buf.d_embedding.ptr, norm_ptr, (vocab * d_model) as i32);
            native_ops::grad_norm_squared(buf.d_final_norm_gamma.ptr, norm_ptr, d_model as i32);

            for aidx in 0..n_attn {
                let daw = &buf.d_attn_layers[aidx];
                native_ops::grad_norm_squared(daw.d_q_proj, norm_ptr, (d_model * d_model) as i32);
                native_ops::grad_norm_squared(daw.d_k_proj, norm_ptr, (d_model * kv_dim) as i32);
                native_ops::grad_norm_squared(daw.d_v_proj, norm_ptr, (d_model * kv_dim) as i32);
                native_ops::grad_norm_squared(daw.d_attn_out_proj, norm_ptr, (d_model * d_model) as i32);
                native_ops::grad_norm_squared(daw.d_mlp_gate, norm_ptr, (d_model * mlp_dim) as i32);
                native_ops::grad_norm_squared(daw.d_mlp_up, norm_ptr, (d_model * mlp_dim) as i32);
                native_ops::grad_norm_squared(daw.d_mlp_down, norm_ptr, (mlp_dim * d_model) as i32);
                native_ops::grad_norm_squared(daw.d_attn_norm_gamma, norm_ptr, d_model as i32);
                native_ops::grad_norm_squared(daw.d_mlp_norm_gamma, norm_ptr, d_model as i32);
            }

            // Fused clipped AdamW — each kernel reads norm_sq from device, clips inline
            // Mamba layers
            for layer in 0..n_mamba {
                native_ops::adamw_clipped_update(buf.layers[layer].in_proj.ptr, buf.d_layers[layer].d_in_proj,
                    buf.adam_m[layer].m.ptr, buf.adam_m[layer].v.ptr,
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, wd, adam_step, (d_model * in_proj_out) as i32);
                native_ops::adamw_clipped_update(buf.layers[layer].out_proj.ptr, buf.d_layers[layer].d_out_proj,
                    buf.adam_m[layer].m.offset(d_model * in_proj_out), buf.adam_m[layer].v.offset(d_model * in_proj_out),
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, wd, adam_step, (d_inner * d_model) as i32);
                native_ops::adamw_clipped_update(buf.layers[layer].norm_gamma.ptr, buf.d_layers[layer].d_norm_gamma,
                    buf.small_adam[layer].norm_m, buf.small_adam[layer].norm_v,
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, 0.0, adam_step, d_model as i32);
                native_ops::adamw_clipped_update(buf.layers[layer].b_gamma.ptr, buf.d_layers[layer].d_b_gamma,
                    buf.small_adam[layer].b_gamma_m, buf.small_adam[layer].b_gamma_v,
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, 0.0, adam_step, bc_size as i32);
                native_ops::adamw_clipped_update(buf.layers[layer].c_gamma.ptr, buf.d_layers[layer].d_c_gamma,
                    buf.small_adam[layer].c_gamma_m, buf.small_adam[layer].c_gamma_v,
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, 0.0, adam_step, bc_size as i32);
                native_ops::adamw_clipped_update(buf.layers[layer].b_bias.ptr, buf.d_layers[layer].d_b_bias,
                    buf.small_adam[layer].b_bias_m, buf.small_adam[layer].b_bias_v,
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, 0.0, adam_step, bc_size as i32);
                native_ops::adamw_clipped_update(buf.layers[layer].c_bias.ptr, buf.d_layers[layer].d_c_bias,
                    buf.small_adam[layer].c_bias_m, buf.small_adam[layer].c_bias_v,
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, 0.0, adam_step, bc_size as i32);
                native_ops::adamw_clipped_update(buf.layers[layer].d_skip.ptr, buf.d_layers[layer].d_d_skip,
                    buf.small_adam[layer].d_skip_m, buf.small_adam[layer].d_skip_v,
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, 0.0, adam_step, n_heads as i32);
                native_ops::adamw_clipped_update(buf.layers[layer].dt_bias.ptr, buf.d_layers[layer].d_dt_bias,
                    buf.small_adam[layer].dt_bias_m, buf.small_adam[layer].dt_bias_v,
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, 0.0, adam_step, n_heads as i32);
                native_ops::adamw_clipped_update(buf.layers[layer].h_init.ptr, buf.d_layers[layer].d_h_init,
                    buf.small_adam[layer].h_init_m, buf.small_adam[layer].h_init_v,
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, 0.0, adam_step, (n_heads * d_state) as i32);
            }

            // Attention layers
            for aidx in 0..n_attn {
                let aw = &buf.attn_layers[aidx];
                let daw = &buf.d_attn_layers[aidx];
                let adam = &buf.attn_adam[aidx];
                let mut off = 0usize;
                let n_q = (d_model * d_model) as i32;
                let n_k = (d_model * kv_dim) as i32;
                let n_mlp = (d_model * mlp_dim) as i32;

                native_ops::adamw_clipped_update(aw.q_proj.ptr, daw.d_q_proj, adam.m.offset(off), adam.v.offset(off),
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, wd, adam_step, n_q);
                off += d_model * d_model;
                native_ops::adamw_clipped_update(aw.k_proj.ptr, daw.d_k_proj, adam.m.offset(off), adam.v.offset(off),
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, wd, adam_step, n_k);
                off += d_model * kv_dim;
                native_ops::adamw_clipped_update(aw.v_proj.ptr, daw.d_v_proj, adam.m.offset(off), adam.v.offset(off),
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, wd, adam_step, n_k);
                off += d_model * kv_dim;
                native_ops::adamw_clipped_update(aw.attn_out_proj.ptr, daw.d_attn_out_proj, adam.m.offset(off), adam.v.offset(off),
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, wd, adam_step, n_q);
                off += d_model * d_model;
                native_ops::adamw_clipped_update(aw.mlp_gate.ptr, daw.d_mlp_gate, adam.m.offset(off), adam.v.offset(off),
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, wd, adam_step, n_mlp);
                off += d_model * mlp_dim;
                native_ops::adamw_clipped_update(aw.mlp_up.ptr, daw.d_mlp_up, adam.m.offset(off), adam.v.offset(off),
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, wd, adam_step, n_mlp);
                off += d_model * mlp_dim;
                native_ops::adamw_clipped_update(aw.mlp_down.ptr, daw.d_mlp_down, adam.m.offset(off), adam.v.offset(off),
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, wd, adam_step, (mlp_dim * d_model) as i32);

                native_ops::adamw_clipped_update(aw.attn_norm_gamma.ptr, daw.d_attn_norm_gamma,
                    adam.norm_m.ptr, adam.norm_v.ptr,
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, 0.0, adam_step, d_model as i32);
                native_ops::adamw_clipped_update(aw.mlp_norm_gamma.ptr, daw.d_mlp_norm_gamma,
                    adam.mlp_norm_m.ptr, adam.mlp_norm_v.ptr,
                    norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, 0.0, adam_step, d_model as i32);
            }

            // Global params
            native_ops::adamw_clipped_update(buf.final_norm_gamma.ptr, buf.d_final_norm_gamma.ptr,
                buf.final_norm_adam_m.ptr, buf.final_norm_adam_v.ptr,
                norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, 0.0, adam_step, d_model as i32);
            native_ops::adamw_clipped_update(buf.embedding.ptr, buf.d_embedding.ptr,
                buf.embedding_adam_m.ptr, buf.embedding_adam_v.ptr,
                norm_ptr, max_grad_norm, lr, beta1, beta2, adam_eps, wd, adam_step, (vocab * d_model) as i32);
        }

        // ──── CLAMP SSM PARAMS (prevent gradient explosion feedback loop) ────
        unsafe {
            for layer in 0..n_mamba {
                native_ops::clamp_values(buf.layers[layer].dt_bias.ptr, -6.0, 2.0, n_heads as i32);
            }
        }

        // ──── LOGGING ────
        loss_val_last = micro_loss;

        if micro_loss.is_nan() || micro_loss.is_infinite() {
            eprintln!("FATAL: NaN/inf loss at step {} — stopping training", step);
            break;
        }

        if step % 10 == 0 || step == config.max_steps - 1 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let tok_per_sec = ((step + 1 - start_step) * grad_accum) as f32 * bs as f32 / elapsed;
            eprintln!(
                "step {:5} | loss {:.4} | lr {:.2e} | {:.0} tok/s",
                step, micro_loss, lr * grad_accum as f32, tok_per_sec,
            );
        }

        // ──── CHECKPOINT ────
        // Skip the resume step — we just loaded these weights, re-saving would
        // only overwrite a pristine source-of-truth checkpoint with itself (and
        // clobber the original meta.json in the process).
        if step > 0 && step != start_step && step % config.checkpoint_interval == 0 {
            let ckpt_dir = format!("checkpoints/step-{:06}", step);
            let meta = serde_json::json!({
                "step": step,
                "loss": micro_loss,
                "epoch": epoch,
                "sample_pos": sample_pos,
                "tokens_seen": tokens_seen,
                "dataset_windows": dataset.len(),
                "format": "native_raw",
            });
            if let Err(e) = crate::native_checkpoint::save_native_checkpoint(&buf, &ckpt_dir, meta, &config.model) {
                eprintln!("WARNING: checkpoint save failed: {}", e);
            }
            if let Err(e) = crate::native_checkpoint::save_optimizer_state(&buf, &ckpt_dir) {
                eprintln!("WARNING: optimizer state save failed: {}", e);
            }
            eprintln!("Checkpoint saved: {} (step={}, loss={:.4}, epoch={}, pos={}, tokens={})",
                ckpt_dir, step, micro_loss, epoch, sample_pos, tokens_seen);
            crate::checkpoint::clean_checkpoints("checkpoints", 3, best_val_ckpt.as_deref());
        }

        // ──── VALIDATION LOSS ────
        if let Some(ref val_ds) = val_dataset {
            if step > 0 && step % config.eval_interval == 0 {
                let mut val_loss_sum = 0.0f32;
                let n_val_batches = 10;
                // Seed from config.seed ^ step so val windows are identical for a given
                // step regardless of how many times training has been restarted.
                let mut val_rng = {
                    use rand::SeedableRng;
                    rand::rngs::StdRng::seed_from_u64(config.seed ^ step as u64)
                };
                for _ in 0..n_val_batches {
                    let (vi, vt) = get_batch_i32(val_ds, batch, seq, &mut val_rng);
                    upload_batch(&buf.input_ids, &buf.target_ids, &vi, &vt);
                    // Forward only (reuses same buffers — no backward)
                    unsafe {
                        native_ops::embedding_lookup(
                            buf.embedding.ptr, buf.input_ids.ptr as *const i32,
                            buf.x.ptr, bs as i32, d_model as i32,
                        );
                    }
                    for vlayer in 0..n_layers {
                        use crate::gpu_memory::LayerType;
                        match buf.layer_types[vlayer] {
                        LayerType::Mamba(midx) => {
                        let dst = MambaForwardDst {
                            residual: buf.residual.ptr,
                            x_norm: buf.x_norm.ptr,
                            x_ssm_raw: buf.x_act.ptr,       // in-place SiLU OK
                            x_act: buf.x_act.ptr,
                            z_buf: buf.z_buf.ptr,
                            b_raw: buf.b_raw.ptr,
                            c_raw: buf.c_raw.ptr,
                            b_norm: buf.b_norm_buf.ptr,
                            c_norm: buf.c_norm_buf.ptr,
                            dt_buf: buf.dt_buf.ptr,
                            lambda_raw: buf.lambda_raw.ptr,
                            dd_a_raw: buf.a_vals_buf.ptr,
                            theta_raw: buf.x_ssm_raw.ptr,
                            ssm_out: buf.ssm_out.ptr,
                            y_gated: buf.y_gated.ptr,
                        };
                        unsafe {
                            mamba_block_forward(
                                buf.x.ptr, buf.projected.ptr, buf.lambda_buf.ptr, buf.a_vals_buf.ptr, buf.block_out.ptr,
                                &dst, &buf.layers[midx], &buf,
                                bs as i32, d_model as i32, d_inner as i32, in_proj_out as i32,
                                n_heads as i32, n_groups as i32, d_state as i32,
                                bc_size as i32, batch as i32, seq as i32, head_dim as i32, eps,
                                mp,
                            );
                        }
                        }
                        LayerType::Attention(aidx) => {
                        let ws = config.model.attn_window_size(aidx) as i32;
                        unsafe {
                            attention_block_forward(
                                buf.x.ptr, &buf, &buf.attn_layers[aidx], None,
                                bs as i32, d_model as i32, batch as i32, seq as i32,
                                config.model.attn_n_heads as i32, config.model.attn_kv_heads as i32,
                                config.model.attn_head_dim() as i32, config.model.attn_kv_dim() as i32,
                                config.model.attn_mlp_dim() as i32, ws, eps,
                                mp,
                            );
                        }
                        }
                        } // match
                    }
                    unsafe {
                        native_ops::rmsnorm_fwd(buf.x.ptr, buf.x_norm.ptr,
                            buf.final_norm_gamma.ptr, eps, bs as i32, d_model as i32);
                        native_ops::matmul_f32_bt(buf.x_norm.ptr, buf.embedding.ptr,
                            buf.logits.ptr, bs as i32, vocab as i32, d_model as i32);
                        native_ops::sparse_cross_entropy_fwd(buf.logits.ptr,
                            buf.target_ids.ptr as *const i32,
                            buf.loss.ptr, buf.per_token_loss.ptr, bs as i32, vocab as i32);
                    }
                    // CPU-side loss computation (matches training — avoids GPU reduce precision issues on sm_120)
                    let val_ptl = buf.per_token_loss.to_host();
                    val_loss_sum += val_ptl.iter().sum::<f32>() / val_ptl.len() as f32;
                }
                let val_loss = val_loss_sum / n_val_batches as f32;
                eprintln!("  val_loss: {:.4}", val_loss);
                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    // Only protect checkpoint if one exists at this step
                    let ckpt_name = format!("step-{:06}", step);
                    if std::path::Path::new(&format!("checkpoints/{}", ckpt_name)).exists() {
                        best_val_ckpt = Some(ckpt_name);
                    }
                    eprintln!("  new best val_loss!");
                }

            }
        }
    }

    let total = start_time.elapsed().as_secs_f32();
    eprintln!("\nNative training complete. {} steps in {:.1}s.", config.max_steps, total);
    eprintln!("Epochs: {}, tokens seen: {}, coverage: {:.1}%",
        epoch, tokens_seen, tokens_seen as f64 / (dataset.len() as f64 * seq as f64) * 100.0);

    // Append training log
    let log_entry = serde_json::json!({
        "timestamp": chrono_timestamp(),
        "steps": config.max_steps,
        "tokens_seen": tokens_seen,
        "epochs_completed": epoch,
        "sample_pos": sample_pos,
        "dataset_tokens": dataset.tokens.len(),
        "best_val_loss": best_val_loss,
        "final_loss": loss_val_last,
        "lr": config.learning_rate,
        "batch_size": config.batch_size,
        "seq_len": seq,
        "duration_secs": total,
    });
    if let Ok(mut log) = std::fs::read_to_string("training_log.json")
        .and_then(|s| Ok(serde_json::from_str::<Vec<serde_json::Value>>(&s).unwrap_or_default()))
    {
        log.push(log_entry);
        std::fs::write("training_log.json", serde_json::to_string_pretty(&log).unwrap()).ok();
    } else {
        std::fs::write("training_log.json", serde_json::to_string_pretty(&vec![log_entry]).unwrap()).ok();
    }
    eprintln!("Training log updated: training_log.json");

    unsafe { native_ops::cublas_destroy(); }
}

/// Smoke test: allocate small buffers, run one forward pass, check each kernel.
#[cfg(feature = "cuda")]
pub fn smoke_test_forward(config: &TrainingConfig) {
    use crate::gpu_memory::GpuBuf;
    use crate::native_ops;

    let batch: usize = 2;
    let seq: usize = 32;
    let d_model = config.model.d_model;
    let d_inner = config.model.d_inner();
    let n_heads = config.model.n_heads;
    let d_state = config.model.d_state;
    let n_groups = config.model.n_groups;
    let head_dim = d_inner / n_heads;
    let bs = batch * seq;

    eprintln!("=== CUDA Smoke Test — batch={}, seq={} ===", batch, seq);
    eprintln!("Model: d_model={}, n_layers={}, n_heads={}, d_state={}", d_model, config.model.n_layers, n_heads, d_state);

    // Init CUDA
    eprintln!("[1/8] cuBLAS init...");
    unsafe { native_ops::cublas_init(); }
    eprintln!("  OK");

    // Allocate small test buffers
    eprintln!("[2/8] Allocating test buffers...");
    let x = GpuBuf::from_host(&vec![0.01f32; bs * d_model]);
    let w = GpuBuf::from_host(&vec![0.01f32; d_model * d_inner]);
    let out = GpuBuf::alloc(bs * d_inner);
    eprintln!("  OK");

    // Test matmul
    eprintln!("[3/8] cuBLAS matmul ({} x {})...", bs, d_inner);
    unsafe {
        native_ops::matmul_f32(x.ptr, w.ptr, out.ptr, bs as i32, d_inner as i32, d_model as i32);
        native_ops::cudaDeviceSynchronize();
    }
    let result = out.to_host();
    let has_nan = result.iter().any(|v| v.is_nan());
    eprintln!("  {} (sample: {:.6})", if has_nan { "FAIL — NaN" } else { "OK" }, result[0]);

    // Test RMSNorm
    eprintln!("[4/8] RMSNorm...");
    let gamma = GpuBuf::from_host(&vec![1.0f32; d_model]);
    let x_norm = GpuBuf::alloc(bs * d_model);
    unsafe {
        native_ops::rmsnorm_fwd(x.ptr, x_norm.ptr, gamma.ptr, config.model.norm_eps as f32, bs as i32, d_model as i32);
        native_ops::cudaDeviceSynchronize();
    }
    let result = x_norm.to_host();
    eprintln!("  {} (sample: {:.6})", if result.iter().any(|v| v.is_nan()) { "FAIL" } else { "OK" }, result[0]);

    // Test SiLU
    eprintln!("[5/8] SiLU...");
    let silu_out = GpuBuf::alloc(bs * d_inner);
    unsafe {
        native_ops::silu_fwd(out.ptr, silu_out.ptr, (bs * d_inner) as i32);
        native_ops::cudaDeviceSynchronize();
    }
    let result = silu_out.to_host();
    eprintln!("  {} (sample: {:.6})", if result.iter().any(|v| v.is_nan()) { "FAIL" } else { "OK" }, result[0]);

    // Test embedding
    eprintln!("[6/8] Embedding lookup...");
    let embed = GpuBuf::from_host(&vec![0.01f32; config.model.vocab_size * d_model]);
    let ids_data: Vec<i32> = (0..bs as i32).map(|i| i % 100).collect();
    let ids = GpuBuf::from_host(unsafe { std::slice::from_raw_parts(ids_data.as_ptr() as *const f32, bs) });
    let embed_out = GpuBuf::alloc(bs * d_model);
    unsafe {
        native_ops::embedding_lookup(embed.ptr, ids.ptr as *const i32, embed_out.ptr, bs as i32, d_model as i32);
        native_ops::cudaDeviceSynchronize();
    }
    let result = embed_out.to_host();
    eprintln!("  {} (sample: {:.6})", if result.iter().any(|v| v.is_nan()) { "FAIL" } else { "OK" }, result[0]);

    // Test SSM forward
    eprintln!("[7/8] SSM forward scan (batch={}, seq={}, heads={}, d_state={})...", batch, seq, n_heads, d_state);
    let ssm_x = GpuBuf::from_host(&vec![0.01f32; bs * n_heads * head_dim]);
    let ssm_dt = GpuBuf::from_host(&vec![0.0f32; bs * n_heads]);
    let ssm_b = GpuBuf::from_host(&vec![0.01f32; bs * n_groups * d_state]);
    let ssm_c = GpuBuf::from_host(&vec![0.01f32; bs * n_groups * d_state]);
    let ssm_d = GpuBuf::from_host(&vec![1.0f32; n_heads]);
    let ssm_dt_bias = GpuBuf::from_host(&vec![-3.0f32; n_heads]);
    let ssm_h_init = GpuBuf::from_host(&vec![0.0f32; n_heads * d_state]);
    let ssm_lambda = GpuBuf::from_host(&vec![0.5f32; bs * n_heads]);
    let ssm_a_vals = GpuBuf::from_host(&vec![-0.1f32; bs * n_heads]);
    let ssm_z = GpuBuf::from_host(&vec![0.01f32; bs * n_heads * head_dim]);
    let ssm_y = GpuBuf::alloc(bs * n_heads * head_dim);
    let ssm_y_gated = GpuBuf::alloc(bs * n_heads * head_dim);

    unsafe {
        native_ops::ssm_scan_fwd_gpu(
            ssm_x.ptr, ssm_dt.ptr, ssm_b.ptr, ssm_c.ptr,
            ssm_d.ptr, ssm_dt_bias.ptr, ssm_h_init.ptr,
            ssm_lambda.ptr,
            std::ptr::null(), // no DD-RoPE for smoke test
            ssm_a_vals.ptr,
            ssm_z.ptr,
            ssm_y.ptr, ssm_y_gated.ptr,
            batch as i32, seq as i32, n_heads as i32,
            head_dim as i32, d_state as i32, n_groups as i32,
        );
        native_ops::cudaDeviceSynchronize();
    }
    let result = ssm_y_gated.to_host();
    let has_nan = result.iter().any(|v| v.is_nan());
    let has_inf = result.iter().any(|v| v.is_infinite());
    eprintln!("  {} (sample: {:.6}, nan={}, inf={})",
        if has_nan || has_inf { "FAIL" } else { "OK" }, result[0], has_nan, has_inf);

    // Test sparse cross-entropy
    eprintln!("[8/8] Sparse cross-entropy...");
    let logits = GpuBuf::from_host(&vec![0.01f32; bs * config.model.vocab_size]);
    let targets_data: Vec<i32> = (0..bs as i32).map(|i| i % 100).collect();
    let targets = GpuBuf::from_host(unsafe { std::slice::from_raw_parts(targets_data.as_ptr() as *const f32, bs) });
    let losses = GpuBuf::alloc(bs);
    unsafe {
        let loss_scalar = GpuBuf::alloc(1);
        native_ops::sparse_cross_entropy_fwd(logits.ptr, targets.ptr as *const i32, loss_scalar.ptr, losses.ptr, bs as i32, config.model.vocab_size as i32);
        native_ops::cudaDeviceSynchronize();
    }
    let result = losses.to_host();
    let has_nan = result.iter().any(|v| v.is_nan());
    eprintln!("  {} (sample loss: {:.4})", if has_nan { "FAIL" } else { "OK" }, result[0]);

    unsafe { native_ops::cublas_destroy(); }
    eprintln!("\n=== Smoke test complete ===");
}

/// Extended smoke test: full forward pass for every config branch.
/// Tests pure Mamba, hybrid (interval + explicit), windowed attention,
/// and byte-level vocab — all with small dimensions on GPU.
#[cfg(feature = "cuda")]
pub fn smoke_test_configs() {
    use crate::config::{ModelConfig, TrainingConfig};
    use crate::gpu_memory::{GpuBuf, LayerType};
    use crate::native_ops;

    let batch: usize = 2;
    let seq: usize = 32;

    // Base small config shared by all variants
    let base = ModelConfig {
        d_model: 128,
        n_layers: 8,
        d_state: 16,
        expand: 2,
        n_heads: 8,
        n_groups: 2,
        chunk_size: 64,
        vocab_size: 1000,
        max_seq_len: 64,
        norm_eps: 1e-5,
        attention_interval: 0,
        attn_n_heads: 4,
        attn_kv_heads: 2,
        attn_mlp_expand: 4,
        attn_window_sizes: vec![],
        attention_layers: vec![],
        byte_level: false,
        bwd_chunk_size: 8,
    };

    let configs: Vec<(&str, ModelConfig)> = vec![
        ("pure_mamba", base.clone()),
        ("hybrid_interval", ModelConfig { attention_interval: 4, ..base.clone() }),
        ("hybrid_explicit", ModelConfig { attention_layers: vec![2, 6], ..base.clone() }),
        ("windowed_attention", ModelConfig {
            attention_interval: 4,
            attn_window_sizes: vec![16, 16],
            ..base.clone()
        }),
        ("byte_level", ModelConfig { byte_level: true, vocab_size: 259, ..base.clone() }),
    ];

    eprintln!("=== Extended CUDA Smoke Test — {} config variants ===\n", configs.len());
    unsafe { native_ops::cublas_init(); }

    let mut pass_count = 0;
    let mut fail_count = 0;

    for (name, model_config) in &configs {
        let training_config = TrainingConfig {
            model: model_config.clone(),
            batch_size: batch,
            max_steps: 1,
            ..TrainingConfig::default()
        };

        let d_model = model_config.d_model;
        let d_inner = model_config.d_inner();
        let n_heads = model_config.n_heads;
        let n_groups = model_config.n_groups;
        let d_state = model_config.d_state;
        let n_layers = model_config.n_layers;
        let vocab = model_config.vocab_size;
        let head_dim = d_inner / n_heads;
        let bc_size = n_groups * d_state;
        let theta_proj = n_heads * d_state / 2;
        let in_proj_out = d_inner + d_inner + bc_size * 2 + n_heads * 3 + theta_proj;
        let bs = batch * seq;
        let eps = model_config.norm_eps as f32;

        eprint!("  {:24} (mamba={}, attn={}, vocab={}) ... ",
            name, model_config.n_mamba_layers(), model_config.n_attn_layers(), vocab);

        // Allocate and init
        let mut buf = TrainingBuffers::allocate(&model_config, batch, seq, false, false);
        init_weights_random(&mut buf, &training_config);

        // Create fake input/target data (token IDs in valid range)
        let input_data: Vec<i32> = (0..bs).map(|i| (i % vocab) as i32).collect();
        let target_data: Vec<i32> = (0..bs).map(|i| ((i + 1) % vocab) as i32).collect();
        let input_buf = GpuBuf::from_host(unsafe {
            std::slice::from_raw_parts(input_data.as_ptr() as *const f32, bs)
        });
        let target_buf = GpuBuf::from_host(unsafe {
            std::slice::from_raw_parts(target_data.as_ptr() as *const f32, bs)
        });
        buf.input_ids = input_buf;
        buf.target_ids = target_buf;

        // ──── FORWARD PASS ────
        // 1. Embedding lookup
        unsafe {
            native_ops::embedding_lookup(
                buf.embedding.ptr, buf.input_ids.ptr as *const i32,
                buf.x.ptr, bs as i32, d_model as i32,
            );
        }

        // 2. Layer loop (mirrors validation forward pass)
        for layer in 0..n_layers {
            match buf.layer_types[layer] {
                LayerType::Mamba(midx) => {
                    let dst = MambaForwardDst {
                        residual: buf.residual.ptr,
                        x_norm: buf.x_norm.ptr,
                        x_ssm_raw: buf.x_act.ptr,
                        x_act: buf.x_act.ptr,
                        z_buf: buf.z_buf.ptr,
                        b_raw: buf.b_raw.ptr,
                        c_raw: buf.c_raw.ptr,
                        b_norm: buf.b_norm_buf.ptr,
                        c_norm: buf.c_norm_buf.ptr,
                        dt_buf: buf.dt_buf.ptr,
                        lambda_raw: buf.lambda_raw.ptr,
                        dd_a_raw: buf.a_vals_buf.ptr,
                        theta_raw: buf.x_ssm_raw.ptr,
                        ssm_out: buf.ssm_out.ptr,
                        y_gated: buf.y_gated.ptr,
                    };
                    unsafe {
                        mamba_block_forward(
                            buf.x.ptr, buf.projected.ptr, buf.lambda_buf.ptr,
                            buf.a_vals_buf.ptr, buf.block_out.ptr,
                            &dst, &buf.layers[midx], &buf,
                            bs as i32, d_model as i32, d_inner as i32, in_proj_out as i32,
                            n_heads as i32, n_groups as i32, d_state as i32,
                            bc_size as i32, batch as i32, seq as i32, head_dim as i32, eps,
                            false,
                        );
                    }
                }
                LayerType::Attention(aidx) => {
                    let ws = model_config.attn_window_size(aidx) as i32;
                    unsafe {
                        attention_block_forward(
                            buf.x.ptr, &buf, &buf.attn_layers[aidx], None,
                            bs as i32, d_model as i32, batch as i32, seq as i32,
                            model_config.attn_n_heads as i32, model_config.attn_kv_heads as i32,
                            model_config.attn_head_dim() as i32, model_config.attn_kv_dim() as i32,
                            model_config.attn_mlp_dim() as i32, ws, eps,
                            false,
                        );
                    }
                }
            }
        }

        // 3. Final norm → logits → loss
        unsafe {
            native_ops::rmsnorm_fwd(
                buf.x.ptr, buf.x_norm.ptr,
                buf.final_norm_gamma.ptr, eps, bs as i32, d_model as i32,
            );
            native_ops::matmul_f32_bt(
                buf.x_norm.ptr, buf.embedding.ptr, buf.logits.ptr,
                bs as i32, vocab as i32, d_model as i32,
            );
            native_ops::sparse_cross_entropy_fwd(
                buf.logits.ptr, buf.target_ids.ptr as *const i32,
                buf.loss.ptr, buf.per_token_loss.ptr, bs as i32, vocab as i32,
            );
            native_ops::cudaDeviceSynchronize();
        }

        // 4. Check results
        let per_token_loss = buf.per_token_loss.to_host();
        let loss = per_token_loss.iter().sum::<f32>() / per_token_loss.len() as f32;
        let has_nan = per_token_loss.iter().any(|v| v.is_nan());
        let has_inf = per_token_loss.iter().any(|v| v.is_infinite());

        if has_nan || has_inf {
            eprintln!("FAIL (loss={:.4}, nan={}, inf={})", loss, has_nan, has_inf);
            fail_count += 1;
        } else {
            eprintln!("PASS (loss={:.4})", loss);
            pass_count += 1;
        }
    }

    unsafe { native_ops::cublas_destroy(); }
    eprintln!("\n=== Config smoke test: {}/{} passed ===",
        pass_count, pass_count + fail_count);
    if fail_count > 0 {
        eprintln!("WARNING: {} config(s) FAILED!", fail_count);
    }
}

/// Initialize weights for training from scratch.
/// Embedding uses random init (breaks token symmetry). Projections use a single
/// pre-generated random buffer sliced per layer (fast). Small params use constants.
#[cfg(feature = "cuda")]
fn init_weights_random(buf: &mut TrainingBuffers, config: &TrainingConfig) {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let std_dev = 0.02f32;
    let n_layers = config.model.n_layers;
    let residual_scale = 1.0 / (2.0 * n_layers as f32).sqrt();

    let init_start = std::time::Instant::now();

    // Embedding — random (must break symmetry between tokens)
    eprintln!("Initializing weights...");
    let embed_data: Vec<f32> = (0..buf.vocab * buf.d_model)
        .map(|_| rng.gen::<f32>() * std_dev * 2.0 - std_dev)
        .collect();
    buf.embedding = GpuBuf::from_host(&embed_data);
    eprintln!("  Embedding: {:.1}s", init_start.elapsed().as_secs_f32());

    // Final norm gamma = ones
    buf.final_norm_gamma = GpuBuf::from_host(&vec![1.0f32; buf.d_model]);

    // Pre-generate one random buffer for the largest projection, reuse across layers
    // Each layer gets a different slice via rotating offset to avoid exact symmetry
    let max_proj_size = buf.d_model * buf.in_proj_out;
    let rand_pool: Vec<f32> = (0..max_proj_size)
        .map(|_| rng.gen::<f32>() * std_dev * 2.0 - std_dev)
        .collect();

    // Per-layer weights — projections from random pool, small params as constants
    for (i, layer) in buf.layers.iter_mut().enumerate() {
        eprint!("\r  Mamba layer {}/{} ({:.0}%)", i + 1, n_layers, (i + 1) as f32 / n_layers as f32 * 100.0);

        // in_proj: use random pool with per-layer rotation
        let in_size = buf.d_model * buf.in_proj_out;
        let offset = (i * 7919) % rand_pool.len(); // prime offset per layer
        let in_proj_data: Vec<f32> = (0..in_size)
            .map(|j| rand_pool[(offset + j) % rand_pool.len()])
            .collect();
        layer.in_proj = GpuBuf::from_host(&in_proj_data);

        // out_proj: from pool with residual scaling
        let out_size = buf.d_inner * buf.d_model;
        let offset2 = (i * 6271 + 3137) % rand_pool.len();
        let out_proj_data: Vec<f32> = (0..out_size)
            .map(|j| rand_pool[(offset2 + j) % rand_pool.len()] * residual_scale)
            .collect();
        layer.out_proj = GpuBuf::from_host(&out_proj_data);

        // Constants for small params
        layer.norm_gamma = GpuBuf::from_host(&vec![1.0f32; buf.d_model]);
        layer.b_gamma = GpuBuf::from_host(&vec![1.0f32; buf.n_groups * buf.d_state]);
        layer.c_gamma = GpuBuf::from_host(&vec![1.0f32; buf.n_groups * buf.d_state]);
        layer.b_bias = GpuBuf::from_host(&vec![1.0f32; buf.n_groups * buf.d_state]);
        layer.c_bias = GpuBuf::from_host(&vec![1.0f32; buf.n_groups * buf.d_state]);
        layer.d_skip = GpuBuf::from_host(&vec![1.0f32; buf.n_heads]);
        layer.h_init = GpuBuf::from_host(&vec![0.01f32; buf.n_heads * buf.d_state]);

        // dt_bias needs specific range — small vec, fast
        let dt_bias_data: Vec<f32> = (0..buf.n_heads)
            .map(|_| rng.gen::<f32>() * 2.0 - 4.0)
            .collect();
        layer.dt_bias = GpuBuf::from_host(&dt_bias_data);
    }
    eprintln!("\r  Mamba layers: {:.1}s              ", init_start.elapsed().as_secs_f32());

    // Attention layer weights — from same random pool with Xavier scaling
    let kv_dim = config.model.attn_kv_dim();
    let mlp_dim = config.model.attn_mlp_dim();
    let d_model = config.model.d_model;

    for (i, layer) in buf.attn_layers.iter_mut().enumerate() {
        let xavier_qo = (6.0f32 / (d_model + d_model) as f32).sqrt();
        let xavier_kv = (6.0f32 / (d_model + kv_dim) as f32).sqrt();
        let xavier_gate_up = (6.0f32 / (d_model + mlp_dim) as f32).sqrt();
        let xavier_down = (6.0f32 / (mlp_dim + d_model) as f32).sqrt();

        // Scale random pool values to Xavier range
        let off = (i * 4937) % rand_pool.len();
        let q_data: Vec<f32> = (0..d_model * d_model)
            .map(|j| rand_pool[(off + j) % rand_pool.len()] / std_dev * xavier_qo)
            .collect();
        layer.q_proj = GpuBuf::from_host(&q_data);

        let off = (i * 4937 + d_model * d_model) % rand_pool.len();
        let k_data: Vec<f32> = (0..d_model * kv_dim)
            .map(|j| rand_pool[(off + j) % rand_pool.len()] / std_dev * xavier_kv)
            .collect();
        layer.k_proj = GpuBuf::from_host(&k_data);

        let off = (i * 4937 + d_model * d_model + d_model * kv_dim) % rand_pool.len();
        let v_data: Vec<f32> = (0..d_model * kv_dim)
            .map(|j| rand_pool[(off + j) % rand_pool.len()] / std_dev * xavier_kv)
            .collect();
        layer.v_proj = GpuBuf::from_host(&v_data);

        let off = (i * 3571) % rand_pool.len();
        let out_data: Vec<f32> = (0..d_model * d_model)
            .map(|j| rand_pool[(off + j) % rand_pool.len()] / std_dev * xavier_qo * residual_scale)
            .collect();
        layer.attn_out_proj = GpuBuf::from_host(&out_data);

        layer.attn_norm_gamma = GpuBuf::from_host(&vec![1.0f32; d_model]);
        layer.mlp_norm_gamma = GpuBuf::from_host(&vec![1.0f32; d_model]);

        let off = (i * 2741) % rand_pool.len();
        let gate_data: Vec<f32> = (0..d_model * mlp_dim)
            .map(|j| rand_pool[(off + j) % rand_pool.len()] / std_dev * xavier_gate_up)
            .collect();
        layer.mlp_gate = GpuBuf::from_host(&gate_data);

        let off = (i * 2741 + d_model * mlp_dim) % rand_pool.len();
        let up_data: Vec<f32> = (0..d_model * mlp_dim)
            .map(|j| rand_pool[(off + j) % rand_pool.len()] / std_dev * xavier_gate_up)
            .collect();
        layer.mlp_up = GpuBuf::from_host(&up_data);

        let off = (i * 1999) % rand_pool.len();
        let down_data: Vec<f32> = (0..mlp_dim * d_model)
            .map(|j| rand_pool[(off + j) % rand_pool.len()] / std_dev * xavier_down * residual_scale)
            .collect();
        layer.mlp_down = GpuBuf::from_host(&down_data);
    }
    eprintln!("  Weight init complete: {:.1}s", init_start.elapsed().as_secs_f32());
}

/// Get a batch as i32 token IDs (for GPU embedding lookup).
#[cfg(feature = "cuda")]
fn get_batch_i32(dataset: &TokenDataset, batch: usize, seq: usize, rng: &mut impl rand::Rng) -> (Vec<i32>, Vec<i32>) {
    let max_idx = dataset.len();
    let mut inputs = Vec::with_capacity(batch * seq);
    let mut targets = Vec::with_capacity(batch * seq);

    for _ in 0..batch {
        let idx = rng.gen_range(0..max_idx);
        for j in 0..seq {
            inputs.push(dataset.tokens[idx + j] as i32);
            targets.push(dataset.tokens[idx + j + 1] as i32);
        }
    }

    (inputs, targets)
}

/// Get a batch using sequential coprime-stride sampling.
/// Guarantees every window is visited exactly once per epoch.
#[cfg(feature = "cuda")]
fn get_batch_sequential(
    dataset: &TokenDataset, batch: usize, seq: usize,
    stride: usize, pos: &mut usize, epoch: &mut usize,
) -> (Vec<i32>, Vec<i32>) {
    let n = dataset.len();
    let mut inputs = Vec::with_capacity(batch * seq);
    let mut targets = Vec::with_capacity(batch * seq);

    for _ in 0..batch {
        let idx = (*pos * stride) % n;
        *pos += 1;
        if *pos >= n {
            *pos = 0;
            *epoch += 1;
            eprintln!("  epoch {} complete", *epoch);
        }
        for j in 0..seq {
            inputs.push(dataset.tokens[idx + j] as i32);
            targets.push(dataset.tokens[idx + j + 1] as i32);
        }
    }

    (inputs, targets)
}

/// Upload token IDs to GPU.
#[cfg(feature = "cuda")]
fn upload_batch(input_buf: &GpuBuf, target_buf: &GpuBuf, inputs: &[i32], targets: &[i32]) {
    unsafe {
        cuda_memcpy_h2d(input_buf.ptr as *mut std::ffi::c_void,
                        inputs.as_ptr() as *const std::ffi::c_void,
                        inputs.len() * 4);
        cuda_memcpy_h2d(target_buf.ptr as *mut std::ffi::c_void,
                        targets.as_ptr() as *const std::ffi::c_void,
                        targets.len() * 4);
    }
}

#[cfg(feature = "cuda")]
extern "C" {
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                  count: usize, kind: i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_set_device(device: i32) -> i32 { cudaSetDevice(device) }

#[cfg(feature = "cuda")]
unsafe fn cuda_memcpy_h2d(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, bytes: usize) {
    cudaMemcpy(dst, src, bytes, 1); // cudaMemcpyHostToDevice
}

#[cfg(feature = "cuda")]
unsafe fn cuda_memcpy_d2d(dst: *mut f32, src: *const f32, n: usize) {
    cudaMemcpy(dst as *mut std::ffi::c_void, src as *const std::ffi::c_void,
               n * 4, 3); // cudaMemcpyDeviceToDevice
}

#[cfg(feature = "cuda")]
unsafe fn cuda_convert_f32_to_bf16(dst: *mut u16, src: *const f32, n: usize) {
    native_ops::convert_f32_to_bf16(src, dst, n as i32);
}

#[cfg(feature = "cuda")]
unsafe fn cuda_convert_bf16_to_f32(dst: *mut f32, src: *const u16, n: usize) {
    native_ops::convert_bf16_to_f32(src, dst, n as i32);
}

/// Copy live FP32 forward scratch into BF16 per-layer storage.
/// Only called in the bf16_activations path after a Mamba forward into shared scratch.
#[cfg(feature = "cuda")]
unsafe fn save_mamba_layer_bf16(
    buf: &TrainingBuffers,
    saved: &crate::gpu_memory::PerLayerSavedBf16,
    bs: usize, d_model: usize, d_inner: usize,
    bc_size: usize, n_heads: usize, theta_proj: usize,
) {
    // x_act = SiLU(x_ssm_raw) and y_gated = ssm_out * SiLU(z_buf) are NOT saved —
    // both are recomputed in backward from tensors already present here.
    cuda_convert_f32_to_bf16(saved.residual.ptr,   buf.residual.ptr,   bs * d_model);
    cuda_convert_f32_to_bf16(saved.x_norm.ptr,     buf.x_norm.ptr,     bs * d_model);
    cuda_convert_f32_to_bf16(saved.x_ssm_raw.ptr,  buf.x_ssm_raw.ptr,  bs * d_inner);
    cuda_convert_f32_to_bf16(saved.z_buf.ptr,      buf.z_buf.ptr,      bs * d_inner);
    cuda_convert_f32_to_bf16(saved.b_raw.ptr,      buf.b_raw.ptr,      bs * bc_size);
    cuda_convert_f32_to_bf16(saved.c_raw.ptr,      buf.c_raw.ptr,      bs * bc_size);
    cuda_convert_f32_to_bf16(saved.b_norm.ptr,     buf.b_norm_buf.ptr, bs * bc_size);
    cuda_convert_f32_to_bf16(saved.c_norm.ptr,     buf.c_norm_buf.ptr, bs * bc_size);
    cuda_convert_f32_to_bf16(saved.dt_buf.ptr,     buf.dt_buf.ptr,     bs * n_heads);
    cuda_convert_f32_to_bf16(saved.lambda_raw.ptr, buf.lambda_raw.ptr, bs * n_heads);
    cuda_convert_f32_to_bf16(saved.dd_a_raw.ptr,   buf.dd_a_raw.ptr,   bs * n_heads);
    cuda_convert_f32_to_bf16(saved.theta_raw.ptr,  buf.theta_raw.ptr,  bs * theta_proj);
    cuda_convert_f32_to_bf16(saved.ssm_out.ptr,    buf.ssm_out.ptr,    bs * d_inner);
}

/// Stub for non-CUDA
#[cfg(not(feature = "cuda"))]
pub fn train_native(_config: &crate::config::TrainingConfig, _device_id: i32) {
    panic!("Native training requires --features cuda");
}
