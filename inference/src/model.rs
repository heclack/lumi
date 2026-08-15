/// Forward pass: per-layer Mamba block decode step, chained into the full model.
///
/// Op order mirrors `archive/metal-inference/src/model.rs`'s
/// `MambaBlock::forward_step` (~line 233, CPU-fallback path) and
/// `NmModel::forward_step` (~line 766), translated from Candle tensor ops to
/// direct kernel calls on `GpuBuf`s — the same translation
/// `training/src/native_trainer.rs::mamba_block_forward` already did for the
/// training/windowed path, which this module steals its idioms from.
use lumi_kernels::buf::{memcpy_d2d, memcpy_h2d, GpuBuf};
use lumi_kernels::ops;

use crate::config::ModelConfig;
use crate::state::LayerState;
use crate::weights::ModelWeights;

pub struct Model {
    config: ModelConfig,
    weights: ModelWeights,
    states: Vec<LayerState>,

    // ── Work buffers, allocated once and reused across every decode step ──
    token_id_dev: GpuBuf, // [1] i32 bit-pattern stored in an f32-typed slot (see embed_token)
    x: GpuBuf,             // [d_model] residual stream, mutated in place layer to layer
    residual: GpuBuf,      // [d_model] x saved before norm, added back after the block
    x_norm: GpuBuf,        // [d_model]
    projected: GpuBuf,     // [in_proj_out] in_proj output, sliced below by pointer offset
    b_normed: GpuBuf,      // [bc_size]
    c_normed: GpuBuf,      // [bc_size]
    lambda_buf: GpuBuf,    // [n_heads] post-sigmoid
    a_vals_buf: GpuBuf,    // [n_heads] post neg-softplus-clamp
    x_act: GpuBuf,         // [d_inner] SiLU(x_ssm)
    y: GpuBuf,              // [d_inner] SSM output, already silu(z)-gated by the kernel
    block_out: GpuBuf,      // [d_model] out_proj(y)
    x_final: GpuBuf,        // [d_model] after the model's final norm
    logits: GpuBuf,          // [vocab]
}

impl Model {
    /// Allocate every work buffer once and initialize per-layer SSM state from
    /// each layer's learned `h_init`. `weights` is moved in — the model owns
    /// its weights for the rest of the process's life.
    pub fn new(config: ModelConfig, weights: ModelWeights) -> Self {
        let d_model = config.d_model;
        let d_inner = config.d_inner();
        let bc_size = config.bc_size();
        let n_heads = config.n_heads;
        let in_proj_out = config.in_proj_out();
        let vocab = config.vocab_size;

        let mut states: Vec<LayerState> =
            (0..config.n_layers).map(|_| LayerState::new(&config)).collect();
        for (state, layer) in states.iter_mut().zip(weights.layers.iter()) {
            state.reset(&config, layer);
        }

        Self {
            token_id_dev: GpuBuf::alloc(1),
            x: GpuBuf::alloc(d_model),
            residual: GpuBuf::alloc(d_model),
            x_norm: GpuBuf::alloc(d_model),
            projected: GpuBuf::alloc(in_proj_out),
            b_normed: GpuBuf::alloc(bc_size),
            c_normed: GpuBuf::alloc(bc_size),
            lambda_buf: GpuBuf::alloc(n_heads),
            a_vals_buf: GpuBuf::alloc(n_heads),
            x_act: GpuBuf::alloc(d_inner),
            y: GpuBuf::alloc(d_inner),
            block_out: GpuBuf::alloc(d_model),
            x_final: GpuBuf::alloc(d_model),
            logits: GpuBuf::alloc(vocab),
            config,
            weights,
            states,
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Reset every layer's SSM state to a fresh generation (h <- learned
    /// h_init broadcast across head_dim, prev_bx/cum_angle <- 0). Call before
    /// starting a new prompt so state doesn't leak across generations.
    pub fn reset_state(&mut self) {
        for (state, layer) in self.states.iter_mut().zip(self.weights.layers.iter()) {
            state.reset(&self.config, layer);
        }
    }

    /// Upload `token_id` and write its embedding row into `self.x`.
    ///
    /// `GpuBuf` is nominally an f32 buffer, but `embedding_lookup` wants
    /// `*const i32` token ids. Following `upload_batch` in
    /// `training/src/native_trainer.rs`, we reinterpret the single 4-byte slot:
    /// copy the i32's raw bytes in via `memcpy_h2d`, then hand the same
    /// pointer to the kernel cast as `*const i32`.
    fn embed_token(&mut self, token_id: u32) {
        let id = token_id as i32;
        unsafe {
            // SAFETY: token_id_dev is a 1-element (4-byte) GpuBuf; `id` is one
            // i32 (4 bytes). f32 and i32 share size and alignment, so writing
            // the i32's bit pattern into the f32 slot and reading it back out
            // as `*const i32` in embedding_lookup below is a sound reinterpret.
            memcpy_h2d(
                self.token_id_dev.ptr as *mut std::ffi::c_void,
                (&id as *const i32) as *const std::ffi::c_void,
                4,
            );
            ops::embedding_lookup(
                self.weights.embedding.ptr,
                self.token_id_dev.ptr as *const i32,
                self.x.ptr,
                1,
                self.config.d_model as i32,
            );
        }
    }

    /// Run one Mamba block's decode step in place on `self.x`.
    fn mamba_block_step(&mut self, layer_idx: usize) {
        let c = &self.config;
        let d_model = c.d_model as i32;
        let d_inner = c.d_inner() as i32;
        let bc_size = c.bc_size() as i32;
        let n_heads = c.n_heads as i32;
        let n_groups = c.n_groups as i32;
        let d_state = c.d_state as i32;
        let head_dim = c.head_dim() as i32;
        let eps = c.norm_eps as f32;
        let in_proj_out = c.in_proj_out() as i32;

        let layer = &self.weights.layers[layer_idx];
        let state = &mut self.states[layer_idx];

        // Save the residual before norm into a dedicated buffer, rather than
        // relying on the final elemwise_add to alias one of its own inputs
        // with its output — native_trainer.rs's mamba_block_forward does the
        // same (writes block_out + a separate `residual` into x, never
        // aliasing x with either add input) and elemwise_add's kernel makes
        // no documented aliasing guarantee.
        unsafe {
            // SAFETY: residual and x are both [d_model]-element GpuBufs
            // (allocated together in `new`), so copying d_model elements
            // device-to-device stays within both allocations.
            memcpy_d2d(
                self.residual.ptr as *mut std::ffi::c_void,
                self.x.ptr as *const std::ffi::c_void,
                d_model as usize * 4,
            );
        }

        unsafe {
            // 1. Pre-block RMSNorm.
            ops::rmsnorm_fwd(self.x.ptr, self.x_norm.ptr, layer.norm_gamma.ptr, eps, 1, d_model);

            // 2. in_proj: [1, d_model] @ [d_model, in_proj_out] -> [1, in_proj_out].
            ops::matmul_f32(
                self.x_norm.ptr,
                layer.in_proj.ptr,
                self.projected.ptr,
                1,
                in_proj_out,
                d_model,
            );
        }

        // 3. Slice `projected` by pointer offset. Batch=1 means every segment
        // is a contiguous run of columns starting at a fixed offset, so no
        // split kernel is needed. `GpuBuf::offset` bound-checks (offset <
        // len); `ModelConfig::in_proj_out` is defined as the exact sum of
        // these six segment lengths, so every offset+length pair below stays
        // within the `projected` allocation.
        let d_inner_u = d_inner as usize;
        let bc_size_u = bc_size as usize;
        let n_heads_u = n_heads as usize;
        let x_ssm_ptr = self.projected.offset(0);
        let z_ptr = self.projected.offset(d_inner_u);
        let b_ptr = self.projected.offset(2 * d_inner_u);
        let c_ptr = self.projected.offset(2 * d_inner_u + bc_size_u);
        let dt_ptr = self.projected.offset(2 * d_inner_u + 2 * bc_size_u);
        let lambda_ptr = self.projected.offset(2 * d_inner_u + 2 * bc_size_u + n_heads_u);
        let dd_a_ptr = self.projected.offset(2 * d_inner_u + 2 * bc_size_u + 2 * n_heads_u);
        let theta_ptr = self.projected.offset(2 * d_inner_u + 2 * bc_size_u + 3 * n_heads_u);

        unsafe {
            // 4. BCNorm: the whole bc_size-wide B (resp. C) vector is ONE
            // RMSNorm row (rows=1, dim=bc_size) — i.e. a single RMS scalar
            // over all n_groups*d_state elements, not a per-group norm. This
            // matches archive/metal-inference's RmsNorm::forward (mean over
            // the full last dim of the single bc_size-length tensor for one
            // token) and native_trainer.rs::mamba_block_forward's
            // `rmsnorm_bias_fwd(..., bs, bc_size)` call.
            ops::rmsnorm_bias_fwd(
                b_ptr, self.b_normed.ptr, layer.b_gamma.ptr, layer.b_bias.ptr, eps, 1, bc_size,
            );
            ops::rmsnorm_bias_fwd(
                c_ptr, self.c_normed.ptr, layer.c_gamma.ptr, layer.c_bias.ptr, eps, 1, bc_size,
            );

            // 5. Elementwise prep. dt/theta pass through raw — ssm_step_gpu
            // applies softplus(dt+dt_bias) and tanh(theta)*pi internally (see
            // its doc comment in kernels/src/ops.rs).
            ops::sigmoid_fwd(lambda_ptr, self.lambda_buf.ptr, n_heads);
            ops::neg_softplus_clamp(dd_a_ptr, self.a_vals_buf.ptr, ops::A_VAL_CLAMP_MIN, ops::A_VAL_CLAMP_MAX, n_heads);
            ops::silu_fwd(x_ssm_ptr, self.x_act.ptr, d_inner);

            // 6. Single-token SSM step: mutates state.h / state.prev_bx /
            // state.cum_angle in place, writes the silu(z)-gated output into
            // self.y.
            ops::ssm_step_gpu(
                self.x_act.ptr,
                dt_ptr,
                self.b_normed.ptr,
                self.c_normed.ptr,
                layer.d_skip.ptr,
                layer.dt_bias.ptr,
                self.lambda_buf.ptr,
                theta_ptr,
                self.a_vals_buf.ptr,
                z_ptr,
                state.h.ptr,
                state.prev_bx.ptr,
                state.cum_angle.ptr,
                self.y.ptr,
                n_heads,
                head_dim,
                d_state,
                n_groups,
            );

            // 7. out_proj + residual add. block_out and residual are distinct
            // allocations from the add's output (self.x), so neither input
            // aliases the output.
            ops::matmul_f32(self.y.ptr, layer.out_proj.ptr, self.block_out.ptr, 1, d_model, d_inner);
            ops::elemwise_add(self.block_out.ptr, self.residual.ptr, self.x.ptr, d_model);
        }
    }

    /// Run one full decode step: embed `token_id`, push it through every
    /// layer, apply the final norm, and project to logits. Returns the
    /// logits copied to host. Mirrors `NmModel::forward_step` in the archived
    /// reference (embedding -> blocks -> final_norm -> x @ embedding^T).
    ///
    /// For prompt prefill, call this once per prompt token in order and
    /// ignore every return value except the last — each call still advances
    /// every layer's SSM state, which is the only thing prefill needs.
    pub fn forward_step(&mut self, token_id: u32) -> Vec<f32> {
        self.embed_token(token_id);

        for layer_idx in 0..self.config.n_layers {
            self.mamba_block_step(layer_idx);
        }

        let d_model = self.config.d_model as i32;
        let vocab = self.config.vocab_size as i32;
        let eps = self.config.norm_eps as f32;
        unsafe {
            ops::rmsnorm_fwd(
                self.x.ptr,
                self.x_final.ptr,
                self.weights.final_norm_gamma.ptr,
                eps,
                1,
                d_model,
            );
            // logits = x_final @ embedding^T : [1, d_model] x [vocab, d_model]^T -> [1, vocab].
            ops::matmul_f32_bt(
                self.x_final.ptr,
                self.weights.embedding.ptr,
                self.logits.ptr,
                1,
                vocab,
                d_model,
            );
            // Kernels run async on the default stream. GpuBuf::to_host's copy
            // is synchronous and would itself serialize, but sync explicitly
            // here so a hang or an illegal-memory-access from the matmul
            // surfaces at this call site rather than downstream.
            ops::cudaDeviceSynchronize();
        }
        self.logits.to_host()
    }
}
