/// Lumi Mamba-3 + Attention hybrid model for inference using Candle + Metal.
///
/// Implements single-token autoregressive generation with:
/// - Persistent SSM state for Mamba blocks (O(1) memory)
/// - KV cache for Attention blocks (grows linearly with sequence)

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use std::sync::Arc;

/// Metal-compatible sigmoid: 1/(1+exp(-x)). Avoids candle_nn::ops::sigmoid which lacks Metal support.
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let ones = x.ones_like()?;
    let exp_neg = x.neg()?.exp()?;
    let denom = ones.broadcast_add(&exp_neg)?;
    ones.broadcast_div(&denom)
}

/// Model configuration (mirrors the training config).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub d_state: usize,
    pub expand: usize,
    pub n_heads: usize,
    pub n_groups: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    #[serde(default)]
    pub attention_interval: usize,
    #[serde(default = "default_attn_n_heads")]
    pub attn_n_heads: usize,
    #[serde(default = "default_attn_kv_heads")]
    pub attn_kv_heads: usize,
    #[serde(default = "default_attn_mlp_expand")]
    pub attn_mlp_expand: usize,
    #[serde(default)]
    pub attn_window_sizes: Vec<usize>,
    #[serde(default)]
    pub attention_layers: Vec<usize>,
    /// Byte-level tokenization mode (vocab_size=259, no BPE tokenizer needed).
    #[serde(default)]
    pub byte_level: bool,
}
fn default_attn_n_heads() -> usize { 16 }
fn default_attn_kv_heads() -> usize { 4 }
fn default_attn_mlp_expand() -> usize { 4 }

impl ModelConfig {
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }
    pub fn head_dim(&self) -> usize {
        self.d_inner() / self.n_heads
    }
    pub fn attn_head_dim(&self) -> usize {
        self.d_model / self.attn_n_heads
    }
    pub fn attn_kv_dim(&self) -> usize {
        self.attn_kv_heads * self.attn_head_dim()
    }
    pub fn attn_mlp_dim(&self) -> usize {
        self.attn_mlp_expand * self.d_model
    }
    pub fn is_attention_layer(&self, i: usize) -> bool {
        if !self.attention_layers.is_empty() {
            return self.attention_layers.contains(&i);
        }
        self.attention_interval > 0 && (i + 1) % self.attention_interval == 0
    }
    pub fn n_attn_layers(&self) -> usize {
        if !self.attention_layers.is_empty() {
            return self.attention_layers.len();
        }
        if self.attention_interval == 0 { return 0; }
        (0..self.n_layers).filter(|i| self.is_attention_layer(*i)).count()
    }
    pub fn attn_window_size(&self, attn_idx: usize) -> usize {
        if attn_idx < self.attn_window_sizes.len() {
            self.attn_window_sizes[attn_idx]
        } else {
            0
        }
    }
}

/// RMSNorm layer.
struct RmsNorm {
    gamma: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self { gamma, eps: 1e-5 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1)?;
        let norm = (mean_sq + self.eps)?.sqrt()?.recip()?;
        let normalized = x.broadcast_mul(&norm)?;
        normalized.broadcast_mul(&self.gamma)
    }
}

/// SSM hidden state for one Mamba block.
/// On Metal: uses fused GPU-resident state buffers (MetalSsmState).
/// On CPU: uses Candle tensors + Vec for angles.
pub struct SsmState {
    pub h: Tensor,            // [n_heads, head_dim, d_state]
    pub prev_bx: Tensor,      // [n_heads, head_dim, d_state]
    pub cum_angle: Vec<f32>,  // [n_heads * d_state/2] (CPU fallback for DD-RoPE)
    pub metal_state: Option<crate::metal_ssm::MetalSsmState>, // GPU-resident state for fused kernel
}

/// KV cache for one Attention block. FIFO ring buffer when window_size > 0.
pub struct KvCache {
    k: Option<Tensor>,  // [kv_heads, seq_so_far, head_dim] or None if empty
    v: Option<Tensor>,  // [kv_heads, seq_so_far, head_dim] or None if empty
    window_size: usize,  // 0 = unbounded, >0 = FIFO truncation
}

impl KvCache {
    fn new(window_size: usize) -> Self {
        Self { k: None, v: None, window_size }
    }

    /// Append new K, V vectors and return full cache tensors.
    /// If window_size > 0, truncates to keep only the most recent entries.
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (new_k, new_v) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                (Tensor::cat(&[prev_k, k], 1)?, Tensor::cat(&[prev_v, v], 1)?)
            }
            _ => (k.clone(), v.clone()),
        };
        // FIFO: truncate to window size
        let (new_k, new_v) = if self.window_size > 0 {
            let seq_len = new_k.dim(1)?;
            if seq_len > self.window_size {
                let start = seq_len - self.window_size;
                (new_k.narrow(1, start, self.window_size)?,
                 new_v.narrow(1, start, self.window_size)?)
            } else { (new_k, new_v) }
        } else { (new_k, new_v) };
        self.k = Some(new_k.clone());
        self.v = Some(new_v.clone());
        Ok((new_k, new_v))
    }
}

/// Per-layer state: either SSM state (Mamba) or KV cache (Attention).
pub enum LayerState {
    Mamba(SsmState),
    Attention(KvCache),
}

/// Single Mamba block for inference.
struct MambaBlock {
    norm: RmsNorm,
    in_proj: Linear,
    b_norm: RmsNorm,
    c_norm: RmsNorm,
    b_bias: Tensor,
    c_bias: Tensor,
    d_skip: Tensor,
    dt_bias: Tensor,
    h_init: Option<Tensor>,
    out_proj: Linear,
    config: ModelConfig,
    metal_pipeline: Option<Arc<crate::metal_ssm::SsmStepPipeline>>,
    window_pipeline: Option<Arc<crate::metal_ssm::SsmWindowPipeline>>,
    pre_ssm_pipeline: Option<Arc<crate::metal_ssm::PreSsmPipeline>>,
    pre_ssm_buffers: Option<Arc<crate::metal_ssm::PreSsmBuffers>>,
    rms_norm_pipeline: Option<Arc<crate::metal_ssm::RmsNormPipeline>>,
    gemv_pipeline: Option<Arc<crate::metal_ssm::GemvPipeline>>,
    gemv_residual_pipeline: Option<Arc<crate::metal_ssm::GemvResidualPipeline>>,
    gemv_buffers: Option<Arc<crate::metal_ssm::GemvBuffers>>,
}

impl MambaBlock {
    fn load(
        vb: VarBuilder,
        config: &ModelConfig,
        metal_pipeline: Option<Arc<crate::metal_ssm::SsmStepPipeline>>,
        window_pipeline: Option<Arc<crate::metal_ssm::SsmWindowPipeline>>,
        pre_ssm_pipeline: Option<Arc<crate::metal_ssm::PreSsmPipeline>>,
        pre_ssm_buffers: Option<Arc<crate::metal_ssm::PreSsmBuffers>>,
        rms_norm_pipeline: Option<Arc<crate::metal_ssm::RmsNormPipeline>>,
        gemv_pipeline: Option<Arc<crate::metal_ssm::GemvPipeline>>,
        gemv_residual_pipeline: Option<Arc<crate::metal_ssm::GemvResidualPipeline>>,
        gemv_buffers: Option<Arc<crate::metal_ssm::GemvBuffers>>,
    ) -> Result<Self> {
        let d_inner = config.d_inner();
        let bc_size = config.n_groups * config.d_state;
        let theta_proj = config.n_heads * config.d_state / 2;
        let in_proj_out = d_inner + d_inner + bc_size * 2 + config.n_heads + config.n_heads + config.n_heads + theta_proj;

        let norm = RmsNorm::load(vb.pp("norm"), config.d_model)?;
        let in_proj = candle_nn::linear_no_bias(config.d_model, in_proj_out, vb.pp("in_proj"))?;
        let b_norm = RmsNorm::load(vb.pp("b_norm"), bc_size)?;
        let c_norm = RmsNorm::load(vb.pp("c_norm"), bc_size)?;
        let d_skip = vb.pp("ssm").get(config.n_heads, "d")?;
        let dt_bias = vb.pp("ssm").get(config.n_heads, "dt_bias")?;
        // Mamba-3: BC biases (fallback to ones for old checkpoints without them)
        let b_bias = vb.get(bc_size, "b_bias")
            .unwrap_or_else(|_| Tensor::ones(bc_size, DType::F32, vb.device()).unwrap());
        let c_bias = vb.get(bc_size, "c_bias")
            .unwrap_or_else(|_| Tensor::ones(bc_size, DType::F32, vb.device()).unwrap());
        // Learned initial hidden state (optional — old checkpoints init to zeros)
        let h_init = vb.pp("ssm").get((config.n_heads, config.d_state), "h_init").ok();
        let out_proj = candle_nn::linear_no_bias(d_inner, config.d_model, vb.pp("out_proj"))?;

        Ok(Self {
            norm, in_proj, b_norm, c_norm,
            b_bias, c_bias, d_skip, dt_bias, h_init,
            out_proj,
            config: config.clone(),
            metal_pipeline,
            window_pipeline,
            pre_ssm_pipeline,
            pre_ssm_buffers,
            rms_norm_pipeline,
            gemv_pipeline,
            gemv_residual_pipeline,
            gemv_buffers,
        })
    }

    fn forward_step(&self, x: &Tensor, state: &mut SsmState) -> Result<Tensor> {
        let residual = x.clone();
        let d_inner = self.config.d_inner();
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim();
        let d_state = self.config.d_state;
        let n_groups = self.config.n_groups;
        let bc_size = n_groups * d_state;
        let heads_per_group = n_heads / n_groups;

        let d_model = self.config.d_model;
        let in_proj_out_dim = d_inner + d_inner + bc_size * 2 + n_heads * 3
            + n_heads * d_state / 2;

        // ── Fully fused Metal path: 5 dispatches total ──
        if let (
            Some(gemv_pipe), Some(gemv_res_pipe), Some(gemv_bufs),
            Some(pre_pipeline), Some(ssm_pipeline), Some(metal_state), Some(pre_bufs),
            Some(rms_pipe),
        ) = (
            &self.gemv_pipeline, &self.gemv_residual_pipeline, &self.gemv_buffers,
            &self.pre_ssm_pipeline, &self.metal_pipeline, &state.metal_state, &self.pre_ssm_buffers,
            &self.rms_norm_pipeline,
        ) {
            let metal_dev = x.device().as_metal_device()?;

            // 1. RMSNorm (fused, 1 dispatch)
            let normed = crate::metal_ssm::rms_norm_metal(rms_pipe, metal_dev, x, &self.norm.gamma, d_model, 1e-5)?;

            // 2. in_proj GEMV (1 dispatch) — replaces unsqueeze+matmul+squeeze
            let in_proj_w = self.in_proj.weight();
            crate::metal_ssm::gemv_metal(
                gemv_pipe, metal_dev, in_proj_w, &normed, &gemv_bufs.in_proj_out,
                in_proj_out_dim, d_model,
            )?;

            // 3. pre-SSM (fused, 1 dispatch)
            crate::metal_ssm::pre_ssm_metal(
                pre_pipeline, metal_dev, &gemv_bufs.in_proj_out,
                &self.b_norm.gamma, &self.c_norm.gamma,
                &self.b_bias, &self.c_bias, &self.dt_bias,
                pre_bufs, d_inner, bc_size, n_heads, head_dim, d_state, n_groups, 1e-5,
            )?;

            // 4. SSM step (fused, 1 dispatch)
            let y_heads = crate::metal_ssm::ssm_step_metal(
                ssm_pipeline, metal_dev,
                &pre_bufs.x_heads, &pre_bufs.b_expanded, &pre_bufs.c_expanded,
                &pre_bufs.a_bar, &pre_bufs.lambda_vals, &self.d_skip,
                &pre_bufs.theta, &pre_bufs.dt_pos, &pre_bufs.z,
                metal_state, n_heads, head_dim, d_state,
            )?;
            let y = y_heads.reshape(d_inner)?;

            // 5. out_proj GEMV + residual add (fused, 1 dispatch)
            let out_proj_w = self.out_proj.weight();
            crate::metal_ssm::gemv_residual_metal(
                gemv_res_pipe, metal_dev, out_proj_w, &y, &residual, &gemv_bufs.out_proj_out,
                d_model, d_inner,
            )?;
            return Ok(gemv_bufs.out_proj_out.clone());
        }

        // ── CPU fallback: Candle tensor ops ──
        let x = self.norm.forward(x)?;
        let x_2d = x.unsqueeze(0)?;
        let projected = self.in_proj.forward(&x_2d)?.squeeze(0)?;

        let x_ssm = projected.narrow(0, 0, d_inner)?;
        let z = projected.narrow(0, d_inner, d_inner)?;
        let b_proj = projected.narrow(0, 2 * d_inner, bc_size)?;
        let c_proj = projected.narrow(0, 2 * d_inner + bc_size, bc_size)?;
        let dt = projected.narrow(0, 2 * d_inner + 2 * bc_size, n_heads)?;
        let lambda_raw = projected.narrow(0, 2 * d_inner + 2 * bc_size + n_heads, n_heads)?;
        let lambda_vals = sigmoid(&lambda_raw)?;
        let dd_a_raw = projected.narrow(0, 2 * d_inner + 2 * bc_size + n_heads + n_heads, n_heads)?;

        let x_act = candle_nn::ops::silu(&x_ssm)?;
        let b_normed = (self.b_norm.forward(&b_proj)? + &self.b_bias)?;
        let c_normed = (self.c_norm.forward(&c_proj)? + &self.c_bias)?;

        let dt_biased = (&dt + &self.dt_bias)?;
        let dt_pos = softplus(&dt_biased)?;
        let a_vals = softplus(&dd_a_raw)?.neg()?.clamp(-1e6f64, -1e-4f64)?;
        let a_bar = (&a_vals * &dt_pos)?.exp()?;

        let x_heads = x_act.reshape((n_heads, head_dim))?;
        let b_groups = b_normed.reshape((n_groups, d_state))?;
        let c_groups = c_normed.reshape((n_groups, d_state))?;
        let b_expanded = b_groups.unsqueeze(1)?
            .expand((n_groups, heads_per_group, d_state))?
            .reshape((n_heads, d_state))?;
        let c_expanded = c_groups.unsqueeze(1)?
            .expand((n_groups, heads_per_group, d_state))?
            .reshape((n_heads, d_state))?;

        let half_ds = d_state / 2;
        let theta_offset = 2 * d_inner + 2 * bc_size + n_heads + n_heads + n_heads;
        let theta_raw: Vec<f32> = projected.narrow(0, theta_offset, n_heads * half_ds)?.to_vec1()?;
        let dt_pos_vec: Vec<f32> = dt_pos.to_vec1()?;
        let b_vec: Vec<f32> = b_expanded.flatten_all()?.to_vec1()?;
        let c_vec: Vec<f32> = c_expanded.flatten_all()?.to_vec1()?;
        let mut b_rot = vec![0.0f32; n_heads * d_state];
        let mut c_rot = vec![0.0f32; n_heads * d_state];
        let two_pi = 2.0 * std::f32::consts::PI;
        for h in 0..n_heads {
            let dtp = dt_pos_vec[h];
            for k in 0..half_ds {
                let angle = theta_raw[h * half_ds + k].tanh() * std::f32::consts::PI;
                state.cum_angle[h * half_ds + k] = (state.cum_angle[h * half_ds + k] + dtp * angle) % two_pi;
                let ca = state.cum_angle[h * half_ds + k].cos();
                let sa = state.cum_angle[h * half_ds + k].sin();
                b_rot[h * d_state + 2*k]     = ca * b_vec[h * d_state + 2*k] - sa * b_vec[h * d_state + 2*k + 1];
                b_rot[h * d_state + 2*k + 1] = sa * b_vec[h * d_state + 2*k] + ca * b_vec[h * d_state + 2*k + 1];
                c_rot[h * d_state + 2*k]     = ca * c_vec[h * d_state + 2*k] - sa * c_vec[h * d_state + 2*k + 1];
                c_rot[h * d_state + 2*k + 1] = sa * c_vec[h * d_state + 2*k] + ca * c_vec[h * d_state + 2*k + 1];
            }
        }
        let b_exp = Tensor::from_vec(b_rot, (n_heads, d_state), x.device())?;
        let c_exp = Tensor::from_vec(c_rot, (n_heads, d_state), x.device())?;

        let bx = x_heads.unsqueeze(2)?.broadcast_mul(&b_exp.unsqueeze(1)?)?;
        let lam_3d = lambda_vals.unsqueeze(1)?.unsqueeze(2)?;
        let one_minus_lam = lam_3d.affine(-1.0, 1.0)?;
        let a_bar_3d = a_bar.unsqueeze(1)?.unsqueeze(2)?;
        let term1 = a_bar_3d.broadcast_mul(&state.h)?;
        let term2 = (&a_bar_3d * &one_minus_lam)?.broadcast_mul(&state.prev_bx)?;
        let term3 = lam_3d.broadcast_mul(&bx)?;
        let new_h = (term1 + term2)?.add(&term3)?;
        state.h = new_h;
        state.prev_bx = bx;

        let c_3d = c_exp.unsqueeze(1)?.expand((n_heads, head_dim, d_state))?;
        let y_heads = (c_3d * &state.h)?.sum(2)?;
        let d_2d = self.d_skip.unsqueeze(1)?.expand((n_heads, head_dim))?;
        let y_heads_final = (y_heads + (d_2d * &x_heads)?)?;
        let y = y_heads_final.reshape(d_inner)?;
        let y = (y * candle_nn::ops::silu(&z)?)?;

        let y_2d = y.unsqueeze(0)?;
        let out = self.out_proj.forward(&y_2d)?.squeeze(0)?;
        out + residual
    }

    /// Windowed forward pass for eval. Batches all per-token projections and
    /// runs the SSM scan in a single Metal dispatch via `ssm_window_metal`.
    /// State is mutated in-place through the metal buffers in `state`.
    /// Input: [seq_len, d_model], Output: [seq_len, d_model]
    fn forward_window(&self, x: &Tensor, state: &mut SsmState) -> Result<Tensor> {
        let residual = x.clone();
        let seq_len = x.dim(0)?;
        let d_inner = self.config.d_inner();
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim();
        let d_state = self.config.d_state;
        let n_groups = self.config.n_groups;
        let bc_size = n_groups * d_state;
        let heads_per_group = n_heads / n_groups;

        // ── Batched projections ──
        let normed = self.norm.forward(x)?;                      // [S, d_model]
        let projected = self.in_proj.forward(&normed)?;          // [S, in_proj_out]

        let x_ssm = projected.narrow(1, 0, d_inner)?;
        let z = projected.narrow(1, d_inner, d_inner)?;
        let b_proj = projected.narrow(1, 2 * d_inner, bc_size)?;
        let c_proj = projected.narrow(1, 2 * d_inner + bc_size, bc_size)?;
        let dt = projected.narrow(1, 2 * d_inner + 2 * bc_size, n_heads)?;
        let lambda_raw = projected.narrow(1, 2 * d_inner + 2 * bc_size + n_heads, n_heads)?;
        let lambda_vals = sigmoid(&lambda_raw)?;
        let dd_a_raw = projected.narrow(1, 2 * d_inner + 2 * bc_size + 2 * n_heads, n_heads)?;

        let x_act = candle_nn::ops::silu(&x_ssm)?;
        let b_normed = self.b_norm.forward(&b_proj)?.broadcast_add(&self.b_bias)?;
        let c_normed = self.c_norm.forward(&c_proj)?.broadcast_add(&self.c_bias)?;

        let dt_biased = dt.broadcast_add(&self.dt_bias)?;
        let dt_pos = softplus(&dt_biased)?;                          // [S, n_heads]
        let a_vals = softplus(&dd_a_raw)?.neg()?.clamp(-1e6f64, -1e-4f64)?;
        let a_bar = (&a_vals * &dt_pos)?.exp()?;                     // [S, n_heads]

        // ── Reshape and group-expand B/C to per-head layout ──
        let x_heads = x_act.reshape((seq_len, n_heads, head_dim))?;
        let b_groups = b_normed.reshape((seq_len, n_groups, d_state))?;
        let b_expanded = b_groups.unsqueeze(2)?
            .expand((seq_len, n_groups, heads_per_group, d_state))?
            .reshape((seq_len, n_heads, d_state))?;
        let c_groups = c_normed.reshape((seq_len, n_groups, d_state))?;
        let c_expanded = c_groups.unsqueeze(2)?
            .expand((seq_len, n_groups, heads_per_group, d_state))?
            .reshape((seq_len, n_heads, d_state))?;

        // ── Single Metal dispatch over the whole window ──
        let pipeline = self.window_pipeline.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("window_pipeline missing — Metal required for forward_window".into()))?;
        let metal_state = state.metal_state.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("metal_state missing — Metal required for forward_window".into()))?;
        let metal_dev = x.device().as_metal_device()?;

        let half_ds = d_state / 2;
        let theta_offset = 2 * d_inner + 2 * bc_size + 3 * n_heads;
        let theta_seq = projected.narrow(1, theta_offset, n_heads * half_ds)?
            .reshape((seq_len, n_heads, half_ds))?;

        let z_heads = z.reshape((seq_len, n_heads, head_dim))?;
        let y_heads = crate::metal_ssm::ssm_window_metal(
            pipeline.as_ref(), metal_dev,
            &x_heads, &b_expanded, &c_expanded,
            &a_bar, &lambda_vals, &self.d_skip,
            &theta_seq, &dt_pos, &z_heads,
            metal_state,
            seq_len, n_heads, head_dim, d_state,
        )?;                                                            // [S, n_heads, head_dim]

        // ── Batched output projection (z-gating fused into kernel) ──
        let y = y_heads.reshape((seq_len, d_inner))?;
        let out = self.out_proj.forward(&y)?;                          // [S, d_model]
        out + residual
    }
}

/// Attention block with GQA + SwiGLU MLP for inference.
struct AttentionBlock {
    attn_norm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    attn_out: Linear,
    mlp_norm: RmsNorm,
    mlp_gate: Linear,
    mlp_up: Linear,
    mlp_down: Linear,
    config: ModelConfig,
}

impl AttentionBlock {
    fn load(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let d_model = config.d_model;
        let kv_dim = config.attn_kv_dim();
        let mlp_dim = config.attn_mlp_dim();

        let attn_norm = RmsNorm::load(vb.pp("attn_norm"), d_model)?;
        let q_proj = candle_nn::linear_no_bias(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(d_model, kv_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(d_model, kv_dim, vb.pp("v_proj"))?;
        let attn_out = candle_nn::linear_no_bias(d_model, d_model, vb.pp("attn_out_proj"))?;
        let mlp_norm = RmsNorm::load(vb.pp("mlp_norm"), d_model)?;
        let mlp_gate = candle_nn::linear_no_bias(d_model, mlp_dim, vb.pp("mlp_gate"))?;
        let mlp_up = candle_nn::linear_no_bias(d_model, mlp_dim, vb.pp("mlp_up"))?;
        let mlp_down = candle_nn::linear_no_bias(mlp_dim, d_model, vb.pp("mlp_down"))?;

        Ok(Self {
            attn_norm, q_proj, k_proj, v_proj, attn_out,
            mlp_norm, mlp_gate, mlp_up, mlp_down,
            config: config.clone(),
        })
    }

    /// Single-token forward with KV cache.
    fn forward_step(&self, x: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let n_heads = self.config.attn_n_heads;
        let kv_heads = self.config.attn_kv_heads;
        let head_dim = self.config.attn_head_dim();
        let heads_per_kv = n_heads / kv_heads;

        // ── Attention sub-block ──
        let residual = x.clone();
        let normed = self.attn_norm.forward(x)?;

        // Project Q, K, V — need 2D for linear
        let normed_2d = normed.unsqueeze(0)?;
        let q = self.q_proj.forward(&normed_2d)?.squeeze(0)?;  // [d_model]
        let k = self.k_proj.forward(&normed_2d)?.squeeze(0)?;  // [kv_dim]
        let v = self.v_proj.forward(&normed_2d)?.squeeze(0)?;  // [kv_dim]

        // Reshape to heads: Q [n_heads, head_dim], K/V [kv_heads, head_dim]
        let q = q.reshape((n_heads, head_dim))?;
        let k = k.reshape((kv_heads, 1, head_dim))?;   // [kv_heads, 1, head_dim]
        let v = v.reshape((kv_heads, 1, head_dim))?;

        // Append to KV cache, get full cache
        let (k_cache, v_cache) = cache.append(&k, &v)?;
        // k_cache: [kv_heads, seq_len, head_dim]
        // v_cache: [kv_heads, seq_len, head_dim]

        // GQA: expand KV to match query heads
        let seq_len = k_cache.dim(1)?;
        let k_expanded = k_cache.unsqueeze(1)?
            .expand((kv_heads, heads_per_kv, seq_len, head_dim))?
            .reshape((n_heads, seq_len, head_dim))?;
        let v_expanded = v_cache.unsqueeze(1)?
            .expand((kv_heads, heads_per_kv, seq_len, head_dim))?
            .reshape((n_heads, seq_len, head_dim))?;

        // Attention scores: Q @ K^T / sqrt(head_dim)
        // Q: [n_heads, head_dim] → [n_heads, 1, head_dim]
        let q = q.unsqueeze(1)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        // scores = Q @ K^T: [n_heads, 1, head_dim] @ [n_heads, head_dim, seq_len] → [n_heads, 1, seq_len]
        let scores = q.matmul(&k_expanded.transpose(1, 2)?)?.affine(scale, 0.0)?;

        // Softmax (no causal mask needed — single token can attend to all)
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;

        // Context = weights @ V: [n_heads, 1, seq_len] @ [n_heads, seq_len, head_dim] → [n_heads, 1, head_dim]
        let context = weights.matmul(&v_expanded)?.squeeze(1)?; // [n_heads, head_dim]

        // Reshape to [d_model] and project
        let context_flat = context.reshape(self.config.d_model)?;
        let attn_out = self.attn_out.forward(&context_flat.unsqueeze(0)?)?.squeeze(0)?;
        let x = (attn_out + residual)?;

        // ── SwiGLU MLP sub-block ──
        let residual = x.clone();
        let normed = self.mlp_norm.forward(&x)?;
        let normed_2d = normed.unsqueeze(0)?;

        let gate = sigmoid(&self.mlp_gate.forward(&normed_2d)?)?;
        let up = self.mlp_up.forward(&normed_2d)?;
        let hidden = (gate * up)?;
        let mlp_out = self.mlp_down.forward(&hidden)?.squeeze(0)?;

        mlp_out + residual
    }

    /// Windowed forward pass for eval. Standard causal self-attention + SwiGLU MLP
    /// over the entire sequence in batched Candle ops. Does not touch KvCache —
    /// the windowed eval path always starts from a fresh context.
    /// Input: [seq_len, d_model], Output: [seq_len, d_model]
    fn forward_window(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(0)?;
        let n_heads = self.config.attn_n_heads;
        let kv_heads = self.config.attn_kv_heads;
        let head_dim = self.config.attn_head_dim();
        let heads_per_kv = n_heads / kv_heads;

        // ── Attention sub-block ──
        let residual = x.clone();
        let normed = self.attn_norm.forward(x)?;                       // [S, d_model]

        let q = self.q_proj.forward(&normed)?
            .reshape((seq_len, n_heads, head_dim))?
            .transpose(0, 1)?;                                          // [n_heads, S, head_dim]
        let k = self.k_proj.forward(&normed)?
            .reshape((seq_len, kv_heads, head_dim))?
            .transpose(0, 1)?;                                          // [kv_heads, S, head_dim]
        let v = self.v_proj.forward(&normed)?
            .reshape((seq_len, kv_heads, head_dim))?
            .transpose(0, 1)?;                                          // [kv_heads, S, head_dim]

        // GQA expand to query heads
        let k = k.unsqueeze(1)?
            .expand((kv_heads, heads_per_kv, seq_len, head_dim))?
            .reshape((n_heads, seq_len, head_dim))?;
        let v = v.unsqueeze(1)?
            .expand((kv_heads, heads_per_kv, seq_len, head_dim))?
            .reshape((n_heads, seq_len, head_dim))?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(1, 2)?)?.affine(scale, 0.0)?;  // [n_heads, S, S]
        let mask = build_causal_mask(seq_len, x.device())?;               // [S, S]
        let scores = scores.broadcast_add(&mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = weights.matmul(&v)?
            .transpose(0, 1)?
            .reshape((seq_len, self.config.d_model))?;                    // [S, d_model]

        let attn_out = self.attn_out.forward(&context)?;
        let x = (attn_out + residual)?;

        // ── SwiGLU MLP sub-block ──
        let residual = x.clone();
        let normed = self.mlp_norm.forward(&x)?;
        let gate = sigmoid(&self.mlp_gate.forward(&normed)?)?;
        let up = self.mlp_up.forward(&normed)?;
        let hidden = (gate * up)?;
        let mlp_out = self.mlp_down.forward(&hidden)?;
        mlp_out + residual
    }
}

/// Build a causal mask [seq_len, seq_len]: 0.0 on lower triangle, -inf on upper.
fn build_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(mask, (seq_len, seq_len), device)
}

/// Block enum for hybrid dispatch.
enum Block {
    Mamba(MambaBlock),
    Attention(AttentionBlock),
}

/// Full Lumi model for inference.
pub struct NmModel {
    embedding: Embedding,
    blocks: Vec<Block>,
    final_norm: RmsNorm,
    config: ModelConfig,
    rms_norm_pipeline: Option<Arc<crate::metal_ssm::RmsNormPipeline>>,
    gemv_pipeline: Option<Arc<crate::metal_ssm::GemvPipeline>>,
    gemv_buffers: Option<Arc<crate::metal_ssm::GemvBuffers>>,
}

impl NmModel {
    pub fn load(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let embedding = candle_nn::embedding(config.vocab_size, config.d_model, vb.pp("embedding"))?;

        let n_mamba = config.n_layers - config.n_attn_layers();
        let n_attn = config.n_attn_layers();

        // Compile fused Metal kernels once, share across all layers.
        let d_inner = config.d_inner();
        let bc_size = config.n_groups * config.d_state;
        let theta_proj = config.n_heads * config.d_state / 2;
        let in_proj_out_dim = d_inner + d_inner + bc_size * 2
            + config.n_heads + config.n_heads + config.n_heads + theta_proj;

        let (metal_pipeline, window_pipeline, pre_ssm_pipeline, pre_ssm_buffers,
             rms_norm_pipeline, gemv_pipeline, gemv_residual_pipeline, gemv_buffers) =
            if let Ok(metal_dev) = vb.device().as_metal_device() {
                let step = match crate::metal_ssm::SsmStepPipeline::new(metal_dev) {
                    Ok(p) => { eprintln!("  Fused Metal SSM step kernel compiled."); Some(Arc::new(p)) }
                    Err(e) => { eprintln!("  Warning: Metal SSM step compile failed: {}, using fallback", e); None }
                };
                let window = match crate::metal_ssm::SsmWindowPipeline::new(metal_dev) {
                    Ok(p) => { eprintln!("  Fused Metal SSM window kernel compiled."); Some(Arc::new(p)) }
                    Err(e) => { eprintln!("  Warning: Metal SSM window compile failed: {}", e); None }
                };
                let pre_ssm = match crate::metal_ssm::PreSsmPipeline::new(metal_dev) {
                    Ok(p) => { eprintln!("  Fused Metal pre-SSM kernel compiled."); Some(Arc::new(p)) }
                    Err(e) => { eprintln!("  Warning: Metal pre-SSM compile failed: {}", e); None }
                };
                let bufs = match crate::metal_ssm::PreSsmBuffers::new(
                    vb.device(), d_inner, config.n_heads, config.d_state,
                ) {
                    Ok(b) => Some(Arc::new(b)),
                    Err(e) => { eprintln!("  Warning: PreSsmBuffers alloc failed: {}", e); None }
                };
                let rms_norm = match crate::metal_ssm::RmsNormPipeline::new(metal_dev) {
                    Ok(p) => { eprintln!("  Fused Metal RMSNorm kernel compiled."); Some(Arc::new(p)) }
                    Err(e) => { eprintln!("  Warning: Metal RMSNorm compile failed: {}", e); None }
                };
                let gemv = match crate::metal_ssm::GemvPipeline::new(metal_dev) {
                    Ok(p) => { eprintln!("  Fused Metal GEMV kernel compiled."); Some(Arc::new(p)) }
                    Err(e) => { eprintln!("  Warning: Metal GEMV compile failed: {}", e); None }
                };
                let gemv_res = match crate::metal_ssm::GemvResidualPipeline::new(metal_dev) {
                    Ok(p) => { eprintln!("  Fused Metal GEMV+residual kernel compiled."); Some(Arc::new(p)) }
                    Err(e) => { eprintln!("  Warning: Metal GEMV+residual compile failed: {}", e); None }
                };
                let gemv_bufs = match crate::metal_ssm::GemvBuffers::new(
                    vb.device(), in_proj_out_dim, config.d_model, config.vocab_size,
                ) {
                    Ok(b) => Some(Arc::new(b)),
                    Err(e) => { eprintln!("  Warning: GemvBuffers alloc failed: {}", e); None }
                };
                (step, window, pre_ssm, bufs, rms_norm, gemv, gemv_res, gemv_bufs)
            } else { (None, None, None, None, None, None, None, None) };

        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            if config.is_attention_layer(i) {
                let block = AttentionBlock::load(vb.pp(&format!("blocks.{}", i)), config)?;
                blocks.push(Block::Attention(block));
            } else {
                let block = MambaBlock::load(
                    vb.pp(&format!("blocks.{}", i)),
                    config,
                    metal_pipeline.clone(),
                    window_pipeline.clone(),
                    pre_ssm_pipeline.clone(),
                    pre_ssm_buffers.clone(),
                    rms_norm_pipeline.clone(),
                    gemv_pipeline.clone(),
                    gemv_residual_pipeline.clone(),
                    gemv_buffers.clone(),
                )?;
                blocks.push(Block::Mamba(block));
            }
        }

        eprintln!("  Layers: {} Mamba + {} Attention", n_mamba, n_attn);

        let final_norm = RmsNorm::load(vb.pp("final_norm"), config.d_model)?;

        Ok(Self {
            embedding, blocks, final_norm, config: config.clone(),
            rms_norm_pipeline, gemv_pipeline, gemv_buffers,
        })
    }

    /// Initialize per-layer states (SSM for Mamba, KV cache for Attention).
    pub fn init_states(&self, device: &Device) -> Result<Vec<LayerState>> {
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim();
        let d_state = self.config.d_state;

        let mut attn_idx = 0usize;
        self.blocks.iter().map(|block| {
            match block {
                Block::Mamba(mb) => {
                    let metal_state = if let Ok(metal_dev) = device.as_metal_device() {
                        Some(crate::metal_ssm::MetalSsmState::new(metal_dev, n_heads, head_dim, d_state)?)
                    } else { None };
                    // Use learned h_init if available: broadcast [n_heads, d_state] → [n_heads, head_dim, d_state]
                    let h = if let Some(ref h_init) = mb.h_init {
                        h_init.unsqueeze(1)?.expand((n_heads, head_dim, d_state))?.contiguous()?
                    } else {
                        Tensor::zeros((n_heads, head_dim, d_state), DType::F32, device)?
                    };
                    Ok(LayerState::Mamba(SsmState {
                        h,
                        prev_bx: Tensor::zeros((n_heads, head_dim, d_state), DType::F32, device)?,
                        cum_angle: vec![0.0f32; n_heads * d_state / 2],
                        metal_state,
                    }))
                },
                Block::Attention(_) => {
                    let ws = self.config.attn_window_size(attn_idx);
                    attn_idx += 1;
                    Ok(LayerState::Attention(KvCache::new(ws)))
                }
            }
        }).collect()
    }

    /// Forward one token, updating states. Returns logits [vocab_size].
    pub fn forward_step(
        &self,
        token_id: u32,
        states: &mut [LayerState],
        device: &Device,
    ) -> Result<Tensor> {
        let token = Tensor::new(&[token_id], device)?;
        let mut x = self.embedding.forward(&token)?.squeeze(0)?;

        for (block, state) in self.blocks.iter().zip(states.iter_mut()) {
            x = match (block, state) {
                (Block::Mamba(b), LayerState::Mamba(s)) => b.forward_step(&x, s)?,
                (Block::Attention(b), LayerState::Attention(s)) => b.forward_step(&x, s)?,
                _ => unreachable!("block/state type mismatch"),
            };
        }

        x = if let (Some(rms_pipe), Ok(metal_dev)) = (&self.rms_norm_pipeline, x.device().as_metal_device()) {
            crate::metal_ssm::rms_norm_metal(rms_pipe, metal_dev, &x, &self.final_norm.gamma, self.config.d_model, 1e-5)?
        } else {
            self.final_norm.forward(&x)?
        };
        // Logit projection: x @ embedding^T → [vocab_size]
        if let (Some(gemv_pipe), Some(gemv_bufs), Ok(metal_dev)) =
            (&self.gemv_pipeline, &self.gemv_buffers, x.device().as_metal_device())
        {
            let embed_weight = self.embedding.embeddings();
            crate::metal_ssm::gemv_metal(
                gemv_pipe, metal_dev, embed_weight, &x, &gemv_bufs.logit_out,
                self.config.vocab_size, self.config.d_model,
            )?;
            return Ok(gemv_bufs.logit_out.clone());
        }
        let embed_weight = self.embedding.embeddings();
        x.unsqueeze(0)?.matmul(&embed_weight.t()?)?.squeeze(0)
    }

    /// Forward an entire window in batched form. Returns [seq_len, vocab_size].
    /// Used by eval (perplexity, MC scoring). State is mutated through metal
    /// buffers; for attention layers it is unused.
    pub fn forward_window(
        &self,
        tokens: &[u32],
        states: &mut [LayerState],
        device: &Device,
    ) -> Result<Tensor> {
        let token_tensor = Tensor::new(tokens, device)?;             // [S]
        let mut x = self.embedding.forward(&token_tensor)?;          // [S, d_model]

        for (block, state) in self.blocks.iter().zip(states.iter_mut()) {
            x = match (block, state) {
                (Block::Mamba(b), LayerState::Mamba(s)) => b.forward_window(&x, s)?,
                (Block::Attention(b), LayerState::Attention(_)) => b.forward_window(&x)?,
                _ => unreachable!("block/state type mismatch"),
            };
        }

        x = self.final_norm.forward(&x)?;                            // [S, d_model]
        let embed_weight = self.embedding.embeddings();              // [vocab, d_model]
        x.matmul(&embed_weight.t()?)                                  // [S, vocab]
    }
}

/// Softplus: log(1 + exp(x)), numerically stable.
fn softplus(x: &Tensor) -> Result<Tensor> {
    (x.exp()? + 1.0)?.log()
}
