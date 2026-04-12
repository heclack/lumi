/// GPU memory manager for native training.
///
/// Pre-allocates ALL training buffers at init. Zero cudaMalloc during training.
/// Activations reuse buffers across layers (ping-pong pattern).

/// Backward kernel checkpoint granularity: timesteps replayed between saved states.
/// Distinct from config.chunk_size (SSD forward tiling). Must match CHUNK_SIZE in csrc/ssm_scan.cu backward kernel.
pub const SSM_BWD_CHUNK_SIZE: usize = 8;

use std::ptr;

use crate::config::ModelConfig;


/// GPU buffer — owns a device pointer and knows its size.
pub struct GpuBuf {
    pub ptr: *mut f32,
    pub len: usize, // number of f32 elements
}

impl GpuBuf {
    pub fn alloc(n: usize) -> Self {
        let mut ptr: *mut f32 = ptr::null_mut();
        let bytes = n * std::mem::size_of::<f32>();
        unsafe {
            let err = cuda_malloc(&mut ptr as *mut *mut f32 as *mut *mut std::ffi::c_void, bytes);
            assert!(err == 0, "cudaMalloc failed: {}", err);
        }
        Self { ptr, len: n }
    }

    pub fn zero(&self) {
        unsafe {
            cuda_memset(self.ptr as *mut std::ffi::c_void, 0,
                        self.len * std::mem::size_of::<f32>());
        }
    }

    /// Copy from host Vec to device.
    pub fn from_host(data: &[f32]) -> Self {
        let buf = Self::alloc(data.len());
        unsafe {
            cuda_memcpy(buf.ptr as *mut std::ffi::c_void,
                        data.as_ptr() as *const std::ffi::c_void,
                        data.len() * 4, 1); // cudaMemcpyHostToDevice = 1
        }
        buf
    }

    /// Copy device to host Vec.
    pub fn to_host(&self) -> Vec<f32> {
        let mut data = vec![0.0f32; self.len];
        unsafe {
            cuda_memcpy(data.as_mut_ptr() as *mut std::ffi::c_void,
                        self.ptr as *const std::ffi::c_void,
                        self.len * 4, 2); // cudaMemcpyDeviceToHost = 2
        }
        data
    }

    /// Copy host data into existing allocation (no realloc — preserves sub-pointers).
    pub fn copy_from_host(&self, data: &[f32]) {
        assert!(data.len() == self.len, "copy_from_host size mismatch: {} vs {}", data.len(), self.len);
        unsafe {
            cuda_memcpy(self.ptr as *mut std::ffi::c_void,
                        data.as_ptr() as *const std::ffi::c_void,
                        data.len() * 4, 1); // cudaMemcpyHostToDevice = 1
        }
    }

    /// Offset pointer (for splitting projected buffer into sub-tensors).
    pub fn offset(&self, offset: usize) -> *mut f32 {
        assert!(offset < self.len, "offset {} >= len {}", offset, self.len);
        unsafe { self.ptr.add(offset) }
    }
}

impl Drop for GpuBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cuda_free(self.ptr as *mut std::ffi::c_void); }
            self.ptr = ptr::null_mut();
        }
    }
}

unsafe impl Send for GpuBuf {}
unsafe impl Sync for GpuBuf {}

/// Weights for a single Mamba block on GPU.
pub struct LayerWeights {
    pub in_proj: GpuBuf,    // [d_model, in_proj_out]
    pub out_proj: GpuBuf,   // [d_inner, d_model]
    pub norm_gamma: GpuBuf, // [d_model]
    pub b_gamma: GpuBuf,    // [n_groups * d_state]
    pub c_gamma: GpuBuf,    // [n_groups * d_state]
    pub b_bias: GpuBuf,     // [n_groups * d_state] (Mamba-3 implicit conv)
    pub c_bias: GpuBuf,     // [n_groups * d_state]
    pub d_skip: GpuBuf,     // [n_heads]
    pub dt_bias: GpuBuf,    // [n_heads]
    pub h_init: GpuBuf,     // [n_heads * d_state] — learned initial hidden state
}

/// Per-layer saved activations for backward pass.
/// Each layer needs its own copy since shared buffers get overwritten.
pub struct PerLayerSaved {
    pub residual: GpuBuf,    // [bs, d_model] — input to this layer (for norm backward)
    pub x_norm: GpuBuf,      // [bs, d_model] — post-norm (for in_proj weight grad)
    pub x_ssm_raw: GpuBuf,   // [bs, d_inner] — pre-SiLU x (for SiLU backward)
    pub z_buf: GpuBuf,       // [bs, d_inner] — raw z (for SiLU backward on gate)
    pub b_raw: GpuBuf,       // [bs, bc_size] — pre-BCNorm B (for BCNorm backward)
    pub c_raw: GpuBuf,       // [bs, bc_size] — pre-BCNorm C (for BCNorm backward)
    pub b_norm: GpuBuf,      // [bs, bc_size] — post-BCNorm B (for SSM backward)
    pub c_norm: GpuBuf,      // [bs, bc_size] — post-BCNorm C (for SSM backward)
    pub dt_buf: GpuBuf,      // [bs, n_heads] — raw dt (for SSM backward)
    pub lambda_raw: GpuBuf,  // [bs, n_heads] — pre-sigmoid λ (for sigmoid backward)
    pub dd_a_raw: GpuBuf,    // [bs, n_heads] — raw dd_A projection (for softplus backward)
    pub ssm_out: GpuBuf,     // [bs, d_inner] — SSM output (for gate backward)
    pub z_act: GpuBuf,       // [bs, d_inner] — SiLU(z) (for gate backward)
    pub y_gated: GpuBuf,     // [bs, d_inner] — gated output (for out_proj weight grad)
    pub x_act: GpuBuf,       // [bs, d_inner] — post-SiLU x (for SSM input)
    pub theta_raw: GpuBuf,   // [bs, n_heads * d_state/2] — DD-RoPE theta (per timestep)
}

/// Which type of block is at each layer position.
#[derive(Clone, Copy, Debug)]
pub enum LayerType {
    Mamba(usize),     // index into mamba weight/grad vecs
    Attention(usize), // index into attention weight/grad vecs
}

/// Weights for a single attention block on GPU.
pub struct AttentionLayerWeights {
    pub attn_norm_gamma: GpuBuf,  // [d_model]
    pub q_proj: GpuBuf,           // [d_model, d_model]
    pub k_proj: GpuBuf,           // [d_model, kv_dim]
    pub v_proj: GpuBuf,           // [d_model, kv_dim]
    pub attn_out_proj: GpuBuf,    // [d_model, d_model]
    pub mlp_norm_gamma: GpuBuf,   // [d_model]
    pub mlp_gate: GpuBuf,         // [d_model, mlp_dim]
    pub mlp_up: GpuBuf,           // [d_model, mlp_dim]
    pub mlp_down: GpuBuf,         // [mlp_dim, d_model]
}

/// Gradient accumulators for one attention block.
/// Single contiguous GPU allocation — one cudaMemset zeros everything.
pub struct AttentionLayerGrads {
    _buf: GpuBuf,
    pub d_attn_norm_gamma: *mut f32,
    pub d_q_proj: *mut f32,
    pub d_k_proj: *mut f32,
    pub d_v_proj: *mut f32,
    pub d_attn_out_proj: *mut f32,
    pub d_mlp_norm_gamma: *mut f32,
    pub d_mlp_gate: *mut f32,
    pub d_mlp_up: *mut f32,
    pub d_mlp_down: *mut f32,
}

impl AttentionLayerGrads {
    pub fn alloc(d_model: usize, kv_dim: usize, mlp_dim: usize) -> Self {
        let sizes = [
            d_model,               // attn_norm_gamma
            d_model * d_model,     // q_proj
            d_model * kv_dim,      // k_proj
            d_model * kv_dim,      // v_proj
            d_model * d_model,     // attn_out_proj
            d_model,               // mlp_norm_gamma
            d_model * mlp_dim,     // mlp_gate
            d_model * mlp_dim,     // mlp_up
            mlp_dim * d_model,     // mlp_down
        ];
        let total: usize = sizes.iter().sum();
        let buf = GpuBuf::alloc(total);
        let mut off = 0usize;
        let ptrs: Vec<*mut f32> = sizes.iter().map(|&sz| {
            let p = unsafe { buf.ptr.add(off) };
            off += sz;
            p
        }).collect();
        Self {
            _buf: buf,
            d_attn_norm_gamma: ptrs[0], d_q_proj: ptrs[1],
            d_k_proj: ptrs[2], d_v_proj: ptrs[3], d_attn_out_proj: ptrs[4],
            d_mlp_norm_gamma: ptrs[5], d_mlp_gate: ptrs[6],
            d_mlp_up: ptrs[7], d_mlp_down: ptrs[8],
        }
    }

    pub fn zero_all(&self) { self._buf.zero(); }
}

/// Per-layer saved activations for attention backward.
pub struct AttentionPerLayerSaved {
    pub residual: GpuBuf,      // [bs, d_model]
    pub x_norm: GpuBuf,        // [bs, d_model] — for Q/K/V weight grads
    pub q: GpuBuf,             // [bs, d_model] — for attention backward
    pub k: GpuBuf,             // [bs, kv_dim] — for attention backward
    pub v: GpuBuf,             // [bs, kv_dim] — for attention backward
    pub attn_weights: GpuBuf,  // [batch, n_heads, seq, seq] — for V gradient
    pub context: GpuBuf,       // [bs, d_model] — for out_proj weight grad
    pub mlp_input: GpuBuf,     // [bs, d_model] — pre-MLP (for residual)
    pub mlp_norm_out: GpuBuf,  // [bs, d_model] — for MLP weight grads
    pub gate_raw: GpuBuf,      // [bs, mlp_dim] — pre-sigmoid gate (for sigmoid bwd)
    pub up_out: GpuBuf,        // [bs, mlp_dim] — up projection output
    pub gate_act: GpuBuf,      // [bs, mlp_dim] — sigmoid(gate) for mul backward
    pub y_gated: GpuBuf,       // [bs, mlp_dim] — gate * up (for down proj weight grad)
}

/// Adam m/v states for attention layer weights.
pub struct AttentionAdam {
    pub m: GpuBuf, // all attention weight m states concatenated
    pub v: GpuBuf, // all attention weight v states concatenated
    pub norm_m: GpuBuf, pub norm_v: GpuBuf,         // attn_norm [d_model]
    pub mlp_norm_m: GpuBuf, pub mlp_norm_v: GpuBuf, // mlp_norm [d_model]
}

/// Adam m/v states for small per-layer parameters.
/// Two contiguous allocations (m_buf, v_buf). Raw pointers into sub-regions.
pub struct SmallParamAdam {
    pub m_buf: GpuBuf, pub v_buf: GpuBuf,
    pub norm_m: *mut f32, pub norm_v: *mut f32,
    pub b_gamma_m: *mut f32, pub b_gamma_v: *mut f32,
    pub c_gamma_m: *mut f32, pub c_gamma_v: *mut f32,
    pub b_bias_m: *mut f32, pub b_bias_v: *mut f32,
    pub c_bias_m: *mut f32, pub c_bias_v: *mut f32,
    pub d_skip_m: *mut f32, pub d_skip_v: *mut f32,
    pub dt_bias_m: *mut f32, pub dt_bias_v: *mut f32,
    pub h_init_m: *mut f32, pub h_init_v: *mut f32,
}

impl SmallParamAdam {
    pub fn alloc(d_model: usize, bc_size: usize, n_heads: usize, d_state: usize) -> Self {
        let sizes = [d_model, bc_size, bc_size, bc_size, bc_size, n_heads, n_heads, n_heads * d_state];
        let total: usize = sizes.iter().sum();
        let m_buf = GpuBuf::alloc(total);
        let v_buf = GpuBuf::alloc(total);
        let mut m_off = 0usize;
        let mut v_off = 0usize;
        let m_ptrs: Vec<*mut f32> = sizes.iter().map(|&sz| {
            let p = unsafe { m_buf.ptr.add(m_off) }; m_off += sz; p
        }).collect();
        let v_ptrs: Vec<*mut f32> = sizes.iter().map(|&sz| {
            let p = unsafe { v_buf.ptr.add(v_off) }; v_off += sz; p
        }).collect();
        Self {
            m_buf, v_buf,
            norm_m: m_ptrs[0], norm_v: v_ptrs[0],
            b_gamma_m: m_ptrs[1], b_gamma_v: v_ptrs[1],
            c_gamma_m: m_ptrs[2], c_gamma_v: v_ptrs[2],
            b_bias_m: m_ptrs[3], b_bias_v: v_ptrs[3],
            c_bias_m: m_ptrs[4], c_bias_v: v_ptrs[4],
            d_skip_m: m_ptrs[5], d_skip_v: v_ptrs[5],
            dt_bias_m: m_ptrs[6], dt_bias_v: v_ptrs[6],
            h_init_m: m_ptrs[7], h_init_v: v_ptrs[7],
        }
    }
}

/// All training buffers on GPU.
pub struct TrainingBuffers {
    // Activation buffers (reused across layers)
    pub x: GpuBuf,            // [batch*seq, d_model]
    pub x_norm: GpuBuf,       // [batch*seq, d_model]
    pub projected: GpuBuf,    // [batch*seq, in_proj_out]
    pub x_ssm_raw: GpuBuf,    // [batch*seq, d_inner] (before SiLU, saved for backward)
    pub x_act: GpuBuf,        // [batch*seq, d_inner] (after SiLU)
    pub z_buf: GpuBuf,        // [batch*seq, d_inner] (gate branch)
    pub z_act: GpuBuf,        // [batch*seq, d_inner] (SiLU(z))
    pub b_raw: GpuBuf,        // [batch*seq, n_groups*d_state] — before BCNorm
    pub c_raw: GpuBuf,        // [batch*seq, n_groups*d_state] — before BCNorm
    pub b_norm_buf: GpuBuf,   // [batch*seq, n_groups*d_state] — after BCNorm
    pub c_norm_buf: GpuBuf,   // [batch*seq, n_groups*d_state] — after BCNorm
    pub dt_buf: GpuBuf,       // [batch*seq, n_heads]
    pub lambda_raw: GpuBuf,   // [batch*seq, n_heads] — raw λ logits
    pub lambda_buf: GpuBuf,   // [batch*seq, n_heads] — sigmoid(λ) ∈ [0,1]
    pub a_vals_buf: GpuBuf,   // [batch*seq, n_heads] — data-dependent A values (negative)
    pub ssm_out: GpuBuf,      // [batch*seq, d_inner] (= n_heads * head_dim)
    pub y_gated: GpuBuf,      // [batch*seq, d_inner]
    pub block_out: GpuBuf,    // [batch*seq, d_model]
    pub residual: GpuBuf,     // [batch*seq, d_model]

    // Gradient buffers (same sizes as activations)
    pub d_x: GpuBuf,
    pub d_x_norm: GpuBuf,
    pub d_projected: GpuBuf,
    pub d_x_act: GpuBuf,
    pub d_z: GpuBuf,
    pub d_ssm_out: GpuBuf,
    pub d_y_gated: GpuBuf,
    pub d_block_out: GpuBuf,
    pub d_b_norm: GpuBuf,
    pub d_c_norm: GpuBuf,
    pub d_dt: GpuBuf,
    pub d_lambda: GpuBuf,     // [batch*seq, n_heads] — gradient for λ
    pub d_a_vals: GpuBuf,     // [batch*seq, n_heads] — gradient for A values
    pub d_theta_raw: GpuBuf,  // [batch*seq, n_heads * d_state/2] — DD-RoPE theta gradient

    // SSM backward workspace (pre-allocated, reused across layers)
    pub ws_d_d_buf: GpuBuf,      // [batch * n_heads] partial sums
    pub ws_d_dtb_buf: GpuBuf,   // [batch * n_heads] partial sums

    // SSM forward→backward checkpoints (reused across layers)
    pub ssm_h_checkpoints: GpuBuf,   // [batch*n_heads * n_chunks * 256 * max_elems]
    pub ssm_pbx_checkpoints: GpuBuf, // same size
    pub ssm_h_saved: GpuBuf,        // [batch*n_heads * CHUNK_SIZE * 256 * max_elems]
    pub ssm_pbx_saved: GpuBuf,      // same size

    // Output + loss
    pub logits: GpuBuf,       // [batch*seq, vocab]
    pub d_logits: GpuBuf,     // [batch*seq, vocab]
    pub per_token_loss: GpuBuf, // [batch*seq]
    pub loss: GpuBuf,         // [1]
    pub grad_norm_scratch: GpuBuf, // [1] — scratch for gradient norm (separate from loss)

    // Token IDs
    pub input_ids: GpuBuf,    // [batch*seq] (i32 stored as f32 for simplicity)
    pub target_ids: GpuBuf,   // [batch*seq]

    // Shared attention buffers (reused across attention layers)
    pub attn_q: GpuBuf,           // [bs, d_model]
    pub attn_k: GpuBuf,           // [bs, kv_dim]
    pub attn_v: GpuBuf,           // [bs, kv_dim]
    pub attn_q_t: GpuBuf,        // [batch, n_heads, seq, head_dim] — transposed for batched matmul
    pub attn_k_t: GpuBuf,        // [batch, n_heads, seq, head_dim] — after GQA expand
    pub attn_v_t: GpuBuf,        // [batch, n_heads, seq, head_dim] — after GQA expand
    pub attn_context_t: GpuBuf,  // [batch, n_heads, seq, head_dim] — before transpose back
    pub attn_scores: GpuBuf,      // [batch, attn_n_heads, seq, seq]
    pub attn_weights: GpuBuf,     // [batch, attn_n_heads, seq, seq]
    pub attn_context: GpuBuf,     // [bs, d_model]
    pub attn_mlp_hidden: GpuBuf,  // [bs, mlp_dim]
    pub attn_mlp_gated: GpuBuf,   // [bs, mlp_dim]
    pub attn_kv_temp: GpuBuf,     // [batch, kv_heads, seq, head_dim] — temp for transpose before GQA expand
    pub d_attn_q: GpuBuf,         // gradient buffers for attention
    pub d_attn_k: GpuBuf,
    pub d_attn_v: GpuBuf,
    pub d_attn_q_t: GpuBuf,       // [batch, n_heads, seq, head_dim]
    pub d_attn_k_t: GpuBuf,       // [batch, n_heads, seq, head_dim]
    pub d_attn_v_t: GpuBuf,       // [batch, n_heads, seq, head_dim]
    pub d_attn_context_t: GpuBuf, // [batch, n_heads, seq, head_dim]
    pub d_attn_scores: GpuBuf,
    pub d_attn_context: GpuBuf,
    pub d_attn_mlp_hidden: GpuBuf,
    pub d_attn_mlp_gated: GpuBuf,

    // Model weights
    pub embedding: GpuBuf,    // [vocab, d_model]
    pub final_norm_gamma: GpuBuf, // [d_model]
    pub layers: Vec<LayerWeights>,          // Mamba blocks
    pub attn_layers: Vec<AttentionLayerWeights>,  // Attention blocks

    // Gradient accumulators for weights
    pub d_embedding: GpuBuf,
    pub d_final_norm_gamma: GpuBuf,
    pub d_layers: Vec<LayerGrads>,            // Mamba grads
    pub d_attn_layers: Vec<AttentionLayerGrads>, // Attention grads

    // Per-layer saved activations (for backward pass)
    pub saved: Vec<PerLayerSaved>,                  // Mamba saved
    pub attn_saved: Vec<AttentionPerLayerSaved>,    // Attention saved

    // Layer dispatch map
    pub layer_types: Vec<LayerType>,

    // Adam state for large params (in_proj + out_proj)
    pub adam_m: Vec<ParamAdamState>,

    // Adam state for small per-layer params
    pub small_adam: Vec<SmallParamAdam>,

    // Adam state for attention layers
    pub attn_adam: Vec<AttentionAdam>,

    // Adam state for global params
    pub final_norm_adam_m: GpuBuf,
    pub final_norm_adam_v: GpuBuf,
    pub embedding_adam_m: GpuBuf,
    pub embedding_adam_v: GpuBuf,

    // Dimensions
    pub batch_seq: usize,
    pub d_model: usize,
    pub d_inner: usize,
    pub in_proj_out: usize,
    pub n_heads: usize,
    pub n_groups: usize,
    pub d_state: usize,
    pub vocab: usize,
}

/// Gradient accumulators for one Mamba layer.
/// Single contiguous GPU allocation — one cudaMemset zeros everything.
pub struct LayerGrads {
    _buf: GpuBuf,  // owns the contiguous allocation
    pub d_in_proj: *mut f32,
    pub d_out_proj: *mut f32,
    pub d_norm_gamma: *mut f32,
    pub d_b_gamma: *mut f32,
    pub d_c_gamma: *mut f32,
    pub d_b_bias: *mut f32,
    pub d_c_bias: *mut f32,
    pub d_d_skip: *mut f32,
    pub d_dt_bias: *mut f32,
    pub d_h_init: *mut f32,
}

impl LayerGrads {
    pub fn alloc(d_model: usize, in_proj_out: usize, d_inner: usize,
                 bc_size: usize, n_heads: usize, d_state: usize) -> Self {
        let sizes = [
            d_model * in_proj_out, d_inner * d_model, d_model,
            bc_size, bc_size, bc_size, bc_size,
            n_heads, n_heads, n_heads * d_state,
        ];
        let total: usize = sizes.iter().sum();
        let buf = GpuBuf::alloc(total);
        let mut off = 0usize;
        let ptrs: Vec<*mut f32> = sizes.iter().map(|&sz| {
            let p = unsafe { buf.ptr.add(off) };
            off += sz;
            p
        }).collect();
        Self {
            _buf: buf,
            d_in_proj: ptrs[0], d_out_proj: ptrs[1], d_norm_gamma: ptrs[2],
            d_b_gamma: ptrs[3], d_c_gamma: ptrs[4],
            d_b_bias: ptrs[5], d_c_bias: ptrs[6],
            d_d_skip: ptrs[7], d_dt_bias: ptrs[8], d_h_init: ptrs[9],
        }
    }

    /// Zero all gradient accumulators in one cudaMemset call.
    pub fn zero_all(&self) { self._buf.zero(); }
}

pub struct ParamAdamState {
    pub m: GpuBuf,
    pub v: GpuBuf,
}

impl TrainingBuffers {
    pub fn allocate(config: &ModelConfig, batch: usize, seq: usize) -> Self {
        let bs = batch * seq;
        let d_model = config.d_model;
        let d_inner = config.d_inner();
        let n_heads = config.n_heads;
        let n_groups = config.n_groups;
        let d_state = config.d_state;
        let bc_size = n_groups * d_state;
        let theta_proj = n_heads * d_state / 2; // DD-RoPE always enabled
        let in_proj_out = d_inner + d_inner + bc_size * 2 + n_heads + n_heads + n_heads + theta_proj; // +n_heads for dd_A
        let vocab = config.vocab_size;
        let n_layers = config.n_layers;

        eprintln!("Allocating native GPU training buffers...");
        let total_mb = (
            // Activations + gradients (each pair)
            bs * d_model * 2 * 4 +        // x, d_x
            bs * d_model * 2 * 2 +        // x_norm, residual, etc
            bs * in_proj_out * 2 * 2 +    // projected
            bs * d_inner * 2 * 4 +        // x_act, z, ssm_out, y_gated
            bs * bc_size * 2 * 2 +        // b_norm, c_norm
            bs * n_heads * 2 +            // dt
            bs * vocab * 2 * 2 +          // logits
            // Weights
            n_layers * (d_model * in_proj_out + d_inner * d_model + d_model + bc_size * 2 + n_heads * 3) * 2 + // weights + grads
            vocab * d_model * 2 +         // embedding + grad
            // Adam state (2x weights)
            n_layers * (d_model * in_proj_out + d_inner * d_model) * 2 * 2
        ) * 4 / 1024 / 1024;
        eprintln!("  Estimated GPU memory: ~{}MB", total_mb);

        // Allocate activations
        let x = GpuBuf::alloc(bs * d_model);
        let x_norm = GpuBuf::alloc(bs * d_model);
        let projected = GpuBuf::alloc(bs * in_proj_out);
        let x_ssm_raw = GpuBuf::alloc(bs * d_inner);
        let x_act = GpuBuf::alloc(bs * d_inner);
        let z_buf = GpuBuf::alloc(bs * d_inner);
        let z_act = GpuBuf::alloc(bs * d_inner);
        let b_raw = GpuBuf::alloc(bs * bc_size);
        let c_raw = GpuBuf::alloc(bs * bc_size);
        let b_norm_buf = GpuBuf::alloc(bs * bc_size);
        let c_norm_buf = GpuBuf::alloc(bs * bc_size);
        let dt_buf = GpuBuf::alloc(bs * n_heads);
        let lambda_raw = GpuBuf::alloc(bs * n_heads);  // raw λ logits from in_proj
        let lambda_buf = GpuBuf::alloc(bs * n_heads);  // sigmoid(λ) ∈ [0,1]
        let a_vals_buf = GpuBuf::alloc(bs * n_heads);  // data-dependent A values
        let ssm_out = GpuBuf::alloc(bs * d_inner);
        let y_gated = GpuBuf::alloc(bs * d_inner);
        let block_out = GpuBuf::alloc(bs * d_model);
        let residual = GpuBuf::alloc(bs * d_model);

        // Gradient buffers
        let d_x = GpuBuf::alloc(bs * d_model);
        let d_x_norm = GpuBuf::alloc(bs * d_model);
        let d_projected = GpuBuf::alloc(bs * in_proj_out);
        let d_x_act = GpuBuf::alloc(bs * d_inner);
        let d_z = GpuBuf::alloc(bs * d_inner);
        let d_ssm_out = GpuBuf::alloc(bs * d_inner);
        let d_y_gated = GpuBuf::alloc(bs * d_inner);
        let d_block_out = GpuBuf::alloc(bs * d_model);
        let d_b_norm = GpuBuf::alloc(bs * bc_size);
        let d_c_norm = GpuBuf::alloc(bs * bc_size);
        let d_dt = GpuBuf::alloc(bs * n_heads);
        let d_lambda = GpuBuf::alloc(bs * n_heads);
        let d_a_vals = GpuBuf::alloc(bs * n_heads);
        let d_theta_raw = GpuBuf::alloc(bs * n_heads * d_state / 2);

        // SSM backward workspace (batch * n_heads — reused across layers)
        let ws_d_d_buf = GpuBuf::alloc(batch * n_heads);
        let ws_d_dtb_buf = GpuBuf::alloc(batch * n_heads);

        // SSM forward→backward checkpoints (reused across layers)
        let n_threads = 256;
        let state_size = (d_model * config.expand / config.n_heads) * config.d_state;
        let max_elems = (state_size + n_threads - 1) / n_threads;
        let n_chunks = (seq + SSM_BWD_CHUNK_SIZE - 1) / SSM_BWD_CHUNK_SIZE;
        let ckpt_size = batch * config.n_heads * n_chunks * n_threads * max_elems;
        let ssm_h_checkpoints = GpuBuf::alloc(ckpt_size);
        let ssm_pbx_checkpoints = GpuBuf::alloc(ckpt_size);

        // SSM backward within-chunk saved states (reused across layers)
        // Layout: [batch*n_heads, CHUNK_SIZE, n_threads, max_elems]
        let saved_size = batch * config.n_heads * SSM_BWD_CHUNK_SIZE * n_threads * max_elems;
        let ssm_h_saved = GpuBuf::alloc(saved_size);
        let ssm_pbx_saved = GpuBuf::alloc(saved_size);

        // Output
        let logits = GpuBuf::alloc(bs * vocab);
        let d_logits = GpuBuf::alloc(bs * vocab);
        let per_token_loss = GpuBuf::alloc(bs);
        let loss = GpuBuf::alloc(1);
        let grad_norm_scratch = GpuBuf::alloc(1);
        let input_ids = GpuBuf::alloc(bs);
        let target_ids = GpuBuf::alloc(bs);

        // Model weights + grads (initialized randomly, or loaded from checkpoint)
        let embedding = GpuBuf::alloc(vocab * d_model);
        let final_norm_gamma = GpuBuf::alloc(d_model);
        let d_embedding = GpuBuf::alloc(vocab * d_model);
        let d_final_norm_gamma = GpuBuf::alloc(d_model);

        let mut layers = Vec::with_capacity(n_layers);
        let mut d_layers = Vec::with_capacity(n_layers);
        let mut adam_m = Vec::with_capacity(n_layers);
        let mut small_adam = Vec::with_capacity(n_layers);
        let mut saved = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            layers.push(LayerWeights {
                in_proj: GpuBuf::alloc(d_model * in_proj_out),
                out_proj: GpuBuf::alloc(d_inner * d_model),
                norm_gamma: GpuBuf::alloc(d_model),
                b_gamma: GpuBuf::alloc(bc_size),
                c_gamma: GpuBuf::alloc(bc_size),
                b_bias: GpuBuf::alloc(bc_size),
                c_bias: GpuBuf::alloc(bc_size),
                d_skip: GpuBuf::alloc(n_heads),
                dt_bias: GpuBuf::alloc(n_heads),
                h_init: GpuBuf::alloc(n_heads * d_state),
            });
            d_layers.push(LayerGrads::alloc(d_model, in_proj_out, d_inner, bc_size, n_heads, d_state));
            // Adam m and v for in_proj + out_proj (the big ones)
            adam_m.push(ParamAdamState {
                m: GpuBuf::alloc(d_model * in_proj_out + d_inner * d_model),
                v: GpuBuf::alloc(d_model * in_proj_out + d_inner * d_model),
            });
            // Adam m/v for small per-layer params
            small_adam.push(SmallParamAdam::alloc(d_model, bc_size, n_heads, d_state));
            // Per-layer saved activations
            saved.push(PerLayerSaved {
                residual: GpuBuf::alloc(bs * d_model),
                x_norm: GpuBuf::alloc(bs * d_model),
                x_ssm_raw: GpuBuf::alloc(bs * d_inner),
                z_buf: GpuBuf::alloc(bs * d_inner),
                b_raw: GpuBuf::alloc(bs * bc_size),
                c_raw: GpuBuf::alloc(bs * bc_size),
                b_norm: GpuBuf::alloc(bs * bc_size),
                c_norm: GpuBuf::alloc(bs * bc_size),
                dt_buf: GpuBuf::alloc(bs * n_heads),
                lambda_raw: GpuBuf::alloc(bs * n_heads),
                dd_a_raw: GpuBuf::alloc(bs * n_heads),
                ssm_out: GpuBuf::alloc(bs * d_inner),
                z_act: GpuBuf::alloc(bs * d_inner),
                y_gated: GpuBuf::alloc(bs * d_inner),
                x_act: GpuBuf::alloc(bs * d_inner),
                theta_raw: GpuBuf::alloc(bs * n_heads * d_state / 2),
            });
        }

        // Attention layer allocation
        let attn_n_heads = config.attn_n_heads;
        let kv_dim = config.attn_kv_dim();
        let mlp_dim = config.attn_mlp_dim();
        let n_attn = config.n_attn_layers();
        let n_mamba = config.n_mamba_layers();

        // Build layer type dispatch map
        let mut layer_types = Vec::with_capacity(n_layers);
        let mut mamba_idx = 0usize;
        let mut attn_idx = 0usize;
        for i in 0..n_layers {
            if config.is_attention_layer(i) {
                layer_types.push(LayerType::Attention(attn_idx));
                attn_idx += 1;
            } else {
                layer_types.push(LayerType::Mamba(mamba_idx));
                mamba_idx += 1;
            }
        }
        eprintln!("  Layer types: {} Mamba + {} Attention", n_mamba, n_attn);

        // Shared attention activation buffers (reused across attention layers)
        let attn_scores_size = batch * attn_n_heads * seq * seq;
        let attn_q = GpuBuf::alloc(bs * d_model);
        let attn_k = GpuBuf::alloc(bs * kv_dim);
        let attn_v = GpuBuf::alloc(bs * kv_dim);
        let attn_q_t = GpuBuf::alloc(bs * d_model);  // [batch, n_heads, seq, head_dim]
        let attn_k_t = GpuBuf::alloc(bs * d_model);  // expanded from kv_heads to n_heads
        let attn_v_t = GpuBuf::alloc(bs * d_model);
        let attn_context_t = GpuBuf::alloc(bs * d_model);
        let attn_scores = GpuBuf::alloc(attn_scores_size);
        let attn_weights = GpuBuf::alloc(attn_scores_size);
        let attn_context = GpuBuf::alloc(bs * d_model);
        let attn_mlp_hidden = GpuBuf::alloc(bs * mlp_dim);
        let attn_mlp_gated = GpuBuf::alloc(bs * mlp_dim);
        let attn_kv_temp = GpuBuf::alloc(bs * kv_dim);  // temp for transpose before GQA expand (avoids in-place race)
        let d_attn_q = GpuBuf::alloc(bs * d_model);
        let d_attn_k = GpuBuf::alloc(bs * kv_dim);
        let d_attn_v = GpuBuf::alloc(bs * kv_dim);
        let d_attn_q_t = GpuBuf::alloc(bs * d_model);
        let d_attn_k_t = GpuBuf::alloc(bs * d_model);
        let d_attn_v_t = GpuBuf::alloc(bs * d_model);
        let d_attn_context_t = GpuBuf::alloc(bs * d_model);
        let d_attn_scores = GpuBuf::alloc(attn_scores_size);
        let d_attn_context = GpuBuf::alloc(bs * d_model);
        let d_attn_mlp_hidden = GpuBuf::alloc(bs * mlp_dim);
        let d_attn_mlp_gated = GpuBuf::alloc(bs * mlp_dim);

        // Per-attention-layer weights, gradients, saved activations, Adam
        let mut attn_layers_vec = Vec::with_capacity(n_attn);
        let mut d_attn_layers = Vec::with_capacity(n_attn);
        let mut attn_saved = Vec::with_capacity(n_attn);
        let mut attn_adam = Vec::with_capacity(n_attn);

        let attn_large_size = d_model * d_model * 2 + d_model * kv_dim * 2 // Q,K,V,out projections
            + d_model * mlp_dim * 2 + mlp_dim * d_model; // gate,up,down

        for _ in 0..n_attn {
            attn_layers_vec.push(AttentionLayerWeights {
                attn_norm_gamma: GpuBuf::alloc(d_model),
                q_proj: GpuBuf::alloc(d_model * d_model),
                k_proj: GpuBuf::alloc(d_model * kv_dim),
                v_proj: GpuBuf::alloc(d_model * kv_dim),
                attn_out_proj: GpuBuf::alloc(d_model * d_model),
                mlp_norm_gamma: GpuBuf::alloc(d_model),
                mlp_gate: GpuBuf::alloc(d_model * mlp_dim),
                mlp_up: GpuBuf::alloc(d_model * mlp_dim),
                mlp_down: GpuBuf::alloc(mlp_dim * d_model),
            });
            d_attn_layers.push(AttentionLayerGrads::alloc(d_model, kv_dim, mlp_dim));
            attn_saved.push(AttentionPerLayerSaved {
                residual: GpuBuf::alloc(bs * d_model),
                x_norm: GpuBuf::alloc(bs * d_model),
                q: GpuBuf::alloc(bs * d_model),
                k: GpuBuf::alloc(bs * kv_dim),
                v: GpuBuf::alloc(bs * kv_dim),
                attn_weights: GpuBuf::alloc(attn_scores_size),
                context: GpuBuf::alloc(bs * d_model),
                mlp_input: GpuBuf::alloc(bs * d_model),
                mlp_norm_out: GpuBuf::alloc(bs * d_model),
                gate_raw: GpuBuf::alloc(bs * mlp_dim),
                up_out: GpuBuf::alloc(bs * mlp_dim),
                gate_act: GpuBuf::alloc(bs * mlp_dim),
                y_gated: GpuBuf::alloc(bs * mlp_dim),
            });
            attn_adam.push(AttentionAdam {
                m: GpuBuf::alloc(attn_large_size),
                v: GpuBuf::alloc(attn_large_size),
                norm_m: GpuBuf::alloc(d_model),
                norm_v: GpuBuf::alloc(d_model),
                mlp_norm_m: GpuBuf::alloc(d_model),
                mlp_norm_v: GpuBuf::alloc(d_model),
            });
        }

        // Global Adam states
        let final_norm_adam_m = GpuBuf::alloc(d_model);
        let final_norm_adam_v = GpuBuf::alloc(d_model);
        let embedding_adam_m = GpuBuf::alloc(vocab * d_model);
        let embedding_adam_v = GpuBuf::alloc(vocab * d_model);

        eprintln!("  Native GPU buffers allocated.");

        Self {
            x, x_norm, projected, x_ssm_raw, x_act, z_buf, z_act,
            b_raw, c_raw, b_norm_buf, c_norm_buf, dt_buf, lambda_raw, lambda_buf, a_vals_buf,
            ssm_out, y_gated, block_out, residual,
            d_x, d_x_norm, d_projected, d_x_act, d_z,
            d_ssm_out, d_y_gated, d_block_out, d_b_norm, d_c_norm, d_dt, d_lambda, d_a_vals, d_theta_raw,
            ws_d_d_buf, ws_d_dtb_buf,
            ssm_h_checkpoints, ssm_pbx_checkpoints, ssm_h_saved, ssm_pbx_saved,

            attn_q, attn_k, attn_v, attn_q_t, attn_k_t, attn_v_t, attn_context_t,
            attn_scores, attn_weights, attn_context,
            attn_mlp_hidden, attn_mlp_gated, attn_kv_temp,
            d_attn_q, d_attn_k, d_attn_v, d_attn_q_t, d_attn_k_t, d_attn_v_t, d_attn_context_t,
            d_attn_scores, d_attn_context,
            d_attn_mlp_hidden, d_attn_mlp_gated,
            logits, d_logits, per_token_loss, loss, grad_norm_scratch,
            input_ids, target_ids,
            embedding, final_norm_gamma,
            d_embedding, d_final_norm_gamma,
            layers, d_layers,
            attn_layers: attn_layers_vec, d_attn_layers,
            saved, attn_saved,
            layer_types,
            adam_m, small_adam, attn_adam,
            final_norm_adam_m, final_norm_adam_v,
            embedding_adam_m, embedding_adam_v,
            batch_seq: bs, d_model, d_inner, in_proj_out,
            n_heads, n_groups, d_state, vocab,
        }
    }
}

// Raw CUDA FFI for memory management
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                  count: usize, kind: i32) -> i32;
    fn cudaMemset(devPtr: *mut std::ffi::c_void, value: i32, count: usize) -> i32;
}

unsafe fn cuda_malloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32 {
    cudaMalloc(ptr, size)
}
unsafe fn cuda_free(ptr: *mut std::ffi::c_void) -> i32 {
    cudaFree(ptr)
}
pub(crate) unsafe fn cuda_memcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                      count: usize, kind: i32) -> i32 {
    cudaMemcpy(dst, src, count, kind)
}
unsafe fn cuda_memset(ptr: *mut std::ffi::c_void, value: i32, count: usize) -> i32 {
    cudaMemset(ptr, value, count)
}
