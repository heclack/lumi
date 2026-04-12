/// Fused Metal SSM step kernel bridge.
///
/// Replaces ~40 individual Candle tensor ops with a single Metal dispatch
/// for the SSM state update + DD-RoPE + output contraction.

use candle_core::{DType, Device, MetalDevice, Result, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use std::sync::Arc;

const SHADER_SOURCE: &str = include_str!("ssm_step.metal");

/// Compiled Metal pipeline for the fused SSM step kernel.
/// Shared across all Mamba layers (same kernel, different state buffers).
pub struct SsmStepPipeline {
    pipeline: ComputePipeline,
}

unsafe impl Send for SsmStepPipeline {}
unsafe impl Sync for SsmStepPipeline {}

impl SsmStepPipeline {
    /// Compile the Metal shader. Call once at model load time.
    pub fn new(device: &MetalDevice) -> Result<Self> {
        let raw_device = device.device();
        let lib = raw_device
            .new_library_with_source(SHADER_SOURCE, None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal shader compile: {:?}", e)))?;
        let func = lib
            .get_function("ssm_step_fused", None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal function: {:?}", e)))?;
        let pipeline = raw_device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("Metal pipeline: {:?}", e)))?;
        Ok(Self { pipeline })
    }
}

/// Compiled Metal pipeline for the windowed SSM kernel.
/// Used by `forward_window` (eval); shared across all Mamba layers.
pub struct SsmWindowPipeline {
    pipeline: ComputePipeline,
}

unsafe impl Send for SsmWindowPipeline {}
unsafe impl Sync for SsmWindowPipeline {}

impl SsmWindowPipeline {
    /// Compile the windowed kernel. Call once at model load time.
    pub fn new(device: &MetalDevice) -> Result<Self> {
        let raw_device = device.device();
        let lib = raw_device
            .new_library_with_source(SHADER_SOURCE, None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal shader compile: {:?}", e)))?;
        let func = lib
            .get_function("ssm_window_fused", None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal function: {:?}", e)))?;
        let pipeline = raw_device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("Metal pipeline: {:?}", e)))?;
        Ok(Self { pipeline })
    }
}

/// Compiled Metal pipeline for the fused pre-SSM kernel.
/// Replaces ~35 individual Candle ops with 1 dispatch.
pub struct PreSsmPipeline {
    pipeline: ComputePipeline,
}

unsafe impl Send for PreSsmPipeline {}
unsafe impl Sync for PreSsmPipeline {}

impl PreSsmPipeline {
    pub fn new(device: &MetalDevice) -> Result<Self> {
        let raw_device = device.device();
        let lib = raw_device
            .new_library_with_source(SHADER_SOURCE, None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal shader compile: {:?}", e)))?;
        let func = lib
            .get_function("pre_ssm_fused", None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal function: {:?}", e)))?;
        let pipeline = raw_device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("Metal pipeline: {:?}", e)))?;
        Ok(Self { pipeline })
    }
}

/// Pre-allocated GPU buffers for pre-SSM kernel outputs.
/// Shared across all Mamba layers (forward is sequential, so no race).
pub struct PreSsmBuffers {
    pub x_heads: Tensor,      // [d_inner] = [n_heads * head_dim]
    pub z: Tensor,            // [d_inner]
    pub b_expanded: Tensor,   // [n_heads * d_state]
    pub c_expanded: Tensor,   // [n_heads * d_state]
    pub a_bar: Tensor,        // [n_heads]
    pub lambda_vals: Tensor,  // [n_heads]
    pub dt_pos: Tensor,       // [n_heads]
    pub theta: Tensor,        // [n_heads * half_ds]
}

impl PreSsmBuffers {
    /// Allocate all pre-SSM output buffers on the Metal device.
    pub fn new(device: &Device, d_inner: usize, n_heads: usize, d_state: usize) -> Result<Self> {
        let half_ds = d_state / 2;
        Ok(Self {
            x_heads: Tensor::zeros(d_inner, DType::F32, device)?,
            z: Tensor::zeros(d_inner, DType::F32, device)?,
            b_expanded: Tensor::zeros(n_heads * d_state, DType::F32, device)?,
            c_expanded: Tensor::zeros(n_heads * d_state, DType::F32, device)?,
            a_bar: Tensor::zeros(n_heads, DType::F32, device)?,
            lambda_vals: Tensor::zeros(n_heads, DType::F32, device)?,
            dt_pos: Tensor::zeros(n_heads, DType::F32, device)?,
            theta: Tensor::zeros(n_heads * half_ds, DType::F32, device)?,
        })
    }
}

/// Parameters struct matching the Metal kernel's PreSsmParams.
#[repr(C)]
struct PreSsmParams {
    d_inner: i32,
    bc_size: i32,
    n_heads: i32,
    head_dim: i32,
    d_state: i32,
    n_groups: i32,
    heads_per_group: i32,
    half_ds: i32,
    theta_proj: i32,
    norm_eps: f32,
}

/// Execute the fused pre-SSM kernel. Writes to pre-allocated buffers.
pub fn pre_ssm_metal(
    pipeline: &PreSsmPipeline,
    device: &MetalDevice,
    projected: &Tensor,       // [in_proj_out] -- flat in_proj output
    b_norm_gamma: &Tensor,    // [bc_size]
    c_norm_gamma: &Tensor,    // [bc_size]
    b_bias: &Tensor,          // [bc_size]
    c_bias: &Tensor,          // [bc_size]
    dt_bias: &Tensor,         // [n_heads]
    buffers: &PreSsmBuffers,
    d_inner: usize,
    bc_size: usize,
    n_heads: usize,
    head_dim: usize,
    d_state: usize,
    n_groups: usize,
    norm_eps: f32,
) -> Result<()> {
    let projected = projected.contiguous()?;
    let proj_buf = metal_buffer(&projected)?;
    let b_gamma_buf = metal_buffer(b_norm_gamma)?;
    let c_gamma_buf = metal_buffer(c_norm_gamma)?;
    let b_bias_buf = metal_buffer(b_bias)?;
    let c_bias_buf = metal_buffer(c_bias)?;
    let dt_bias_buf = metal_buffer(dt_bias)?;

    let x_heads_buf = metal_buffer(&buffers.x_heads)?;
    let z_buf = metal_buffer(&buffers.z)?;
    let b_exp_buf = metal_buffer(&buffers.b_expanded)?;
    let c_exp_buf = metal_buffer(&buffers.c_expanded)?;
    let a_bar_buf = metal_buffer(&buffers.a_bar)?;
    let lam_buf = metal_buffer(&buffers.lambda_vals)?;
    let dt_pos_buf = metal_buffer(&buffers.dt_pos)?;
    let theta_buf = metal_buffer(&buffers.theta)?;

    let half_ds = d_state / 2;
    let heads_per_group = n_heads / n_groups;

    let params = PreSsmParams {
        d_inner: d_inner as i32,
        bc_size: bc_size as i32,
        n_heads: n_heads as i32,
        head_dim: head_dim as i32,
        d_state: d_state as i32,
        n_groups: n_groups as i32,
        heads_per_group: heads_per_group as i32,
        half_ds: half_ds as i32,
        theta_proj: (n_heads * half_ds) as i32,
        norm_eps: norm_eps,
    };

    let encoder = device.command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline.pipeline);
    encoder.set_buffer(0, Some(&proj_buf), 0);
    encoder.set_buffer(1, Some(&b_gamma_buf), 0);
    encoder.set_buffer(2, Some(&c_gamma_buf), 0);
    encoder.set_buffer(3, Some(&b_bias_buf), 0);
    encoder.set_buffer(4, Some(&c_bias_buf), 0);
    encoder.set_buffer(5, Some(&dt_bias_buf), 0);
    encoder.set_buffer(6, Some(&x_heads_buf), 0);
    encoder.set_buffer(7, Some(&z_buf), 0);
    encoder.set_buffer(8, Some(&b_exp_buf), 0);
    encoder.set_buffer(9, Some(&c_exp_buf), 0);
    encoder.set_buffer(10, Some(&a_bar_buf), 0);
    encoder.set_buffer(11, Some(&lam_buf), 0);
    encoder.set_buffer(12, Some(&dt_pos_buf), 0);
    encoder.set_buffer(13, Some(&theta_buf), 0);
    encoder.set_bytes_directly(
        14,
        std::mem::size_of::<PreSsmParams>(),
        &params as *const PreSsmParams as *const std::ffi::c_void,
    );

    // 1 threadgroup of 1024 threads
    let threads_per_group = objc2_metal::MTLSize { width: 1024, height: 1, depth: 1 };
    let threadgroups = objc2_metal::MTLSize { width: 1, height: 1, depth: 1 };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);

    Ok(())
}

/// Compiled Metal pipeline for fused RMSNorm.
/// Shared across all norm operations (Mamba block norms, final norm).
pub struct RmsNormPipeline {
    pipeline: ComputePipeline,
}

unsafe impl Send for RmsNormPipeline {}
unsafe impl Sync for RmsNormPipeline {}

impl RmsNormPipeline {
    /// Compile the RMSNorm kernel. Call once at model load time.
    pub fn new(device: &MetalDevice) -> Result<Self> {
        let raw_device = device.device();
        let lib = raw_device
            .new_library_with_source(SHADER_SOURCE, None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal shader compile: {:?}", e)))?;
        let func = lib
            .get_function("rms_norm_fused", None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal function: {:?}", e)))?;
        let pipeline = raw_device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("Metal pipeline: {:?}", e)))?;
        Ok(Self { pipeline })
    }
}

/// Parameters struct matching the Metal kernel's RmsNormParams.
#[repr(C)]
struct RmsNormParams {
    dim: i32,
    eps: f32,
}

/// Execute the fused RMSNorm kernel: out = x * rsqrt(mean(x^2) + eps) * gamma.
/// Single dispatch replaces 4 separate Candle ops.
pub fn rms_norm_metal(
    pipeline: &RmsNormPipeline,
    device: &MetalDevice,
    x: &Tensor,         // [dim]
    gamma: &Tensor,      // [dim]
    dim: usize,
    eps: f32,
) -> Result<Tensor> {
    let x = x.contiguous()?;
    let x_buf = metal_buffer(&x)?;
    let gamma_buf = metal_buffer(gamma)?;

    let candle_device = Device::Metal(device.clone());
    let output_tensor = Tensor::zeros(dim, DType::F32, &candle_device)?;
    let output_buf = metal_buffer(&output_tensor)?;

    let params = RmsNormParams {
        dim: dim as i32,
        eps,
    };

    let encoder = device.command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline.pipeline);
    encoder.set_buffer(0, Some(&x_buf), 0);
    encoder.set_buffer(1, Some(&gamma_buf), 0);
    encoder.set_buffer(2, Some(output_buf.as_ref()), 0);
    encoder.set_bytes_directly(
        3,
        std::mem::size_of::<RmsNormParams>(),
        &params as *const RmsNormParams as *const std::ffi::c_void,
    );

    // 1 threadgroup of 32 threads (1 SIMD group) — enough for d_model=1024
    let threads_per_group = objc2_metal::MTLSize { width: 32, height: 1, depth: 1 };
    let threadgroups = objc2_metal::MTLSize { width: 1, height: 1, depth: 1 };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);

    Ok(output_tensor)
}

/// Persistent GPU-resident state for one SSM layer.
pub struct MetalSsmState {
    pub h: Arc<Buffer>,          // [n_heads, head_dim, d_state]
    pub prev_bx: Arc<Buffer>,    // [n_heads, head_dim, d_state]
    pub cum_angle: Arc<Buffer>,  // [n_heads, half_ds]
}

impl MetalSsmState {
    /// Allocate zero-initialized state buffers on the Metal device.
    pub fn new(device: &MetalDevice, n_heads: usize, head_dim: usize, d_state: usize) -> Result<Self> {
        let state_size = n_heads * head_dim * d_state;
        let angle_size = n_heads * d_state / 2;

        let h = device.allocate_zeros(state_size * std::mem::size_of::<f32>())?;
        let prev_bx = device.allocate_zeros(state_size * std::mem::size_of::<f32>())?;
        let cum_angle = device.allocate_zeros(angle_size * std::mem::size_of::<f32>())?;

        Ok(Self { h, prev_bx, cum_angle })
    }
}

/// Parameters struct matching the Metal kernel's SsmStepParams.
#[repr(C)]
struct SsmStepParams {
    n_heads: i32,
    head_dim: i32,
    d_state: i32,
    half_ds: i32,
}

/// Parameters struct matching the Metal kernel's SsmWindowParams.
#[repr(C)]
struct SsmWindowParams {
    n_heads: i32,
    head_dim: i32,
    d_state: i32,
    half_ds: i32,
    seq_len: i32,
}

/// Get the raw Metal buffer from a Candle tensor. Tensor must be contiguous and on Metal.
fn metal_buffer(tensor: &Tensor) -> Result<Arc<Buffer>> {
    let (storage, _layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Metal(ms) => Ok(Arc::new(ms.buffer().clone())),
        _ => candle_core::bail!("expected Metal storage, got CPU"),
    }
}

/// Execute the fused SSM step kernel.
pub fn ssm_step_metal(
    pipeline: &SsmStepPipeline,
    device: &MetalDevice,
    // Inputs
    x_heads: &Tensor,      // [n_heads, head_dim]
    b_expanded: &Tensor,    // [n_heads, d_state]
    c_expanded: &Tensor,    // [n_heads, d_state]
    a_bar: &Tensor,         // [n_heads]
    lambda_vals: &Tensor,   // [n_heads]
    d_skip: &Tensor,        // [n_heads]
    theta_raw: &Tensor,     // [n_heads, half_ds]
    dt_pos: &Tensor,        // [n_heads]
    z: &Tensor,             // [n_heads, head_dim] -- z gate for fused SiLU output gating
    // State (mutated in-place)
    state: &MetalSsmState,
    // Config
    n_heads: usize,
    head_dim: usize,
    d_state: usize,
) -> Result<Tensor> {
    // Ensure contiguous
    let x_heads = x_heads.contiguous()?;
    let b_expanded = b_expanded.contiguous()?;
    let c_expanded = c_expanded.contiguous()?;
    let a_bar = a_bar.contiguous()?;
    let lambda_vals = lambda_vals.contiguous()?;

    let x_buf = metal_buffer(&x_heads)?;
    let b_buf = metal_buffer(&b_expanded)?;
    let c_buf = metal_buffer(&c_expanded)?;
    let ab_buf = metal_buffer(&a_bar)?;
    let lam_buf = metal_buffer(&lambda_vals)?;
    let d_buf = metal_buffer(d_skip)?;

    // DD-RoPE buffers (always required)
    let theta_buf = metal_buffer(&theta_raw.contiguous()?)?;
    let dt_buf = metal_buffer(&dt_pos.contiguous()?)?;

    // Z gate buffer
    let z_buf = metal_buffer(&z.contiguous()?)?;

    // Output: allocate via Candle (ensures proper tensor wrapping)
    let candle_device = Device::Metal(device.clone());
    let output_tensor = Tensor::zeros((n_heads, head_dim), DType::F32, &candle_device)?;
    let output_buf = metal_buffer(&output_tensor)?;

    // Params via set_bytes_directly
    let params = SsmStepParams {
        n_heads: n_heads as i32,
        head_dim: head_dim as i32,
        d_state: d_state as i32,
        half_ds: (d_state / 2) as i32,
    };

    // Encode
    let encoder = device.command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline.pipeline);
    encoder.set_buffer(0, Some(&x_buf), 0);
    encoder.set_buffer(1, Some(&b_buf), 0);
    encoder.set_buffer(2, Some(&c_buf), 0);
    encoder.set_buffer(3, Some(&ab_buf), 0);
    encoder.set_buffer(4, Some(&lam_buf), 0);
    encoder.set_buffer(5, Some(&d_buf), 0);
    encoder.set_buffer(6, Some(&theta_buf), 0);
    encoder.set_buffer(7, Some(&dt_buf), 0);
    encoder.set_buffer(8, Some(&state.h), 0);
    encoder.set_buffer(9, Some(&state.prev_bx), 0);
    encoder.set_buffer(10, Some(&state.cum_angle), 0);
    encoder.set_buffer(11, Some(output_buf.as_ref()), 0);
    encoder.set_bytes_directly(
        12,
        std::mem::size_of::<SsmStepParams>(),
        &params as *const SsmStepParams as *const std::ffi::c_void,
    );
    encoder.set_buffer(13, Some(&z_buf), 0);

    // Dispatch: n_heads threadgroups × head_dim threads each
    let threads_per_group = objc2_metal::MTLSize { width: head_dim, height: 1, depth: 1 };
    let threadgroups = objc2_metal::MTLSize { width: n_heads, height: 1, depth: 1 };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    // Don't call end_encoding -- Candle's MetalDevice manages the shared command encoder lifecycle

    Ok(output_tensor)
}

/// Execute the windowed SSM kernel over an entire sequence in one dispatch.
///
/// Inputs are 3-D `[seq_len, n_heads, ...]` tensors. State is mutated in-place
/// (the kernel loads `h`/`prev_bx` into private memory at start, scans the
/// window, and writes them back at end). Returns `[seq_len, n_heads, head_dim]`.
pub fn ssm_window_metal(
    pipeline: &SsmWindowPipeline,
    device: &MetalDevice,
    // Inputs
    x_seq: &Tensor,         // [seq, n_heads, head_dim]
    b_seq: &Tensor,         // [seq, n_heads, d_state]
    c_seq: &Tensor,         // [seq, n_heads, d_state]
    a_bar_seq: &Tensor,     // [seq, n_heads]
    lambda_seq: &Tensor,    // [seq, n_heads]
    d_skip: &Tensor,        // [n_heads]
    theta_seq: &Tensor,     // [seq, n_heads, half_ds]
    dt_pos_seq: &Tensor,    // [seq, n_heads]
    z_seq: &Tensor,         // [seq, n_heads, head_dim] -- z gate for fused SiLU output gating
    // State (mutated in-place)
    state: &MetalSsmState,
    // Config
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    d_state: usize,
) -> Result<Tensor> {
    // Ensure contiguous
    let x_seq = x_seq.contiguous()?;
    let b_seq = b_seq.contiguous()?;
    let c_seq = c_seq.contiguous()?;
    let a_bar_seq = a_bar_seq.contiguous()?;
    let lambda_seq = lambda_seq.contiguous()?;

    let x_buf  = metal_buffer(&x_seq)?;
    let b_buf  = metal_buffer(&b_seq)?;
    let c_buf  = metal_buffer(&c_seq)?;
    let ab_buf = metal_buffer(&a_bar_seq)?;
    let lam_buf = metal_buffer(&lambda_seq)?;
    let d_buf  = metal_buffer(d_skip)?;

    // DD-RoPE buffers (always required)
    let theta_buf = metal_buffer(&theta_seq.contiguous()?)?;
    let dt_buf = metal_buffer(&dt_pos_seq.contiguous()?)?;

    // Z gate buffer
    let z_buf = metal_buffer(&z_seq.contiguous()?)?;

    let candle_device = Device::Metal(device.clone());
    let output_tensor = Tensor::zeros((seq_len, n_heads, head_dim), DType::F32, &candle_device)?;
    let output_buf = metal_buffer(&output_tensor)?;

    let params = SsmWindowParams {
        n_heads: n_heads as i32,
        head_dim: head_dim as i32,
        d_state: d_state as i32,
        half_ds: (d_state / 2) as i32,
        seq_len: seq_len as i32,
    };

    let encoder = device.command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline.pipeline);
    encoder.set_buffer(0, Some(&x_buf), 0);
    encoder.set_buffer(1, Some(&b_buf), 0);
    encoder.set_buffer(2, Some(&c_buf), 0);
    encoder.set_buffer(3, Some(&ab_buf), 0);
    encoder.set_buffer(4, Some(&lam_buf), 0);
    encoder.set_buffer(5, Some(&d_buf), 0);
    encoder.set_buffer(6, Some(&theta_buf), 0);
    encoder.set_buffer(7, Some(&dt_buf), 0);
    encoder.set_buffer(8, Some(&state.h), 0);
    encoder.set_buffer(9, Some(&state.prev_bx), 0);
    encoder.set_buffer(10, Some(&state.cum_angle), 0);
    encoder.set_buffer(11, Some(output_buf.as_ref()), 0);
    encoder.set_bytes_directly(
        12,
        std::mem::size_of::<SsmWindowParams>(),
        &params as *const SsmWindowParams as *const std::ffi::c_void,
    );
    encoder.set_buffer(13, Some(&z_buf), 0);

    let threads_per_group = objc2_metal::MTLSize { width: head_dim, height: 1, depth: 1 };
    let threadgroups = objc2_metal::MTLSize { width: n_heads, height: 1, depth: 1 };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);

    Ok(output_tensor)
}

// ─── Fused GEMV (matrix-vector multiply) ─────────────────────────────────────

/// Compiled Metal pipeline for fused GEMV: y = W @ x.
pub struct GemvPipeline {
    pipeline: ComputePipeline,
}

unsafe impl Send for GemvPipeline {}
unsafe impl Sync for GemvPipeline {}

impl GemvPipeline {
    pub fn new(device: &MetalDevice) -> Result<Self> {
        let raw_device = device.device();
        let lib = raw_device
            .new_library_with_source(SHADER_SOURCE, None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal shader compile: {:?}", e)))?;
        let func = lib
            .get_function("gemv_fused", None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal function: {:?}", e)))?;
        let pipeline = raw_device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("Metal pipeline: {:?}", e)))?;
        Ok(Self { pipeline })
    }
}

/// Compiled Metal pipeline for fused GEMV + residual: y = W @ x + residual.
pub struct GemvResidualPipeline {
    pipeline: ComputePipeline,
}

unsafe impl Send for GemvResidualPipeline {}
unsafe impl Sync for GemvResidualPipeline {}

impl GemvResidualPipeline {
    pub fn new(device: &MetalDevice) -> Result<Self> {
        let raw_device = device.device();
        let lib = raw_device
            .new_library_with_source(SHADER_SOURCE, None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal shader compile: {:?}", e)))?;
        let func = lib
            .get_function("gemv_residual_fused", None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal function: {:?}", e)))?;
        let pipeline = raw_device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("Metal pipeline: {:?}", e)))?;
        Ok(Self { pipeline })
    }
}

/// Pre-allocated GPU buffers for GEMV outputs.
/// Shared across all Mamba layers (forward is sequential).
pub struct GemvBuffers {
    pub in_proj_out: Tensor,   // [in_proj_out_dim]
    pub out_proj_out: Tensor,  // [d_model]
    pub logit_out: Tensor,     // [vocab_size]
}

impl GemvBuffers {
    pub fn new(device: &Device, in_proj_out_dim: usize, d_model: usize, vocab_size: usize) -> Result<Self> {
        Ok(Self {
            in_proj_out: Tensor::zeros(in_proj_out_dim, DType::F32, device)?,
            out_proj_out: Tensor::zeros(d_model, DType::F32, device)?,
            logit_out: Tensor::zeros(vocab_size, DType::F32, device)?,
        })
    }
}

/// GEMV parameters matching Metal kernel's GemvParams.
#[repr(C)]
struct GemvParams {
    out_dim: i32,
    in_dim: i32,
}

/// Execute fused GEMV: y = W @ x. Single Metal dispatch.
pub fn gemv_metal(
    pipeline: &GemvPipeline,
    device: &MetalDevice,
    weight: &Tensor,    // [out_dim, in_dim] — weight matrix
    x: &Tensor,         // [in_dim] — input vector
    output: &Tensor,    // [out_dim] — pre-allocated output buffer
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    let w_buf = metal_buffer(&weight.contiguous()?)?;
    let x_buf = metal_buffer(&x.contiguous()?)?;
    let y_buf = metal_buffer(output)?;

    let params = GemvParams {
        out_dim: out_dim as i32,
        in_dim: in_dim as i32,
    };

    let encoder = device.command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline.pipeline);
    encoder.set_buffer(0, Some(&w_buf), 0);
    encoder.set_buffer(1, Some(&x_buf), 0);
    encoder.set_buffer(2, Some(&y_buf), 0);
    encoder.set_bytes_directly(
        3,
        std::mem::size_of::<GemvParams>(),
        &params as *const GemvParams as *const std::ffi::c_void,
    );

    // 8 rows per threadgroup (8 SIMD groups × 32 threads = 256 threads)
    let n_threadgroups = (out_dim + 7) / 8;
    let threadgroups = objc2_metal::MTLSize { width: n_threadgroups, height: 1, depth: 1 };
    let threads_per_group = objc2_metal::MTLSize { width: 256, height: 1, depth: 1 };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);

    Ok(())
}

/// Execute fused GEMV + residual: y = W @ x + residual. Single Metal dispatch.
pub fn gemv_residual_metal(
    pipeline: &GemvResidualPipeline,
    device: &MetalDevice,
    weight: &Tensor,    // [out_dim, in_dim]
    x: &Tensor,         // [in_dim]
    residual: &Tensor,  // [out_dim]
    output: &Tensor,    // [out_dim] — pre-allocated output buffer
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    let w_buf = metal_buffer(&weight.contiguous()?)?;
    let x_buf = metal_buffer(&x.contiguous()?)?;
    let r_buf = metal_buffer(&residual.contiguous()?)?;
    let y_buf = metal_buffer(output)?;

    let params = GemvParams {
        out_dim: out_dim as i32,
        in_dim: in_dim as i32,
    };

    let encoder = device.command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline.pipeline);
    encoder.set_buffer(0, Some(&w_buf), 0);
    encoder.set_buffer(1, Some(&x_buf), 0);
    encoder.set_buffer(2, Some(&r_buf), 0);
    encoder.set_buffer(3, Some(&y_buf), 0);
    encoder.set_bytes_directly(
        4,
        std::mem::size_of::<GemvParams>(),
        &params as *const GemvParams as *const std::ffi::c_void,
    );

    let n_threadgroups = (out_dim + 7) / 8;
    let threadgroups = objc2_metal::MTLSize { width: n_threadgroups, height: 1, depth: 1 };
    let threads_per_group = objc2_metal::MTLSize { width: 256, height: 1, depth: 1 };
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);

    Ok(())
}
