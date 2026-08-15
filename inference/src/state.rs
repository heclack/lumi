/// Per-layer decode state for the SSM recurrence: `h`, `prev_bx`, `cum_angle`.
/// These are read-modify-write across every `ssm_step_gpu` call for a layer, so
/// each `LayerState` is allocated once per layer and mutated in place for the
/// lifetime of a generation (see `ssm_step_gpu`'s doc comment in
/// `kernels/src/ops.rs` for the exact recurrence).
use lumi_kernels::buf::GpuBuf;

use crate::config::ModelConfig;
use crate::weights::MambaLayerWeights;

pub struct LayerState {
    /// `[n_heads, head_dim, d_state]`
    pub h: GpuBuf,
    /// `[n_heads, head_dim, d_state]`
    pub prev_bx: GpuBuf,
    /// `[n_heads, d_state/2]` — DD-RoPE running angle.
    pub cum_angle: GpuBuf,
}

impl LayerState {
    pub fn new(config: &ModelConfig) -> Self {
        let n_heads = config.n_heads;
        let head_dim = config.head_dim();
        let d_state = config.d_state;
        let hpds = n_heads * head_dim * d_state;
        let half_ds = n_heads * d_state / 2;
        Self {
            h: GpuBuf::alloc(hpds),
            prev_bx: GpuBuf::alloc(hpds),
            cum_angle: GpuBuf::alloc(half_ds),
        }
    }

    /// Reset to a fresh-generation state: `h` broadcasts the layer's learned
    /// `h_init` (`[n_heads, d_state]`) across `head_dim` — every head_dim slot
    /// for a given head starts from the same per-(head, state) value, i.e.
    /// `h[head][p][s] = h_init[head*d_state + s]` for all p. `prev_bx` and
    /// `cum_angle` always start at zero (no learned initial value for either).
    pub fn reset(&mut self, config: &ModelConfig, layer: &MambaLayerWeights) {
        let n_heads = config.n_heads;
        let head_dim = config.head_dim();
        let d_state = config.d_state;

        let h_init = layer.h_init.to_host(); // [n_heads * d_state]
        let mut h_host = vec![0.0f32; n_heads * head_dim * d_state];
        for head in 0..n_heads {
            for p in 0..head_dim {
                let dst = (head * head_dim + p) * d_state;
                let src = head * d_state;
                h_host[dst..dst + d_state].copy_from_slice(&h_init[src..src + d_state]);
            }
        }
        self.h.copy_from_host(&h_host);
        self.prev_bx.zero();
        self.cum_angle.zero();
    }
}
