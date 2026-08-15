/// Model configuration for native GPU inference.
///
/// Mirrors `archive/metal-inference/src/model.rs::ModelConfig` (the reference
/// semantics) and `training/src/config.rs::ModelConfig` (the field set actually
/// written by the trainer's exported `config.json`). Only the fields inference
/// needs are kept; anything attention-related is accepted for deserialization
/// compatibility but rejected in `validate()` since attention layers are not
/// wired up on the native decode path yet.
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Hidden dimension (residual stream width).
    pub d_model: usize,
    /// Total number of blocks.
    pub n_layers: usize,
    /// SSM state dimension. Must be 64 — hard requirement of `ssm_step_gpu`
    /// (static shared-memory sizing in the kernel), checked in `validate()`.
    pub d_state: usize,
    /// Expansion factor: d_inner = expand * d_model.
    pub expand: usize,
    /// Number of SSM heads.
    pub n_heads: usize,
    /// Number of groups for the B/C matrices (GQA-style sharing).
    pub n_groups: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length the model was trained for.
    pub max_seq_len: usize,
    /// Attention block interval (0 = pure Mamba). Accepted for deserialization
    /// but any nonzero value fails `validate()` — see module docs.
    #[serde(default)]
    pub attention_interval: usize,
    /// Explicit attention layer indices. Same rejection rule as above.
    #[serde(default)]
    pub attention_layers: Vec<usize>,
    /// Byte-level tokenization mode (vocab_size=259: pad/bos/eos + 256 bytes).
    #[serde(default)]
    pub byte_level: bool,
    /// RMSNorm epsilon.
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
}

fn default_norm_eps() -> f64 {
    1e-5
}

impl ModelConfig {
    /// A small config for `lumi-infer smoke` — no checkpoint files required.
    pub fn smoke_default() -> Self {
        Self {
            d_model: 256,
            n_layers: 4,
            d_state: 64,
            expand: 2,
            n_heads: 16,
            n_groups: 8,
            vocab_size: 259,
            max_seq_len: 2048,
            attention_interval: 0,
            attention_layers: Vec::new(),
            byte_level: true,
            norm_eps: 1e-5,
        }
    }

    /// Inner (expanded) dimension.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// Per-head dimension within the SSM.
    pub fn head_dim(&self) -> usize {
        self.d_inner() / self.n_heads
    }

    /// Combined width of one B or C projection (all groups concatenated).
    pub fn bc_size(&self) -> usize {
        self.n_groups * self.d_state
    }

    /// Width of the DD-RoPE theta projection (n_heads angles of d_state/2 each).
    pub fn theta_size(&self) -> usize {
        self.n_heads * self.d_state / 2
    }

    /// Total width of `in_proj`'s output, laid out as:
    /// [x_ssm: d_inner][z: d_inner][b: bc_size][c: bc_size][dt: n_heads]
    /// [lambda: n_heads][dd_a: n_heads][theta: n_heads*d_state/2]
    pub fn in_proj_out(&self) -> usize {
        2 * self.d_inner() + 2 * self.bc_size() + 3 * self.n_heads + self.theta_size()
    }

    /// Reject configs this decode path can't run correctly instead of silently
    /// mis-slicing `in_proj`'s output or under-sizing kernel shared memory.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.attention_interval > 0 || !self.attention_layers.is_empty() {
            anyhow::bail!("attention layers are not supported by native inference yet");
        }
        if self.d_state != 64 {
            anyhow::bail!(
                "d_state must be 64 (kernel requirement of ssm_step_gpu), got {}",
                self.d_state
            );
        }
        if self.n_heads == 0 || !self.d_inner().is_multiple_of(self.n_heads) {
            anyhow::bail!(
                "d_inner ({}) must be evenly divisible by n_heads ({})",
                self.d_inner(),
                self.n_heads
            );
        }
        if self.n_groups == 0 || !self.n_heads.is_multiple_of(self.n_groups) {
            anyhow::bail!(
                "n_heads ({}) must be evenly divisible by n_groups ({})",
                self.n_heads,
                self.n_groups
            );
        }
        Ok(())
    }
}
