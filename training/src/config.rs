/// Configuration for the hybrid Mamba-3 model, training, and Goldilocks sampling.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Hidden dimension.
    pub d_model: usize,
    /// Total number of blocks (Mamba + Attention).
    pub n_layers: usize,
    /// SSM state dimension.
    pub d_state: usize,
    /// Causal conv kernel size.
    pub d_conv: usize,
    /// Expansion factor (d_inner = expand * d_model).
    pub expand: usize,
    /// Number of SSM heads.
    pub n_heads: usize,
    /// Number of groups for B, C matrices.
    pub n_groups: usize,
    /// Chunk size for SSD algorithm.
    pub chunk_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
    /// Attention block interval (0 = pure Mamba, 8 = attention every 8 layers).
    /// When > 0, layer i is attention if (i+1) % attention_interval == 0.
    #[serde(default)]
    pub attention_interval: usize,
    /// Number of query heads in attention blocks (default 16).
    #[serde(default = "default_attn_n_heads")]
    pub attn_n_heads: usize,
    /// Number of KV heads for GQA in attention blocks (default 4).
    #[serde(default = "default_attn_kv_heads")]
    pub attn_kv_heads: usize,
    /// MLP expand factor in attention blocks (default 4).
    #[serde(default = "default_attn_mlp_expand")]
    pub attn_mlp_expand: usize,
    /// Sliding window sizes per attention layer. Empty = full causal (default).
    #[serde(default)]
    pub attn_window_sizes: Vec<usize>,
    /// Explicit attention layer indices (0-indexed). Overrides attention_interval if non-empty.
    #[serde(default)]
    pub attention_layers: Vec<usize>,
    /// Enable data-dependent RoPE on SSM state dimensions (Mamba-3 complex SSM).
    /// Learned rotation frequencies per head per state-pair. theta=0 init = no rotation.
    #[serde(default = "default_dd_rope")]
    pub dd_rope: bool,
    /// Use byte-level tokenization (vocab_size forced to 259: pad/bos/eos + 256 bytes).
    /// When true, no BPE tokenizer is needed — raw UTF-8 bytes are used directly.
    #[serde(default)]
    pub byte_level: bool,
}

fn default_attn_n_heads() -> usize { 16 }
fn default_attn_kv_heads() -> usize { 4 }
fn default_attn_mlp_expand() -> usize { 4 }
fn default_dd_rope() -> bool { true }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            d_model: 1024,
            n_layers: 48,
            d_state: 64,
            d_conv: 4,
            expand: 2,
            n_heads: 64,
            n_groups: 8,
            chunk_size: 256,
            vocab_size: 32000,
            max_seq_len: 2048,
            norm_eps: 1e-5,
            attention_interval: 0, // 0 = pure Mamba (default), 8 = hybrid
            attn_n_heads: 16,
            attn_kv_heads: 4,
            attn_mlp_expand: 4,
            attn_window_sizes: vec![],
            attention_layers: vec![],
            dd_rope: true,
            byte_level: false,
        }
    }
}

impl ModelConfig {
    /// Inner dimension after expansion.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// Dimension per SSM head.
    pub fn head_dim(&self) -> usize {
        self.d_inner() / self.n_heads
    }

    /// KV dimension for GQA attention (attn_kv_heads * head_dim).
    pub fn attn_kv_dim(&self) -> usize {
        self.attn_kv_heads * (self.d_model / self.attn_n_heads)
    }

    /// Attention head dimension.
    pub fn attn_head_dim(&self) -> usize {
        self.d_model / self.attn_n_heads
    }

    /// MLP hidden dimension in attention blocks.
    pub fn attn_mlp_dim(&self) -> usize {
        self.attn_mlp_expand * self.d_model
    }

    /// Number of attention layers.
    pub fn n_attn_layers(&self) -> usize {
        if !self.attention_layers.is_empty() {
            return self.attention_layers.len();
        }
        if self.attention_interval == 0 { return 0; }
        (0..self.n_layers).filter(|i| (i + 1) % self.attention_interval == 0).count()
    }

    /// Number of Mamba layers.
    pub fn n_mamba_layers(&self) -> usize {
        self.n_layers - self.n_attn_layers()
    }

    /// Is layer i an attention layer?
    pub fn is_attention_layer(&self, i: usize) -> bool {
        if !self.attention_layers.is_empty() {
            return self.attention_layers.contains(&i);
        }
        self.attention_interval > 0 && (i + 1) % self.attention_interval == 0
    }

    /// Window size for the i-th attention layer (0-indexed among attention layers).
    /// Returns 0 for full causal (no window).
    pub fn attn_window_size(&self, attn_idx: usize) -> usize {
        if attn_idx < self.attn_window_sizes.len() {
            self.attn_window_sizes[attn_idx]
        } else {
            0
        }
    }

    /// Estimated parameter count.
    pub fn param_count(&self) -> usize {
        let embedding = self.vocab_size * self.d_model;
        let d_inner = self.d_inner();
        let kv_dim = self.attn_kv_dim();
        let mlp_dim = self.attn_mlp_dim();

        // Mamba block params
        let theta_size = if self.dd_rope { self.n_heads * self.d_state / 2 } else { 0 };
        // in_proj: x(d_inner) + z(d_inner) + B(n_groups*d_state) + C(n_groups*d_state) + dt(n_heads) + lambda(n_heads) + dd_A(n_heads) + theta(optional)
        let in_proj = self.d_model * (d_inner + d_inner + self.n_groups * self.d_state * 2 + self.n_heads + self.n_heads + self.n_heads + theta_size);
        let out_proj = d_inner * self.d_model;
        let norm = self.d_model;
        let ssm_params = self.n_heads * 2 + self.n_heads * self.d_state; // dt_bias, D, h_init
        let bc_params = self.n_groups * self.d_state * 2; // b_bias, c_bias
        let per_mamba = in_proj + out_proj + norm + ssm_params + bc_params;

        // Attention block params
        let q_proj = self.d_model * self.d_model;
        let k_proj = self.d_model * kv_dim;
        let v_proj = self.d_model * kv_dim;
        let attn_out = self.d_model * self.d_model;
        let mlp_gate = self.d_model * mlp_dim;
        let mlp_up = self.d_model * mlp_dim;
        let mlp_down = mlp_dim * self.d_model;
        let attn_norms = self.d_model * 2;
        let per_attn = q_proj + k_proj + v_proj + attn_out + mlp_gate + mlp_up + mlp_down + attn_norms;

        let total_mamba = self.n_mamba_layers() * per_mamba;
        let total_attn = self.n_attn_layers() * per_attn;
        let final_norm = self.d_model;

        embedding + total_mamba + total_attn + final_norm
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    /// Peak learning rate.
    pub learning_rate: f64,
    /// LR floor (33% of peak — lilm finding).
    pub min_lr: f64,
    /// Linear warmup steps.
    pub warmup_steps: usize,
    /// Total training steps.
    pub max_steps: usize,
    /// Per-GPU micro batch size.
    pub batch_size: usize,
    /// Gradient accumulation steps.
    pub gradient_accumulation: usize,
    /// AdamW weight decay.
    pub weight_decay: f64,
    /// Steps between checkpoint saves.
    pub checkpoint_interval: usize,
    /// Steps between validation runs.
    pub eval_interval: usize,
    /// Steps between sample generation.
    pub sample_interval: usize,
    /// RNG seed.
    pub seed: u64,
    /// LR schedule: "wsd" (warmup-stable-decay, default) or "cosine".
    #[serde(default = "default_lr_schedule")]
    pub lr_schedule: String,
    /// For WSD: fraction of steps spent in decay phase (default 0.2 = last 20%).
    #[serde(default = "default_decay_fraction")]
    pub decay_fraction: f64,
    /// Offset warmup to start at this step (for checkpoint resume with re-warm).
    /// When set, warmup runs from warmup_offset to warmup_offset + warmup_steps.
    #[serde(default)]
    pub warmup_offset: usize,
}

fn default_lr_schedule() -> String { "wsd".to_string() }
fn default_decay_fraction() -> f64 { 0.2 }

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            learning_rate: 3e-4,
            min_lr: 1e-4,
            warmup_steps: 2000,
            max_steps: 15000,
            batch_size: 32,
            gradient_accumulation: 4,
            weight_decay: 0.1,
            checkpoint_interval: 500,
            eval_interval: 200,
            sample_interval: 1000,
            seed: 42,
            lr_schedule: "wsd".to_string(),
            decay_fraction: 0.2,
            warmup_offset: 0,
        }
    }
}

impl TrainingConfig {
    /// Learning rate at a given step. Supports "cosine" and "wsd" schedules.
    pub fn lr_at_step(&self, step: usize) -> f64 {
        let warmup_end = self.warmup_offset + self.warmup_steps;
        if step >= self.warmup_offset && step < warmup_end {
            // Linear warmup from warmup_offset (supports re-warm on resume)
            return self.learning_rate * ((step - self.warmup_offset) as f64 / self.warmup_steps as f64);
        }
        if step < self.warmup_offset {
            // Before warmup offset (shouldn't happen in normal use)
            return 0.0;
        }

        match self.lr_schedule.as_str() {
            "wsd" => {
                // Warmup-Stable-Decay: hold peak LR, then decay in final fraction
                let decay_start = self.max_steps - (self.max_steps as f64 * self.decay_fraction) as usize;
                if step < decay_start {
                    // Stable phase: hold at peak LR
                    self.learning_rate
                } else {
                    // Decay phase: cosine decay to floor
                    let progress = (step - decay_start) as f64
                        / (self.max_steps - decay_start).max(1) as f64;
                    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                    self.min_lr + (self.learning_rate - self.min_lr) * cosine
                }
            }
            _ => {
                // Cosine decay with floor (original)
                let progress = (step - self.warmup_steps) as f64
                    / (self.max_steps - self.warmup_steps).max(1) as f64;
                let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                self.min_lr + (self.learning_rate - self.min_lr) * cosine
            }
        }
    }

    /// Load config from JSON file.
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let mut config: Self = serde_json::from_str(&contents)?;
        if config.model.byte_level {
            config.model.vocab_size = 259;
        }
        Ok(config)
    }

    /// Save config to JSON file.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
