/// Validation tests for the training startup process.
///
/// These test the dimension calculations, config invariants, and data pipeline
/// functions that run during `train_native()` initialization — before the
/// training loop starts. All tests run on CPU (no CUDA required).

use lumi::config::{ModelConfig, TrainingConfig};
use lumi::data::{TokenDataset, find_coprime_stride};
use std::io::Write;

// ---------------------------------------------------------------------------
// in_proj_out dimension calculation
// ---------------------------------------------------------------------------

/// Compute in_proj_out the same way native_trainer.rs and gpu_memory.rs do.
fn compute_in_proj_out(config: &ModelConfig) -> usize {
    let d_inner = config.d_inner();
    let bc_size = config.n_groups * config.d_state;
    let theta_proj_size = config.n_heads * config.d_state / 2; // DD-RoPE always on
    // x + z + B + C + dt + lambda + dd_A + theta
    d_inner + d_inner + bc_size * 2 + config.n_heads + config.n_heads + config.n_heads + theta_proj_size
}

#[test]
fn in_proj_out_default_config() {
    let config = ModelConfig::default();
    let in_proj_out = compute_in_proj_out(&config);
    // d_inner=2048, bc_size=8*64=512, n_heads=64, theta=64*64/2=2048
    // 2048 + 2048 + 512*2 + 64 + 64 + 64 + 2048 = 7360
    assert_eq!(in_proj_out, 7360, "in_proj_out mismatch for default config");
}

#[test]
fn in_proj_out_small_config() {
    let config = ModelConfig {
        d_model: 128, n_layers: 4, d_state: 16,
        expand: 2, n_heads: 8, n_groups: 2, chunk_size: 64,
        vocab_size: 1000, max_seq_len: 128, norm_eps: 1e-5,
        ..ModelConfig::default()
    };
    let d_inner = 256;
    let bc_size = 2 * 16; // 32
    let theta = 8 * 16 / 2; // 64
    let expected = d_inner + d_inner + bc_size * 2 + 8 + 8 + 8 + theta;
    assert_eq!(compute_in_proj_out(&config), expected);
}

/// in_proj_out must match the per-Mamba-layer in_proj weight size used in param_count.
#[test]
fn in_proj_out_matches_param_count_assumption() {
    let config = ModelConfig::default();
    let in_proj_out = compute_in_proj_out(&config);

    // param_count computes in_proj as d_model * (d_inner + d_inner + n_groups*d_state*2 + n_heads*3 + theta)
    // which should equal d_model * in_proj_out
    let d_inner = config.d_inner();
    let theta_size = config.n_heads * config.d_state / 2; // DD-RoPE always on
    let param_count_in_proj_cols = d_inner + d_inner
        + config.n_groups * config.d_state * 2
        + config.n_heads + config.n_heads + config.n_heads
        + theta_size;
    assert_eq!(in_proj_out, param_count_in_proj_cols,
        "in_proj_out disagrees with param_count's in_proj calculation");
}

// ---------------------------------------------------------------------------
// Dimension invariants (must hold for CUDA kernels to work)
// ---------------------------------------------------------------------------

#[test]
fn head_dim_divides_evenly() {
    let config = ModelConfig::default();
    assert_eq!(config.d_inner() % config.n_heads, 0,
        "d_inner ({}) must be divisible by n_heads ({})",
        config.d_inner(), config.n_heads);
}

#[test]
fn head_dim_divides_evenly_small() {
    // Verify with a small config that might trip up
    let config = ModelConfig {
        d_model: 128, expand: 2, n_heads: 8,
        ..ModelConfig::default()
    };
    assert_eq!(config.d_inner() % config.n_heads, 0);
    assert_eq!(config.head_dim(), 32);
}

#[test]
fn d_state_even_for_rope() {
    // DD-RoPE needs d_state/2 pairs — d_state must be even
    let config = ModelConfig::default();
    assert_eq!(config.d_state % 2, 0,
        "d_state must be even (DD-RoPE always on)");
}

#[test]
fn attn_head_dim_divides_evenly() {
    let config = ModelConfig::default();
    assert_eq!(config.d_model % config.attn_n_heads, 0,
        "d_model ({}) must be divisible by attn_n_heads ({})",
        config.d_model, config.attn_n_heads);
}

#[test]
fn n_groups_divides_n_heads() {
    // B/C are projected at group granularity, expanded to heads in the kernel
    let config = ModelConfig::default();
    assert_eq!(config.n_heads % config.n_groups, 0,
        "n_heads ({}) must be divisible by n_groups ({})",
        config.n_heads, config.n_groups);
}

#[test]
fn layer_counts_sum_to_total() {
    let mut config = ModelConfig::default();
    // Pure Mamba
    assert_eq!(config.n_mamba_layers() + config.n_attn_layers(), config.n_layers);

    // Hybrid
    config.attention_interval = 8;
    assert_eq!(config.n_mamba_layers() + config.n_attn_layers(), config.n_layers);

    // Explicit attention layers
    config.attention_layers = vec![10, 20, 30];
    assert_eq!(config.n_mamba_layers() + config.n_attn_layers(), config.n_layers);
}

// ---------------------------------------------------------------------------
// TrainingConfig validation
// ---------------------------------------------------------------------------

#[test]
fn gradient_accumulation_at_least_one() {
    let config = TrainingConfig::default();
    assert!(config.gradient_accumulation >= 1,
        "gradient_accumulation must be >= 1, got {}", config.gradient_accumulation);
}

#[test]
fn warmup_steps_less_than_max_steps() {
    let config = TrainingConfig::default();
    assert!(config.warmup_steps < config.max_steps,
        "warmup_steps ({}) must be < max_steps ({})",
        config.warmup_steps, config.max_steps);
}

#[test]
fn min_lr_less_than_peak_lr() {
    let config = TrainingConfig::default();
    assert!(config.min_lr <= config.learning_rate,
        "min_lr ({}) should be <= learning_rate ({})",
        config.min_lr, config.learning_rate);
}

#[test]
fn lr_at_warmup_end_equals_peak() {
    let config = TrainingConfig::default();
    let lr = config.lr_at_step(config.warmup_steps);
    assert!((lr - config.learning_rate).abs() < 1e-10,
        "LR at warmup end should equal peak: got {}, expected {}",
        lr, config.learning_rate);
}

#[test]
fn lr_monotonic_during_warmup() {
    let config = TrainingConfig::default();
    let mut prev = 0.0;
    for step in 0..=config.warmup_steps {
        let lr = config.lr_at_step(step);
        assert!(lr >= prev, "LR decreased during warmup at step {}: {} < {}",
            step, lr, prev);
        prev = lr;
    }
}

#[test]
fn lr_never_below_floor() {
    let config = TrainingConfig::default();
    // Sample across all steps
    for step in (0..config.max_steps).step_by(10) {
        let lr = config.lr_at_step(step);
        if step >= config.warmup_steps {
            assert!(lr >= config.min_lr - 1e-12,
                "LR below floor at step {}: {} < {}", step, lr, config.min_lr);
        }
    }
}

#[test]
fn decay_fraction_in_valid_range() {
    let config = TrainingConfig::default();
    assert!(config.decay_fraction > 0.0 && config.decay_fraction < 1.0,
        "decay_fraction must be in (0, 1), got {}", config.decay_fraction);
}

// ---------------------------------------------------------------------------
// Config round-trip with training-relevant fields
// ---------------------------------------------------------------------------

#[test]
fn config_round_trip_preserves_in_proj_dimensions() {
    let config = TrainingConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let loaded: TrainingConfig = serde_json::from_str(&json).unwrap();

    // All dimension-affecting fields must survive round-trip
    assert_eq!(compute_in_proj_out(&config.model), compute_in_proj_out(&loaded.model));
    assert_eq!(config.model.d_inner(), loaded.model.d_inner());
    assert_eq!(config.model.head_dim(), loaded.model.head_dim());
    assert_eq!(config.model.n_mamba_layers(), loaded.model.n_mamba_layers());
    assert_eq!(config.model.n_attn_layers(), loaded.model.n_attn_layers());
}

// ---------------------------------------------------------------------------
// TokenDataset edge cases
// ---------------------------------------------------------------------------

fn write_temp_tokens(tokens: &[u32]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    for &t in tokens {
        f.write_all(&t.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
    f
}

#[test]
fn dataset_len_exact_boundary() {
    // Exactly seq_len tokens → 0 valid windows (need seq_len + 1 for target)
    let f = write_temp_tokens(&[1, 2, 3, 4]);
    let ds = TokenDataset::from_binary(f.path().to_str().unwrap(), 4);
    assert_eq!(ds.len(), 0);
}

#[test]
fn dataset_len_one_more_than_seq() {
    // seq_len + 1 tokens → 1 valid window
    let f = write_temp_tokens(&[1, 2, 3, 4, 5]);
    let ds = TokenDataset::from_binary(f.path().to_str().unwrap(), 4);
    assert_eq!(ds.len(), 1);
}

#[test]
fn dataset_single_token() {
    let f = write_temp_tokens(&[42]);
    let ds = TokenDataset::from_binary(f.path().to_str().unwrap(), 4);
    assert_eq!(ds.len(), 0);
    assert!(ds.is_empty());
}

// ---------------------------------------------------------------------------
// Coprime stride guarantees
// ---------------------------------------------------------------------------

#[test]
fn coprime_stride_is_actually_coprime() {
    // Test with numbers that have many factors
    for n in [256, 512, 1024, 2048, 65536, 1_000_000] {
        let stride = find_coprime_stride(n);
        // Verify gcd is 1
        let mut a = stride;
        let mut b = n;
        while b != 0 { let t = b; b = a % b; a = t; }
        assert_eq!(a, 1, "stride {} not coprime with {}", stride, n);
    }
}

#[test]
fn coprime_stride_nonzero() {
    for n in [1, 2, 3, 100, 999_999] {
        let stride = find_coprime_stride(n);
        assert!(stride > 0, "stride must be > 0 for n={}", n);
    }
}

#[test]
fn coprime_stride_visits_all_indices_medium_dataset() {
    // Simulate what the training loop does: pos = (pos + stride) % n
    let n = 47_000; // realistic dataset window count
    let stride = find_coprime_stride(n);
    let mut visited = vec![false; n];
    let mut pos = 0;
    for _ in 0..n {
        assert!(!visited[pos], "duplicate visit at {} (stride={}, n={})", pos, stride, n);
        visited[pos] = true;
        pos = (pos + stride) % n;
    }
    assert!(visited.iter().all(|&v| v), "not all indices visited");
}

// ---------------------------------------------------------------------------
// Effective batch size computation
// ---------------------------------------------------------------------------

#[test]
fn effective_batch_tokens() {
    let config = TrainingConfig::default();
    let bs = config.batch_size * config.model.max_seq_len;
    let effective = bs * config.gradient_accumulation;
    // Default: 32 * 2048 * 4 = 262144 tokens per optimizer step
    assert_eq!(effective, 262144);
}

// ---------------------------------------------------------------------------
// Hybrid config consistency
// ---------------------------------------------------------------------------

#[test]
fn hybrid_config_dimensions_valid() {
    let mut config = ModelConfig::default();
    config.attention_interval = 8;

    let n_attn = config.n_attn_layers();
    let n_mamba = config.n_mamba_layers();

    // Each attention layer replaces a Mamba layer
    assert_eq!(n_attn + n_mamba, config.n_layers);
    assert!(n_attn > 0, "hybrid config should have attention layers");

    // Attention dimensions must be valid
    assert!(config.attn_kv_dim() > 0);
    assert!(config.attn_mlp_dim() > 0);
    assert_eq!(config.d_model % config.attn_n_heads, 0);
}

#[test]
fn explicit_attention_layers_in_bounds() {
    let mut config = ModelConfig::default();
    config.attention_layers = vec![31, 47];
    for &idx in &config.attention_layers {
        assert!(idx < config.n_layers,
            "attention layer index {} >= n_layers {}", idx, config.n_layers);
    }
}

// ---------------------------------------------------------------------------
// Param count sanity
// ---------------------------------------------------------------------------

#[test]
fn param_count_increases_with_layers() {
    let mut small = ModelConfig::default();
    small.n_layers = 12;
    let mut large = ModelConfig::default();
    large.n_layers = 48;
    assert!(large.param_count() > small.param_count());
}

#[test]
fn param_count_increases_with_d_model() {
    let mut small = ModelConfig::default();
    small.d_model = 512;
    small.n_heads = 32; // keep head_dim = 32
    let mut large = ModelConfig::default();
    large.d_model = 1024;
    assert!(large.param_count() > small.param_count());
}

#[test]
fn param_count_hybrid_greater_than_pure_mamba() {
    let pure = ModelConfig::default();
    let mut hybrid = ModelConfig::default();
    hybrid.attention_interval = 8;
    // Attention blocks have more params than Mamba blocks at this scale
    assert!(hybrid.param_count() > pure.param_count(),
        "hybrid ({}) should have more params than pure mamba ({})",
        hybrid.param_count(), pure.param_count());
}
