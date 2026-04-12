/// Tests for config serialization, LR schedule, and model config helpers.

use lumi::config::{ModelConfig, TrainingConfig};

#[test]
fn config_round_trip() {
    let config = TrainingConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let loaded: TrainingConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.learning_rate, loaded.learning_rate);
    assert_eq!(config.max_steps, loaded.max_steps);
    assert_eq!(config.warmup_steps, loaded.warmup_steps);
    assert_eq!(config.batch_size, loaded.batch_size);
    assert_eq!(config.model.d_model, loaded.model.d_model);
    assert_eq!(config.model.n_layers, loaded.model.n_layers);
    assert_eq!(config.model.vocab_size, loaded.model.vocab_size);
    assert_eq!(config.model.attention_interval, loaded.model.attention_interval);
    assert_eq!(config.model.attn_n_heads, loaded.model.attn_n_heads);
    assert_eq!(config.model.attn_kv_heads, loaded.model.attn_kv_heads);
}

#[test]
fn config_deserialize_without_optional_fields() {
    // Older configs without attention or goldilocks fields should still load
    let json = r#"{
        "model": {
            "d_model": 256, "n_layers": 4, "d_state": 16, "d_conv": 4,
            "expand": 2, "n_heads": 8, "n_groups": 2, "chunk_size": 64,
            "vocab_size": 1000, "max_seq_len": 128, "norm_eps": 1e-5
        },
        "learning_rate": 1e-3, "min_lr": 1e-4,
        "warmup_steps": 100, "max_steps": 1000,
        "batch_size": 4, "gradient_accumulation": 1,
        "weight_decay": 0.1, "checkpoint_interval": 500,
        "eval_interval": 200, "sample_interval": 500, "seed": 42
    }"#;

    let config: TrainingConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.model.attention_interval, 0);
    assert_eq!(config.model.attn_n_heads, 16);
    assert_eq!(config.model.attn_kv_heads, 4);
    assert_eq!(config.lr_schedule, "wsd");
}

#[test]
fn lr_schedule_wsd() {
    let config = TrainingConfig {
        learning_rate: 3e-4,
        min_lr: 1e-4,
        warmup_steps: 100,
        max_steps: 1000,
        lr_schedule: "wsd".to_string(),
        decay_fraction: 0.2,
        ..TrainingConfig::default()
    };

    // Step 0: LR = 0
    assert_eq!(config.lr_at_step(0), 0.0);

    // Mid-warmup: linear interpolation
    let lr_50 = config.lr_at_step(50);
    assert!((lr_50 - 1.5e-4).abs() < 1e-8, "warmup mid: {}", lr_50);

    // End of warmup: peak LR
    let lr_100 = config.lr_at_step(100);
    assert!((lr_100 - 3e-4).abs() < 1e-8, "warmup end: {}", lr_100);

    // Stable phase: peak LR
    let lr_500 = config.lr_at_step(500);
    assert!((lr_500 - 3e-4).abs() < 1e-8, "stable: {}", lr_500);

    // Decay start (step 800): still near peak
    let lr_800 = config.lr_at_step(800);
    assert!((lr_800 - 3e-4).abs() < 1e-6, "decay start: {}", lr_800);

    // End of training: at floor
    let lr_999 = config.lr_at_step(999);
    assert!(lr_999 >= 1e-4 && lr_999 <= 1.1e-4, "decay end: {}", lr_999);
}

#[test]
fn lr_schedule_cosine() {
    let config = TrainingConfig {
        learning_rate: 3e-4,
        min_lr: 1e-4,
        warmup_steps: 100,
        max_steps: 1000,
        lr_schedule: "cosine".to_string(),
        ..TrainingConfig::default()
    };

    // Warmup same as WSD
    assert_eq!(config.lr_at_step(0), 0.0);
    let lr_100 = config.lr_at_step(100);
    assert!((lr_100 - 3e-4).abs() < 1e-8);

    // Cosine decays immediately after warmup
    let lr_500 = config.lr_at_step(500);
    assert!(lr_500 < 3e-4 && lr_500 > 1e-4, "cosine mid: {}", lr_500);

    // End at floor
    let lr_999 = config.lr_at_step(999);
    assert!(lr_999 >= 1e-4 && lr_999 <= 1.1e-4, "cosine end: {}", lr_999);
}

#[test]
fn model_config_helpers() {
    let config = ModelConfig::default();

    assert_eq!(config.d_inner(), 2048);
    assert_eq!(config.head_dim(), 32);
    assert_eq!(config.attn_head_dim(), 64);
    assert_eq!(config.attn_kv_dim(), 256);
    assert_eq!(config.attn_mlp_dim(), 4096);

    // Pure Mamba (attention_interval = 0)
    assert!(!config.is_attention_layer(0));
    assert!(!config.is_attention_layer(7));
    assert_eq!(config.n_attn_layers(), 0);
    assert_eq!(config.n_mamba_layers(), 48);
}

#[test]
fn model_config_hybrid_layers() {
    let mut config = ModelConfig::default();
    config.attention_interval = 8;

    // Layer 7 (0-indexed) → (7+1) % 8 == 0 → attention
    assert!(config.is_attention_layer(7));
    assert!(config.is_attention_layer(15));
    assert!(config.is_attention_layer(23));
    assert!(!config.is_attention_layer(0));
    assert!(!config.is_attention_layer(6));
    assert!(!config.is_attention_layer(8));

    assert_eq!(config.n_attn_layers(), 6);
    assert_eq!(config.n_mamba_layers(), 42);
}

#[test]
fn param_count_pure_mamba() {
    let config = ModelConfig::default();
    let params = config.param_count();
    // ~495M for default pure Mamba config (DD-RoPE always on adds ~100M theta params)
    assert!(params > 450_000_000 && params < 550_000_000,
        "Expected ~495M params, got {}", params);
}

#[test]
fn param_count_hybrid() {
    let mut config = ModelConfig::default();
    config.attention_interval = 8;
    let params = config.param_count();
    // ~529M for hybrid (6 attention blocks replace 6 Mamba blocks, net ~33M more)
    assert!(params > 480_000_000 && params < 580_000_000,
        "Expected ~529M params, got {}", params);
}

#[test]
fn attn_window_size_empty_is_full_causal() {
    let config = ModelConfig::default();
    assert!(config.attn_window_sizes.is_empty());
    assert_eq!(config.attn_window_size(0), 0);
    assert_eq!(config.attn_window_size(5), 0);
}

#[test]
fn attn_window_size_per_layer() {
    let mut config = ModelConfig::default();
    config.attn_window_sizes = vec![256, 256, 256, 512, 512, 512];
    assert_eq!(config.attn_window_size(0), 256);
    assert_eq!(config.attn_window_size(2), 256);
    assert_eq!(config.attn_window_size(3), 512);
    assert_eq!(config.attn_window_size(5), 512);
    // Out of bounds falls back to full causal
    assert_eq!(config.attn_window_size(6), 0);
}

#[test]
fn attn_window_sizes_deserialize_missing() {
    let json = r#"{
        "model": {
            "d_model": 256, "n_layers": 4, "d_state": 16, "d_conv": 4,
            "expand": 2, "n_heads": 8, "n_groups": 2, "chunk_size": 64,
            "vocab_size": 1000, "max_seq_len": 128, "norm_eps": 1e-5
        },
        "learning_rate": 1e-3, "min_lr": 1e-4,
        "warmup_steps": 100, "max_steps": 1000,
        "batch_size": 4, "gradient_accumulation": 1,
        "weight_decay": 0.1, "checkpoint_interval": 500,
        "eval_interval": 200, "sample_interval": 500, "seed": 42
    }"#;
    let config: TrainingConfig = serde_json::from_str(json).unwrap();
    assert!(config.model.attn_window_sizes.is_empty());
}

#[test]
fn attention_layers_explicit_placement() {
    let mut config = ModelConfig::default();
    config.attention_layers = vec![31, 47];
    assert!(config.is_attention_layer(31));
    assert!(config.is_attention_layer(47));
    assert!(!config.is_attention_layer(0));
    assert!(!config.is_attention_layer(7));
    assert!(!config.is_attention_layer(30));
    assert_eq!(config.n_attn_layers(), 2);
    assert_eq!(config.n_mamba_layers(), 46);
}

#[test]
fn attention_layers_overrides_interval() {
    let mut config = ModelConfig::default();
    config.attention_interval = 8; // would give layers 7,15,23,31,39,47
    config.attention_layers = vec![31, 47]; // explicit overrides interval
    assert_eq!(config.n_attn_layers(), 2); // not 6
    assert!(!config.is_attention_layer(7)); // interval ignored
    assert!(config.is_attention_layer(31));
}

#[test]
fn coprime_stride_covers_full_dataset() {
    use lumi::data::find_coprime_stride;
    let n = 1000;
    let stride = find_coprime_stride(n);
    // Verify every index 0..n is visited exactly once
    let mut visited = vec![false; n];
    for i in 0..n {
        let idx = (i * stride) % n;
        assert!(!visited[idx], "Index {} visited twice (stride={}, i={})", idx, stride, i);
        visited[idx] = true;
    }
    assert!(visited.iter().all(|&v| v), "Not all indices visited");
}

#[test]
fn coprime_stride_different_sizes() {
    use lumi::data::find_coprime_stride;
    // Test various dataset sizes
    for n in [100, 1000, 10000, 93170931] {
        let stride = find_coprime_stride(n);
        assert!(stride > 0);
        assert!(n % stride != 0, "Stride {} not coprime with {}", stride, n);
    }
}

#[test]
fn byte_level_defaults_false() {
    let config = ModelConfig::default();
    assert!(!config.byte_level);
    assert_eq!(config.vocab_size, 32000);
}

#[test]
fn byte_level_backward_compat() {
    // Old JSON configs without byte_level should deserialize with byte_level=false
    let json = r#"{
        "model": {
            "d_model": 256, "n_layers": 4, "d_state": 16, "d_conv": 4,
            "expand": 2, "n_heads": 8, "n_groups": 2, "chunk_size": 64,
            "vocab_size": 1000, "max_seq_len": 128, "norm_eps": 1e-5
        },
        "learning_rate": 1e-3, "min_lr": 1e-4,
        "warmup_steps": 100, "max_steps": 1000,
        "batch_size": 4, "gradient_accumulation": 1,
        "weight_decay": 0.1, "checkpoint_interval": 500,
        "eval_interval": 200, "sample_interval": 500, "seed": 42
    }"#;
    let config: TrainingConfig = serde_json::from_str(json).unwrap();
    assert!(!config.model.byte_level);
    assert_eq!(config.model.vocab_size, 1000);
}

#[test]
fn byte_level_forces_vocab_size() {
    // When byte_level is true, from_file should force vocab_size to 259
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    let json = r#"{
        "model": {
            "d_model": 256, "n_layers": 4, "d_state": 16, "d_conv": 4,
            "expand": 2, "n_heads": 8, "n_groups": 2, "chunk_size": 64,
            "vocab_size": 32000, "max_seq_len": 128, "norm_eps": 1e-5,
            "byte_level": true
        },
        "learning_rate": 1e-3, "min_lr": 1e-4,
        "warmup_steps": 100, "max_steps": 1000,
        "batch_size": 4, "gradient_accumulation": 1,
        "weight_decay": 0.1, "checkpoint_interval": 500,
        "eval_interval": 200, "sample_interval": 500, "seed": 42
    }"#;
    std::fs::write(&path, json).unwrap();
    let config = TrainingConfig::from_file(path.to_str().unwrap()).unwrap();
    assert!(config.model.byte_level);
    assert_eq!(config.model.vocab_size, 259);
}

#[test]
fn minimal_config_deserializes() {
    let json = r#"{
        "model": { "d_model": 256, "n_layers": 4, "vocab_size": 1000 },
        "max_steps": 500
    }"#;
    let config: TrainingConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.model.d_model, 256);
    assert_eq!(config.model.n_layers, 4);
    assert_eq!(config.model.vocab_size, 1000);
    // Everything else from Default
    assert_eq!(config.model.d_state, 64);
    assert_eq!(config.model.expand, 2);
    assert_eq!(config.max_steps, 500);
    assert_eq!(config.learning_rate, 3e-4);
    assert_eq!(config.train_data, "data/train.bin");
    assert_eq!(config.val_data, "data/val.bin");
    assert_eq!(config.seq_len, 0);
}

#[test]
fn seq_len_override() {
    let json = r#"{
        "model": { "max_seq_len": 2048 },
        "seq_len": 512
    }"#;
    let config: TrainingConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.seq_len, 512);
    assert_eq!(config.model.max_seq_len, 2048);
}

#[test]
fn data_paths_configurable() {
    let json = r#"{
        "train_data": "/mnt/data/train.bin",
        "val_data": "/mnt/data/val.bin"
    }"#;
    let config: TrainingConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.train_data, "/mnt/data/train.bin");
    assert_eq!(config.val_data, "/mnt/data/val.bin");
}
