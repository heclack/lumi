use std::collections::HashSet;
use std::fs;

use lumi::checkpoint::{CheckpointMeta, clean_checkpoints, find_latest_checkpoint};
use lumi::config::TrainingConfig;
use tempfile::TempDir;

// ─── CheckpointMeta round-trip ──────────────────────────────

#[test]
fn checkpoint_meta_round_trip() {
    let dir = TempDir::new().unwrap();
    let dir_path = dir.path().to_str().unwrap();

    let config = TrainingConfig::default();
    let meta = CheckpointMeta {
        step: 1500,
        epoch: 3,
        train_loss: 2.345,
        val_loss: Some(2.567),
        config: config.clone(),
    };

    // Save as meta.json
    let meta_path = dir.path().join("meta.json");
    let json = serde_json::to_string_pretty(&meta).unwrap();
    fs::write(&meta_path, &json).unwrap();

    // Load it back
    let loaded = CheckpointMeta::load(dir_path).unwrap();

    assert_eq!(loaded.step, 1500);
    assert_eq!(loaded.epoch, 3);
    assert!((loaded.train_loss - 2.345).abs() < 1e-6);
    assert!(loaded.val_loss.is_some());
    assert!((loaded.val_loss.unwrap() - 2.567).abs() < 1e-6);
    assert_eq!(loaded.config.model.d_model, config.model.d_model);
    assert_eq!(loaded.config.model.n_layers, config.model.n_layers);
    assert_eq!(loaded.config.max_steps, config.max_steps);
    assert_eq!(loaded.config.batch_size, config.batch_size);
}

// ─── find_latest_checkpoint ─────────────────────────────────

#[test]
fn find_latest_checkpoint_returns_highest_step() {
    let dir = TempDir::new().unwrap();
    let dir_path = dir.path().to_str().unwrap();

    // Create checkpoint dirs in non-sorted order
    fs::create_dir(dir.path().join("step-000100")).unwrap();
    fs::create_dir(dir.path().join("step-000200")).unwrap();
    fs::create_dir(dir.path().join("step-000050")).unwrap();

    let latest = find_latest_checkpoint(dir_path);
    assert!(latest.is_some());
    let latest = latest.unwrap();
    assert!(latest.ends_with("step-000200"), "Expected step-000200, got {}", latest);
}

#[test]
fn find_latest_checkpoint_empty_dir() {
    let dir = TempDir::new().unwrap();
    let dir_path = dir.path().to_str().unwrap();

    let result = find_latest_checkpoint(dir_path);
    assert!(result.is_none());
}

// ─── clean_checkpoints ──────────────────────────────────────

#[test]
fn clean_checkpoints_keeps_newest() {
    let dir = TempDir::new().unwrap();
    let dir_path = dir.path().to_str().unwrap();

    // Create 5 checkpoint dirs
    for step in &["step-000100", "step-000200", "step-000300", "step-000400", "step-000500"] {
        fs::create_dir(dir.path().join(step)).unwrap();
    }

    clean_checkpoints(dir_path, 2, None);

    // Collect remaining checkpoint dirs
    let remaining: Vec<String> = fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|name| name.starts_with("step-"))
        .collect();

    assert_eq!(remaining.len(), 2, "Expected 2 checkpoints, got {:?}", remaining);

    // The two newest should survive (step-000400 and step-000500)
    let remaining_set: HashSet<String> = remaining.into_iter().collect();
    assert!(remaining_set.contains("step-000400"), "step-000400 should be kept");
    assert!(remaining_set.contains("step-000500"), "step-000500 should be kept");
}

#[test]
fn clean_checkpoints_protects_best() {
    let dir = TempDir::new().unwrap();
    let dir_path = dir.path().to_str().unwrap();

    for step in &["step-000100", "step-000200", "step-000300", "step-000400", "step-000500"] {
        fs::create_dir(dir.path().join(step)).unwrap();
    }

    // Protect step-000200 (best val loss) while keeping 2 most recent
    clean_checkpoints(dir_path, 2, Some("step-000200"));

    let remaining: HashSet<String> = fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|name| name.starts_with("step-"))
        .collect();

    assert!(remaining.contains("step-000200"), "protected checkpoint should survive");
    assert!(remaining.contains("step-000500"), "newest should survive");
    assert!(remaining.len() == 3, "Expected 3 (2 newest + 1 protected), got {:?}", remaining);
}

#[test]
fn clean_checkpoints_fewer_than_keep() {
    let dir = TempDir::new().unwrap();
    let dir_path = dir.path().to_str().unwrap();

    fs::create_dir(dir.path().join("step-000100")).unwrap();
    fs::create_dir(dir.path().join("step-000200")).unwrap();

    clean_checkpoints(dir_path, 5, None);

    let remaining: Vec<String> = fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|name| name.starts_with("step-"))
        .collect();

    assert_eq!(remaining.len(), 2, "Nothing should be deleted when fewer than keep");
}

