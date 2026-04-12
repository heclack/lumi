/// Checkpoint utilities for training: metadata, discovery, cleanup.
///
/// Model weights are saved/loaded via native_checkpoint.rs (raw binary format).

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::TrainingConfig;

/// Metadata saved alongside the model checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub step: usize,
    pub epoch: usize,
    pub train_loss: f32,
    pub val_loss: Option<f32>,
    pub config: TrainingConfig,
}

impl CheckpointMeta {
    pub fn load(dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let path = Path::new(dir).join("meta.json");
        let json = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

/// Find the latest checkpoint in a directory.
pub fn find_latest_checkpoint(dir: &str) -> Option<String> {
    let entries = fs::read_dir(dir).ok()?;
    let mut checkpoints: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("step-"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    checkpoints.sort();
    checkpoints.last().cloned()
}

/// Remove old checkpoints, keeping the most recent `keep` plus any protected paths.
pub fn clean_checkpoints(dir: &str, keep: usize, protect: Option<&str>) {
    if let Ok(entries) = fs::read_dir(dir) {
        let mut checkpoints: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("step-"))
            .map(|e| e.path().to_string_lossy().to_string())
            .collect();
        checkpoints.sort();

        if checkpoints.len() > keep {
            let to_remove = checkpoints.len() - keep;
            for path in checkpoints.iter().take(to_remove) {
                if let Some(protected) = protect {
                    if path.ends_with(protected) {
                        continue;
                    }
                }
                if let Err(e) = fs::remove_dir_all(path) {
                    eprintln!("Warning: failed to remove old checkpoint {}: {}", path, e);
                }
            }
        }
    }
}
