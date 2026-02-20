// ============================================================
// Layer 6 — Checkpoint Manager
// ============================================================
// Saves and restores model weights using Burn's CompactRecorder.
//
// What gets saved per checkpoint:
//   1. Model weights (.mpk.gz file) — all learned parameters
//   2. latest_epoch.json            — which epoch was last saved
//   3. train_config.json            — model architecture config
//
// Why save the config separately?
//   When loading for inference, we need to know the exact
//   model architecture (d_model, num_layers, etc.) to rebuild
//   the model before loading the weights into it.
//   Without the config, we can't reconstruct the model.
//
// Burn's CompactRecorder:
//   - Serialises model parameters to MessagePack format
//   - Compresses with gzip for smaller file size
//   - Type-safe: loading fails if architecture doesn't match
//
// File naming convention:
//   checkpoints/
//     model_epoch_1.mpk.gz   ← weights after epoch 1
//     model_epoch_2.mpk.gz   ← weights after epoch 2
//     ...
//     latest_epoch.json      ← contains the number of latest epoch
//     train_config.json      ← model hyperparameters
//
// Reference: Burn Book §5 (Records and Checkpointing)
//            Rust Book §9 (Error Handling)

use anyhow::{Context, Result};
use std::{fs, path::PathBuf};
use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};
use serde_json;

use crate::application::train_use_case::TrainConfig;
use crate::ml::model::TransformerQaModel;

/// Manages saving and loading of model checkpoints.
/// All files are stored in the configured directory.
pub struct CheckpointManager {
    /// Path to the directory where checkpoints are stored
    dir: PathBuf,
}

impl CheckpointManager {
    /// Create a new CheckpointManager.
    /// Creates the directory if it doesn't already exist.
    pub fn new(dir: impl Into<String>) -> Self {
        let dir = PathBuf::from(dir.into());
        // create_dir_all creates parent directories too, like `mkdir -p`
        // .ok() ignores the error if the directory already exists
        fs::create_dir_all(&dir).ok();
        Self { dir }
    }

    /// Save model weights for a given epoch.
    ///
    /// Uses Burn's CompactRecorder which:
    ///   1. Calls model.into_record() to extract all parameters
    ///   2. Serialises to MessagePack binary format
    ///   3. Compresses with gzip
    ///   4. Writes to {dir}/model_epoch_{epoch}.mpk.gz
    pub fn save_model<B: AutodiffBackend>(
        &self,
        model: &TransformerQaModel<B>,
        epoch: usize,
    ) -> Result<()> {
        // Build the file path (without extension — recorder adds it)
        let path = self.dir.join(format!("model_epoch_{epoch}"));

        // Save model weights using CompactRecorder
        CompactRecorder::new()
            .record(model.clone().into_record(), path.clone())
            .with_context(|| {
                format!("Failed to save checkpoint to '{}'", path.display())
            })?;

        // Update the latest epoch pointer
        // This tells the inferencer which file to load
        let latest_path = self.dir.join("latest_epoch.json");
        fs::write(&latest_path, serde_json::to_string(&epoch)?)
            .with_context(|| "Failed to write latest_epoch.json")?;

        tracing::debug!("Saved checkpoint: epoch {}", epoch);
        Ok(())
    }

    /// Load model weights from the latest saved checkpoint.
    ///
    /// Steps:
    ///   1. Read latest_epoch.json to find the epoch number
    ///   2. Load the corresponding .mpk.gz file
    ///   3. Call model.load_record() to restore weights
    ///
    /// The model parameter must have the correct architecture
    /// (matching the saved checkpoint) or loading will fail.
    pub fn load_model<B: Backend>(
        &self,
        model:  TransformerQaModel<B>,
        device: &B::Device,
    ) -> Result<TransformerQaModel<B>> {
        // Find out which epoch was saved last
        let epoch = self.latest_epoch()?;
        let path  = self.dir.join(format!("model_epoch_{epoch}"));

        tracing::info!("Loading checkpoint from epoch {}", epoch);

        // Load the serialised record from disk
        let record = CompactRecorder::new()
            .load(path.clone(), device)
            .with_context(|| {
                format!("Cannot load checkpoint '{}'. Have you trained the model first?",
                    path.display())
            })?;

        // Restore the weights into the model
        // load_record() returns a new model with the loaded weights
        Ok(model.load_record(record))
    }

    /// Save the training configuration to JSON.
    ///
    /// This must be called before training starts so the
    /// inferencer can reconstruct the exact model architecture.
    pub fn save_config(&self, cfg: &TrainConfig) -> Result<()> {
        let path = self.dir.join("train_config.json");

        // serde_json::to_string_pretty adds indentation for readability
        let json = serde_json::to_string_pretty(cfg)?;

        fs::write(&path, json)
            .with_context(|| {
                format!("Cannot write config to '{}'", path.display())
            })?;

        tracing::debug!("Saved training config to '{}'", path.display());
        Ok(())
    }

    /// Load the training configuration from JSON.
    ///
    /// Called by the Inferencer to know what model architecture
    /// was used during training so it can rebuild the same model.
    pub fn load_config(&self) -> Result<TrainConfig> {
        let path = self.dir.join("train_config.json");

        let json = fs::read_to_string(&path)
            .with_context(|| {
                format!(
                    "Cannot read config from '{}'. \
                     Make sure you have run 'train' before 'ask'.",
                    path.display()
                )
            })?;

        // Deserialise JSON back into TrainConfig struct
        Ok(serde_json::from_str(&json)?)
    }

    /// Read latest_epoch.json and return the epoch number.
    /// Returns an error if training hasn't been run yet.
    fn latest_epoch(&self) -> Result<usize> {
        let path = self.dir.join("latest_epoch.json");

        let s = fs::read_to_string(&path)
            .with_context(|| {
                "Cannot find 'latest_epoch.json'. \
                 Have you run 'train' first?"
            })?;

        Ok(serde_json::from_str::<usize>(&s)?)
    }
}
