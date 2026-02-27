// ============================================================
// Layer 6 — Checkpoint Manager
// ============================================================
use anyhow::Result;
use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    prelude::*,
};
use std::path::PathBuf;
use crate::application::train_use_case::TrainConfig;

pub struct CheckpointManager {
    dir: PathBuf,
}

impl CheckpointManager {
    pub fn new(dir: impl Into<String>) -> Self {
        Self { dir: PathBuf::from(dir.into()) }
    }

    pub fn save_model<B: Backend, M: Module<B>>(
        &self,
        model: &M,
        epoch: usize,
    ) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.dir.join(format!("model_epoch_{}", epoch));
        CompactRecorder::new()
            .record(model.clone().into_record(), path.clone())
            .map_err(|e| anyhow::anyhow!("Save error: {e}"))?;

        // Write latest epoch number
        std::fs::write(
            self.dir.join("latest_epoch.json"),
            serde_json::to_string(&serde_json::json!({ "epoch": epoch }))?,
        )?;
        Ok(())
    }

    pub fn load_model<B: Backend, M: Module<B>>(
        &self,
        model: M,
        device: &B::Device,
    ) -> Result<M> {
        let epoch = self.best_epoch()?;
        let path  = self.dir.join(format!("model_epoch_{}", epoch));
        tracing::info!("Loading checkpoint from epoch {}", epoch);
        let record = CompactRecorder::new()
            .load(path, device)
            .map_err(|e| anyhow::anyhow!("Load error: {e}"))?;
        Ok(model.load_record(record))
    }

    /// Returns the best epoch (epoch 6 if it exists, else latest)
    pub fn best_epoch(&self) -> Result<usize> {
        // Check if epoch 6 exists — that was our best training epoch
        let best = self.dir.join("model_epoch_6.mpk");
        if best.exists() {
            return Ok(6);
        }
        // Fall back to latest
        let json = std::fs::read_to_string(self.dir.join("latest_epoch.json"))
            .map_err(|_| anyhow::anyhow!("No checkpoint found. Please train first."))?;
        let val: serde_json::Value = serde_json::from_str(&json)?;
        Ok(val["epoch"].as_u64().unwrap_or(1) as usize)
    }

    pub fn save_config(&self, cfg: &TrainConfig) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        std::fs::write(
            self.dir.join("train_config.json"),
            serde_json::to_string_pretty(cfg)?,
        )?;
        Ok(())
    }

    pub fn load_config(&self) -> Result<TrainConfig> {
        let json = std::fs::read_to_string(self.dir.join("train_config.json"))
            .map_err(|_| anyhow::anyhow!("No train_config.json found."))?;
        Ok(serde_json::from_str(&json)?)
    }
}
