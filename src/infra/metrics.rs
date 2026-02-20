// ============================================================
// Layer 6 — Metrics Logger
// ============================================================
// Records training metrics to a CSV file after each epoch.
//
// Why log metrics to CSV?
//   - Easy to open in Excel or Google Sheets
//   - Can plot learning curves to diagnose training issues
//   - Provides a permanent record of each training run
//   - Required for the assignment report
//
// Metrics recorded per epoch:
//   - epoch:      the epoch number (1, 2, 3, ...)
//   - train_loss: average cross-entropy loss on training set
//   - val_loss:   average cross-entropy loss on validation set
//   - start_acc:  % of start positions predicted correctly
//   - end_acc:    % of end positions predicted correctly
//
// Output file: checkpoints/metrics.csv
//
// Example CSV output:
//   epoch,train_loss,val_loss,start_acc,end_acc
//   1,3.124500,3.089200,0.123000,0.118000
//   2,2.890100,2.854300,0.184000,0.172000
//   ...
//
// How to read the metrics:
//   - Loss should decrease each epoch (model is learning)
//   - If val_loss increases while train_loss decreases → overfitting
//   - Accuracy should increase each epoch
//   - start_acc and end_acc close together → balanced learning
//
// Reference: Rust Book §9 (Error Handling)
//            Rust Book §12 (I/O and File Handling)

use anyhow::Result;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
};
use serde::{Deserialize, Serialize};

/// One row of metrics data for a single training epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    /// The epoch number (starts at 1)
    pub epoch: usize,

    /// Average cross-entropy loss over all training batches
    /// Lower is better. Random initialisation gives ~ln(seq_len)
    pub train_loss: f64,

    /// Average cross-entropy loss on the validation set
    /// Should track train_loss — divergence indicates overfitting
    pub val_loss: f64,

    /// Fraction of start positions predicted exactly correctly
    /// Range: [0.0, 1.0] — 1.0 means perfect start prediction
    pub start_acc: f64,

    /// Fraction of end positions predicted exactly correctly
    /// Range: [0.0, 1.0] — 1.0 means perfect end prediction
    pub end_acc: f64,
}

impl EpochMetrics {
    /// Create a new EpochMetrics record
    pub fn new(
        epoch:      usize,
        train_loss: f64,
        val_loss:   f64,
        start_acc:  f64,
        end_acc:    f64,
    ) -> Self {
        Self { epoch, train_loss, val_loss, start_acc, end_acc }
    }

    /// Returns true if this epoch improved over the previous best val_loss
    pub fn is_improvement(&self, best_val_loss: f64) -> bool {
        self.val_loss < best_val_loss
    }
}

/// Logs epoch metrics to a CSV file for later analysis.
pub struct MetricsLogger {
    /// Full path to the CSV file
    csv_path: PathBuf,
}

impl MetricsLogger {
    /// Create a new MetricsLogger.
    /// Writes the CSV header if the file doesn't exist yet.
    pub fn new(dir: impl Into<String>) -> Result<Self> {
        let dir = PathBuf::from(dir.into());

        // Create directory if it doesn't exist
        fs::create_dir_all(&dir)?;

        let csv_path = dir.join("metrics.csv");

        // Write CSV header only if file is new
        // This allows appending to an existing log across runs
        if !csv_path.exists() {
            let mut f = fs::File::create(&csv_path)?;
            // Write the header row
            writeln!(f, "epoch,train_loss,val_loss,start_acc,end_acc")?;
            tracing::debug!("Created metrics CSV: '{}'", csv_path.display());
        }

        Ok(Self { csv_path })
    }

    /// Append one epoch's metrics as a new row in the CSV.
    ///
    /// Uses OpenOptions with append=true so we add to the file
    /// without overwriting previous epochs.
    pub fn log(&self, m: &EpochMetrics) -> Result<()> {
        // Open in append mode — adds to end of file
        let mut f = OpenOptions::new()
            .append(true)
            .open(&self.csv_path)?;

        // Write one CSV row with 6 decimal places for each metric
        writeln!(
            f,
            "{},{:.6},{:.6},{:.6},{:.6}",
            m.epoch,
            m.train_loss,
            m.val_loss,
            m.start_acc,
            m.end_acc,
        )?;

        tracing::debug!(
            "Logged epoch {} metrics: train_loss={:.4}, val_loss={:.4}",
            m.epoch,
            m.train_loss,
            m.val_loss,
        );

        Ok(())
    }

    /// Return the path to the metrics CSV file
    pub fn csv_path(&self) -> &PathBuf {
        &self.csv_path
    }
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_improvement() {
        let m = EpochMetrics::new(2, 2.5, 2.3, 0.2, 0.2);
        // 2.3 < 3.0 → this is an improvement
        assert!(m.is_improvement(3.0));
        // 2.3 is NOT less than 2.0 → not an improvement
        assert!(!m.is_improvement(2.0));
    }
}
