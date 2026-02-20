// ============================================================
// Layer 1 — CLI Commands and Arguments
// ============================================================
// Defines the two subcommands: `train` and `ask`
// and all their configurable flags.
//
// clap's derive macros automatically generate:
//   - help text (--help)
//   - error messages for missing args
//   - type conversion (string → usize, f64, etc.)
//
// Reference: Rust Book §12 (Building a CLI Program)

use clap::{Args, Subcommand};
use crate::application::train_use_case::TrainConfig;

/// The two top-level subcommands available to the user
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Train the Q&A model on .docx documents
    Train(TrainArgs),

    /// Ask a question using a trained checkpoint
    Ask(AskArgs),
}

/// All arguments for the `train` command.
/// Each field becomes a --flag on the command line.
#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Directory containing .docx files to train on
    #[arg(long, default_value = "data/docx_files")]
    pub docs_dir: String,

    /// Directory to save model checkpoints and tokenizer
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: String,

    /// Maximum number of tokens per input sequence
    /// Format: [CLS] question [SEP] context [SEP] + padding
    #[arg(long, default_value_t = 512)]
    pub max_seq_len: usize,

    /// Number of samples processed together in one forward pass
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    /// Number of full passes through the training data
    #[arg(long, default_value_t = 10)]
    pub epochs: usize,

    /// How fast the model learns — too high causes instability,
    /// too low causes slow convergence
    #[arg(long, default_value_t = 2e-4)]
    pub lr: f64,

    /// Hidden dimension of the transformer (d_model in the paper)
    /// Every token is represented as a vector of this size
    #[arg(long, default_value_t = 256)]
    pub d_model: usize,

    /// Number of attention heads in multi-head attention
    /// d_model must be divisible by num_heads
    #[arg(long, default_value_t = 8)]
    pub num_heads: usize,

    /// Number of stacked encoder layers (minimum 6 per assignment spec)
    #[arg(long, default_value_t = 6)]
    pub num_layers: usize,

    /// Inner dimension of the feed-forward network
    /// Typically 4x d_model
    #[arg(long, default_value_t = 1024)]
    pub d_ff: usize,

    /// Dropout probability — randomly zeroes activations during training
    /// to prevent overfitting
    #[arg(long, default_value_t = 0.1)]
    pub dropout: f64,

    /// Total number of unique tokens the model can recognise
    #[arg(long, default_value_t = 30522)]
    pub vocab_size: usize,
}

/// Convert CLI TrainArgs into the application-layer TrainConfig.
/// This is the boundary between Layer 1 and Layer 2 —
/// the application layer never sees clap types.
impl From<TrainArgs> for TrainConfig {
    fn from(a: TrainArgs) -> Self {
        TrainConfig {
            docs_dir:       a.docs_dir,
            checkpoint_dir: a.checkpoint_dir,
            max_seq_len:    a.max_seq_len,
            batch_size:     a.batch_size,
            epochs:         a.epochs,
            lr:             a.lr,
            d_model:        a.d_model,
            num_heads:      a.num_heads,
            num_layers:     a.num_layers,
            d_ff:           a.d_ff,
            dropout:        a.dropout,
            vocab_size:     a.vocab_size,
        }
    }
}

/// All arguments for the `ask` command
#[derive(Args, Debug)]
pub struct AskArgs {
    /// The natural language question to answer
    #[arg(long)]
    pub question: String,

    /// Directory with .docx files (same as used during training)
    #[arg(long, default_value = "data/docx_files")]
    pub docs_dir: String,

    /// Directory where checkpoints were saved during training
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: String,
}
