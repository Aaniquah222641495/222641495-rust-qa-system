// ============================================================
// Layer 4 — Q&A Dataset
// ============================================================
// Implements Burn's Dataset trait for our Q&A samples.
//
// What is a Dataset in Burn?
//   A Dataset is anything that can:
//     1. Return an item by index: .get(index) -> Option<Item>
//     2. Report its size:         .len()       -> usize
//
//   The DataLoader uses these two methods to feed batches
//   to the training loop efficiently.
//
// Our QaSample struct holds one fully tokenised training example:
//   - input_ids:      the token ID sequence (padded to max_seq_len)
//   - attention_mask: 1 for real tokens, 0 for padding
//   - start_position: where the answer begins in the sequence
//   - end_position:   where the answer ends in the sequence
//
// All sequences are pre-padded to max_seq_len so that
// batching is a simple stack operation (no dynamic padding needed).
//
// Reference: Burn Book §4 (Datasets)
//            Rust Book §10 (Traits)
//            Devlin et al. (2019) BERT paper

use burn::data::dataset::Dataset;
use serde::{Deserialize, Serialize};

// ─── QaSample ─────────────────────────────────────────────────────────────────
/// One fully tokenised and padded training sample.
///
/// The input sequence format is:
///   [CLS] q_1 q_2 ... q_n [SEP] c_1 c_2 ... c_m [SEP] [PAD] ... [PAD]
///    101                   102                    102    0          0
///
/// start_position and end_position are absolute indices
/// into this full sequence (not into just the context).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaSample {
    /// Token IDs for the full input sequence.
    /// Length is always exactly max_seq_len (padded with 0s).
    /// Each number maps to a word/subword in the vocabulary.
    pub input_ids: Vec<u32>,

    /// Attention mask — tells the model which tokens are real.
    /// 1 = real token (pay attention to this)
    /// 0 = padding token (ignore this)
    /// Same length as input_ids.
    pub attention_mask: Vec<u32>,

    /// Index of the first answer token in the full input sequence.
    /// Must be >= the position of the first context token
    /// (i.e., after [CLS] + question + [SEP])
    pub start_position: usize,

    /// Index of the last answer token in the full input sequence.
    /// Must be >= start_position.
    /// The answer span is input_ids[start_position..=end_position]
    pub end_position: usize,
}

impl QaSample {
    /// Returns the length of the answer span in tokens
    pub fn span_length(&self) -> usize {
        self.end_position.saturating_sub(self.start_position) + 1
    }

    /// Returns only the answer token IDs (the predicted span)
    pub fn answer_ids(&self) -> &[u32] {
        &self.input_ids[self.start_position..=self.end_position]
    }
}

// ─── QaDataset ────────────────────────────────────────────────────────────────
/// An in-memory dataset backed by a Vec<QaSample>.
///
/// For large datasets you could replace this with a disk-backed
/// implementation that reads samples lazily — the Dataset trait
/// makes this swap transparent to the training loop.
pub struct QaDataset {
    /// All training samples stored in memory
    samples: Vec<QaSample>,
}

impl QaDataset {
    /// Create a new dataset from a vector of samples
    pub fn new(samples: Vec<QaSample>) -> Self {
        Self { samples }
    }

    /// Returns the number of samples in this dataset
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

// ─── Burn Dataset Trait Implementation ────────────────────────────────────────
// This is what makes QaDataset work with Burn's DataLoader.
// The DataLoader will call .get(i) for random indices during training.
impl Dataset<QaSample> for QaDataset {
    /// Return the sample at the given index, or None if out of bounds.
    /// Burn's DataLoader uses this during shuffled iteration.
    fn get(&self, index: usize) -> Option<QaSample> {
        // .cloned() creates an owned copy from the borrowed reference
        self.samples.get(index).cloned()
    }

    /// Return the total number of samples.
    /// The DataLoader uses this to know when an epoch is complete.
    fn len(&self) -> usize {
        self.samples.len()
    }
}
