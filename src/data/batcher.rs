// ============================================================
// Layer 4 — Q&A Batcher
// ============================================================
// Implements Burn's Batcher trait to convert a Vec<QaSample>
// into GPU-ready tensors.
//
// What is a Batcher?
//   A Batcher takes a list of individual samples and stacks
//   them into a single batch tensor. This is necessary because
//   GPUs are most efficient when processing many samples at once.
//
// How batching works here:
//   Input:  Vec of N QaSamples, each with sequences of length S
//   Output: QaBatch with tensors of shape [N, S]
//
//   We flatten all input_ids into one long Vec, then reshape:
//   [s1_t1, s1_t2, ..., s1_tS, s2_t1, ..., sN_tS] → [N, S]
//
// Why is this easy here?
//   Because all sequences are already padded to the same length
//   in QaSample. If they weren't, we'd need dynamic padding here.
//
// Reference: Burn Book §4 (Batcher)
//            Rust Book §8 (Vectors)

use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
};

use crate::data::dataset::QaSample;

// ─── QaBatch ──────────────────────────────────────────────────────────────────
/// A batch of Q&A samples ready for the model forward pass.
/// All tensors have batch_size as their first dimension.
///
/// B is the Burn Backend (e.g. Wgpu, NdArray) —
/// generic so the same batcher works on any device.
#[derive(Debug, Clone)]
pub struct QaBatch<B: Backend> {
    /// Token ID sequences — shape: [batch_size, seq_len]
    /// Each row is one sample's input_ids
    pub input_ids: Tensor<B, 2, Int>,

    /// Attention masks — shape: [batch_size, seq_len]
    /// 1 = real token, 0 = padding
    pub attention_mask: Tensor<B, 2, Int>,

    /// Ground truth start positions — shape: [batch_size]
    /// One integer per sample indicating where the answer starts
    pub start_positions: Tensor<B, 1, Int>,

    /// Ground truth end positions — shape: [batch_size]
    /// One integer per sample indicating where the answer ends
    pub end_positions: Tensor<B, 1, Int>,
}

// ─── QaBatcher ────────────────────────────────────────────────────────────────
/// The batcher struct — holds the target device so tensors
/// are created on the correct GPU/CPU.
#[derive(Clone, Debug)]
pub struct QaBatcher<B: Backend> {
    /// The device to create tensors on (e.g. GPU index 0)
    pub device: B::Device,
}

impl<B: Backend> QaBatcher<B> {
    /// Create a new batcher for the given device
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

// ─── Burn Batcher Trait Implementation ────────────────────────────────────────
// This is what makes QaBatcher work with Burn's DataLoader.
// The DataLoader calls .batch(items) with each mini-batch of samples.
impl<B: Backend> Batcher<QaSample, QaBatch<B>> for QaBatcher<B> {
    /// Convert a Vec of QaSamples into a single QaBatch.
    ///
    /// Steps:
    ///   1. Flatten all input_ids into one Vec<i32>
    ///   2. Create a 1D tensor from the flat Vec
    ///   3. Reshape to [batch_size, seq_len]
    ///   4. Repeat for attention_mask
    ///   5. Create 1D tensors for start/end positions
    fn batch(&self, items: Vec<QaSample>) -> QaBatch<B> {
        let batch_size = items.len();
        // All sequences have the same length (pre-padded)
        let seq_len    = items[0].input_ids.len();

        // ── Flatten input_ids ─────────────────────────────────────────────────
        // We go from Vec<Vec<u32>> to Vec<i32> (Burn uses i32 for Int tensors)
        // by iterating over samples and their tokens in order
        let input_flat: Vec<i32> = items
            .iter()
            .flat_map(|s| s.input_ids.iter().map(|&x| x as i32))
            .collect();

        // ── Flatten attention_mask ────────────────────────────────────────────
        let mask_flat: Vec<i32> = items
            .iter()
            .flat_map(|s| s.attention_mask.iter().map(|&x| x as i32))
            .collect();

        // ── Collect start and end positions ───────────────────────────────────
        // These are scalar values per sample, not sequences
        let starts: Vec<i32> = items
            .iter()
            .map(|s| s.start_position as i32)
            .collect();

        let ends: Vec<i32> = items
            .iter()
            .map(|s| s.end_position as i32)
            .collect();

        // ── Create tensors ────────────────────────────────────────────────────
        // Tensor::from_ints creates a 1D tensor from a slice,
        // then .reshape() gives it the correct 2D shape [batch, seq]

        let input_ids = Tensor::<B, 1, Int>::from_ints(
            input_flat.as_slice(), &self.device
        ).reshape([batch_size, seq_len]);

        let attention_mask = Tensor::<B, 1, Int>::from_ints(
            mask_flat.as_slice(), &self.device
        ).reshape([batch_size, seq_len]);

        // Start and end positions stay as 1D tensors [batch_size]
        let start_positions = Tensor::<B, 1, Int>::from_ints(
            starts.as_slice(), &self.device
        );

        let end_positions = Tensor::<B, 1, Int>::from_ints(
            ends.as_slice(), &self.device
        );

        QaBatch {
            input_ids,
            attention_mask,
            start_positions,
            end_positions,
        }
    }
}
