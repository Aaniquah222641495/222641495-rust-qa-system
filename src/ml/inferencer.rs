// ============================================================
// Layer 5 — Inferencer
// ============================================================
// Loads a trained model checkpoint and runs extractive Q&A.
//
// Inference algorithm:
//
//   1. Tokenise the input:
//      [CLS] question tokens [SEP] context tokens [SEP]
//
//   2. Run forward pass → start_logits, end_logits
//      Shape: [1, seq_len] each
//
//   3. Convert logits to probabilities via softmax:
//      start_probs = softmax(start_logits)
//      end_probs   = softmax(end_logits)
//
//   4. Find the best valid answer span:
//      score(s, e) = start_probs[s] * end_probs[e]
//      subject to: s >= context_start
//                  e >= s
//                  e - s <= MAX_ANSWER_LEN
//
//      Joint probability scoring ensures the model commits
//      to a consistent start AND end position.
//
//   5. Decode the token IDs back to text using the tokenizer
//
//   6. Return (answer_text, confidence_score)
//      If confidence < threshold → "I don't know"
//
// Reference: Devlin et al. (2019) BERT paper §4.2
//            Burn Book §6 (Inference)
//            Rust Book §6 (Enums and Pattern Matching)

use anyhow::Result;
use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use tokenizers::Tokenizer;

use crate::infra::checkpoint::CheckpointManager;
use crate::ml::model::{TransformerQaConfig, TransformerQaModel};

// Use the base WGPU backend (no autodiff needed at inference time)
// This is faster than the Autodiff wrapper since we skip gradient tracking
type InferBackend = burn::backend::Wgpu;

/// Maximum number of tokens an answer span can contain.
/// Prevents the model from returning the entire document as an answer.
const MAX_ANSWER_LEN: usize = 30;

/// The inference engine — holds the loaded model and device
pub struct Inferencer {
    /// The loaded model with trained weights
    model:       TransformerQaModel<InferBackend>,
    /// Maximum sequence length (same as used during training)
    max_seq_len: usize,
    /// The device to run inference on
    device:      burn::backend::wgpu::WgpuDevice,
}

impl Inferencer {
    /// Load the latest checkpoint and build the inferencer.
    ///
    /// Steps:
    ///   1. Load TrainConfig to know the model architecture
    ///   2. Build an empty model with those dimensions
    ///   3. Load saved weights into the model
    pub fn from_checkpoint(
        ckpt_manager: &CheckpointManager,
        _tokenizer:   &Tokenizer,
    ) -> Result<Self> {
        let device = burn::backend::wgpu::WgpuDevice::default();

        // Load the config that was saved during training
        // This tells us vocab_size, d_model, num_layers, etc.
        let cfg = ckpt_manager.load_config()?;

        // Build the model architecture (same as during training)
        // but with dropout=0.0 since we are doing inference
        let model_cfg = TransformerQaConfig::new(
            cfg.vocab_size,
            cfg.max_seq_len,
            cfg.d_model,
            cfg.num_heads,
            cfg.num_layers,
            cfg.d_ff,
            0.0, // no dropout at inference time
        );

        let model: TransformerQaModel<InferBackend> = model_cfg.init(&device);

        // Load the saved weights from the latest epoch checkpoint
        let model = ckpt_manager.load_model(model, &device)?;

        tracing::info!("Model loaded from checkpoint successfully");

        Ok(Self {
            model,
            max_seq_len: cfg.max_seq_len,
            device,
        })
    }

    /// Predict the answer span for a (question, context) pair.
    ///
    /// # Arguments
    /// * `question`  - The natural language question
    /// * `context`   - The passage to search for the answer
    /// * `tokenizer` - The same tokenizer used during training
    ///
    /// # Returns
    /// `(answer_text, confidence)` where confidence is
    /// P(best_start) * P(best_end) — a joint probability score
    pub fn predict(
        &self,
        question:  &str,
        context:   &str,
        tokenizer: &Tokenizer,
    ) -> Result<(String, f32)> {

        // ── Step 1: Tokenise Input ────────────────────────────────────────────
        // Special token IDs (standard BERT vocabulary)
        let cls_id = 101u32; // [CLS] — start of sequence
        let sep_id = 102u32; // [SEP] — separator

        // Tokenise the question
        let q_enc = tokenizer
            .encode(question, false)
            .map_err(|e| anyhow::anyhow!("Question tokenisation error: {e}"))?;

        // Tokenise the context
        let c_enc = tokenizer
            .encode(context, false)
            .map_err(|e| anyhow::anyhow!("Context tokenisation error: {e}"))?;

        // Build the full input sequence:
        // [CLS] q_1 ... q_n [SEP] c_1 ... c_m [SEP]
        let mut input_ids: Vec<u32> = vec![cls_id];
        input_ids.extend_from_slice(q_enc.get_ids());
        input_ids.push(sep_id);

        // Record where the context starts in the sequence
        // The answer MUST start at or after this position
        let context_start = input_ids.len();

        input_ids.extend_from_slice(c_enc.get_ids());
        input_ids.push(sep_id);

        // Truncate if the combined sequence is too long
        input_ids.truncate(self.max_seq_len);
        let seq_len = input_ids.len(); // actual length before padding

        // Pad to max_seq_len with zeros (0 = [PAD] token)
        while input_ids.len() < self.max_seq_len {
            input_ids.push(0);
        }

        // ── Step 2: Build Input Tensor ────────────────────────────────────────
        // Convert Vec<u32> to a Burn Int tensor of shape [1, max_seq_len]
        // (batch_size=1 for single-sample inference)
        let input_flat: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
        let input_tensor = Tensor::<InferBackend, 1, Int>::from_ints(
            input_flat.as_slice(),
            &self.device,
        )
        .unsqueeze::<2>(); // Add batch dimension → [1, max_seq_len]

        // ── Step 3: Forward Pass ──────────────────────────────────────────────
        // Run the model to get start and end logits
        // No gradient tracking needed — use base backend
        let output = self.model.forward(input_tensor);

        // Remove the batch dimension → [max_seq_len]
        let start_logits = output.start_logits.squeeze::<1>(0);
        let end_logits   = output.end_logits.squeeze::<1>(0);

        // Only consider the actual (non-padded) portion of the sequence
        let start_logits = start_logits.slice([0..seq_len]);
        let end_logits   = end_logits.slice([0..seq_len]);

        // ── Step 4: Convert Logits to Probabilities ───────────────────────────
        // softmax turns raw logit scores into a probability distribution
        // that sums to 1.0 over all positions
        let start_probs: Vec<f32> = burn::tensor::activation::softmax(
            start_logits.unsqueeze::<2>(), 1,
        )
        .squeeze::<1>(0)
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default();

        let end_probs: Vec<f32> = burn::tensor::activation::softmax(
            end_logits.unsqueeze::<2>(), 1,
        )
        .squeeze::<1>(0)
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default();

        // ── Step 5: Find Best Valid Answer Span ───────────────────────────────
        // Score every (start, end) pair and find the highest scoring
        // valid span. A valid span must:
        //   - Start within the context (not in the question)
        //   - Have end >= start
        //   - Have length <= MAX_ANSWER_LEN tokens
        //
        // score(s, e) = P(start=s) * P(end=e)
        // Joint probability ensures consistent start AND end selection.
        let mut best_score = f32::NEG_INFINITY;
        let mut best_start = context_start;
        let mut best_end   = context_start;

        for s in context_start..seq_len {
            for e in s..(s + MAX_ANSWER_LEN).min(seq_len) {
                let score = start_probs[s] * end_probs[e];
                if score > best_score {
                    best_score = score;
                    best_start = s;
                    best_end   = e;
                }
            }
        }

        let confidence = best_score;

        // ── Step 6: Decode Answer Tokens Back to Text ─────────────────────────
        // Convert the token ID span back to a human-readable string.
        // skip_special_tokens=true removes [CLS], [SEP], [PAD] etc.
        let answer_ids: Vec<u32> = input_ids[best_start..=best_end].to_vec();
        let answer = tokenizer
            .decode(&answer_ids, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {e}"))?;

        tracing::debug!(
            "Predicted span: [{}, {}], confidence: {:.4}, answer: '{}'",
            best_start,
            best_end,
            confidence,
            answer
        );

        Ok((answer, confidence))
    }
}
