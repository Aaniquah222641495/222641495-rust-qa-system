// ============================================================
// Layer 5 — Inferencer
// ============================================================
use anyhow::Result;
use burn::prelude::*;
use tokenizers::Tokenizer;

use crate::infra::checkpoint::CheckpointManager;
use crate::ml::model::{TransformerQaConfig, TransformerQaModel};

type InferBackend = burn::backend::Wgpu;

const MAX_ANSWER_LEN: usize = 30;
// Return a window of tokens around the predicted span for richer answers
const CONTEXT_WINDOW: usize = 5;

pub struct Inferencer {
    model:       TransformerQaModel<InferBackend>,
    max_seq_len: usize,
    device:      burn::backend::wgpu::WgpuDevice,
}

impl Inferencer {
    pub fn from_checkpoint(
        ckpt_manager: &CheckpointManager,
        _tokenizer:   &Tokenizer,
    ) -> Result<Self> {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let cfg    = ckpt_manager.load_config()?;
        let model_cfg = TransformerQaConfig::new(
            cfg.vocab_size, cfg.max_seq_len, cfg.d_model,
            cfg.num_heads, cfg.num_layers, cfg.d_ff, 0.0,
        );
        let model: TransformerQaModel<InferBackend> = model_cfg.init(&device);
        let model = ckpt_manager.load_model(model, &device)?;
        tracing::info!("Model loaded from checkpoint");
        Ok(Self { model, max_seq_len: cfg.max_seq_len, device })
    }

    pub fn predict(
        &self,
        question:  &str,
        context:   &str,
        tokenizer: &Tokenizer,
    ) -> Result<(String, f32)> {
        let cls_id = 101u32;
        let sep_id = 102u32;

        // Build [CLS] question [SEP] context [SEP]
        let q_enc = tokenizer.encode(question, false)
            .map_err(|e| anyhow::anyhow!("Q tokenise: {e}"))?;
        let c_enc = tokenizer.encode(context, false)
            .map_err(|e| anyhow::anyhow!("C tokenise: {e}"))?;

        let mut input_ids: Vec<u32> = vec![cls_id];
        input_ids.extend_from_slice(q_enc.get_ids());
        input_ids.push(sep_id);
        let context_start = input_ids.len();
        input_ids.extend_from_slice(c_enc.get_ids());
        input_ids.push(sep_id);
        input_ids.truncate(self.max_seq_len);
        let seq_len = input_ids.len();
        while input_ids.len() < self.max_seq_len { input_ids.push(0); }

        // Forward pass
        let input_flat: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
        let input_tensor = Tensor::<InferBackend, 1, Int>::from_ints(
            input_flat.as_slice(), &self.device,
        ).unsqueeze::<2>();

        let output      = self.model.forward(input_tensor);
        let start_logits = output.start_logits.squeeze::<1>();
        let end_logits   = output.end_logits.squeeze::<1>();
        let start_logits = start_logits.slice([0..seq_len]);
        let end_logits   = end_logits.slice([0..seq_len]);

        // Softmax probabilities
        let start_probs: Vec<f32> = burn::tensor::activation::softmax(
            start_logits.unsqueeze::<2>(), 1,
        ).squeeze::<1>().into_data().to_vec::<f32>().unwrap_or_default();

        let end_probs: Vec<f32> = burn::tensor::activation::softmax(
            end_logits.unsqueeze::<2>(), 1,
        ).squeeze::<1>().into_data().to_vec::<f32>().unwrap_or_default();

        // Find best valid span
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

        // Expand window slightly to capture full phrases
        // e.g. "start" → "START OF TERM 2"
        let window_start = best_start.saturating_sub(1);
        let window_end   = (best_end + CONTEXT_WINDOW).min(seq_len - 1);

        let answer_ids: Vec<u32> = input_ids[window_start..=window_end].to_vec();
        let answer = tokenizer.decode(&answer_ids, true)
            .map_err(|e| anyhow::anyhow!("Decode: {e}"))?;

        // Clean up the answer — remove special tokens and extra whitespace
        let answer = answer
            .replace("[CLS]", "")
            .replace("[SEP]", "")
            .replace("[PAD]", "")
            .trim()
            .to_string();

        tracing::debug!("Span [{},{}] conf={:.4} answer='{}'",
            best_start, best_end, best_score, answer);

        Ok((answer, best_score))
    }
}
