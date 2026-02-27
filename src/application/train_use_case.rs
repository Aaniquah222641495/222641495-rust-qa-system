// ============================================================
// Layer 2 — Train Use Case
// ============================================================
use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::data::{
    chunker::Chunker,
    dataset::{QaDataset, QaSample},
    loader::DocxLoader,
    splitter::split_train_val,
};
use crate::domain::traits::DocumentSource;
use crate::infra::{checkpoint::CheckpointManager, tokenizer_store::TokenizerStore};
use crate::ml::trainer::run_training;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub vocab_size:     usize,
    pub max_seq_len:    usize,
    pub d_model:        usize,
    pub num_heads:      usize,
    pub num_layers:     usize,
    pub d_ff:           usize,
    pub dropout:        f64,
    pub batch_size:     usize,
    pub epochs:         usize,
    pub lr:             f64,
    pub docs_dir:       String,
    pub checkpoint_dir: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522, max_seq_len: 512, d_model: 256,
            num_heads: 8, num_layers: 6, d_ff: 1024, dropout: 0.1,
            batch_size: 8, epochs: 10, lr: 2e-4,
            docs_dir: "data/docx_files".to_string(),
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}

pub struct TrainUseCase { cfg: TrainConfig }

impl TrainUseCase {
    pub fn new(cfg: TrainConfig) -> Self { Self { cfg } }

    pub fn execute(self) -> Result<()> {
        let cfg = &self.cfg;
        tracing::info!("Loading .docx files from '{}'", cfg.docs_dir);
        let loader = DocxLoader::new(&cfg.docs_dir);
        let docs   = loader.load_all()?;
        tracing::info!("Loaded {} documents", docs.len());

        let chunker = Chunker::new(cfg.max_seq_len / 2, 50);
        let chunks: Vec<String> = docs.iter().flat_map(|d| chunker.chunk(&d.text)).collect();
        tracing::info!("Created {} context chunks", chunks.len());

        let all_texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
        let tok_store = TokenizerStore::new(&cfg.checkpoint_dir);
        let tokenizer = tok_store.load_or_build(&all_texts, cfg.vocab_size)?;

        let real_pairs = real_qa_pairs();
        let mut samples: Vec<QaSample> = Vec::new();

        for chunk in &chunks {
            let chunk_lower = chunk.to_lowercase();
            for (question, answer) in &real_pairs {
                let answer_lower = answer.to_lowercase();
                if let Some(char_pos) = chunk_lower.find(&answer_lower) {
                    let context_before = &chunk[..char_pos];
                    let answer_text    = &chunk[char_pos..char_pos + answer.len()];
                    let input = format!("[CLS] {} [SEP] {} [SEP]", question, chunk);
                    if let Ok(encoding) = tokenizer.encode(input.as_str(), false) {
                        let ids     = encoding.get_ids();
                        let seq_len = ids.len().min(cfg.max_seq_len);
                        let q_input   = format!("[CLS] {} [SEP]", question);
                        let q_enc     = tokenizer.encode(q_input.as_str(), false).unwrap();
                        let ctx_start = q_enc.get_ids().len();
                        let before_enc = tokenizer.encode(context_before, false).unwrap();
                        let start_pos  = (ctx_start + before_enc.get_ids().len()).min(seq_len.saturating_sub(1));
                        let ans_enc = tokenizer.encode(answer_text, false).unwrap();
                        let end_pos = (start_pos + ans_enc.get_ids().len().max(1) - 1).min(seq_len.saturating_sub(1));
                        let mut input_ids: Vec<u32> = ids[..seq_len].iter().map(|&x| x).collect();
                        let mut attention: Vec<u32> = vec![1u32; seq_len];
                        while input_ids.len() < cfg.max_seq_len { input_ids.push(0); attention.push(0); }
                        samples.push(QaSample { input_ids, attention_mask: attention, start_position: start_pos, end_position: end_pos });
                    }
                }
            }
        }

        if samples.is_empty() {
            tracing::warn!("No real Q&A pairs matched — using synthetic fallback");
            samples = build_synthetic_samples(&chunks, &tokenizer, cfg);
        }

        tracing::info!("Built {} training samples", samples.len());
        let (train_samples, val_samples) = split_train_val(samples, 0.8);
        tracing::info!("Split: {} train, {} validation", train_samples.len(), val_samples.len());

        let train_dataset = QaDataset::new(train_samples);
        let val_dataset   = QaDataset::new(val_samples);
        let ckpt_manager  = CheckpointManager::new(&cfg.checkpoint_dir);
        ckpt_manager.save_config(cfg)?;
        run_training(cfg, train_dataset, val_dataset, ckpt_manager)
    }
}

fn real_qa_pairs() -> Vec<(String, String)> {
    vec![
        ("When is the Senate meeting in November 2026?".to_string(), "Senate".to_string()),
        ("When does the academic year start in 2026?".to_string(), "January".to_string()),
        ("When is the Autumn Graduation in 2026?".to_string(), "April".to_string()),
        ("When does Term 1 start in 2026?".to_string(), "START OF TERM".to_string()),
        ("When does Term 2 start in 2025?".to_string(), "START OF TERM".to_string()),
        ("When does Term 1 start in 2025?".to_string(), "January".to_string()),
        ("When is Workers Day in 2025?".to_string(), "WORKERS DAY".to_string()),
        ("When is Heritage Day in 2025?".to_string(), "HERITAGE DAY".to_string()),
        ("When does Term 2 start in 2024?".to_string(), "START OF TERM".to_string()),
        ("When is Workers Day in 2024?".to_string(), "WORKERS DAY".to_string()),
        ("When is Heritage Day in 2024?".to_string(), "HERITAGE DAY".to_string()),
        ("When is Good Friday in 2024?".to_string(), "GOOD FRIDAY".to_string()),
        ("When is Day of Reconciliation?".to_string(), "DAY OF RECONCILIATION".to_string()),
        ("When does Quality Month start?".to_string(), "Quality Month".to_string()),
        ("When is the Higher Degrees Committee meeting?".to_string(), "Higher Degrees Committee".to_string()),
        ("When is the Management Committee meeting?".to_string(), "Management Committee".to_string()),
    ]
}

fn build_synthetic_samples(chunks: &[String], tokenizer: &tokenizers::Tokenizer, cfg: &TrainConfig) -> Vec<QaSample> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut samples = Vec::new();
    for chunk in chunks {
        let input = format!("[CLS] question [SEP] {} [SEP]", chunk);
        if let Ok(enc) = tokenizer.encode(input.as_str(), false) {
            let ids = enc.get_ids();
            let seq_len = ids.len().min(cfg.max_seq_len);
            if seq_len < 4 { continue; }
            let start = rng.gen_range(1..seq_len.saturating_sub(1));
            let end   = (start + rng.gen_range(1..5)).min(seq_len - 1);
            let mut input_ids: Vec<u32> = ids[..seq_len].iter().map(|&x| x).collect();
            let mut attention: Vec<u32> = vec![1u32; seq_len];
            while input_ids.len() < cfg.max_seq_len { input_ids.push(0); attention.push(0); }
            samples.push(QaSample { input_ids, attention_mask: attention, start_position: start, end_position: end });
        }
    }
    samples
}
