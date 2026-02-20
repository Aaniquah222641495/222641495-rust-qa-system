// ============================================================
// Layer 2 — TrainUseCase
// ============================================================
// Orchestrates the full training pipeline in order:
//
//   Step 1: Load .docx files          (Layer 4 - data)
//   Step 2: Clean the text            (Layer 4 - data)
//   Step 3: Chunk into passages       (Layer 4 - data)
//   Step 4: Build tokenizer           (Layer 6 - infra)
//   Step 5: Create training samples   (Layer 4 - data)
//   Step 6: Split train/validation    (Layer 4 - data)
//   Step 7: Build datasets            (Layer 4 - data)
//   Step 8: Save config               (Layer 6 - infra)
//   Step 9: Run training loop         (Layer 5 - ml)
//
// Reference: Rust Book §13 (Iterators and Closures)
//            Burn Book §5 (Training)

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::data::{
    loader::DocxLoader,
    preprocessor::Preprocessor,
    chunker::Chunker,
    dataset::QaSample,
    dataset::QaDataset,
    splitter::split_train_val,
};
use crate::ml::trainer::run_training;
use crate::infra::{
    tokenizer_store::TokenizerStore,
    checkpoint::CheckpointManager,
};

// ─── Training Configuration ──────────────────────────────────────────────────
// All hyperparameters for a training run.
// Serialisable so it can be saved to disk and reloaded for inference.
// The #[derive(Serialize, Deserialize)] macros from serde handle
// reading/writing this struct to JSON automatically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub docs_dir:       String,
    pub checkpoint_dir: String,
    pub max_seq_len:    usize,
    pub batch_size:     usize,
    pub epochs:         usize,
    pub lr:             f64,
    pub d_model:        usize,
    pub num_heads:      usize,
    pub num_layers:     usize,
    pub d_ff:           usize,
    pub dropout:        f64,
    pub vocab_size:     usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            docs_dir:       "data/docx_files".to_string(),
            checkpoint_dir: "checkpoints".to_string(),
            max_seq_len:    512,
            batch_size:     8,
            epochs:         10,
            lr:             2e-4,
            d_model:        256,
            num_heads:      8,
            num_layers:     6,
            d_ff:           1024,
            dropout:        0.1,
            vocab_size:     30522,
        }
    }
}

// ─── TrainUseCase ─────────────────────────────────────────────────────────────
// Owns the config and runs the full training pipeline.
pub struct TrainUseCase {
    config: TrainConfig,
}

impl TrainUseCase {
    /// Create a new TrainUseCase with the given configuration
    pub fn new(config: TrainConfig) -> Self {
        Self { config }
    }

    /// Execute the full training pipeline end to end
    pub fn execute(&self) -> Result<()> {
        let cfg = &self.config;

        // ── Step 1: Load all .docx documents ─────────────────────────────────
        // DocxLoader walks the directory and parses each .docx file
        tracing::info!("Loading .docx files from '{}'", cfg.docs_dir);
        let loader   = DocxLoader::new(&cfg.docs_dir);
        let raw_docs = loader.load_all()?;
        tracing::info!("Loaded {} documents", raw_docs.len());

        // ── Step 2: Clean / normalise text ────────────────────────────────────
        // Removes extra whitespace, control characters, etc.
        let preprocessor = Preprocessor::new();
        let clean_docs: Vec<String> = raw_docs
            .iter()
            .map(|d| preprocessor.clean(&d.text))
            .collect();

        // ── Step 3: Chunk documents into context windows ──────────────────────
        // Long documents are split into overlapping passages.
        // chunk_size = half of max_seq_len, overlap = 50 words
        // This ensures answer spans are never cut off at boundaries.
        let chunker = Chunker::new(cfg.max_seq_len / 2, 50);
        let chunks: Vec<String> = clean_docs
            .iter()
            .flat_map(|doc| chunker.chunk(doc))
            .collect();
        tracing::info!("Created {} context chunks", chunks.len());

        // ── Step 4: Build / load tokenizer ────────────────────────────────────
        // If a tokenizer was already trained and saved, load it.
        // Otherwise train a new BPE tokenizer on the document corpus.
        let tok_store  = TokenizerStore::new(&cfg.checkpoint_dir);
        let tokenizer  = tok_store.load_or_build(&chunks, cfg.vocab_size)?;

        // ── Step 5: Build training samples ────────────────────────────────────
        // Creates (input_ids, attention_mask, start_pos, end_pos) tuples
        // from the document chunks.
        let samples = build_synthetic_samples(&chunks, &tokenizer, cfg)?;
        tracing::info!("Built {} training samples", samples.len());

        // ── Step 6: Train / validation split (80/20) ──────────────────────────
        // Shuffle and split so the model is evaluated on unseen data
        let (train_samples, val_samples) = split_train_val(samples, 0.8);
        tracing::info!(
            "Split: {} train, {} validation",
            train_samples.len(),
            val_samples.len()
        );

        // ── Step 7: Build Burn datasets ───────────────────────────────────────
        // QaDataset implements Burn's Dataset trait so the DataLoader
        // can call .get(index) and .len() on it
        let train_dataset = QaDataset::new(train_samples);
        let val_dataset   = QaDataset::new(val_samples);

        // ── Step 8: Save config for inference ─────────────────────────────────
        // The inferencer needs to know the model architecture to rebuild it
        let ckpt_manager = CheckpointManager::new(&cfg.checkpoint_dir);
        ckpt_manager.save_config(cfg)?;

        // ── Step 9: Run training loop (Layer 5) ───────────────────────────────
        run_training(cfg, train_dataset, val_dataset, ckpt_manager)?;

        Ok(())
    }
}

// ─── Synthetic Sample Generation ─────────────────────────────────────────────
// In production you would load a labelled SQuAD-format dataset.
// Here we generate synthetic (question, context, start, end) triples
// from document chunks to demonstrate the full pipeline end-to-end.
// Each chunk becomes a context, and a random span within it becomes
// the "answer" with a placeholder question.
fn build_synthetic_samples(
    chunks:    &[String],
    tokenizer: &tokenizers::Tokenizer,
    cfg:       &TrainConfig,
) -> Result<Vec<QaSample>> {
    use rand::Rng;
    let mut rng     = rand::thread_rng();
    let mut samples = Vec::new();

    for chunk in chunks {
        // Tokenise the context passage
        let enc = tokenizer
            .encode(chunk.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenisation error: {e}"))?;
        let context_ids: Vec<u32> = enc.get_ids().to_vec();

        // Skip chunks that are too short to form a meaningful sample
        if context_ids.len() < 10 {
            continue;
        }

        // Pick a random answer span within the context
        let max_start = context_ids.len().saturating_sub(5);
        let start     = rng.gen_range(0..max_start);
        let end       = (start + rng.gen_range(1..5)).min(context_ids.len() - 1);

        // Create a placeholder question from the span positions
        let question = format!("What is mentioned at position {} to {}?", start, end);
        let q_enc    = tokenizer
            .encode(question.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenisation error: {e}"))?;
        let question_ids: Vec<u32> = q_enc.get_ids().to_vec();

        // Build the full input sequence:
        // [CLS] question tokens [SEP] context tokens [SEP]
        // This is the standard BERT input format for extractive Q&A
        let cls_id = 101u32; // [CLS] token id
        let sep_id = 102u32; // [SEP] token id

        let mut input_ids = vec![cls_id];
        input_ids.extend_from_slice(&question_ids);
        input_ids.push(sep_id);

        // Record where context starts (used to offset the answer span)
        let context_offset = input_ids.len();
        input_ids.extend_from_slice(&context_ids);
        input_ids.push(sep_id);

        // Truncate to maximum allowed length
        input_ids.truncate(cfg.max_seq_len);

        // Convert relative span positions to absolute positions
        // in the full [CLS]+Q+[SEP]+C+[SEP] sequence
        let abs_start = (context_offset + start).min(cfg.max_seq_len - 1);
        let abs_end   = (context_offset + end  ).min(cfg.max_seq_len - 1);

        // Attention mask: 1 for real tokens, 0 for padding
        let seq_len         = input_ids.len();
        let mut attn_mask   = vec![1u32; seq_len];

        // Pad both input_ids and attention_mask to max_seq_len
        while input_ids.len() < cfg.max_seq_len {
            input_ids.push(0);  // 0 = [PAD] token
            attn_mask.push(0);  // 0 = ignore this position
        }

        samples.push(QaSample {
            input_ids:      input_ids,
            attention_mask: attn_mask,
            start_position: abs_start,
            end_position:   abs_end,
        });
    }

    Ok(samples)
}
