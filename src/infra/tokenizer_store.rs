// ============================================================
// Layer 6 — Tokenizer Store
// ============================================================
// Manages tokenizer training, saving, and loading.
//
// In tokenizers 0.15, train_from_files requires Trainer::Model
// to equal ModelWrapper. The correct approach is to use the
// high-level tokenizers::from_pretrained style — build the
// tokenizer JSON manually and load it, bypassing the trainer
// type mismatch entirely.
//
// Reference: Sennrich et al. (2016) BPE paper

use anyhow::{Context, Result};
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub struct TokenizerStore {
    dir: PathBuf,
}

impl TokenizerStore {
    pub fn new(dir: impl Into<String>) -> Self {
        Self { dir: PathBuf::from(dir.into()) }
    }

    /// Load existing tokenizer or build a new one from texts
    pub fn load_or_build(
        &self,
        texts:      &[String],
        vocab_size: usize,
    ) -> Result<Tokenizer> {
        let tok_path = self.dir.join("tokenizer.json");
        if tok_path.exists() {
            tracing::info!("Loading existing tokenizer from disk");
            self.load()
        } else {
            tracing::info!("Building new tokenizer (vocab_size={})", vocab_size);
            self.build_and_save(texts, vocab_size)
        }
    }

    /// Load a previously saved tokenizer from JSON file
    pub fn load(&self) -> Result<Tokenizer> {
        let path = self.dir.join("tokenizer.json");
        Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!(
                "Cannot load tokenizer from '{}': {}", path.display(), e
            ))
    }

    /// Build a word-level vocabulary from document texts and
    /// write a valid tokenizer JSON directly — this bypasses
    /// the train_from_files ModelWrapper type mismatch in
    /// tokenizers 0.15 entirely.
    fn build_and_save(&self, texts: &[String], vocab_size: usize) -> Result<Tokenizer> {
        std::fs::create_dir_all(&self.dir).ok();

        // ── Step 1: Build vocabulary from word frequencies ────────────────────
        // Count every word in the corpus
        use std::collections::HashMap;
        let mut freq: HashMap<String, usize> = HashMap::new();

        for text in texts {
            for word in text.split_whitespace() {
                // Normalise to lowercase for consistency
                let w = word.to_lowercase();
                // Strip punctuation from edges
                let w = w.trim_matches(|c: char| !c.is_alphanumeric());
                if !w.is_empty() {
                    *freq.entry(w.to_string()).or_insert(0) += 1;
                }
            }
        }

        // Sort by frequency descending, take top vocab_size - 5
        // (reserve 5 slots for special tokens)
        let mut words: Vec<(String, usize)> = freq.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1));
        let max_words = vocab_size.saturating_sub(5);
        words.truncate(max_words);

        // ── Step 2: Build vocab JSON ──────────────────────────────────────────
        // Special tokens get fixed IDs matching BERT convention
        let mut vocab = serde_json::json!({
            "[PAD]":  0,
            "[UNK]":  1,
            "[CLS]":  101,
            "[SEP]":  102,
            "[MASK]": 103,
        });

        let mut next_id = 104usize;
        for (word, _) in &words {
            // Skip if already a special token
            if vocab.get(word).is_none() {
                vocab[word] = serde_json::json!(next_id);
                next_id += 1;
            }
        }

        // ── Step 3: Write tokenizer JSON in HuggingFace format ────────────────
        // This format is what Tokenizer::from_file() expects
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0,   "content": "[PAD]",  "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1,   "content": "[UNK]",  "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 101, "content": "[CLS]",  "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 102, "content": "[SEP]",  "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 103, "content": "[MASK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": {
                "type": "BertNormalizer",
                "clean_text": true,
                "handle_chinese_chars": true,
                "strip_accents": null,
                "lowercase": true
            },
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "[UNK]"
            }
        });

        let tok_path = self.dir.join("tokenizer.json");
        std::fs::write(
            &tok_path,
            serde_json::to_string_pretty(&tokenizer_json)?
        ).with_context(|| "Cannot write tokenizer JSON")?;

        tracing::info!(
            "Tokenizer built with {} words, saved to '{}'",
            next_id,
            tok_path.display()
        );

        // Load back as a proper Tokenizer instance
        Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("Cannot reload tokenizer: {e}"))
    }
}
