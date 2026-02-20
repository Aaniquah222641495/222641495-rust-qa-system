// ============================================================
// Layer 6 — Tokenizer Store
// ============================================================
// Manages training, saving, and loading of the BPE tokenizer.
//
// What is a tokenizer?
//   A tokenizer converts raw text into token IDs (numbers)
//   that the model can process. For example:
//   "graduation ceremony" → [4521, 892]
//
// What is BPE (Byte Pair Encoding)?
//   BPE is a subword tokenization algorithm:
//   1. Start with individual characters as tokens
//   2. Repeatedly merge the most frequent adjacent pairs
//   3. Stop when vocabulary size is reached
//
//   This handles unknown words by breaking them into
//   known subword pieces:
//   "CPUT" → ["CP", "UT"] or ["C", "PUT"] etc.
//
// Why train our own tokenizer?
//   A domain-specific tokenizer on our documents means
//   the vocabulary is tailored to our content.
//   Words like "graduation", "HDC", "ceremony" get their
//   own tokens instead of being split into subwords.
//
// Saved files:
//   checkpoints/tokenizer.json — full tokenizer config + vocab
//
// Reference: HuggingFace tokenizers crate documentation
//            Sennrich et al. (2016) BPE paper
//            Rust Book §9 (Error Handling)

use anyhow::{Context, Result};
use std::path::PathBuf;
use tokenizers::{
    models::bpe::{BpeTrainerBuilder, BPE},
    normalizers::BertNormalizer,
    pre_tokenizers::whitespace::Whitespace,
    AddedToken, Tokenizer, TokenizerBuilder,
};

/// Manages tokenizer persistence — load existing or train new.
pub struct TokenizerStore {
    /// Directory where tokenizer.json is saved
    dir: PathBuf,
}

impl TokenizerStore {
    /// Create a new TokenizerStore pointing at the given directory
    pub fn new(dir: impl Into<String>) -> Self {
        Self {
            dir: PathBuf::from(dir.into()),
        }
    }

    /// Load an existing tokenizer or train a new one from texts.
    ///
    /// If tokenizer.json exists in the checkpoint directory,
    /// load and return it (fast path).
    ///
    /// Otherwise train a new BPE tokenizer on the provided texts,
    /// save it, and return it (slow path — only happens once).
    pub fn load_or_build(
        &self,
        texts:      &[String],
        vocab_size: usize,
    ) -> Result<Tokenizer> {
        let tok_path = self.dir.join("tokenizer.json");

        if tok_path.exists() {
            tracing::info!(
                "Loading existing tokenizer from '{}'",
                tok_path.display()
            );
            self.load()
        } else {
            tracing::info!(
                "Training new BPE tokenizer on {} chunks (vocab_size={})",
                texts.len(),
                vocab_size
            );
            self.build_and_save(texts, vocab_size)
        }
    }

    /// Load a previously saved tokenizer from disk.
    /// Called by the inferencer to ensure the same vocabulary
    /// is used at inference time as during training.
    pub fn load(&self) -> Result<Tokenizer> {
        let path = self.dir.join("tokenizer.json");

        Tokenizer::from_file(&path)
            .map_err(|e| {
                anyhow::anyhow!(
                    "Cannot load tokenizer from '{}': {}. \
                     Have you run 'train' first?",
                    path.display(),
                    e
                )
            })
    }

    /// Train a new BPE tokenizer on the document corpus and save it.
    ///
    /// Training steps:
    ///   1. Write corpus to a temporary file
    ///   2. Configure BPE trainer with special tokens
    ///   3. Train on the corpus file
    ///   4. Save tokenizer.json
    ///   5. Clean up temporary file
    fn build_and_save(
        &self,
        texts:      &[String],
        vocab_size: usize,
    ) -> Result<Tokenizer> {
        // Create the checkpoint directory if needed
        std::fs::create_dir_all(&self.dir).ok();

        // Write all document chunks to a temporary file.
        // The tokenizers trainer requires file paths, not in-memory strings.
        let tmp_file = self.dir.join("_corpus_tmp.txt");
        std::fs::write(&tmp_file, texts.join("\n"))
            .with_context(|| "Cannot write temporary corpus file")?;

        tracing::info!(
            "Written corpus to temp file: {} chars total",
            texts.iter().map(|t| t.len()).sum::<usize>()
        );

        // Configure the BPE trainer with special tokens.
        // Special tokens are added to the vocabulary with fixed IDs:
        //   [PAD]  = padding token (ID 0)
        //   [UNK]  = unknown token (ID 1)
        //   [CLS]  = classification token, start of sequence (ID 101 by convention)
        //   [SEP]  = separator token between question and context (ID 102)
        //   [MASK] = mask token for masked language modelling (ID 103)
        let mut trainer = BpeTrainerBuilder::new()
            .vocab_size(vocab_size)
            .special_tokens(vec![
                AddedToken::from("[PAD]",  true), // padding
                AddedToken::from("[UNK]",  true), // unknown
                AddedToken::from("[CLS]",  true), // start of sequence
                AddedToken::from("[SEP]",  true), // separator
                AddedToken::from("[MASK]", true), // mask
            ])
            .build();

        // Build the tokenizer with:
        //   - BPE model (the core algorithm)
        //   - BertNormalizer (lowercase + strip accents)
        //   - Whitespace pre-tokenizer (split on spaces first)
        let mut tokenizer = TokenizerBuilder::new()
            .with_model(BPE::default())
            .with_normalizer(Some(BertNormalizer::default()))
            .with_pre_tokenizer(Some(Whitespace::default()))
            .build()
            .map_err(|e| anyhow::anyhow!("TokenizerBuilder error: {e}"))?;

        // Train the tokenizer on our corpus file.
        // This runs the BPE merge algorithm to build the vocabulary.
        tokenizer
            .train_from_files(&mut trainer, &[tmp_file.to_str().unwrap()])
            .map_err(|e| anyhow::anyhow!("Tokenizer training error: {e}"))?;

        // Save the trained tokenizer to JSON for later use
        let tok_path = self.dir.join("tokenizer.json");
        tokenizer
            .save(&tok_path, false)
            .map_err(|e| anyhow::anyhow!("Cannot save tokenizer: {e}"))?;

        tracing::info!(
            "Tokenizer trained and saved to '{}'",
            tok_path.display()
        );

        // Clean up the temporary corpus file
        std::fs::remove_file(&tmp_file).ok();

        Ok(tokenizer)
    }
}
