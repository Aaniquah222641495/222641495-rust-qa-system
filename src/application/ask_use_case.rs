// ============================================================
// Layer 2 — AskUseCase
// ============================================================
// Orchestrates the full inference pipeline in order:
//
//   Step 1: Load tokenizer from checkpoint
//   Step 2: Load and chunk documents
//   Step 3: Retrieve relevant chunks for the question
//   Step 4: Load model from checkpoint
//   Step 5: Run inference over each chunk
//   Step 6: Return best answer or "I don't know"
//
// The confidence threshold determines when the model admits
// it cannot find a good answer in the documents.
//
// Reference: Rust Book §13 (Iterators)
//            Burn Book §6 (Inference)

use anyhow::Result;

use crate::data::{
    loader::DocxLoader,
    preprocessor::Preprocessor,
    chunker::Chunker,
};
use crate::infra::{
    tokenizer_store::TokenizerStore,
    checkpoint::CheckpointManager,
};
use crate::ml::inferencer::Inferencer;

// If the model's confidence score is below this value,
// we respond with "I don't know" rather than guessing.
// This is the joint probability P(start) * P(end) of the best span.
const CONFIDENCE_THRESHOLD: f32 = 0.10;

pub struct AskUseCase {
    checkpoint_dir: String,
    docs_dir:       String,
}

impl AskUseCase {
    /// Create a new AskUseCase with paths to checkpoint and documents
    pub fn new(checkpoint_dir: String, docs_dir: String) -> Result<Self> {
        Ok(Self { checkpoint_dir, docs_dir })
    }

    /// Run the full inference pipeline and return an answer string
    pub fn answer(&self, question: &str) -> Result<String> {

        // ── Step 1: Load the saved tokenizer ─────────────────────────────────
        // The tokenizer must be the same one used during training
        // so token IDs map to the same vocabulary
        let tok_store  = TokenizerStore::new(&self.checkpoint_dir);
        let tokenizer  = tok_store.load()?;

        // ── Step 2: Load and chunk all documents ─────────────────────────────
        // We need to search through the documents to find
        // the passage most likely to contain the answer
        let loader  = DocxLoader::new(&self.docs_dir);
        let raw_docs = loader.load_all()?;
        let prep    = Preprocessor::new();
        let chunker = Chunker::new(200, 50);

        // Clean each document and split into overlapping chunks
        let chunks: Vec<String> = raw_docs
            .iter()
            .flat_map(|d| chunker.chunk(&prep.clean(&d.text)))
            .collect();

        // If no documents were found, we cannot answer anything
        if chunks.is_empty() {
            return Ok("I don't know based on the documents.".to_string());
        }

        // ── Step 3: Retrieve top-3 most relevant chunks ───────────────────────
        // Uses simple word overlap scoring to find passages
        // that share the most words with the question.
        // A production system would use BM25 or dense retrieval here.
        let top_chunks = retrieve_top_chunks(question, &chunks, 3);

        // ── Step 4: Load model from latest checkpoint ─────────────────────────
        // Rebuilds the model architecture and loads the saved weights
        let ckpt_manager = CheckpointManager::new(&self.checkpoint_dir);
        let inferencer   = Inferencer::from_checkpoint(&ckpt_manager, &tokenizer)?;

        // ── Step 5: Run inference over each retrieved chunk ───────────────────
        // We try each chunk and keep the answer with the highest confidence
        let mut best_answer     = String::new();
        let mut best_confidence = 0.0f32;

        for context in &top_chunks {
            let (answer, confidence) = inferencer.predict(
                question,
                context,
                &tokenizer
            )?;

            // Keep track of the most confident prediction
            if confidence > best_confidence {
                best_confidence = confidence;
                best_answer     = answer;
            }
        }

        tracing::info!("Best confidence score: {:.4}", best_confidence);

        // ── Step 6: Return answer or fallback ─────────────────────────────────
        // If confidence is too low, admit we don't know rather than
        // returning a wrong answer with false confidence
        if best_confidence < CONFIDENCE_THRESHOLD || best_answer.trim().is_empty() {
            Ok("I don't know based on the documents.".to_string())
        } else {
            Ok(best_answer)
        }
    }
}

// ─── Simple Term-Overlap Retrieval ───────────────────────────────────────────
// Scores each chunk by counting how many question words appear in it.
// Returns the top_k chunks sorted by overlap score (highest first).
//
// Example: question = "When is the graduation ceremony?"
//   chunk A contains "graduation" and "ceremony" → score = 2/4 = 0.5
//   chunk B contains only "graduation"           → score = 1/4 = 0.25
//   → chunk A is returned first
fn retrieve_top_chunks(
    question: &str,
    chunks:   &[String],
    top_k:    usize,
) -> Vec<String> {
    // Split question into individual words for matching
    let q_tokens: Vec<&str> = question.split_whitespace().collect();

    // Score every chunk by word overlap with the question
    let mut scored: Vec<(f32, &String)> = chunks
        .iter()
        .map(|chunk| {
            let lower = chunk.to_lowercase();
            // Count how many question words appear in this chunk
            let matches = q_tokens
                .iter()
                .filter(|t| lower.contains(&t.to_lowercase() as &str))
                .count();
            let score = matches as f32 / q_tokens.len().max(1) as f32;
            (score, chunk)
        })
        .collect();

    // Sort by score descending — highest overlap first
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Return the top_k chunk strings
    scored
        .into_iter()
        .take(top_k)
        .map(|(_, c)| c.clone())
        .collect()
}
