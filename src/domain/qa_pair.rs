// ============================================================
// Layer 3 — QaPair Domain Type
// ============================================================
// Represents a single question-answer pair in domain terms.
// This is the core concept of extractive Q&A:
//   - We have a question
//   - We have a context passage
//   - The answer is a SPAN within the context
//     (a start index and end index into the token sequence)
//
// This is different from generative Q&A where the model
// generates new text. Here the model just points to where
// the answer already exists in the document.
//
// Example:
//   Question: "When is the graduation ceremony?"
//   Context:  "The graduation ceremony will be held on 15 April 2026"
//   Answer:   tokens at positions 7 to 11 → "15 April 2026"
//
// Reference: Devlin et al. (2019) - BERT paper
//            Rust Book §5 (Structs)

use serde::{Deserialize, Serialize};

/// A labelled Q&A example with token-level span annotation.
///
/// The start_position and end_position are indices into the
/// combined [CLS] Q [SEP] C [SEP] token sequence,
/// NOT into the raw text string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaPair {
    /// The natural language question being asked
    pub question: String,

    /// The context passage that contains the answer
    pub context: String,

    /// Index of the FIRST answer token in the full input sequence
    /// (after prepending [CLS] and question tokens)
    pub answer_start: usize,

    /// Index of the LAST answer token in the full input sequence
    /// (inclusive — the answer span is [answer_start..=answer_end])
    pub answer_end: usize,
}

impl QaPair {
    /// Create a new QaPair
    pub fn new(
        question:     impl Into<String>,
        context:      impl Into<String>,
        answer_start: usize,
        answer_end:   usize,
    ) -> Self {
        Self {
            question:     question.into(),
            context:      context.into(),
            answer_start,
            answer_end,
        }
    }

    /// Returns the length of the answer span in tokens
    pub fn span_length(&self) -> usize {
        self.answer_end.saturating_sub(self.answer_start) + 1
    }
}
