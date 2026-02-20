// ============================================================
// Layer 3 — Core Traits (Abstractions)
// ============================================================
// Traits are Rust's way of defining shared behaviour —
// similar to interfaces in Java or abstract classes in Python.
//
// By programming against traits instead of concrete types,
// we can swap implementations without changing the code
// that uses them. For example:
//   - DocxLoader implements DocumentSource
//   - A future PdfLoader could also implement DocumentSource
//   - The application layer only sees DocumentSource
//     and works with both without any changes
//
// This is the Dependency Inversion Principle from SOLID,
// applied using Rust's trait system.
//
// Reference: Rust Book §10 (Traits: Defining Shared Behaviour)
//            Rust Book §17 (Object Oriented Patterns)

use anyhow::Result;
use crate::domain::document::Document;

// ─── DocumentSource ───────────────────────────────────────────────────────────
/// Any component that can load documents from a source.
///
/// Implementations:
///   - DocxLoader  → loads from a directory of .docx files
///   - (future) PdfLoader → loads from PDF files
///   - (future) WebLoader → loads from URLs
pub trait DocumentSource {
    /// Load all available documents from this source.
    /// Returns a Vec of Documents or an error.
    fn load_all(&self) -> Result<Vec<Document>>;
}

// ─── QuestionAnswerer ─────────────────────────────────────────────────────────
/// Any component that can answer natural language questions.
///
/// Implementations:
///   - AskUseCase → uses the transformer model
///   - (future) RuleBasedAnswerer → uses keyword matching
pub trait QuestionAnswerer {
    /// Given a question string, return the best answer found.
    /// Returns "I don't know based on the documents." if unsure.
    fn answer(&self, question: &str) -> Result<String>;
}

// ─── Persistable ──────────────────────────────────────────────────────────────
/// Any component whose state can be saved and restored from disk.
///
/// Implementations:
///   - TransformerQaModel → saves/loads weights
///   - Tokenizer          → saves/loads vocabulary
pub trait Persistable: Sized {
    /// Save this component's state to the given path
    fn save(&self, path: &str) -> Result<()>;

    /// Load a component's state from the given path.
    /// Returns Self so callers can use the loaded instance directly.
    fn load(path: &str) -> Result<Self>;
}
