// ============================================================
// Layer 3 — Document Domain Type
// ============================================================
// Represents a single document loaded from disk.
// This is a plain data struct with no behaviour —
// just a source name and the extracted text content.
//
// Using #[derive(Debug, Clone)] gives us:
//   - Debug: lets us print the struct with {:?}
//   - Clone: lets us make copies of the struct
//   - Serialize/Deserialize: lets us save/load as JSON
//
// Reference: Rust Book §5 (Structs and Methods)
//            Rust Book §10 (Derive Macros)

use serde::{Deserialize, Serialize};

/// A raw document loaded from disk.
/// Language-agnostic and format-agnostic —
/// by the time a Document is created, the text
/// has already been extracted from the .docx format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// The filename or path — kept for traceability
    /// so we know which file an answer came from
    pub source: String,

    /// The full extracted text content of the document
    /// before any cleaning or tokenisation
    pub text: String,
}

impl Document {
    /// Create a new Document with a source path and text content.
    /// Uses impl Into<String> so callers can pass &str or String —
    /// this is idiomatic Rust for flexible string arguments.
    ///
    /// Example:
    ///   let doc = Document::new("calendar.docx", "January meeting...");
    pub fn new(source: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            text:   text.into(),
        }
    }
}
