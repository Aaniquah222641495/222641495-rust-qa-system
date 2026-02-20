// ============================================================
// Layer 4 — Document Loader
// ============================================================
// Loads .docx files from a directory using the docx-rs crate.
//
// How .docx files work:
//   A .docx file is actually a ZIP archive containing XML files.
//   docx-rs parses this ZIP and gives us a typed Rust API
//   over the XML content.
//
// The document structure in docx-rs looks like:
//   Document
//     └── children: Vec<DocumentChild>
//           └── Paragraph
//                 └── children: Vec<ParagraphChild>
//                       └── Run
//                             └── children: Vec<RunChild>
//                                   └── Text (the actual words!)
//
// We walk this tree collecting all Text nodes,
// joining them into a single string per paragraph.
//
// Reference: docx-rs crate documentation
//            Rust Book §8 (Collections)
//            Rust Book §9 (Error Handling)

use anyhow::{Context, Result};
use std::{fs, path::Path};
use docx_rs::{read_docx, DocxError};

use crate::domain::document::Document;
use crate::domain::traits::DocumentSource;

/// Loads all .docx files from a given directory.
/// Implements the DocumentSource trait from Layer 3.
pub struct DocxLoader {
    /// Path to the directory containing .docx files
    dir: String,
}

impl DocxLoader {
    /// Create a new DocxLoader pointed at a directory
    pub fn new(dir: impl Into<String>) -> Self {
        Self { dir: dir.into() }
    }
}

/// Implement the DocumentSource trait so the application layer
/// can call load_all() without knowing about .docx internals
impl DocumentSource for DocxLoader {
    fn load_all(&self) -> Result<Vec<Document>> {
        let dir = Path::new(&self.dir);

        // If the directory doesn't exist, return empty rather than crashing.
        // This allows the system to run even without documents (demo mode).
        if !dir.exists() {
            tracing::warn!(
                "Docs directory '{}' does not exist — returning empty corpus",
                self.dir
            );
            return Ok(Vec::new());
        }

        let mut docs = Vec::new();

        // Walk every entry in the directory
        for entry in fs::read_dir(dir)
            .with_context(|| format!("Cannot read directory '{}'", self.dir))?
        {
            let entry = entry?;
            let path  = entry.path();

            // Only process files with the .docx extension
            if path.extension().and_then(|e| e.to_str()) == Some("docx") {
                match load_single_docx(&path) {
                    Ok(doc) => {
                        tracing::debug!(
                            "Loaded: {} ({} chars)",
                            doc.source,
                            doc.text.len()
                        );
                        docs.push(doc);
                    }
                    // Log a warning but continue — don't fail on one bad file
                    Err(e) => {
                        tracing::warn!(
                            "Skipping '{}': {}",
                            path.display(),
                            e
                        );
                    }
                }
            }
        }

        tracing::info!("Successfully loaded {} documents", docs.len());
        Ok(docs)
    }
}

/// Parse a single .docx file and return a Document.
/// This function extracts all paragraph text and joins it with newlines.
fn load_single_docx(path: &Path) -> Result<Document> {
    // Read the raw bytes of the .docx file (which is a ZIP)
    let bytes = fs::read(path)
        .with_context(|| format!("Cannot read '{}'", path.display()))?;

    // Parse the ZIP/XML using docx-rs
    let docx = read_docx(&bytes)
        .map_err(|e: DocxError| {
            anyhow::anyhow!("docx-rs parse error in '{}': {:?}", path.display(), e)
        })?;

    // Walk the document tree collecting paragraph text
    let mut paragraphs: Vec<String> = Vec::new();

    for child in &docx.document.children {
        use docx_rs::DocumentChild;

        // We only care about Paragraph nodes (not tables, images, etc.)
        if let DocumentChild::Paragraph(para) = child {
            let para_text = extract_paragraph_text(para);

            // Skip empty paragraphs (section breaks, blank lines, etc.)
            if !para_text.trim().is_empty() {
                paragraphs.push(para_text);
            }
        }
    }

    // Join all paragraphs with newlines to form the full document text
    let text = paragraphs.join("\n");

    // Use the filename as the source identifier
    let source = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    Ok(Document::new(source, text))
}

/// Extract plain text from a single docx-rs Paragraph node.
///
/// Paragraph → Run → Text is the path through the docx-rs tree.
/// Multiple runs in a paragraph are concatenated with no separator
/// because they are parts of the same sentence.
fn extract_paragraph_text(para: &docx_rs::Paragraph) -> String {
    let mut parts = Vec::new();

    for child in &para.children {
        use docx_rs::ParagraphChild;

        // ParagraphChild::Run contains the actual text
        if let ParagraphChild::Run(run) = child {
            for rc in &run.children {
                use docx_rs::RunChild;

                // RunChild::Text is the leaf node with the actual string
                if let RunChild::Text(t) = rc {
                    parts.push(t.value.clone());
                }
            }
        }
    }

    // Join all text runs in this paragraph
    parts.join("")
}
