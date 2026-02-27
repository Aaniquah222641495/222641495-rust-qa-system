#![allow(dead_code)]
// ============================================================
// Layer 4 — Document Loader
// ============================================================
// Loads .docx files extracting text from BOTH paragraphs
// and tables (calendar grids live in tables).

use anyhow::{Context, Result};
use std::{fs, path::Path};
use docx_rs::read_docx;

use crate::domain::document::Document;
use crate::domain::traits::DocumentSource;

pub struct DocxLoader {
    dir: String,
}

impl DocxLoader {
    pub fn new(dir: impl Into<String>) -> Self {
        Self { dir: dir.into() }
    }
}

impl DocumentSource for DocxLoader {
    fn load_all(&self) -> Result<Vec<Document>> {
        let dir = Path::new(&self.dir);
        if !dir.exists() {
            tracing::warn!("Docs directory '{}' does not exist", self.dir);
            return Ok(Vec::new());
        }

        let mut docs = Vec::new();
        for entry in fs::read_dir(dir)
            .with_context(|| format!("Cannot read directory '{}'", self.dir))?
        {
            let entry = entry?;
            let path  = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("docx") {
                match load_single_docx(&path) {
                    Ok(doc) => {
                        tracing::info!("Loaded: {} ({} chars)", doc.source, doc.text.len());
                        docs.push(doc);
                    }
                    Err(e) => tracing::warn!("Skipping '{}': {}", path.display(), e),
                }
            }
        }
        tracing::info!("Successfully loaded {} documents", docs.len());
        Ok(docs)
    }
}

fn load_single_docx(path: &Path) -> Result<Document> {
    let bytes = fs::read(path)
        .with_context(|| format!("Cannot read '{}'", path.display()))?;

    let docx = read_docx(&bytes)
        .map_err(|e| anyhow::anyhow!("docx-rs parse error: {:?}", e))?;

    let mut text_parts: Vec<String> = Vec::new();

    for child in &docx.document.children {
        use docx_rs::DocumentChild;
        match child {
            DocumentChild::Paragraph(para) => {
                let t = extract_paragraph_text(para);
                if !t.trim().is_empty() {
                    text_parts.push(t);
                }
            }
            DocumentChild::Table(table) => {
                // Calendar data lives in table cells — extract each row
                for row in &table.rows {
                    use docx_rs::TableChild;
                    let TableChild::TableRow(tr) = row;
                    let mut row_texts: Vec<String> = Vec::new();

                    for cell in &tr.cells {
                        use docx_rs::TableRowChild;
                        let TableRowChild::TableCell(tc) = cell;
                        let cell_text: String = tc.children
                            .iter()
                            .filter_map(|c| {
                                use docx_rs::TableCellContent;
                                if let TableCellContent::Paragraph(p) = c {
                                    let t = extract_paragraph_text(p);
                                    if t.trim().is_empty() { None } else { Some(t.trim().to_string()) }
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(" ");

                        if !cell_text.trim().is_empty() {
                            row_texts.push(cell_text);
                        }
                    }

                    if !row_texts.is_empty() {
                        text_parts.push(row_texts.join(" | "));
                    }
                }
            }
            _ => {}
        }
    }

    let text   = text_parts.join("\n");
    let source = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    Ok(Document::new(source, text))
}

fn extract_paragraph_text(para: &docx_rs::Paragraph) -> String {
    let mut parts = Vec::new();
    for child in &para.children {
        use docx_rs::ParagraphChild;
        if let ParagraphChild::Run(run) = child {
            for rc in &run.children {
                use docx_rs::RunChild;
                if let RunChild::Text(t) = rc {
                    parts.push(t.text.clone());
                }
            }
        }
    }
    parts.join("")
}
