#![allow(dead_code)]
use anyhow::{Context, Result};
use std::{fs, path::Path};

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
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.extension().and_then(|e| e.to_str()) == Some("docx")
                && !file_name.starts_with("~$")
            {
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
    use std::io::Read;
    let bytes = fs::read(path)
        .with_context(|| format!("Cannot read '{}'", path.display()))?;

    // Raw ZIP+XML extraction: docx-rs silently drops text inside <mc:AlternateContent>
    // runs (e.g. "SUMMER GRADUATION"). Direct XML scanning finds every <w:t> at any depth.
    let xml = {
        let cursor  = std::io::Cursor::new(&bytes);
        let mut zip = zip::ZipArchive::new(cursor)
            .map_err(|e| anyhow::anyhow!("ZIP open failed: {e}"))?;
        let mut entry = zip.by_name("word/document.xml")
            .map_err(|e| anyhow::anyhow!("word/document.xml not found: {e}"))?;
        let mut s = String::new();
        entry.read_to_string(&mut s)?;
        s
    };

    let text   = extract_text_from_xml(&xml);
    let source = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    Ok(Document::new(source, text))
}

/// Scan OOXML, emitting paragraphs and table rows (cells joined with " | ").
fn extract_text_from_xml(xml: &str) -> String {
    let mut parts: Vec<String> = Vec::new();

    let tokens: Vec<&str> = {
        let mut v   = Vec::new();
        let mut pos = 0;
        let b       = xml.as_bytes();
        while pos < b.len() {
            if b[pos] == b'<' {
                if let Some(end) = xml[pos..].find('>') {
                    v.push(&xml[pos..pos + end + 1]);
                    pos += end + 1;
                } else {
                    pos += 1;
                }
            } else {
                let start = pos;
                while pos < b.len() && b[pos] != b'<' { pos += 1; }
                v.push(&xml[start..pos]);
            }
        }
        v
    };

    let mut in_table     = 0usize; // nesting depth for <w:tbl>
    let mut in_cell      = false;
    let mut in_p_in_cell = false;  // inside a <p> within a table cell
    let mut in_t         = false;
    // Skip <w:t> inside <mc:Choice> — only capture <mc:Fallback> to avoid
    // double-counting text that appears in both branches.
    let mut in_choice    = 0usize;

    let mut para_buf     = String::new();
    let mut para_in_cell = String::new(); // runs within one in-cell paragraph (no space between)
    let mut cell_buf     = String::new(); // paragraphs within a cell (joined with " ")
    let mut row_cells: Vec<String> = Vec::new();

    for tok in &tokens {
        if tok.starts_with('<') {
            let name = xml_tag_name(tok);

            if name == "Choice" && !tok.starts_with("</") {
                in_choice += 1;
                continue;
            } else if name == "Choice" && tok.starts_with("</") {
                in_choice = in_choice.saturating_sub(1);
                continue;
            }

            if name == "tbl" && !tok.starts_with("</") {
                in_table += 1;
            } else if name == "tbl" && tok.starts_with("</") {
                in_table = in_table.saturating_sub(1);

            } else if xml_is_open(tok, "tr") && in_table > 0 {
                row_cells = Vec::new();
            } else if xml_is_close(tok, "tr") && in_table > 0 {
                if in_p_in_cell {
                    let t = para_in_cell.trim().to_string();
                    if !t.is_empty() {
                        if !cell_buf.is_empty() { cell_buf.push(' '); }
                        cell_buf.push_str(&t);
                    }
                    para_in_cell  = String::new();
                    in_p_in_cell  = false;
                }
                if in_cell && !cell_buf.trim().is_empty() {
                    row_cells.push(cell_buf.trim().to_string());
                }
                cell_buf  = String::new();
                in_cell   = false;
                let row_text = row_cells.join(" | ");
                if !row_text.trim().is_empty() {
                    parts.push(row_text);
                }
                row_cells = Vec::new();

            } else if xml_is_open(tok, "tc") && in_table > 0 {
                in_cell       = true;
                cell_buf      = String::new();
            } else if xml_is_close(tok, "tc") && in_table > 0 {
                if in_p_in_cell {
                    let t = para_in_cell.trim().to_string();
                    if !t.is_empty() {
                        if !cell_buf.is_empty() { cell_buf.push(' '); }
                        cell_buf.push_str(&t);
                    }
                    para_in_cell = String::new();
                    in_p_in_cell = false;
                }
                if !cell_buf.trim().is_empty() {
                    row_cells.push(cell_buf.trim().to_string());
                }
                cell_buf = String::new();
                in_cell  = false;

            } else if xml_is_open(tok, "p") && in_cell {
                in_p_in_cell = true;
                para_in_cell = String::new();
            } else if xml_is_close(tok, "p") && in_cell {
                let t = para_in_cell.trim().to_string();
                if !t.is_empty() {
                    if !cell_buf.is_empty() { cell_buf.push(' '); }
                    cell_buf.push_str(&t);
                }
                para_in_cell = String::new();
                in_p_in_cell = false;

            } else if xml_is_open(tok, "p") && in_table == 0 {
                para_buf = String::new();
            } else if xml_is_close(tok, "p") && in_table == 0 {
                if !para_buf.trim().is_empty() {
                    parts.push(para_buf.trim().to_string());
                }
                para_buf = String::new();

            } else if name == "t" && !tok.starts_with("</") {
                in_t = true;
            } else if xml_is_close(tok, "t") {
                in_t = false;
            }
        } else if in_t && in_choice == 0 {
            // Runs within a paragraph are concatenated without spaces so that numbers
            // split across runs (e.g. "2" + "5") stay joined as "25" not "2 5".
            let text = html_unescape(tok);
            if !text.is_empty() {
                if in_p_in_cell {
                    para_in_cell.push_str(&text);
                } else if in_cell {
                    if !cell_buf.is_empty() { cell_buf.push(' '); }
                    cell_buf.push_str(&text);
                } else if in_table == 0 {
                    para_buf.push_str(&text);
                }
            }
        }
    }

    parts.join("\n")
}

fn html_unescape(s: &str) -> String {
    s.replace("&amp;",  "&")
     .replace("&lt;",   "<")
     .replace("&gt;",   ">")
     .replace("&quot;", "\"")
     .replace("&apos;", "'")
}

/// Returns the local tag name. e.g. `"<w:tr …>"` → `"tr"`
fn xml_tag_name(tok: &str) -> &str {
    let inner = tok.trim_start_matches('<')
                   .trim_end_matches('>')
                   .trim_end_matches('/')
                   .trim();
    let name  = inner.split_whitespace().next().unwrap_or("");
    name.split(':').next_back().unwrap_or(name)
}

fn xml_is_open(tok: &str, elem: &str) -> bool {
    !tok.starts_with("</") && tok.starts_with('<') && xml_tag_name(tok) == elem
}

fn xml_is_close(tok: &str, elem: &str) -> bool {
    tok.starts_with("</") && xml_tag_name(tok) == elem
}
