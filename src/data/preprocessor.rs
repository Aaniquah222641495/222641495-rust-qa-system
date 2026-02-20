// ============================================================
// Layer 4 — Text Preprocessor
// ============================================================
// Cleans raw text extracted from .docx files before tokenisation.
//
// Why do we need to clean text?
//   .docx files often contain:
//   - Non-breaking spaces (U+00A0) from Word formatting
//   - Zero-width spaces (U+200B) from copy-pasting
//   - Carriage returns (\r) from Windows line endings
//   - Tab characters from table formatting
//   - Multiple consecutive spaces from indentation
//   - Control characters from special Word features
//
// If we don't clean these, the tokenizer treats them as
// meaningful tokens and wastes vocabulary space on whitespace.
//
// Cleaning steps (applied in order):
//   1. Replace Unicode whitespace variants with plain space
//   2. Replace \r with \n for consistent line endings
//   3. Remove invisible control characters
//   4. Collapse multiple spaces into one per line
//   5. Trim leading/trailing whitespace per line
//   6. Collapse more than 2 consecutive blank lines
//
// Reference: Rust Book §8 (Strings in Rust)
//            Rust Book §13 (Iterators)

pub struct Preprocessor;

impl Preprocessor {
    /// Create a new Preprocessor instance
    pub fn new() -> Self {
        Self
    }

    /// Clean a raw text string for downstream tokenisation.
    /// Takes a &str and returns an owned String.
    pub fn clean(&self, text: &str) -> String {

        // ── Step 1: Normalise individual characters ───────────────────────────
        // Map problematic Unicode characters to their ASCII equivalents.
        // This uses Rust's char-level iterator for safe Unicode handling.
        let step1: String = text
            .chars()
            .map(|c| match c {
                // Tab → space
                '\t' => ' ',
                // Non-breaking space → regular space
                '\u{00A0}' => ' ',
                // Zero-width space → regular space
                '\u{200B}' => ' ',
                // Byte order mark → space
                '\u{FEFF}' => ' ',
                // Windows carriage return → Unix newline
                '\r' => '\n',
                // Any other control character (except newline) → space
                c if c.is_control() && c != '\n' => ' ',
                // All other characters pass through unchanged
                c => c,
            })
            .collect();

        // ── Step 2: Clean each line individually ─────────────────────────────
        // Process line by line so we don't accidentally collapse
        // intentional paragraph breaks
        let step2: String = step1
            .lines()
            .map(|line| {
                // Collapse multiple consecutive spaces into one
                let mut out        = String::with_capacity(line.len());
                let mut last_space = false;

                for c in line.chars() {
                    if c == ' ' {
                        // Only add a space if the last char wasn't a space
                        if !last_space {
                            out.push(' ');
                        }
                        last_space = true;
                    } else {
                        out.push(c);
                        last_space = false;
                    }
                }

                // Trim leading and trailing spaces from each line
                out.trim().to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");

        // ── Step 3: Collapse excessive blank lines ────────────────────────────
        // Allow at most 2 consecutive newlines (one blank line).
        // More than that is just wasted space in the context window.
        let mut result       = String::with_capacity(step2.len());
        let mut newline_count = 0usize;

        for c in step2.chars() {
            if c == '\n' {
                newline_count += 1;
                // Allow up to 2 newlines, ignore the rest
                if newline_count <= 2 {
                    result.push(c);
                }
            } else {
                // Reset counter when we hit a non-newline character
                newline_count = 0;
                result.push(c);
            }
        }

        // Final trim of the whole document
        result.trim().to_string()
    }
}

/// Implement Default so Preprocessor can be created with Preprocessor::default()
impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────
// These tests run with `cargo test` and verify the cleaning logic.
// Reference: Rust Book §11 (Writing Automated Tests)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collapses_multiple_spaces() {
        let p = Preprocessor::new();
        assert_eq!(p.clean("hello   world"), "hello world");
    }

    #[test]
    fn test_trims_edges() {
        let p = Preprocessor::new();
        assert_eq!(p.clean("  hello world  "), "hello world");
    }

    #[test]
    fn test_removes_control_chars() {
        let p = Preprocessor::new();
        // \x01 is a control character that should become a space
        assert_eq!(p.clean("hello\x01world"), "hello world");
    }

    #[test]
    fn test_collapses_blank_lines() {
        let p = Preprocessor::new();
        let input  = "line1\n\n\n\n\nline2";
        let output = p.clean(input);
        // Should have at most 2 newlines between lines
        assert!(!output.contains("\n\n\n"));
    }

    #[test]
    fn test_empty_string() {
        let p = Preprocessor::new();
        assert_eq!(p.clean(""), "");
    }
}
