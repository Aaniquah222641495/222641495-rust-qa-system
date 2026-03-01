pub struct Preprocessor;

impl Preprocessor {
    pub fn new() -> Self { Self }

    pub fn clean(&self, text: &str) -> String {
        // Normalise Unicode whitespace variants and control characters.
        let step1: String = text
            .chars()
            .map(|c| match c {
                '\t' | '\u{00A0}' | '\u{200B}' | '\u{FEFF}' => ' ',
                '\r' => '\n',
                c if c.is_control() && c != '\n' => ' ',
                c => c,
            })
            .collect();

        let step2: String = step1
            .lines()
            .map(|line| {
                let mut out        = String::with_capacity(line.len());
                let mut last_space = false;
                for c in line.chars() {
                    if c == ' ' {
                        if !last_space { out.push(' '); }
                        last_space = true;
                    } else {
                        out.push(c);
                        last_space = false;
                    }
                }
                out.trim().to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Allow at most 2 consecutive newlines.
        let mut result        = String::with_capacity(step2.len());
        let mut newline_count = 0usize;
        for c in step2.chars() {
            if c == '\n' {
                newline_count += 1;
                if newline_count <= 2 { result.push(c); }
            } else {
                newline_count = 0;
                result.push(c);
            }
        }
        result.trim().to_string()
    }
}

impl Default for Preprocessor {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collapses_multiple_spaces() {
        assert_eq!(Preprocessor::new().clean("hello   world"), "hello world");
    }

    #[test]
    fn test_trims_edges() {
        assert_eq!(Preprocessor::new().clean("  hello world  "), "hello world");
    }

    #[test]
    fn test_removes_control_chars() {
        assert_eq!(Preprocessor::new().clean("hello\x01world"), "hello world");
    }

    #[test]
    fn test_collapses_blank_lines() {
        assert!(!Preprocessor::new().clean("line1\n\n\n\n\nline2").contains("\n\n\n"));
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(Preprocessor::new().clean(""), "");
    }
}
