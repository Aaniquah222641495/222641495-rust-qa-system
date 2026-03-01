pub struct Chunker {
    chunk_size: usize,
    overlap:    usize,
}

impl Chunker {
    /// Panics if overlap >= chunk_size (stride would be 0).
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        assert!(
            overlap < chunk_size,
            "overlap ({}) must be less than chunk_size ({})",
            overlap,
            chunk_size
        );
        Self { chunk_size, overlap }
    }

    pub fn chunk(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }
        let stride = self.chunk_size.saturating_sub(self.overlap);
        let mut chunks = Vec::new();
        let mut start  = 0usize;
        loop {
            let end = (start + self.chunk_size).min(words.len());
            chunks.push(words[start..end].join(" "));
            if end == words.len() { break; }
            start += stride;
        }
        chunks
    }

    pub fn num_chunks(&self, word_count: usize) -> usize {
        if word_count == 0 { return 0; }
        let stride = self.chunk_size.saturating_sub(self.overlap);
        word_count.div_ceil(stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let chunks = Chunker::new(5, 2).chunk("a b c d e f g h i j");
        assert_eq!(chunks[0], "a b c d e");
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_overlap_is_correct() {
        // chunk=4, overlap=2, stride=2 → chunk 2 starts 2 words back ("c d e f")
        let chunks = Chunker::new(4, 2).chunk("a b c d e f");
        assert_eq!(chunks[0], "a b c d");
        assert!(chunks[1].starts_with("c d"));
    }

    #[test]
    fn test_short_text_gives_one_chunk() {
        let chunks = Chunker::new(100, 10).chunk("just a few words");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "just a few words");
    }

    #[test]
    fn test_empty_text_gives_no_chunks() {
        assert!(Chunker::new(5, 2).chunk("").is_empty());
    }

    #[test]
    #[should_panic]
    fn test_overlap_must_be_less_than_chunk_size() {
        let _ = Chunker::new(5, 5);
    }
}
