// ============================================================
// Layer 4 — Text Chunker
// ============================================================
// Splits long documents into overlapping windows of text.
//
// Why do we need chunking?
//   Transformer models have a maximum input length (max_seq_len).
//   Our assignment documents may be much longer than this limit.
//   We can't just truncate — we might cut off the answer!
//
// Solution: Sliding window chunking with overlap
//   - Split the document into chunks of `chunk_size` words
//   - Each chunk overlaps with the next by `overlap` words
//   - This ensures answer spans near chunk boundaries
//     appear fully in at least one chunk
//
// Example with chunk_size=5, overlap=2:
//   Document: "A B C D E F G H I J"
//   Chunk 1:  "A B C D E"          (positions 0-4)
//   Chunk 2:  "C D E F G"          (positions 2-6, overlaps by 2)
//   Chunk 3:  "E F G H I"          (positions 4-8, overlaps by 2)
//   Chunk 4:  "G H I J"            (positions 6-9, last chunk)
//
// The stride (step between chunks) = chunk_size - overlap
//
// Reference: Rust Book §8 (Slices)
//            Devlin et al. (2019) BERT paper - sliding window approach

pub struct Chunker {
    /// Target number of words per chunk
    chunk_size: usize,
    /// Number of words shared between adjacent chunks
    overlap: usize,
}

impl Chunker {
    /// Create a new Chunker.
    ///
    /// # Panics
    /// Panics if overlap >= chunk_size, because that would
    /// create an infinite loop (stride would be 0 or negative)
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        assert!(
            overlap < chunk_size,
            "overlap ({}) must be less than chunk_size ({})",
            overlap,
            chunk_size
        );
        Self { chunk_size, overlap }
    }

    /// Split text into overlapping word-level chunks.
    /// Returns a Vec of owned Strings — one per chunk.
    pub fn chunk(&self, text: &str) -> Vec<String> {
        // Split the text on whitespace to get individual words.
        // collect() into Vec<&str> gives us indexed access.
        let words: Vec<&str> = text.split_whitespace().collect();

        // Empty document → return empty vec, nothing to chunk
        if words.is_empty() {
            return Vec::new();
        }

        // stride = how many words we advance between chunks
        // stride = chunk_size - overlap ensures the overlap
        let stride = self.chunk_size.saturating_sub(self.overlap);

        let mut chunks = Vec::new();
        let mut start  = 0usize;

        loop {
            // End of this chunk (clamped to document length)
            let end = (start + self.chunk_size).min(words.len());

            // Join the words in this window back into a string
            chunks.push(words[start..end].join(" "));

            // If we've reached the end of the document, stop
            if end == words.len() {
                break;
            }

            // Advance by stride for the next chunk
            start += stride;
        }

        chunks
    }

    /// Returns how many chunks a text of `word_count` words would produce
    pub fn num_chunks(&self, word_count: usize) -> usize {
        if word_count == 0 {
            return 0;
        }
        let stride = self.chunk_size.saturating_sub(self.overlap);
        (word_count + stride - 1) / stride
    }
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let c      = Chunker::new(5, 2);
        let text   = "a b c d e f g h i j";
        let chunks = c.chunk(text);

        // First chunk should be the first 5 words
        assert_eq!(chunks[0], "a b c d e");
        // Should produce multiple chunks
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_overlap_is_correct() {
        let c      = Chunker::new(4, 2);
        let text   = "a b c d e f";
        let chunks = c.chunk(text);

        // With chunk=4, overlap=2, stride=2:
        // Chunk 1: a b c d
        // Chunk 2: c d e f  ← starts 2 words back from end of chunk 1
        assert_eq!(chunks[0], "a b c d");
        assert!(chunks[1].starts_with("c d"));
    }

    #[test]
    fn test_short_text_gives_one_chunk() {
        let c      = Chunker::new(100, 10);
        let chunks = c.chunk("just a few words");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "just a few words");
    }

    #[test]
    fn test_empty_text_gives_no_chunks() {
        let c      = Chunker::new(5, 2);
        let chunks = c.chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_overlap_must_be_less_than_chunk_size() {
        // This should panic because overlap >= chunk_size
        let _ = Chunker::new(5, 5);
    }
}
