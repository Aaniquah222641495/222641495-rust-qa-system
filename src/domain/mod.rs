// ============================================================
// Layer 3 — Domain Layer
// ============================================================
// This is the heart of the application — pure Rust structs
// and traits that define the core concepts of the system.
//
// Rules for this layer:
//   - NO Burn framework types allowed here
//   - NO file I/O or network calls
//   - NO ML-specific code
//   - Only plain Rust structs, enums, and traits
//
// Why keep this layer pure?
//   - Easy to unit test (no GPU needed)
//   - Easy to understand (no framework noise)
//   - Easy to swap implementations (just implement the trait)
//
// Think of this layer as the "dictionary" of the system —
// it defines what things ARE, not how they work.
//
// Reference: Rust Book §5 (Structs), §10 (Traits)

// A loaded document from disk
pub mod document;

// A question-answer pair with token span annotations
pub mod qa_pair;

// Core abstractions (traits) that other layers implement
pub mod traits;
