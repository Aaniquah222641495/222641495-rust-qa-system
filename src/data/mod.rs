// ============================================================
// Layer 4 — Data Pipeline
// ============================================================
// This layer handles everything from raw .docx files
// all the way to GPU-ready tensor batches.
//
// The pipeline flows in this order:
//
//   .docx files
//       │
//       ▼
//   DocxLoader        → reads files, extracts raw text
//       │
//       ▼
//   Preprocessor      → cleans text (whitespace, encoding)
//       │
//       ▼
//   Chunker           → splits long docs into overlapping windows
//       │
//       ▼
//   Tokenizer         → converts words to token ID numbers
//       │
//       ▼
//   QaDataset         → implements Burn's Dataset trait
//       │
//       ▼
//   QaBatcher         → stacks samples into tensor batches
//       │
//       ▼
//   DataLoader        → feeds batches to the training loop
//
// Each module is responsible for exactly one step.
// This makes each step independently testable and replaceable.
//
// Reference: Burn Book §4 (Datasets and Dataloaders)
//            Rust Book §13 (Iterators and Closures)

/// Loads .docx files from a directory using docx-rs
pub mod loader;

/// Cleans and normalises raw extracted text
pub mod preprocessor;

/// Splits long documents into overlapping chunks
pub mod chunker;

/// Implements Burn's Dataset trait for Q&A samples
pub mod dataset;

/// Implements Burn's Batcher trait to create tensor batches
pub mod batcher;

/// Shuffles and splits data into train/validation sets
pub mod splitter;
