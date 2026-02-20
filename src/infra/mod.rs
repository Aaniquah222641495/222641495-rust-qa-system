// ============================================================
// Layer 6 — Infrastructure Layer
// ============================================================
// Handles all cross-cutting concerns that don't belong in
// any specific business layer:
//
//   checkpoint.rs      — Saving and loading model weights
//                        Uses Burn's CompactRecorder to
//                        serialise model parameters to disk.
//                        Also saves/loads TrainConfig as JSON
//                        so inference can rebuild the model.
//
//   tokenizer_store.rs — Tokenizer persistence
//                        Trains a BPE tokenizer on the document
//                        corpus if none exists, or loads a
//                        previously saved one. Ensures the same
//                        vocabulary is used for training and
//                        inference.
//
//   metrics.rs         — Training metrics logging
//                        Writes epoch-level metrics (loss,
//                        accuracy) to a CSV file for later
//                        analysis and plotting.
//
// Why is this a separate layer?
//   These concerns are used by multiple other layers but
//   don't belong to any one of them. Keeping them here:
//   - Prevents duplication across layers
//   - Makes it easy to swap implementations
//     (e.g. swap file checkpoints for S3 cloud storage)
//   - Keeps other layers focused on their core logic
//
// Reference: Rust Book §7 (Modules)
//            Rust Book §9 (Error Handling with anyhow)
//            Burn Book §5 (Checkpointing)

/// Model checkpoint saving and loading
pub mod checkpoint;

/// Tokenizer training, saving, and loading
pub mod tokenizer_store;

/// Training metrics CSV logger
pub mod metrics;
