// ============================================================
// Layer 2 — Application / Use Cases
// ============================================================
// This layer orchestrates all the other layers to accomplish
// a specific goal (training or answering a question).
//
// Rules for this layer:
//   - No ML math or model code here
//   - No UI or printing here (that's Layer 1)
//   - No direct database/file access (that's Layer 4 and 6)
//   - Only workflow coordination
//
// Think of this layer as the "director" — it tells other
// layers what to do but doesn't do the work itself.
//
// Reference: Clean Architecture pattern
//            Rust Book §7 (Module System)

// The training workflow
pub mod train_use_case;

// The inference/question-answering workflow
pub mod ask_use_case;
