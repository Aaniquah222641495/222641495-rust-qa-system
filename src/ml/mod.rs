// ============================================================
// Layer 5 — ML / Model Layer (Burn)
// ============================================================
// This layer contains ALL Burn framework specific code.
// No other layer imports from burn directly — only this one.
//
// Why isolate Burn code here?
//   - If Burn's API changes, we only update this layer
//   - Other layers are testable without a GPU
//   - The model architecture is clearly separated from
//     data loading and application logic
//
// What's in this layer:
//
//   model.rs     — The transformer encoder architecture
//                  Implements the full 6-layer encoder with:
//                  • Token embeddings
//                  • Positional embeddings
//                  • Multi-head self-attention
//                  • Feed-forward networks (GELU activation)
//                  • Layer normalisation
//                  • Residual connections
//                  • Q&A span prediction head
//
//   trainer.rs   — The training loop
//                  Handles forward pass, loss computation,
//                  backward pass, optimiser step, and
//                  checkpoint saving per epoch
//
//   inferencer.rs — The inference engine
//                  Loads a checkpoint, tokenises input,
//                  runs the model, decodes the answer span
//
// Reference: Burn Book §3 (Building Blocks)
//            Burn Book §5 (Training)
//            Vaswani et al. (2017) Attention Is All You Need
//            Devlin et al. (2019) BERT

/// Transformer encoder Q&A model architecture
pub mod model;

/// Full training loop with validation and checkpointing
pub mod trainer;

/// Inference engine — loads checkpoint and predicts answers
pub mod inferencer;
