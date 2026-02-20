// ============================================================
// Layer 5 — Transformer Encoder Q&A Model
// ============================================================
// Implements the full transformer encoder architecture from
// "Attention Is All You Need" (Vaswani et al., 2017)
// adapted for extractive Question Answering.
//
// Architecture Overview:
//
//   Input: [CLS] question tokens [SEP] context tokens [SEP]
//                                                        │
//   TokenEmbedding(vocab_size → d_model)                 │
//         +                                              │
//   PositionalEmbedding(max_seq_len → d_model)           │
//         │                                              │
//         ▼                                              │
//   ┌─────────────────────┐                              │
//   │   Encoder Layer 1   │  ← repeated num_layers times │
//   │  ┌───────────────┐  │                              │
//   │  │ Multi-Head    │  │  Attention(Q,K,V) =          │
//   │  │ Self-Attention│  │  softmax(QKᵀ/√dk)V          │
//   │  └──────┬────────┘  │                              │
//   │    Add & LayerNorm  │                              │
//   │  ┌───────────────┐  │                              │
//   │  │  FFN (GELU)   │  │  FFN(x) = W2·GELU(W1·x+b1)+b2
//   │  └──────┬────────┘  │                              │
//   │    Add & LayerNorm  │                              │
//   └─────────────────────┘  × 6 layers                 │
//         │                                              │
//   Final LayerNorm                                      │
//         │                                              │
//   Linear(d_model → 2)  ← Q&A head                     │
//         │                                              │
//   ┌─────┴──────┐                                       │
//   start_logits  end_logits  [batch, seq_len]           │
//
// Loss = (CrossEntropy(start) + CrossEntropy(end)) / 2
//
// Reference: Vaswani et al. (2017) https://arxiv.org/abs/1706.03762
//            Devlin et al. (2019) https://arxiv.org/abs/1810.04805
//            Burn Book §3 (Modules and Configs)

use burn::{
    nn::{
        attention::{MultiHeadAttention, MultiHeadAttentionConfig},
        Dropout, DropoutConfig,
        Embedding, EmbeddingConfig,
        LayerNorm, LayerNormConfig,
        Linear, LinearConfig,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use serde::{Deserialize, Serialize};

// ─── Model Configuration ──────────────────────────────────────────────────────
// The #[derive(Config)] macro from Burn generates:
//   - TransformerQaConfig::new(args...) constructor
//   - with_X() builder methods for each field
//   - save/load to JSON
//
// Serializable so we can save it alongside checkpoints
// and reconstruct the exact same model for inference.
#[derive(Config, Debug, Clone, Serialize, Deserialize)]
pub struct TransformerQaConfig {
    /// Number of unique tokens in the vocabulary
    /// Must match the tokenizer vocabulary size
    pub vocab_size: usize,

    /// Maximum input sequence length
    /// Sequences longer than this are truncated
    pub max_seq_len: usize,

    /// Hidden dimension — every token is represented as a
    /// vector of this size throughout the model (d_model in paper)
    pub d_model: usize,

    /// Number of attention heads
    /// d_model must be divisible by num_heads
    /// Each head learns different types of relationships
    pub num_heads: usize,

    /// Number of stacked encoder layers (minimum 6 per spec)
    /// More layers = more abstract representations
    pub num_layers: usize,

    /// Inner dimension of the feed-forward network
    /// Typically 4x d_model (e.g. 256 → 1024)
    pub d_ff: usize,

    /// Dropout probability — randomly zeroes activations
    /// during training to prevent overfitting
    /// Set to 0.0 at inference time
    pub dropout: f64,
}

impl TransformerQaConfig {
    /// Build and initialise the model on the given device.
    /// This allocates all weight tensors and fills them with
    /// random values from the default initialiser.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerQaModel<B> {
        // Token embedding table: maps integer token IDs to d_model vectors
        // Shape: [vocab_size, d_model]
        let token_embedding = EmbeddingConfig::new(self.vocab_size, self.d_model)
            .init(device);

        // Positional embedding table: maps position indices to d_model vectors
        // Shape: [max_seq_len, d_model]
        // Learned PE works as well as sinusoidal in practice
        let position_embedding = EmbeddingConfig::new(self.max_seq_len, self.d_model)
            .init(device);

        // Build num_layers identical encoder blocks
        // Each block has its own independent weights
        let layers: Vec<EncoderBlock<B>> = (0..self.num_layers)
            .map(|_| self.build_encoder_block(device))
            .collect();

        // Final layer norm applied after the last encoder block
        let final_norm = LayerNormConfig::new(self.d_model).init(device);

        // Q&A span head: projects d_model → 2 scalars per token
        // Output[:, :, 0] = start logits
        // Output[:, :, 1] = end logits
        let qa_head = LinearConfig::new(self.d_model, 2).init(device);

        let dropout = DropoutConfig::new(self.dropout).init();

        TransformerQaModel {
            token_embedding,
            position_embedding,
            layers,
            final_norm,
            qa_head,
            dropout,
            max_seq_len: self.max_seq_len,
        }
    }

    /// Build one encoder block with fresh weights
    fn build_encoder_block<B: Backend>(&self, device: &B::Device) -> EncoderBlock<B> {
        // Multi-head self-attention
        // Q = K = V = input (self-attention)
        let self_attn = MultiHeadAttentionConfig::new(self.d_model, self.num_heads)
            .with_dropout(self.dropout)
            .init(device);

        // FFN first linear: d_model → d_ff
        let ffn_linear1 = LinearConfig::new(self.d_model, self.d_ff).init(device);

        // FFN second linear: d_ff → d_model
        let ffn_linear2 = LinearConfig::new(self.d_ff, self.d_model).init(device);

        // Layer norms — one after attention, one after FFN
        let norm1 = LayerNormConfig::new(self.d_model).init(device);
        let norm2 = LayerNormConfig::new(self.d_model).init(device);

        let dropout = DropoutConfig::new(self.dropout).init();

        EncoderBlock {
            self_attn,
            ffn_linear1,
            ffn_linear2,
            norm1,
            norm2,
            dropout,
        }
    }
}

// ─── Encoder Block ────────────────────────────────────────────────────────────
/// One complete transformer encoder layer.
///
/// Mathematical operations (Post-LN variant):
///
///   Self-Attention sub-layer:
///     h' = LayerNorm(x + Dropout(MHSA(x, x, x)))
///
///   Feed-Forward sub-layer:
///     h'' = LayerNorm(h' + Dropout(FFN(h')))
///
///   where MHSA(Q,K,V) = Concat(head_1,...,head_h) * W_O
///         head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^V)
///         Attention(Q,K,V) = softmax(Q*K^T / sqrt(d_k)) * V
///         d_k = d_model / num_heads
///
///   and   FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2
///
/// The residual connections (x + ...) are crucial —
/// they allow gradients to flow directly to earlier layers
/// preventing the vanishing gradient problem.
#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    /// Multi-head self-attention module
    pub self_attn:   MultiHeadAttention<B>,
    /// First linear layer of the FFN (expands d_model → d_ff)
    pub ffn_linear1: Linear<B>,
    /// Second linear layer of the FFN (contracts d_ff → d_model)
    pub ffn_linear2: Linear<B>,
    /// Layer norm applied after attention + residual
    pub norm1:       LayerNorm<B>,
    /// Layer norm applied after FFN + residual
    pub norm2:       LayerNorm<B>,
    /// Dropout applied after attention and FFN outputs
    pub dropout:     Dropout,
}

impl<B: Backend> EncoderBlock<B> {
    /// Forward pass through one encoder layer.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, seq_len, d_model]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, seq_len, d_model]
    /// (same shape as input — encoder preserves dimensions)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // ── Sub-layer 1: Multi-Head Self-Attention ────────────────────────────
        // In self-attention, Q = K = V = x
        // The model learns which other tokens each token should
        // "pay attention to" when building its representation.
        //
        // For Q&A: attention lets question tokens attend to
        // relevant context tokens to find the answer span.
        use burn::nn::attention::MhaInput;

        let attn_output = self
            .self_attn
            .forward(MhaInput::self_attn(x.clone()))
            .context; // .context is the attended output

        // Residual connection: add input to attention output
        // Then apply layer normalisation for stable training
        // LayerNorm normalises across the d_model dimension
        let x = self.norm1.forward(x + self.dropout.forward(attn_output));

        // ── Sub-layer 2: Position-wise Feed-Forward Network ───────────────────
        // Applied independently to each position (token) in the sequence.
        // Two linear layers with GELU activation in between.
        //
        // GELU (Gaussian Error Linear Unit) is used instead of ReLU
        // because it performs better in transformer language models.
        // GELU(x) = x * Φ(x) where Φ is the Gaussian CDF
        let ffn_out = self.ffn_linear2.forward(
            burn::tensor::activation::gelu(
                self.ffn_linear1.forward(x.clone())
            )
        );

        // Residual connection + layer normalisation for FFN
        self.norm2.forward(x + self.dropout.forward(ffn_out))
    }
}

// ─── Full Transformer Q&A Model ───────────────────────────────────────────────
/// The complete transformer encoder with Q&A head.
///
/// #[derive(Module)] generates:
///   - Parameter collection for the optimiser
///   - .valid() method to switch off dropout for inference
///   - .into_record() / .load_record() for checkpointing
#[derive(Module, Debug)]
pub struct TransformerQaModel<B: Backend> {
    /// Maps token IDs to d_model dimensional vectors
    pub token_embedding:    Embedding<B>,
    /// Maps position indices to d_model dimensional vectors
    pub position_embedding: Embedding<B>,
    /// Stack of num_layers encoder blocks
    pub layers:             Vec<EncoderBlock<B>>,
    /// Layer norm applied after the final encoder block
    pub final_norm:         LayerNorm<B>,
    /// Linear projection from d_model to 2 (start/end logits)
    pub qa_head:            Linear<B>,
    /// Dropout applied to the input embeddings
    pub dropout:            Dropout,
    /// Stored for position index generation
    pub max_seq_len:        usize,
}

/// Output of the model's forward pass
pub struct QaModelOutput<B: Backend> {
    /// Logits for the start position — shape: [batch, seq_len]
    /// Higher value = more likely to be the start of the answer
    pub start_logits: Tensor<B, 2>,
    /// Logits for the end position — shape: [batch, seq_len]
    /// Higher value = more likely to be the end of the answer
    pub end_logits: Tensor<B, 2>,
}

impl<B: Backend> TransformerQaModel<B> {
    /// Full forward pass through the model.
    ///
    /// # Arguments
    /// * `input_ids` - Token ID tensor of shape [batch_size, seq_len]
    ///
    /// # Returns
    /// QaModelOutput with start_logits and end_logits,
    /// each of shape [batch_size, seq_len]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> QaModelOutput<B> {
        let [batch_size, seq_len] = input_ids.dims();

        // ── Step 1: Token Embeddings ──────────────────────────────────────────
        // Look up each token ID in the embedding table.
        // Each integer becomes a d_model dimensional float vector.
        // Shape: [batch, seq_len] → [batch, seq_len, d_model]
        let tok_emb = self.token_embedding.forward(input_ids);

        // ── Step 2: Positional Embeddings ─────────────────────────────────────
        // Self-attention is permutation-invariant — it doesn't know
        // the order of tokens. Positional embeddings inject
        // position information into the representations.
        //
        // Create position indices [0, 1, 2, ..., seq_len-1]
        // then look them up in the position embedding table.
        let positions = Tensor::<B, 1, Int>::arange(
            0..seq_len as i64,
            &tok_emb.device()
        )
        .unsqueeze::<2>()              // [1, seq_len]
        .expand([batch_size, seq_len]); // [batch, seq_len]

        let pos_emb = self.position_embedding.forward(positions);

        // ── Step 3: Input Representation ──────────────────────────────────────
        // Combine token and position information by addition.
        // This is the standard approach from the original transformer.
        // x = TokenEmbedding + PositionalEmbedding
        // Shape: [batch, seq_len, d_model]
        let mut x = self.dropout.forward(tok_emb + pos_emb);

        // ── Step 4: Stack of Encoder Layers ───────────────────────────────────
        // Pass through each of the num_layers encoder blocks in sequence.
        // Each layer refines the contextual representations —
        // early layers capture syntax, later layers capture semantics.
        for layer in &self.layers {
            x = layer.forward(x);
        }

        // ── Step 5: Final Layer Normalisation ─────────────────────────────────
        // Applied once after all encoder layers
        let x = self.final_norm.forward(x); // [batch, seq_len, d_model]

        // ── Step 6: Q&A Span Prediction Head ──────────────────────────────────
        // Project each token's d_model representation to 2 scalars.
        // logits[:, :, 0] = start position scores
        // logits[:, :, 1] = end position scores
        // Shape: [batch, seq_len, d_model] → [batch, seq_len, 2]
        let logits = self.qa_head.forward(x);

        // Split the last dimension into separate start and end tensors
        // slice notation: [dim0_range, dim1_range, dim2_range]
        let start_logits = logits
            .clone()
            .slice([0..batch_size, 0..seq_len, 0..1])
            .squeeze::<2>(2); // Remove last dim → [batch, seq_len]

        let end_logits = logits
            .slice([0..batch_size, 0..seq_len, 1..2])
            .squeeze::<2>(2); // Remove last dim → [batch, seq_len]

        QaModelOutput { start_logits, end_logits }
    }

    /// Forward pass that also computes the training loss.
    ///
    /// Loss = (CrossEntropy(start_logits, start_targets)
    ///       + CrossEntropy(end_logits, end_targets)) / 2
    ///
    /// CrossEntropyLoss treats the seq_len positions as classes
    /// and penalises the model for assigning low probability
    /// to the correct start/end positions.
    ///
    /// The where clause adds AutodiffBackend requirement only
    /// for this method — inference uses forward() without it.
    pub fn forward_loss(
        &self,
        input_ids:       Tensor<B, 2, Int>,
        start_positions: Tensor<B, 1, Int>,
        end_positions:   Tensor<B, 1, Int>,
    ) -> (Tensor<B, 1>, QaModelOutput<B>)
    where
        B: AutodiffBackend,
    {
        // Run the full forward pass to get logits
        let output = self.forward(input_ids);

        // Initialise cross entropy loss on the same device as the logits
        let ce = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&output.start_logits.device());

        // Compute start position loss
        // logits: [batch, seq_len], targets: [batch] (class indices)
        let start_loss = ce.forward(
            output.start_logits.clone(),
            start_positions,
        );

        // Compute end position loss
        let end_loss = ce.forward(
            output.end_logits.clone(),
            end_positions,
        );

        // Average the two losses — equal weight to start and end
        let loss = (start_loss + end_loss) / 2.0_f64;

        (loss, output)
    }
}
