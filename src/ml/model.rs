// ============================================================
// Layer 5 — Transformer Encoder Q&A Model
// ============================================================
// Implements the full transformer encoder from
// "Attention Is All You Need" (Vaswani et al., 2017)
// adapted for extractive Question Answering.
//
// Input:  [CLS] question [SEP] context [SEP]
// Output: start_logits, end_logits — shape [batch, seq_len]
//
// Loss = (CrossEntropy(start) + CrossEntropy(end)) / 2
//
// Reference: Vaswani et al. (2017) https://arxiv.org/abs/1706.03762
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

// ─── Model Configuration ──────────────────────────────────────────────────────
// NOTE: #[derive(Config)] already generates Clone and Serialize/Deserialize
// internally — do NOT add them again or you get conflicting impls.
#[derive(Config, Debug)]
pub struct TransformerQaConfig {
    /// Number of unique tokens in the vocabulary
    pub vocab_size: usize,
    /// Maximum input sequence length
    pub max_seq_len: usize,
    /// Hidden dimension (d_model in paper)
    pub d_model: usize,
    /// Number of attention heads (d_model must be divisible by num_heads)
    pub num_heads: usize,
    /// Number of stacked encoder layers (minimum 6 per spec)
    pub num_layers: usize,
    /// Inner dimension of the feed-forward network (typically 4x d_model)
    pub d_ff: usize,
    /// Dropout probability (set to 0.0 at inference time)
    pub dropout: f64,
}

impl TransformerQaConfig {
    /// Build and initialise the model on the given device
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerQaModel<B> {
        // Token embedding: maps token IDs → d_model vectors
        let token_embedding = EmbeddingConfig::new(self.vocab_size, self.d_model)
            .init(device);

        // Positional embedding: maps positions → d_model vectors
        let position_embedding = EmbeddingConfig::new(self.max_seq_len, self.d_model)
            .init(device);

        // Build num_layers identical encoder blocks
        let layers: Vec<EncoderBlock<B>> = (0..self.num_layers)
            .map(|_| self.build_encoder_block(device))
            .collect();

        let final_norm = LayerNormConfig::new(self.d_model).init(device);

        // Q&A head: projects d_model → 2 (start logit, end logit)
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

    /// Build one encoder block
    fn build_encoder_block<B: Backend>(&self, device: &B::Device) -> EncoderBlock<B> {
        let self_attn = MultiHeadAttentionConfig::new(self.d_model, self.num_heads)
            .with_dropout(self.dropout)
            .init(device);
        let ffn_linear1 = LinearConfig::new(self.d_model, self.d_ff).init(device);
        let ffn_linear2 = LinearConfig::new(self.d_ff, self.d_model).init(device);
        let norm1   = LayerNormConfig::new(self.d_model).init(device);
        let norm2   = LayerNormConfig::new(self.d_model).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        EncoderBlock { self_attn, ffn_linear1, ffn_linear2, norm1, norm2, dropout }
    }
}

// ─── Encoder Block ────────────────────────────────────────────────────────────
/// One complete transformer encoder layer.
///
/// Sub-layer 1 (Self-Attention):
///   h' = LayerNorm(x + Dropout(MHSA(x,x,x)))
///
/// Sub-layer 2 (FFN):
///   h'' = LayerNorm(h' + Dropout(FFN(h')))
///
/// FFN(x) = W2 * GELU(W1 * x + b1) + b2
///
/// Residual connections prevent vanishing gradients in deep networks.
#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    pub self_attn:   MultiHeadAttention<B>,
    pub ffn_linear1: Linear<B>,
    pub ffn_linear2: Linear<B>,
    pub norm1:       LayerNorm<B>,
    pub norm2:       LayerNorm<B>,
    pub dropout:     Dropout,
}

impl<B: Backend> EncoderBlock<B> {
    /// Forward pass — input and output shape: [batch, seq_len, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        use burn::nn::attention::MhaInput;

        // Self-attention: Q = K = V = x
        let attn_output = self
            .self_attn
            .forward(MhaInput::self_attn(x.clone()))
            .context;

        // Add & LayerNorm after attention
        let x = self.norm1.forward(x + self.dropout.forward(attn_output));

        // FFN with GELU activation
        // GELU(x) = x * Φ(x) — performs better than ReLU in transformers
        let ffn_out = self.ffn_linear2.forward(
            burn::tensor::activation::gelu(self.ffn_linear1.forward(x.clone()))
        );

        // Add & LayerNorm after FFN
        self.norm2.forward(x + self.dropout.forward(ffn_out))
    }
}

// ─── Full Model ────────────────────────────────────────────────────────────────
#[derive(Module, Debug)]
pub struct TransformerQaModel<B: Backend> {
    pub token_embedding:    Embedding<B>,
    pub position_embedding: Embedding<B>,
    pub layers:             Vec<EncoderBlock<B>>,
    pub final_norm:         LayerNorm<B>,
    pub qa_head:            Linear<B>,
    pub dropout:            Dropout,
    pub max_seq_len:        usize,
}

/// Output of the model forward pass
pub struct QaModelOutput<B: Backend> {
    /// Start position logits — shape: [batch, seq_len]
    pub start_logits: Tensor<B, 2>,
    /// End position logits — shape: [batch, seq_len]
    pub end_logits: Tensor<B, 2>,
}

impl<B: Backend> TransformerQaModel<B> {
    /// Full forward pass
    /// input_ids shape: [batch, seq_len]
    /// returns start_logits and end_logits, each [batch, seq_len]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> QaModelOutput<B> {
        let [batch_size, seq_len] = input_ids.dims();

        // Token embeddings: [batch, seq_len] → [batch, seq_len, d_model]
        let tok_emb = self.token_embedding.forward(input_ids);

        // Positional embeddings — inject position information
        // since self-attention is permutation-invariant
        let positions = Tensor::<B, 1, Int>::arange(
            0..seq_len as i64,
            &tok_emb.device()
        )
        .unsqueeze::<2>()
        .expand([batch_size, seq_len]);

        let pos_emb = self.position_embedding.forward(positions);

        // Combine token + position embeddings
        let mut x = self.dropout.forward(tok_emb + pos_emb);

        // Pass through all encoder layers
        for layer in &self.layers {
            x = layer.forward(x);
        }

        // Final layer normalisation
        let x = self.final_norm.forward(x); // [batch, seq_len, d_model]

        // Q&A head: project to 2 logits per token
        let logits = self.qa_head.forward(x); // [batch, seq_len, 2]

        // Split into start and end logits
        // squeeze::<2>() removes the last dimension (no argument in burn 0.20)
        let start_logits = logits
            .clone()
            .slice([0..batch_size, 0..seq_len, 0..1])
            .reshape([batch_size, seq_len]);

        let end_logits = logits
            .slice([0..batch_size, 0..seq_len, 1..2])
            .reshape([batch_size, seq_len]);

        QaModelOutput { start_logits, end_logits }
    }

    /// Forward pass with loss computation for training
    pub fn forward_loss(
        &self,
        input_ids:       Tensor<B, 2, Int>,
        start_positions: Tensor<B, 1, Int>,
        end_positions:   Tensor<B, 1, Int>,
    ) -> (Tensor<B, 1>, QaModelOutput<B>)
    where
        B: AutodiffBackend,
    {
        let output = self.forward(input_ids);

        let ce = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&output.start_logits.device());

        // Loss = (CE_start + CE_end) / 2
        let start_loss = ce.forward(output.start_logits.clone(), start_positions);
        let end_loss   = ce.forward(output.end_logits.clone(),   end_positions);
        let loss       = (start_loss + end_loss) / 2.0_f64;

        (loss, output)
    }
}
