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

// NOTE: #[derive(Config)] already generates Clone and Serialize/Deserialize
// internally — do NOT add them again or you get conflicting impls.
#[derive(Config, Debug)]
pub struct TransformerQaConfig {
    pub vocab_size:  usize,
    pub max_seq_len: usize,
    pub d_model:     usize,
    pub num_heads:   usize,
    pub num_layers:  usize,
    pub d_ff:        usize,
    pub dropout:     f64,
}

impl TransformerQaConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerQaModel<B> {
        let token_embedding    = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let position_embedding = EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device);
        let layers: Vec<EncoderBlock<B>> = (0..self.num_layers)
            .map(|_| self.build_encoder_block(device))
            .collect();
        let final_norm = LayerNormConfig::new(self.d_model).init(device);
        let qa_head    = LinearConfig::new(self.d_model, 2).init(device);
        let dropout    = DropoutConfig::new(self.dropout).init();
        TransformerQaModel {
            token_embedding, position_embedding, layers,
            final_norm, qa_head, dropout,
            max_seq_len: self.max_seq_len,
        }
    }

    fn build_encoder_block<B: Backend>(&self, device: &B::Device) -> EncoderBlock<B> {
        let self_attn   = MultiHeadAttentionConfig::new(self.d_model, self.num_heads)
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
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        use burn::nn::attention::MhaInput;
        let attn_output = self.self_attn.forward(MhaInput::self_attn(x.clone())).context;
        let x = self.norm1.forward(x + self.dropout.forward(attn_output));
        let ffn_out = self.ffn_linear2.forward(
            burn::tensor::activation::gelu(self.ffn_linear1.forward(x.clone()))
        );
        self.norm2.forward(x + self.dropout.forward(ffn_out))
    }
}

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

pub struct QaModelOutput<B: Backend> {
    pub start_logits: Tensor<B, 2>,
    pub end_logits:   Tensor<B, 2>,
}

impl<B: Backend> TransformerQaModel<B> {
    /// input_ids: [batch, seq_len] → start_logits, end_logits: [batch, seq_len]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> QaModelOutput<B> {
        let [batch_size, seq_len] = input_ids.dims();

        let tok_emb = self.token_embedding.forward(input_ids);

        // Self-attention is permutation-invariant, so position must be injected explicitly.
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &tok_emb.device())
            .unsqueeze::<2>()
            .expand([batch_size, seq_len]);
        let pos_emb = self.position_embedding.forward(positions);

        let mut x = self.dropout.forward(tok_emb + pos_emb);
        for layer in &self.layers {
            x = layer.forward(x);
        }
        let x = self.final_norm.forward(x); // [batch, seq_len, d_model]

        // Project to 2 logits per token then split into start / end.
        let logits = self.qa_head.forward(x); // [batch, seq_len, 2]
        let start_logits = logits.clone()
            .slice([0..batch_size, 0..seq_len, 0..1])
            .reshape([batch_size, seq_len]);
        let end_logits = logits
            .slice([0..batch_size, 0..seq_len, 1..2])
            .reshape([batch_size, seq_len]);

        QaModelOutput { start_logits, end_logits }
    }

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
        let loss = (ce.forward(output.start_logits.clone(), start_positions)
                  + ce.forward(output.end_logits.clone(),   end_positions)) / 2.0_f64;
        (loss, output)
    }
}
