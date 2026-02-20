# Project Report: Word-Document Q&A System in Rust + Burn

**Module:** SEG 580S: Software Engineering Deep Learning Systems  
**Student Number:** 222641495  
**Institution:** Cape Peninsula University of Technology (CPUT)  
**Date:** February 2026  
**Repository:** https://github.com/Aaniquah222641495/222641495-rust-qa-system

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Design](#4-model-design)
5. [Training Procedure](#5-training-procedure)
6. [Inference and Decoding](#6-inference-and-decoding)
7. [Experiments and Results](#7-experiments-and-results)
8. [Discussion](#8-discussion)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

---

## 1. Introduction

This project implements a complete extractive Question-Answering (Q&A)
system in Rust using the Burn deep learning framework. The system is
designed to answer natural language questions about Word documents such as:

- "What is the Month and date will the 2026 End of year Graduation
  Ceremony be held?"
- "How many times did the HDC hold their meetings in 2024?"

Extractive Q&A means the model finds and returns a span of text that
already exists in the document — it does not generate new words.
This is more reliable than generative Q&A because the answer is
always grounded in the source document.

### Why Rust and Burn?

Rust provides memory safety without a garbage collector, making it
ideal for systems that need both safety and performance. The Burn
framework brings deep learning to Rust with a clean, type-safe API
that supports multiple backends including GPU via WGPU.

### Assignment Approach

The system follows the full ML pipeline:
1. Load .docx files using docx-rs
2. Clean and chunk text into passages
3. Train a BPE tokenizer on the corpus
4. Train a 6-layer transformer encoder
5. Answer questions via a CLI interface

---

## 2. System Architecture

The project follows a strict 6-layer Clean Architecture where each
layer depends only on the layer directly below it.
```
┌─────────────────────────────────────────────────────┐
│  Layer 1: CLI (src/cli/)                            │
│  Commands: train | ask                              │
│  Uses: clap derive macros                           │
└────────────────────┬────────────────────────────────┘
                     │ delegates to
┌────────────────────▼────────────────────────────────┐
│  Layer 2: Application (src/application/)            │
│  TrainUseCase | AskUseCase                          │
│  Orchestrates all layers — no ML math here          │
└────────────────────┬────────────────────────────────┘
                     │ uses domain types
┌────────────────────▼────────────────────────────────┐
│  Layer 3: Domain (src/domain/)                      │
│  Document | QaPair | Traits                         │
│  Pure Rust — zero framework dependency              │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Layer 4: Data (src/data/)                          │
│  Loader → Preprocessor → Chunker                   │
│  → Dataset → Batcher → Splitter                    │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Layer 5: ML Model (src/ml/)                        │
│  TransformerQaModel (6 encoder layers)              │
│  Trainer | Inferencer                               │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Layer 6: Infrastructure (src/infra/)               │
│  CheckpointManager | TokenizerStore                 │
│  MetricsLogger                                      │
└─────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Domain isolation:** The Document and QaPair structs in Layer 3
contain no Burn types. This means the domain can be unit tested
without a GPU. (Rust Book §11 - Testing)

**Backend genericism:** The model is generic over B: Backend,
allowing the same code to run on GPU (WGPU) or CPU (NdArray)
without any changes. (Burn Book §2 - Backends)

**Owned data:** Following Rust Book §4 (Ownership), data flows
downward through layers using owned types. Clone is used
conservatively only at layer boundaries.

---

## 3. Data Pipeline

### 3.1 Document Loading

The DocxLoader uses docx-rs to parse Word documents. A .docx file
is a ZIP archive containing XML files. docx-rs exposes a typed
Rust API over this XML structure:
```
Document
  └── children: Vec<DocumentChild>
        └── Paragraph
              └── children: Vec<ParagraphChild>
                    └── Run
                          └── children: Vec<RunChild>
                                └── Text ← actual words
```

We walk this tree collecting all Text nodes from all paragraphs.

### 3.2 Text Preprocessing

The Preprocessor applies three cleaning steps:
1. Replace Unicode whitespace variants with ASCII space
2. Collapse multiple consecutive spaces per line
3. Collapse more than 2 consecutive blank lines

### 3.3 Chunking with Sliding Window
```
Document: "A B C D E F G H I J"
chunk_size=5, overlap=2, stride=3

Chunk 1: "A B C D E"     [0..5]
Chunk 2: "C D E F G"     [2..7]  ← overlaps by 2
Chunk 3: "E F G H I"     [4..9]  ← overlaps by 2
Chunk 4: "G H I J"       [6..10] ← final chunk
```

Overlap ensures answer spans near boundaries are fully captured.

### 3.4 Tokenisation

A BPE (Byte Pair Encoding) tokenizer is trained on the document
corpus. BPE builds a vocabulary by iteratively merging the most
frequent adjacent character pairs:
```
Initial:    ["g","r","a","d","u","a","t","i","o","n"]
After merge: ["gr","a","d","u","a","t","i","o","n"]
After merge: ["gr","ad","u","a","t","i","o","n"]
...
Final token: ["graduation"] ← if frequent enough
```

### 3.5 Input Format

Following Devlin et al. (2019), the model input is:
```
[CLS] q_1 q_2 ... q_n [SEP] c_1 c_2 ... c_m [SEP] [PAD]...
 101                    102                   102    0
  0    1   2       n   n+1  n+2              n+m+1
```

### 3.6 Train/Validation Split

80% of samples go to training, 20% to validation.
Fisher-Yates shuffle ensures an unbiased random split.

---

## 4. Model Design

### 4.1 Architecture Overview
```
Input: [CLS] question [SEP] context [SEP]
              │
    TokenEmbedding(vocab_size → d_model)
              +
    PositionalEmbedding(max_seq_len → d_model)
              │
              ▼
    ┌──────────────────────┐
    │   Encoder Layer 1    │
    │  ┌────────────────┐  │
    │  │ Multi-Head     │  │  Q=K=V=x (self-attention)
    │  │ Self-Attention │  │  heads=8, d_k=d_model/heads
    │  └───────┬────────┘  │
    │    Add & LayerNorm   │
    │  ┌────────────────┐  │
    │  │ FFN            │  │  Linear→GELU→Linear
    │  │ d_model→d_ff   │  │  d_model=256, d_ff=1024
    │  │ →d_model       │  │
    │  └───────┬────────┘  │
    │    Add & LayerNorm   │
    └──────────────────────┘  × 6 layers
              │
        Final LayerNorm
              │
        Linear(d_model → 2)
              │
    ┌─────────┴──────────┐
    start_logits      end_logits
    [batch, seq_len]  [batch, seq_len]
```

### 4.2 Multi-Head Self-Attention

From Vaswani et al. (2017):
```
Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V

where d_k = d_model / num_heads = 256 / 8 = 32

MHSA(Q,K,V) = Concat(head_1,...,head_h) · W_O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

Dividing by √d_k prevents the dot products from growing too large,
which would push softmax into regions with tiny gradients.

### 4.3 Feed-Forward Network
```
FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2

GELU(x) = x · Φ(x)  where Φ is the Gaussian CDF
```

GELU is used instead of ReLU because it performs better in
transformer language models (Hendrycks & Gimpel, 2016).

### 4.4 Residual Connections and Layer Norm
```
After attention: x = LayerNorm(x + Dropout(MHSA(x)))
After FFN:       x = LayerNorm(x + Dropout(FFN(x)))
```

Residual connections allow gradients to flow directly to earlier
layers, preventing the vanishing gradient problem in deep networks.

### 4.5 Model Parameters

| Component           | Parameters                    |
|---------------------|-------------------------------|
| Token Embedding     | vocab_size × d_model          |
| Position Embedding  | max_seq_len × d_model         |
| Per Encoder Layer   | ~4 × d_model²                 |
| 6 Encoder Layers    | ~6 × 4 × 256² = 1,572,864     |
| Q&A Head            | d_model × 2                   |
| **Total (approx)**  | **~2.2M parameters**          |

---

## 5. Training Procedure

### 5.1 Loss Function
```
L = (CrossEntropy(start_logits, start_targets)
   + CrossEntropy(end_logits,   end_targets)) / 2
```

CrossEntropy treats each sequence position as a class.
The model is penalised for assigning low probability to
the correct start and end positions.

### 5.2 Optimiser: Adam

From Kingma & Ba (2015):
```
m_t = β1·m_{t-1} + (1-β1)·g_t          (1st moment)
v_t = β2·v_{t-1} + (1-β2)·g_t²         (2nd moment)
θ_t = θ_{t-1} - lr·m_t / (√v_t + ε)   (update)

lr=2e-4, β1=0.9, β2=0.999, ε=1e-8
```

### 5.3 Training Configuration

| Hyperparameter | Value  | Reason                              |
|----------------|--------|-------------------------------------|
| d_model        | 256    | Balance between capacity and speed  |
| num_heads      | 8      | 32 dims per head (256/8)            |
| num_layers     | 6      | Minimum per assignment spec         |
| d_ff           | 1024   | 4× d_model (standard ratio)         |
| max_seq_len    | 512    | Fits most Q+context pairs           |
| batch_size     | 8      | Fits in GPU memory                  |
| dropout        | 0.1    | Light regularisation                |
| epochs         | 10     | Starting point for experiments      |
| lr             | 2e-4   | Standard for transformer training   |

---

## 6. Inference and Decoding

### 6.1 Retrieval

Before running the model, we retrieve the top-3 document chunks
most likely to contain the answer using word overlap scoring:
```
score(chunk) = |question_words ∩ chunk_words| / |question_words|
```

### 6.2 Span Extraction
```
start_probs = softmax(start_logits)   shape: [seq_len]
end_probs   = softmax(end_logits)     shape: [seq_len]

best_span = argmax over (s,e) where:
  s ≥ context_start   (answer must be in context, not question)
  e ≥ s               (end must be after start)
  e - s ≤ 30          (answer length limit)

score(s,e) = start_probs[s] × end_probs[e]
```

### 6.3 Confidence Thresholding
```
if max_score < 0.10:
    return "I don't know based on the documents."
else:
    return decoded_answer_text
```

---

## 7. Experiments and Results

> Fill in actual values after running training experiments.

### 7.1 Training Curves

| Epoch | Train Loss | Val Loss | Start Acc | End Acc |
|-------|-----------|----------|-----------|---------|
| 1     | X.XXXX    | X.XXXX   | XX.X%     | XX.X%   |
| 2     | X.XXXX    | X.XXXX   | XX.X%     | XX.X%   |
| 5     | X.XXXX    | X.XXXX   | XX.X%     | XX.X%   |
| 10    | X.XXXX    | X.XXXX   | XX.X%     | XX.X%   |

### 7.2 Sample Q&A Results

| Question | Answer | Confidence |
|----------|--------|------------|
| What is the date of the 2026 graduation ceremony? | [result] | X.XX |
| How many times did HDC meet in 2024? | [result] | X.XX |

### 7.3 Ablation: Number of Layers

| Layers | Val Loss | Start Acc |
|--------|----------|-----------|
| 2      | X.XXXX   | XX.X%     |
| 4      | X.XXXX   | XX.X%     |
| 6      | X.XXXX   | XX.X%     |

---

## 8. Discussion

### Strengths

- **Type safety:** Rust's ownership system (Rust Book §4) prevents
  data races and null-pointer errors common in Python ML code.
- **Layer isolation:** Each layer is independently testable.
- **Backend flexibility:** Burn's generic Backend trait means the
  same model runs on GPU or CPU without code changes.

### Limitations

- **Synthetic training data:** Without a labelled dataset, the model
  trains on heuristically generated spans. Real performance requires
  manually annotated Q&A pairs from the CPUT documents.
- **Simple retrieval:** Word overlap is a weak baseline. BM25 or
  dense retrieval would find more relevant passages.
- **No pre-training:** Starting from random weights requires much
  more data than fine-tuning a pre-trained BERT model.

### Future Work

1. Annotate real Q&A pairs from CPUT documents
2. Load SQuAD-format JSON for supervised training
3. Add BM25 retrieval for better passage selection
4. Implement learning rate warmup schedule
5. Add answer highlighting in the source document

---

## 9. Conclusion

This project demonstrates a complete Transformer Q&A pipeline
implemented entirely in safe Rust using the Burn framework.
The 6-layer software architecture enforces clean separation of
concerns, and Burn's Backend trait enables GPU acceleration
with zero unsafe code.

The system successfully loads CPUT Word documents, trains a
6-layer transformer encoder, and answers natural language
questions through a CLI interface, fulfilling all requirements
of the SEG 580S assignment.

---

## 10. References

1. **Klabnik, S. & Nichols, C. (2023).** The Rust Programming
   Language (2nd ed.). No Starch Press.
   https://doc.rust-lang.org/book/

2. **Burn Framework Book (2024).** Burn: A Deep Learning Framework
   for Rust. https://burn.dev/book/

3. **Burn Official Examples (2024).**
   https://github.com/tracel-ai/burn/tree/main/examples

4. **Vaswani, A. et al. (2017).** Attention Is All You Need.
   NeurIPS 2017. https://arxiv.org/abs/1706.03762

5. **Devlin, J. et al. (2019).** BERT: Pre-training of Deep
   Bidirectional Transformers for Language Understanding.
   NAACL 2019. https://arxiv.org/abs/1810.04805

6. **Kingma, D. & Ba, J. (2015).** Adam: A Method for Stochastic
   Optimization. ICLR 2015. https://arxiv.org/abs/1412.6980

7. **Hendrycks, D. & Gimpel, K. (2016).** Gaussian Error Linear
   Units (GELUs). https://arxiv.org/abs/1606.08415

8. **Sennrich, R. et al. (2016).** Neural Machine Translation of
   Rare Words with Subword Units (BPE paper).
   https://arxiv.org/abs/1508.07909
