# 222641495 — Rust Q&A System
## SEG 580S: Software Engineering Deep Learning Systems

A complete extractive Question-Answering system built in Rust
using the Burn deep learning framework. Trains a 6-layer
Transformer encoder on .docx documents and answers natural
language questions through a CLI.

**Student:** 222641495  
**Institution:** CPUT  
**Repository:** https://github.com/Aaniquah222641495/222641495-rust-qa-system

---

## Project Structure
```
word-doc-qa/
├── Cargo.toml                  ← dependencies
├── README.md                   ← this file
├── docs/
│   └── report.md               ← assignment report
├── data/
│   └── docx_files/             ← put your .docx files here
├── checkpoints/                ← created automatically
└── src/
    ├── main.rs                 ← entry point
    ├── cli/                    ← Layer 1: CLI
    │   ├── mod.rs
    │   └── commands.rs
    ├── application/            ← Layer 2: Use Cases
    │   ├── mod.rs
    │   ├── train_use_case.rs
    │   └── ask_use_case.rs
    ├── domain/                 ← Layer 3: Domain Types
    │   ├── mod.rs
    │   ├── document.rs
    │   ├── qa_pair.rs
    │   └── traits.rs
    ├── data/                   ← Layer 4: Data Pipeline
    │   ├── mod.rs
    │   ├── loader.rs
    │   ├── preprocessor.rs
    │   ├── chunker.rs
    │   ├── dataset.rs
    │   ├── batcher.rs
    │   └── splitter.rs
    ├── ml/                     ← Layer 5: ML Model
    │   ├── mod.rs
    │   ├── model.rs
    │   ├── trainer.rs
    │   └── inferencer.rs
    └── infra/                  ← Layer 6: Infrastructure
        ├── mod.rs
        ├── checkpoint.rs
        ├── tokenizer_store.rs
        └── metrics.rs
```

---

## Prerequisites

- Rust 1.75 or newer
- A GPU with Vulkan/Metal/DX12 drivers (or CPU fallback)

---

## Setup

### 1. Add your .docx files
```bash
cp your_documents/*.docx data/docx_files/
```

### 2. Build the project
```bash
cargo build --release
```

### 3. Train the model
```bash
cargo run --release -- train \
  --docs-dir data/docx_files \
  --checkpoint-dir checkpoints \
  --epochs 10
```

Training output:
```
Epoch   1/10 | train_loss=3.1245 | val_loss=3.0892 | start_acc=12.3% | end_acc=11.8%
Epoch   2/10 | train_loss=2.8901 | val_loss=2.8543 | start_acc=18.4% | end_acc=17.2%
...
Training complete. Checkpoint saved.
```

### 4. Ask questions
```bash
cargo run --release -- ask \
  --question "What is the date of the 2026 graduation ceremony?" \
  --docs-dir data/docx_files \
  --checkpoint-dir checkpoints
```
```bash
cargo run --release -- ask \
  --question "How many times did the HDC hold their meetings in 2024?" \
  --docs-dir data/docx_files \
  --checkpoint-dir checkpoints
```

---

## All Training Options
```
--docs-dir        data/docx_files   Directory with .docx files
--checkpoint-dir  checkpoints       Where to save model
--max-seq-len     512               Max token sequence length
--batch-size      8                 Training batch size
--epochs          10                Number of epochs
--lr              0.0002            Learning rate
--d-model         256               Transformer hidden size
--num-heads       8                 Attention heads
--num-layers      6                 Encoder layers (min 6)
--d-ff            1024              FFN inner dimension
--dropout         0.1               Dropout probability
--vocab-size      30522             Vocabulary size
```

---

## Run Tests
```bash
cargo test
```

---

## Model Architecture
```
Input: [CLS] question [SEP] context [SEP]
         │
  TokenEmbedding + PositionalEmbedding
         │
  ┌──────────────────┐
  │  Encoder Layer   │  × 6 layers
  │  - Multi-Head    │
  │    Attention     │
  │  - Add+LayerNorm │
  │  - FFN (GELU)    │
  │  - Add+LayerNorm │
  └──────────────────┘
         │
  Final LayerNorm
         │
  Linear(d_model → 2)
         │
  start_logits + end_logits
```

---

## References

- Vaswani et al. (2017) — Attention Is All You Need
- Devlin et al. (2019) — BERT
- Klabnik & Nichols (2023) — The Rust Programming Language
- Burn Framework Book (2024)
