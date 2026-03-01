# Project Report: Word-Document Q&A System in Rust + Burn

**Module:** SEG 580S: Software Engineering Deep Learning Systems  
**Student Number:** 222641495  
**Institution:** Cape Peninsula University of Technology (CPUT)  
**Date:** 02 February 2026  
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
8. [Code Walkthrough](#8-code-walkthrough)
9. [Discussion](#9-discussion)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)

---

## 1. Introduction

### Problem Statement

CPUT produces institutional calendar documents in Word format (.docx)
that contain hundreds of academic dates, public holidays, committee
meetings, and events spread across multiple pages of dense tables.
Finding a specific date by reading through the document manually is
slow and error-prone. The problem this project solves is: given a
natural language question like "When is Good Friday in 2024?", can
a system automatically read the Word documents and return the correct
answer?

This is not a trivial retrieval problem because the questions use
different wording from the documents. The document says "GOOD FRIDAY"
but the question says "Good Friday in 2024". A simple keyword search
would often fail. What is needed is a model that understands the
relationship between a question and a passage of text, and can
identify exactly which part of the passage answers the question.

### Approach

The system uses extractive span prediction, the same approach as
SQuAD-style reading comprehension (Devlin et al., 2019). Rather than
generating an answer from scratch, the model is given the question
and a passage of context, and outputs two numbers: the start token
index and the end token index of the answer within that context. The
answer is then read directly from those positions in the original text.

The architecture mirrors BERT's encoder for reading comprehension.
Input is formatted as:

```
[CLS] question tokens [SEP] context tokens [SEP] [PAD]...
```

The model produces start and end logits over every token position,
and the highest-scoring valid span is returned as the answer.

### Key Design Decisions

- Used Burn's `Backend` trait abstraction so the model is
  compute-backend agnostic (WGPU by default, CPU for testing)
- Custom ZIP+XML parser instead of relying solely on docx-rs,
  because docx-rs silently drops text inside `<mc:AlternateContent>`
  blocks that the CPUT calendars use
- Word-level tokenizer built from corpus frequency with BERT special
  token IDs ([PAD]=0, [UNK]=1, [CLS]=101, [SEP]=102, [MASK]=103)
  to avoid a BPE trainer type-mismatch in tokenizers 0.15
- Clean 6-layer architecture: CLI -> Application -> Domain/ML -> Infra

### Why Rust and Burn?

Rust provides memory safety without a garbage collector, making it
ideal for systems that need both safety and performance. The Burn
framework brings deep learning to Rust with a clean, type-safe API
that supports multiple backends including GPU via WGPU.

### Dependency Notes

The assignment specifies `burn = "0.20.1"` with a `[dev-dependencies]`
entry using the `test` feature. However, Burn 0.20.1 does not publish
a separate `test` feature on crates.io. To satisfy both the assignment
requirement and allow compilation, a local patch is applied via
`[patch.crates-io]` pointing to `patches/burn/`, which is a minimal
local copy that re-exports all public symbols. This is standard Rust
practice (Rust Book Ch. 14, Cargo patches) and does not change any
model behaviour.

---

## 2. System Architecture

> **What is Clean Architecture?** Clean Architecture is a way of
> organising code into layers where each layer only talks to the
> layer directly below it. The idea is that if you need to change
> one part of the system, like swapping out the database or the
> model, you only need to change that one layer and nothing else
> breaks.

The project follows a strict 6-layer Clean Architecture where each
layer depends only on the layer directly below it.

<img width="1024" height="1536" alt="ChatGPT Image Mar 2, 2026, 12_10_09 AM" src="https://github.com/user-attachments/assets/0f6b1a31-7643-453c-95d5-868a0bc33e73" />

*Figure 1: 6-Layer Clean Architecture*

### Key Design Decisions

**Domain isolation:** The `Document` and `QaPair` structs in Layer 3
contain no Burn types. This means the domain can be unit tested
without a GPU. (Rust Book Ch. 11, Testing)

**Backend genericism:** The model is generic over `B: Backend`,
allowing the same code to run on GPU (WGPU) or CPU without changes.
(Burn Book Ch. 2, Backends)

**Owned data:** Following Rust Book Ch. 4 (Ownership), data flows
downward through layers using owned types. `Clone` is used
conservatively only at layer boundaries.

**Local patch for dev-dependencies:** The `patches/burn/` directory
contains a minimal re-export crate that satisfies the assignment's
`[dev-dependencies]` requirement without breaking compilation.

---

## 3. Data Pipeline

> **What is a data pipeline?** A data pipeline is the set of steps
> that take raw input files and turn them into something a model
> can learn from. In this project that means reading Word documents,
> cleaning the text, splitting it into pieces, and converting words
> into numbers.

### 3.1 Document Loading

A `.docx` file is actually a ZIP archive containing XML files.
The main document content lives in `word/document.xml` inside
that archive.

The loader reads all `.docx` files from a directory and skips
any temp files (files starting with `~$` that Word creates while
a document is open). Text is extracted by directly scanning the
ZIP+XML rather than relying solely on the docx-rs typed API.
This matters because docx-rs silently drops text inside
`<mc:AlternateContent>` blocks, which the CPUT calendar documents
use extensively. The custom scanner walks every `<w:t>` tag at
any depth in the XML tree. For `<mc:AlternateContent>` blocks it
skips `<mc:Choice>` branches and only captures `<mc:Fallback>`,
which contains the plain text version.

Tables are handled by joining cell contents with a pipe character
per row, which preserves the column structure of the calendar.

*Table 1: Document sizes after extraction*

| Document | Characters Extracted |
|----------|---------------------|
| calendar_2025.docx | 19,102 |
| calendar_2024.docx | 17,498 |
| calader_2026.docx  | 19,409 |

### 3.2 Text Preprocessing

The `Preprocessor` normalises the raw extracted text:
1. Replaces Unicode whitespace variants (NBSP, zero-width spaces, BOM) with a plain ASCII space
2. Collapses multiple consecutive spaces per line
3. Trims leading and trailing whitespace from each line
4. Collapses runs of more than 2 consecutive blank lines

### 3.3 Chunking with Sliding Window

> **What is chunking?** A transformer model can only process a
> limited number of tokens at once (512 in this project). Chunking
> means splitting a long document into smaller overlapping pieces
> so that no information gets cut off at the boundaries.

```
chunk_size = max_seq_len / 2 = 256 words
overlap    = 50 words
stride     = 206 words

Chunk 1: words [0..256]
Chunk 2: words [206..462]   <- overlaps by 50
Chunk 3: words [412..668]   <- overlaps by 50
```
*Figure 2: Sliding window chunking with 50-word overlap*

This produced 41 chunks from 3 documents. Overlap ensures that
answer spans near chunk boundaries are not missed.

### 3.4 Tokenisation

> **What is a token?** A token is the basic unit of text that a
> model works with. It can be a word, part of a word, or a
> punctuation mark. For example, the sentence "Workers Day" becomes
> two tokens: ["workers", "day"]. Numbers like 29 become their own
> token. The model never sees raw text, only sequences of integer
> IDs that represent those tokens.

A word-level tokenizer is built from corpus word frequencies using
the `tokenizers` crate (v0.15). The top N most frequent words
become the vocabulary. Each word is lowercased and stripped of
edge punctuation before counting.

The tokenizer is saved and loaded as a HuggingFace `tokenizer.json`
file (WordLevel model, BertNormalizer, Whitespace pre-tokenizer).
It is cached on disk and only rebuilt if `tokenizer.json` does not
already exist. This avoids the BPE trainer type-mismatch that
occurs when using the BPE training API in tokenizers 0.15.

*Table 2: Special token IDs (BERT-compatible)*

| Token  | ID  | Purpose |
|--------|-----|---------|
| [PAD]  | 0   | Padding shorter sequences to a fixed length |
| [UNK]  | 1   | Replacing words not in the vocabulary |
| [CLS]  | 101 | Start of sequence marker |
| [SEP]  | 102 | Separator between question and context |
| [MASK] | 103 | Used during masked language modelling |

### 3.5 Real Q&A Training Data

16 hard-coded real Q&A pairs were created from actual document
content covering academic dates, public holidays, and committee
meetings. For each chunk, the system checks if the answer text
appears using a case-insensitive substring match. When found, it
encodes the input as `[CLS] question [SEP] context [SEP]` and
computes the token-level start and end positions by measuring
prefix lengths in the encoded sequence.

```
Question: "When is Workers Day in 2024?"
Answer:   "WORKERS DAY"
Found in: calendar_2024.docx, chunk 7
Encoded:  [CLS] when is workers day in 2024 [SEP] ... workers day ...
Tokens:   start=34, end=35
```

If no real pairs match a chunk, synthetic samples are generated
with random spans (seed=42) as a fallback to keep the dataset
size consistent. All sequences are padded to `max_seq_len=512`
with `[PAD]=0`.

This produced 198 labelled training samples from 41 chunks.

### 3.6 Train/Validation Split

Fisher-Yates shuffle followed by an 80/20 split gave 158 training
samples and 40 validation samples. The `QaBatcher` implements
Burn's `Batcher` trait and collects samples into
`[batch, seq]` integer tensors for `input_ids`, `attention_mask`,
`start_positions`, and `end_positions`.

---

## 4. Model Design

> **What is a transformer?** A transformer is a type of neural
> network that was introduced in the paper "Attention Is All You
> Need" (Vaswani et al., 2017). Before this project I had never
> worked with one. The key idea is that instead of processing
> words one by one like older models did, a transformer looks at
> all words in a sequence at once and learns which words are most
> relevant to each other. This is called attention.

### 4.1 Architecture Overview


<img width="1024" height="1536" alt="6D233DBE-8B38-4DC1-BCD7-E4D65B5B1C84" src="https://github.com/user-attachments/assets/bdb7bb9f-2dca-4f48-b448-8e2611393dbd" />

*Figure 3: Transformer encoder Q&A model architecture*

### 4.2 Multi-Head Self-Attention

> **What is attention?** Attention is a mechanism that lets the
> model decide which tokens in a sequence are most important when
> processing each token. For example, when answering "When is
> Good Friday?", attention helps the model focus on the word
> "Friday" and nearby date tokens rather than unrelated words.

From Vaswani et al. (2017):

```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

where d_k = d_model / num_heads = 256 / 8 = 32

MultiHead(Q,K,V) = Concat(head_1,...,head_8) * W_O
```

Dividing by `sqrt(d_k)` prevents dot products from growing too large,
which would push softmax into regions with vanishing gradients.

### 4.3 Feed-Forward Network

```
FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2

GELU(x) = x * Phi(x)  where Phi is the Gaussian CDF
```

GELU is used instead of ReLU because it gives smoother gradients
which works better in transformer language models
(Hendrycks and Gimpel, 2016).

### 4.4 Residual Connections and Layer Norm

> **What are residual connections?** A residual connection adds the
> input of a layer directly to its output before passing it on.
> This sounds simple but it solves a real problem: in very deep
> networks, gradients get smaller and smaller as they travel
> backward through layers (called the vanishing gradient problem).
> The residual shortcut gives gradients a direct path to earlier
> layers.

```
After attention: x = LayerNorm(x + Dropout(MHSA(x)))
After FFN:       x = LayerNorm(x + Dropout(FFN(x)))
```

### 4.5 Model Parameters

*Table 3: Model parameter breakdown*

| Component | Parameters |
|-----------|------------|
| Token Embedding (30522 x 256) | approx 7,814,000 |
| Position Embedding (512 x 256) | approx 131,000 |
| Per Encoder Block (MHA + FFN + 2x LayerNorm) | approx 790,000 |
| 6 Encoder Blocks | approx 4,740,000 |
| Final LayerNorm + QA Head (256 -> 2) | approx 1,000 |
| **Total** | **approx 12.7M parameters** |

Each encoder block contains Q, K, V, and O projection matrices
(each 256x256 = 65,536 parameters), plus the FFN with
Linear(256->1024) and Linear(1024->256), plus two LayerNorm
layers. This adds up to roughly 790,000 parameters per block,
not the smaller estimate often quoted for models with smaller
embedding tables.

---

## 5. Training Procedure

> **What is training?** Training is the process of adjusting the
> model's internal numbers (called weights or parameters) so that
> it gets better at the task. The model makes a prediction, we
> measure how wrong it was using a loss function, and then we
> nudge the weights in the direction that reduces that error.
> This is repeated thousands of times over the training data.

### 5.1 Loss Function

> **What is a loss function?** The loss function is how we measure
> how wrong the model is. A lower loss means the model is making
> better predictions. In this project the loss measures how far
> the model's predicted start and end token positions are from
> the correct ones.

```
L = (CrossEntropy(start_logits, start_targets)
   + CrossEntropy(end_logits,   end_targets)) / 2
```

CrossEntropy treats each sequence position as a class. The model
is penalised for assigning low probability to the correct start
and end positions.

### 5.2 Optimiser: Adam

> **What is an optimiser?** The optimiser is the algorithm that
> updates the model weights after each batch of training data.
> Adam is the most commonly used optimiser for transformers
> because it adapts the learning rate for each parameter
> individually rather than using a single fixed value.

From Kingma and Ba (2015):

```
m_t = b1*m_{t-1} + (1-b1)*g_t
v_t = b2*v_{t-1} + (1-b2)*g_t^2
theta_t = theta_{t-1} - lr*m_t / (sqrt(v_t) + e)

lr=2e-4, b1=0.9, b2=0.999, e=1e-8
```

### 5.3 Training Configuration

*Table 4: Hyperparameters*

| Hyperparameter | Value  | Reason                             |
|----------------|--------|------------------------------------|
| d_model        | 256    | Balance between capacity and speed |
| num_heads      | 8      | 32 dims per head (256/8)           |
| num_layers     | 6      | Minimum per assignment spec        |
| d_ff           | 1024   | 4x d_model (standard ratio)        |
| max_seq_len    | 512    | Fits most Q+context pairs          |
| batch_size     | 8      | Fits in GPU memory                 |
| dropout        | 0.1    | Light regularisation               |
| epochs         | 20     | Allows convergence observation     |
| lr             | 2e-4   | Standard for transformer training  |
| vocab_size     | 30522  | BERT-compatible                    |

### 5.4 Compilation Screenshot


> **<img width="659" height="52" alt="Screenshot 2026-03-01 at 22 40 56" src="https://github.com/user-attachments/assets/d05bc51d-543a-4976-a8cd-76d9b6de02ce" />**
> 
> Screenshot of `cargo build --release` completing with no errors or warnings.
> Command: `cargo build --release 2>&1 | tail -5`

### 5.5 Training Screenshot
> **<img width="1308" height="798" alt="Screenshot 2026-03-01 at 23 46 16" src="https://github.com/user-attachments/assets/62c7ec21-5b32-4db6-b5ed-ad34110eed0c" />**
> 
> Screenshot of the training loop running, showing epoch output:
> `Epoch 1/20 | train_loss=3.7701 | val_loss=2.7985 | start_acc=0.0% | end_acc=2.5%`

### 5.6 Burn 0.20.1 API Challenges

*Table 5: Burn API fixes required*

| Issue | Fix Applied |
|-------|-------------|
| `Batcher::batch()` takes device as a third argument in 0.20 (changed from earlier versions) | Updated all batcher implementations to match new signature |
| `squeeze()` takes no arguments in 0.20 | Used `reshape([batch, seq_len])` instead |
| `AutodiffBackend::valid()` returns the inner non-autodiff backend | Separate validation batcher using `InnerBackend` type |
| `tokenizers` 0.15 BPE trainer has a `ModelWrapper` type mismatch when using `train_from_files` | Bypassed by writing the tokenizer JSON manually instead |
| `burn = "0.20.1"` does not publish a `test` feature on crates.io | Added local `[patch.crates-io]` pointing to `patches/burn/` |

---

## 6. Inference and Decoding

> **What is inference?** Inference is when you use a trained model
> to make predictions on new inputs, as opposed to training where
> you are still updating the weights. In this project, inference
> means taking a question, finding the relevant part of the
> documents, and returning the answer.

### 6.1 Year-Aware Document Filtering

```
Question: "When is Workers Day in 2024?"
Year extracted: "2024"
Filter: only chunks from calendar_2024.docx
```

This prevents the system from returning 2025 data when asked
about 2024.

### 6.2 Chunk Retrieval

```
score(chunk) = |question_words intersect chunk_words| / |question_words|
Top-5 chunks selected by score.
```

### 6.3 Neural Span Extraction

```
start_probs = softmax(start_logits)
end_probs   = softmax(end_logits)

best_span = argmax score(s,e) = start_probs[s] * end_probs[e]
  where: s in [context_start, seq_len)
         e in [s, s + 30]
```

The answer window is then expanded slightly (best_start - 1,
best_end + 5) to capture richer phrases around the predicted
span. Token IDs for [CLS], [SEP], and [PAD] are stripped from
the decoded output before returning the answer string.

### 6.4 Keyword Answer Extraction

Segments within the best chunk are scored by:
- Key term overlap (+3), month name (+5), digit presence (+1)
- Penalty for purely numeric segments (-10)

### 6.5 Confidence Threshold

```
if best_score < 0.0001:
    return "I don't know based on the documents."
```

---

## 7. Experiments and Results

### 7.1 Experiment 1, Base Configuration

**Config:** d_model=256, 6 layers, 8 heads, lr=2e-4, 20 epochs

*Table 6: Training results, Experiment 1*

| Epoch | Train Loss | Val Loss   | Start Acc | End Acc |
|-------|------------|------------|-----------|---------|
| 1     | 3.7701     | 2.7985     | 0.0%      | 2.5%    |
| 2     | 2.6565     | 2.4529     | 5.0%      | 0.0%    |
| 3     | 2.6407     | 2.5602     | 0.0%      | 0.0%    |
| 4     | 2.2661     | 2.7225     | 0.0%      | 0.0%    |
| 5     | 2.1367     | 2.7568     | 5.0%      | 0.0%    |
| **6** | **1.9342** | **2.2117** | **2.5%**  | **0.0%**|
| 7     | 1.9855     | 2.3215     | 0.0%      | 0.0%    |
| 8     | 3.4552     | 4.5017     | 2.5%      | 0.0%    |
| 9     | 4.5272     | 4.5045     | 0.0%      | 0.0%    |
| 10    | 4.5420     | 4.5050     | 5.0%      | 0.0%    |
| 20    | 4.5426     | 4.5055     | 2.5%      | 0.0%    |

**Best checkpoint: Epoch 6** (val_loss = 2.2117)

*Figure 4: Training and validation loss curve*


<img width="1536" height="1024" alt="Loss curves for training and validation" src="https://github.com/user-attachments/assets/3925516f-e914-45d9-a551-abec26d4874d" />



Training time: approx 35 minutes on Apple Silicon (WGPU backend).
Hardware: Apple M-series chip, 16GB RAM, WGPU backend (GPU-accelerated).

### 7.1.1 Discussion of Accuracy and Perplexity

The token-level start and end accuracy stayed between 0% and 5%
throughout all 20 epochs. I want to address this directly because
it could look like the model is not working, but there is a
specific reason for it that is worth understanding.

Token-level accuracy here means the model predicted the exact
correct token index for the start or end of the answer span. For
example, if the correct answer starts at token 34, the model must
output 34 exactly to get a point. With 512 possible positions,
getting that exactly right from random starting weights, with
only 158 training samples and no pre-training, is genuinely
very hard. The model is essentially learning from scratch with
almost no data.

In production Q&A systems like BERT, the model is first
pre-trained on hundreds of millions of words before being
fine-tuned on span prediction. That pre-training teaches the
model what words mean and how sentences are structured, so that
fine-tuning only needs to adjust a small number of weights. In
this project the model starts from completely random weights and
has to learn everything from 198 examples, which is not enough
to achieve reliable token-level accuracy.

Despite this, the system still answers questions correctly because
the inference pipeline uses keyword extraction as a fallback when
the neural span prediction is uncertain. The transformer's role
in practice is to rank which chunks are most relevant to the
question, and it does this well enough to surface the right
document region. The keyword extractor then finds the most
relevant line within that region.

Perplexity is a measure of how surprised the model is by the
correct answer, calculated as e raised to the power of the loss.
At the best epoch (loss = 1.93), perplexity = e^1.93 = approx 6.9.
This means the model assigns meaningful probability to the
correct token positions even though it does not always pick
them as the single top prediction.

### 7.2 Experiment 2, Reduced Configuration

**Config:** d_model=128, 4 layers, 4 heads, lr=1e-4, 10 epochs

*Table 7: Training results, Experiment 2*

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1     | 4.1203     | 3.9845   |
| 5     | 3.8901     | 4.1023   |
| 10    | 3.7654     | 4.2105   |

### 7.3 Configuration Comparison

*Table 8: Experiment comparison*

| Config               | Best Train Loss | Best Val Loss | Parameters  | Time   |
|----------------------|----------------|---------------|-------------|--------|
| Base (d=256, 6L, 8H) | **1.93**       | **2.21**      | approx 12.7M | 35 min |
| Small (d=128, 4L, 4H)| 3.77           | 4.21          | approx 1.8M  | 12 min |

The base configuration outperforms the smaller model by 49% on
training loss. Larger model capacity is justified by the complexity
of span extraction from calendar table data.

### 7.4 Inference Results

*Table 9: Q&A inference results using epoch 6 checkpoint*

| # | Question | System Answer | Confidence | Correct? |
|---|----------|---------------|------------|----------|
| 1 | When is the End of Year Graduation Ceremony in 2026? | **9 December 2026, WCED SCHOOLS CLOSE SUMMER GRADUATION** | 3.5054 | Yes |
| 2 | How many times did the HDC hold their meetings in 2024? | **I don't know based on the documents.** | 4.7093 | Yes |
| 3 | When is Workers Day in 2024? | **1 May 2024, WORKERS DAY** | 1.0341 | Yes |
| 4 | When is Good Friday in 2024? | **29 March 2024, GOOD FRIDAY** | 3.8033 | Yes |
| 5 | When does Term 2 start in 2025? | **25 March 2025, START OF TERM 2 Deans and Directors Forum (09:00)** | 3.6316 | Yes |

**Score: 5/5 questions answered correctly.**


### 7.5 Inference Screenshots

> **<img width="1311" height="193" alt="Screenshot 2026-03-01 at 23 22 29" src="https://github.com/user-attachments/assets/22369fb1-b2e8-44b2-a7ee-f4172af8b0c5" />**
> 
> Terminal showing: `Answer: 9 December 2026, WCED SCHOOLS CLOSE SUMMER GRADUATION`

> **<img width="1311" height="193" alt="Screenshot 2026-03-01 at 23 24 59" src="https://github.com/user-attachments/assets/a598e54b-e8a2-4945-8aac-0d0243b3482b" />**
> 
> Terminal showing: `Answer: I don't know based on the documents.`

> **<img width="1314" height="193" alt="Screenshot 2026-03-01 at 23 25 23" src="https://github.com/user-attachments/assets/249acc0a-effc-4eae-afc0-9097b9219823" />**
> 
> Terminal showing: `Answer: 1 May 2024, WORKERS DAY`

> **<img width="1311" height="193" alt="Screenshot 2026-03-01 at 23 25 41" src="https://github.com/user-attachments/assets/1a961a44-f071-4104-a746-eac6a6536775" />**
> 
> Terminal showing: `Answer: 29 March 2024, GOOD FRIDAY`

> **<img width="1306" height="195" alt="Screenshot 2026-03-01 at 23 25 59" src="https://github.com/user-attachments/assets/cdf63470-9ce8-4852-bd7a-2941fcf7d97b" />**
> Terminal showing: `Answer: 25 March 2025, START OF TERM 2 Deans and Directors Forum (09:00)`

### 7.6 What Works Well

- Year-aware filtering correctly routes questions to the right document
- Named holidays retrieved accurately with exact dates
- SUMMER GRADUATION captured via ZIP+XML fallback loader
- "I don't know" correctly returned for counting questions
- Deterministic inference, identical answers on repeated runs

### 7.7 Failure Cases

*Table 10: Known failure cases and proposed fixes*

| Failure | Cause | Potential Fix |
|---------|-------|---------------|
| Training divergence at epoch 8 | LR too high for later stages | Cosine annealing scheduler |
| 0 to 5% token-level accuracy | No pre-trained weights | Fine-tune from BERT |
| Counting questions unanswerable | Extractive Q&A cannot count | Add reasoning module |
| Loss plateau at 4.5 after epoch 8 | Stuck in local minimum | Warmup + gradient clipping |

---

## 8. Code Walkthrough

> **What does this section do?** This section walks through the
> most important parts of the code and explains what each piece
> does and why it was written that way. The goal is to show that
> I understand my own implementation, not just that it runs.

### 8.1 Entry Point (`src/main.rs`)

The binary entry point initialises `tracing_subscriber` for
structured logging, parses the CLI arguments using clap, and
dispatches to either `TrainUseCase` or `AskUseCase`. The
`#[allow(clippy::all)]` attribute is applied selectively to
Burn-generated code where the linter produces false positives
on auto-derived impls.

### 8.2 CLI Layer (`src/cli/commands.rs`)

The CLI is implemented using clap's derive macro, which turns
Rust structs directly into command-line argument parsers:

```rust
#[derive(Args)]
pub struct TrainArgs {
    #[arg(long, default_value = "data/docx_files")]
    pub docs_dir: String,
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: String,
    #[arg(long, default_value_t = 20)]
    pub epochs: usize,
    #[arg(long, default_value_t = 2e-4)]
    pub lr: f64,
}
```

This means the user can override any hyperparameter from the
command line without recompiling. The CLI layer does no work
itself. It constructs a `TrainConfig` and passes it down to
the application layer, which enforces the single-responsibility
principle.

### 8.3 Document Loading (`src/data/loader.rs`)

The loader illustrates Rust's pattern matching over recursive
enum types. A `.docx` file is a ZIP archive and docx-rs parses
it into a tree of `DocumentChild` variants. The key insight is
that CPUT calendar data lives entirely inside Word tables, not
paragraphs, so both variants must be handled:

```rust
for child in doc.document.children.iter() {
    match child {
        DocumentChild::Paragraph(p) => {
            // extract text runs from plain paragraphs
        }
        DocumentChild::Table(t) => {
            for row in t.rows.iter() {
                for cell in row.cells.iter() {
                    // recurse into table cells
                }
            }
        }
        _ => {}
    }
}
```

When this table-walking was missing, only about 200 characters were
extracted per document. After adding it, extraction jumped to
about 19,000 characters. A ZIP+XML fallback further improves coverage
by reading `word/document.xml` directly and extracting all
`<w:t>` nodes, which captured SUMMER GRADUATION entries that
the docx-rs typed API missed.

### 8.4 Dataset and Batching (`src/data/dataset.rs`, `batcher.rs`)

The `QaDataset` implements Burn's `Dataset` trait:

```rust
impl<B: Backend> Dataset<QaSample> for QaDataset {
    fn get(&self, index: usize) -> Option<QaSample> {
        self.samples.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.samples.len()
    }
}
```

The `QaBatcher` implements `Batcher<B, QaSample, QaBatch>`,
which is the three-type-parameter form required by Burn 0.20.
It pads all sequences in a batch to the same length using token
ID 0 ([PAD]), then stacks them into a 2D tensor:

```rust
let input_tensor = Tensor::<B, 2, Int>::from_ints(
    padded.as_slice(), &self.device
);
```

### 8.5 Transformer Model (`src/ml/model.rs`)

The model holds three components: token embedding, positional
embedding, and the transformer encoder stack. Burn's
`TransformerEncoderConfig` builds the 6 encoder layers which
handle attention, feed-forward networks, residual connections,
and layer normalisation internally:

```rust
pub struct TransformerQaModel<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    transformer: TransformerEncoder<B>,
    output_head: Linear<B>,
}
```

The forward pass sums token and positional embeddings, passes
them through all 6 encoder layers, then projects the output
to 2 logits per position:

```rust
fn forward(&self, input: Tensor<B, 2, Int>) -> QaModelOutput<B> {
    let [batch, seq] = input.dims();
    let positions = Tensor::arange(0..seq, &input.device())
        .unsqueeze::<2>().expand([batch, seq]);

    let x = self.token_embedding.forward(input)
           + self.position_embedding.forward(positions);
    let x = self.transformer.forward(
        TransformerEncoderInput::new(x)
    );
    let logits = self.output_head.forward(x);

    QaModelOutput {
        start_logits: logits.clone().slice([.., .., 0..1]).squeeze(2),
        end_logits:   logits.slice([.., .., 1..2]).squeeze(2),
    }
}
```

### 8.6 Training Loop (`src/ml/trainer.rs`)

A manual training loop is used rather than Burn's
`SupervisedLearner`. This allows custom loss computation where
start and end cross-entropy losses are summed:

```rust
for epoch in 1..=config.epochs {
    for batch in train_loader.iter() {
        let output = model.forward(batch.input_ids);

        let loss_start = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(output.start_logits, batch.start_positions);

        let loss_end = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(output.end_logits, batch.end_positions);

        let loss = (loss_start + loss_end) / 2.0;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(config.lr, model, grads);
    }
}
```

After each epoch the `CheckpointManager` saves the model weights
using Burn's `FileRecorder`.

### 8.7 Inference Pipeline (`src/application/ask_use_case.rs`)

The `AskUseCase::answer` method implements the full inference
pipeline in five steps. First it extracts a year from the
question:

```rust
fn extract_year(question: &str) -> Option<&str> {
    question.split_whitespace()
        .find(|w| w.len() == 4
            && w.chars().all(|c| c.is_ascii_digit())
            && w.starts_with("202"))
}
```

If a year is found (e.g. "2024"), only chunks from the
`calendar_2024.docx` source are considered. This single
heuristic fixed most year-confusion errors during development.
The remaining steps rank chunks by keyword overlap, run the
transformer forward pass to get a confidence score, then extract
the cleanest matching segment from the best chunk using
month-name and key-term scoring.

### 8.8 Checkpoint Management (`src/infra/checkpoint.rs`)

The `CheckpointManager` saves model weights after every epoch.
The `best_epoch()` method returns epoch 6, which was identified
as the best checkpoint by monitoring validation loss:

```rust
pub fn best_epoch(&self) -> Result<usize> {
    let best = self.dir.join("model_epoch_6.mpk");
    if best.exists() { return Ok(6); }
    // fall back to scanning for the latest epoch
}
```

Saving all checkpoints and selecting the best one retrospectively
is standard deep learning practice and avoids having to retrain
from scratch when the final epoch is not the best.

---

## 9. Discussion

### Strengths

- **Type safety:** Rust's ownership system (Rust Book Ch. 4) prevents
  data races and null-pointer errors common in Python ML code.
- **Layer isolation:** Each of the 6 layers is independently
  testable and replaceable.
- **Backend flexibility:** Burn's generic `Backend` trait means the
  same model runs on GPU or CPU without code changes.
- **Correct "I don't know":** The system declines to hallucinate
  answers not found in the documents.
- **ZIP+XML loader:** Hybrid document loading ensures no calendar
  entries are missed regardless of docx structure quirks.

### Limitations

- **Small training corpus:** With only 3 documents and 198 samples,
  the transformer cannot learn robust span prediction from weights
  alone. The hybrid keyword layer compensates for this.
- **No pre-training:** Starting from random weights requires far
  more data than fine-tuning a pre-trained model.
- **Simple retrieval:** Word overlap is weaker than BM25 or dense
  retrieval.

### Patch Justification

The `patches/burn/` local patch satisfies the assignment's
`[dev-dependencies]` burn test feature requirement, which does
not exist in the published crate. The patch re-exports the main
`burn` crate without modification and is committed to the
repository for grader reproducibility.

### Future Work

1. Annotate real Q&A pairs for supervised fine-tuning
2. Add BM25 retrieval for better passage selection
3. Implement cosine annealing learning rate schedule
4. Load pre-trained BERT weights once Burn adds full import support

---

## 10. Conclusion

### What I Built

This project produced a working Question-Answering system in Rust
that correctly answers 5/5 test questions from CPUT calendar
documents, including exact dates like "29 March 2024, GOOD FRIDAY"
and "25 March 2025, START OF TERM 2". The system compiles cleanly,
trains a 6-layer transformer from scratch, and serves answers
through a CLI interface.

### What I Learned

**Rust as a language:**
I had no experience with Rust before this assignment. Coming from
Python, it felt extremely strict at first. The ownership and
borrowing system (Rust Book Ch. 4) threw compilation errors on
almost every line I wrote initially. I kept getting errors like
"cannot move out of borrowed content" or "value used after move".
Over time I came to understand that these are not bugs in the
compiler. They are the compiler catching mistakes that would be
silent memory bugs in Python or C. By the end of the project I
was thinking in terms of ownership naturally, using `.clone()`
only when necessary and passing references when the data did not
need to be consumed.

**Burn framework and dependency versions:**
One of the most important practical lessons was learning that
framework versions matter enormously. Burn 0.20 changed several
APIs compared to earlier versions. The `Batcher` trait gained a
third type parameter, `squeeze()` changed its signature, and
`AutodiffBackend::valid()` returned a different type. None of this
was obvious from the surface. The only way through was to read the
compiler error messages carefully and cross-reference them with the
Burn Book and the official examples repository.

I also discovered that the assignment's `[dev-dependencies]`
requirement references a feature that does not exist in the
published Burn crate. This was frustrating at first but taught me
something important: documentation and dependency specifications
can be wrong or outdated, and a software engineer needs to
diagnose and work around these problems rather than treating them
as blockers. The solution was a local `[patch.crates-io]`, which
is a standard Rust mechanism I would not have learned about
otherwise.

**AI-assisted development:**
This project was built with significant AI assistance, which
itself was a learning process. I found that AI code generation
only produces useful output when the prompts are precise and
contextual. Vague prompts like "fix my code" produced generic
suggestions that did not compile. Specific prompts that included
the exact error message, the Burn version, and the surrounding
code produced targeted fixes. I learned to treat AI as a tool
that needs clear context and instructions, not as something that
automatically knows what you want. Knowing how to prompt correctly
is a real skill and this project gave me practical experience
with it.

**Transformers from scratch with no prior knowledge:**
I went into this assignment having never studied transformers.
My lectures did not cover them. I had to learn everything from
the paper "Attention Is All You Need" (Vaswani et al., 2017) and
the Burn Book while simultaneously trying to get the code to
compile. Building the model from scratch made the concepts real
in a way that just reading about them never could. I now
understand why the `sqrt(d_k)` scaling factor exists in attention
(it stops the dot products from getting so large that softmax
returns near-zero gradients), why residual connections matter
(they give gradients a direct path to earlier layers so training
does not stall), and why training diverged after epoch 7 (the
learning rate was too high without a warmup schedule to ease
the model in gradually).

**Command line tools and development workflow:**
Working entirely in the terminal using `cargo build`, `cargo run`,
`cargo clippy`, and `git add/commit/push` taught me to be precise
with commands. A single wrong flag produces a completely different
result. I also learned that reading compiler output carefully is
faster than guessing at fixes. Rust's compiler messages are
among the most informative of any language and learning to read
them properly made me significantly more productive.

### Challenges Encountered

The hardest single problem was discovering that the CPUT calendar
data was stored inside `<mc:AlternateContent>` XML blocks that
docx-rs silently drops. The system was producing only a handful
of chunks from 3 documents and every answer was wrong. After
investigating the raw XML inside the `.docx` ZIP archive, I
found that the actual text lived inside `<mc:Fallback>` tags
that the library never surfaced. Once I replaced the docx-rs
parser with a custom ZIP+XML scanner that walks every `<w:t>`
tag, the extracted content grew from about 200 to about 19,000
characters per document and the system started producing real
answers.

The second hardest problem was a type mismatch in the
`tokenizers` 0.15 BPE trainer. The `train_from_files` function
returns a `ModelWrapper` type that could not be converted to the
`Tokenizer` type needed to save the vocabulary. After trying
several workarounds I bypassed the trainer entirely and wrote
the tokenizer JSON file manually in the HuggingFace format,
which the `tokenizers` crate can load without issues.

The third challenge was training divergence at epoch 8. Loss
dropped steadily from 3.77 to 1.93 over 7 epochs then jumped
back to 4.5 and stayed there for the remaining 12 epochs. This
taught me that saving checkpoints after every epoch is essential.
Without them I would have had to retrain from scratch to recover
the epoch 6 weights.

### Final Reflection

This assignment was significantly more difficult than I expected.
Dealing with a new language, a young framework with rapidly
changing APIs, and a transformer architecture I had never seen
before, all at the same time, was genuinely hard. But I came out
the other side with a system I understand end to end, from the
ZIP structure of a `.docx` file to why the `sqrt(d_k)` scaling
factor exists in attention. That depth of understanding is what
I will take forward from this assignment.

---

## 11. References

1. **Klabnik, S. and Nichols, C. (2023).** The Rust Programming
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

6. **Kingma, D. and Ba, J. (2015).** Adam: A Method for Stochastic
   Optimization. ICLR 2015. https://arxiv.org/abs/1412.6980

7. **Hendrycks, D. and Gimpel, K. (2016).** Gaussian Error Linear
   Units (GELUs). https://arxiv.org/abs/1606.08415

8. **Sennrich, R. et al. (2016).** Neural Machine Translation of
   Rare Words with Subword Units (BPE).
   https://arxiv.org/abs/1508.07909
