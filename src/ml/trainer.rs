// ============================================================
// Layer 5 — Training Loop
// ============================================================
// Implements the full training and validation loop.
//
// Training loop overview (per epoch):
//
//   ┌─────────────────────────────────────────┐
//   │  TRAINING PHASE                         │
//   │  for each batch in train_loader:        │
//   │    1. Forward pass → get logits         │
//   │    2. Compute loss (CE_start + CE_end)  │
//   │    3. Backward pass → compute gradients │
//   │    4. Optimiser step → update weights   │
//   └─────────────────────────────────────────┘
//             ↓
//   ┌─────────────────────────────────────────┐
//   │  VALIDATION PHASE                       │
//   │  for each batch in val_loader:          │
//   │    1. Forward pass (no gradients)       │
//   │    2. Compute val loss                  │
//   │    3. Compute start/end accuracy        │
//   └─────────────────────────────────────────┘
//             ↓
//   Print metrics → Save checkpoint
//
// Key Burn concepts used here:
//   - AutodiffBackend: enables loss.backward()
//   - GradientsParams: typed gradient container
//   - Optimizer::step(): updates model weights
//   - model.valid(): switches off dropout for validation
//
// Reference: Burn Book §5 (Training)
//            Kingma & Ba (2015) Adam optimiser paper
//            Rust Book §13 (Iterators)

use anyhow::Result;
use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};

use crate::application::train_use_case::TrainConfig;
use crate::data::{
    batcher::QaBatcher,
    dataset::QaDataset,
};
use crate::infra::checkpoint::CheckpointManager;
use crate::ml::model::{TransformerQaConfig, TransformerQaModel};

// ─── Backend Type Alias ───────────────────────────────────────────────────────
// Autodiff<Wgpu> wraps the WGPU backend with automatic differentiation.
// This is what enables loss.backward() to work —
// Burn records all operations and replays them in reverse for gradients.
type MyBackend = burn::backend::Autodiff<burn::backend::Wgpu>;

/// Entry point called by TrainUseCase.
/// Selects the WGPU device and launches the training loop.
pub fn run_training(
    cfg:           &TrainConfig,
    train_dataset: QaDataset,
    val_dataset:   QaDataset,
    ckpt_manager:  CheckpointManager,
) -> Result<()> {
    // Get the default WGPU device (first available GPU, or CPU fallback)
    let device = burn::backend::wgpu::WgpuDevice::default();
    tracing::info!("Using WGPU device: {:?}", device);

    train_loop::<MyBackend>(cfg, train_dataset, val_dataset, ckpt_manager, device)
}

/// The actual training loop — generic over any AutodiffBackend.
/// Being generic means we could swap WGPU for NdArray (CPU)
/// without changing any of this code.
fn train_loop<B: AutodiffBackend>(
    cfg:           &TrainConfig,
    train_dataset: QaDataset,
    val_dataset:   QaDataset,
    ckpt_manager:  CheckpointManager,
    device:        B::Device,
) -> Result<()>
where
    B::Device: Clone,
{
    // ── Build Model ───────────────────────────────────────────────────────────
    // Construct the TransformerQaConfig from our TrainConfig
    // then call .init() to allocate all weight tensors
    let model_cfg = TransformerQaConfig::new(
        cfg.vocab_size,
        cfg.max_seq_len,
        cfg.d_model,
        cfg.num_heads,
        cfg.num_layers,
        cfg.d_ff,
        cfg.dropout,
    );
    let mut model: TransformerQaModel<B> = model_cfg.init(&device);
    tracing::info!(
        "Model initialised with {} encoder layers, d_model={}",
        cfg.num_layers,
        cfg.d_model
    );

    // ── Build Optimiser ───────────────────────────────────────────────────────
    // Adam (Adaptive Moment Estimation) is the standard optimiser
    // for transformer fine-tuning.
    //
    // Adam maintains per-parameter learning rates based on:
    //   m_t = β1 * m_{t-1} + (1-β1) * g_t        (1st moment / mean)
    //   v_t = β2 * v_{t-1} + (1-β2) * g_t²       (2nd moment / variance)
    //   θ_t = θ_{t-1} - lr * m_t / (√v_t + ε)   (parameter update)
    //
    // β1=0.9, β2=0.999, ε=1e-8 are the standard defaults
    let optim_cfg = AdamConfig::new().with_epsilon(1e-8);
    let mut optim: burn::optim::Adam<B> = optim_cfg.init();

    // ── Build Data Loaders ────────────────────────────────────────────────────
    // DataLoader wraps the dataset and handles:
    //   - Shuffling (training only)
    //   - Batching (calls QaBatcher.batch())
    //   - Parallel loading (num_workers)
    let train_batcher = QaBatcher::<B>::new(device.clone());
    let val_batcher   = QaBatcher::<B>::new(device.clone());

    let train_loader = DataLoaderBuilder::new(train_batcher)
        .batch_size(cfg.batch_size)
        .shuffle(42)      // seed=42 for reproducibility
        .num_workers(1)
        .build(train_dataset);

    let val_loader = DataLoaderBuilder::new(val_batcher)
        .batch_size(cfg.batch_size)
        .num_workers(1)   // no shuffle for validation
        .build(val_dataset);

    // ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in 1..=cfg.epochs {
        // ── Training Phase ────────────────────────────────────────────────────
        let mut train_loss_sum = 0.0f64;
        let mut train_batches  = 0usize;

        for batch in train_loader.iter() {
            // Forward pass + loss computation
            // forward_loss() runs the model and computes CE loss
            let (loss, _output) = model.forward_loss(
                batch.input_ids,
                batch.start_positions,
                batch.end_positions,
            );

            // Extract scalar loss value for logging BEFORE backward
            // .elem::<f64>() converts the scalar tensor to a Rust f64
            let loss_val: f64 = loss.clone().into_scalar().elem::<f64>();
            train_loss_sum += loss_val;
            train_batches  += 1;

            // Backward pass — Burn's autodiff computes gradients
            // for all parameters by replaying the forward computation
            let grads = loss.backward();

            // Wrap gradients with the model to create GradientsParams
            // This links each gradient to its corresponding parameter
            let grads = GradientsParams::from_grads(grads, &model);

            // Optimiser step — updates all model parameters using Adam
            // Returns the updated model (Burn uses move semantics)
            model = optim.step(cfg.lr, model, grads);
        }

        // Calculate average training loss for this epoch
        let avg_train_loss = if train_batches > 0 {
            train_loss_sum / train_batches as f64
        } else {
            f64::NAN
        };

        // ── Validation Phase ──────────────────────────────────────────────────
        let mut val_loss_sum  = 0.0f64;
        let mut val_batches   = 0usize;
        let mut correct_start = 0usize;
        let mut correct_end   = 0usize;
        let mut total_samples = 0usize;

        // .valid() returns a copy of the model with dropout disabled
        // This gives us deterministic outputs for fair evaluation
        let model_valid = model.clone().valid();

        for batch in val_loader.iter() {
            // Forward pass only — no gradients needed for validation
            let output = model_valid.forward(batch.input_ids.clone());

            // Compute validation loss using same CE formula as training
            let ce = burn::nn::loss::CrossEntropyLossConfig::new()
                .init(&output.start_logits.device());

            let s_loss = ce.forward(
                output.start_logits.clone(),
                batch.start_positions.clone()
            );
            let e_loss = ce.forward(
                output.end_logits.clone(),
                batch.end_positions.clone()
            );

            let batch_loss: f64 = ((s_loss + e_loss) / 2.0_f64)
                .into_scalar()
                .elem::<f64>();

            val_loss_sum += batch_loss;
            val_batches  += 1;

            // ── Accuracy Metrics ──────────────────────────────────────────────
            // Exact match: predicted position == ground truth position
            // argmax(1) finds the position with the highest logit value
            let pred_start = output.start_logits.argmax(1);
            let pred_end   = output.end_logits.argmax(1);

            let bs = batch.start_positions.dims()[0];
            total_samples += bs;

            // Compare predictions to ground truth element-wise
            // .equal() returns a Bool tensor, .int() converts to 0/1
            // .sum() counts correct predictions
            let s_correct: i64 = pred_start
                .equal(batch.start_positions)
                .int()
                .sum()
                .into_scalar()
                .elem::<i64>();

            let e_correct: i64 = pred_end
                .equal(batch.end_positions)
                .int()
                .sum()
                .into_scalar()
                .elem::<i64>();

            correct_start += s_correct as usize;
            correct_end   += e_correct as usize;
        }

        // Calculate validation metrics
        let avg_val_loss = if val_batches > 0 {
            val_loss_sum / val_batches as f64
        } else {
            f64::NAN
        };

        let start_acc = if total_samples > 0 {
            correct_start as f64 / total_samples as f64
        } else {
            0.0
        };

        let end_acc = if total_samples > 0 {
            correct_end as f64 / total_samples as f64
        } else {
            0.0
        };

        // ── Print Epoch Metrics ───────────────────────────────────────────────
        println!(
            "Epoch {:>3}/{} | train_loss={:.4} | val_loss={:.4} | start_acc={:.1}% | end_acc={:.1}%",
            epoch,
            cfg.epochs,
            avg_train_loss,
            avg_val_loss,
            start_acc * 100.0,
            end_acc   * 100.0,
        );

        // ── Save Checkpoint ───────────────────────────────────────────────────
        // Save after every epoch so we can resume if training is interrupted
        // The checkpoint manager saves weights + updates latest_epoch.json
        ckpt_manager.save_model(&model, epoch)?;
        tracing::info!("Checkpoint saved for epoch {}", epoch);
    }

    tracing::info!("Training complete!");
    Ok(())
}
