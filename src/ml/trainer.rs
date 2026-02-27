// ============================================================
// Layer 5 — Training Loop
// ============================================================
// Full train + validation loop using Burn's DataLoader and Adam.
//
// Key Burn 0.20 insight:
//   - Training uses MyBackend (Autodiff<Wgpu>) for gradients
//   - model.valid() returns model on MyInnerBackend (Wgpu)
//   - Validation batcher must also use MyInnerBackend
//   - argmax(1) returns [batch,1] so we squeeze before .equal()
//
// Reference: Burn Book §5, Kingma & Ba (2015) Adam

use anyhow::Result;
use burn::{
    data::dataloader::DataLoaderBuilder,
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
};

use crate::application::train_use_case::TrainConfig;
use crate::data::{batcher::QaBatcher, dataset::QaDataset};
use crate::infra::checkpoint::CheckpointManager;
use crate::ml::model::{TransformerQaConfig, TransformerQaModel};

type MyBackend      = burn::backend::Autodiff<burn::backend::Wgpu>;
type MyInnerBackend = burn::backend::Wgpu;

pub fn run_training(
    cfg:           &TrainConfig,
    train_dataset: QaDataset,
    val_dataset:   QaDataset,
    ckpt_manager:  CheckpointManager,
) -> Result<()> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    tracing::info!("Using WGPU device: {:?}", device);
    train_loop(cfg, train_dataset, val_dataset, ckpt_manager, device)
}

fn train_loop(
    cfg:           &TrainConfig,
    train_dataset: QaDataset,
    val_dataset:   QaDataset,
    ckpt_manager:  CheckpointManager,
    device:        burn::backend::wgpu::WgpuDevice,
) -> Result<()> {

    // ── Build model ───────────────────────────────────────────────────────────
    let model_cfg = TransformerQaConfig::new(
        cfg.vocab_size, cfg.max_seq_len, cfg.d_model,
        cfg.num_heads, cfg.num_layers, cfg.d_ff, cfg.dropout,
    );
    let mut model: TransformerQaModel<MyBackend> = model_cfg.init(&device);
    tracing::info!("Model ready: {} layers, d_model={}", cfg.num_layers, cfg.d_model);

    // ── Adam optimiser ────────────────────────────────────────────────────────
    // m = β1*m + (1-β1)*g        (mean)
    // v = β2*v + (1-β2)*g²       (variance)
    // θ = θ - lr * m / (√v + ε)  (update)
    let optim_cfg = AdamConfig::new().with_epsilon(1e-8);
    let mut optim = optim_cfg.init();

    // ── Training data loader (AutodiffBackend) ────────────────────────────────
    let train_batcher = QaBatcher::<MyBackend>::new(device.clone());
    let train_loader  = DataLoaderBuilder::new(train_batcher)
        .batch_size(cfg.batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(train_dataset);

    // ── Validation data loader (InnerBackend — no autodiff overhead) ──────────
    let val_batcher = QaBatcher::<MyInnerBackend>::new(device.clone());
    let val_loader  = DataLoaderBuilder::new(val_batcher)
        .batch_size(cfg.batch_size)
        .num_workers(1)
        .build(val_dataset);

    // ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in 1..=cfg.epochs {

        // ── Training phase ────────────────────────────────────────────────────
        let mut train_loss_sum = 0.0f64;
        let mut train_batches  = 0usize;

        for batch in train_loader.iter() {
            let (loss, _) = model.forward_loss(
                batch.input_ids,
                batch.start_positions,
                batch.end_positions,
            );

            let loss_val: f64 = loss.clone().into_scalar().elem::<f64>();
            train_loss_sum += loss_val;
            train_batches  += 1;

            // Backward pass + Adam update
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(cfg.lr, model, grads);
        }

        let avg_train_loss = if train_batches > 0 {
            train_loss_sum / train_batches as f64
        } else { f64::NAN };

        // ── Validation phase ──────────────────────────────────────────────────
        // model.valid() → TransformerQaModel<MyInnerBackend>
        // dropout disabled for deterministic evaluation
        let model_valid = model.valid();

        let mut val_loss_sum  = 0.0f64;
        let mut val_batches   = 0usize;
        let mut correct_start = 0usize;
        let mut correct_end   = 0usize;
        let mut total_samples = 0usize;

        for batch in val_loader.iter() {
            let output = model_valid.forward(batch.input_ids);

            let ce = burn::nn::loss::CrossEntropyLossConfig::new()
                .init(&output.start_logits.device());

            let s_loss = ce.forward(
                output.start_logits.clone(),
                batch.start_positions.clone(),
            );
            let e_loss = ce.forward(
                output.end_logits.clone(),
                batch.end_positions.clone(),
            );

            let batch_loss: f64 = ((s_loss + e_loss) / 2.0_f64)
                .into_scalar().elem::<f64>();
            val_loss_sum += batch_loss;
            val_batches  += 1;

            // argmax(1) returns shape [batch, 1] — squeeze to [batch]
            // before comparing with start_positions which is [batch]
            let pred_start = output.start_logits.argmax(1).flatten::<1>(0, 1);
            let pred_end   = output.end_logits.argmax(1).flatten::<1>(0, 1);

            total_samples += batch.start_positions.dims()[0];

            let s_correct: i64 = pred_start
                .equal(batch.start_positions)
                .int().sum().into_scalar().elem::<i64>();
            let e_correct: i64 = pred_end
                .equal(batch.end_positions)
                .int().sum().into_scalar().elem::<i64>();

            correct_start += s_correct as usize;
            correct_end   += e_correct as usize;
        }

        let avg_val_loss = if val_batches   > 0 { val_loss_sum / val_batches as f64 } else { f64::NAN };
        let start_acc    = if total_samples > 0 { correct_start as f64 / total_samples as f64 } else { 0.0 };
        let end_acc      = if total_samples > 0 { correct_end   as f64 / total_samples as f64 } else { 0.0 };

        println!(
            "Epoch {:>3}/{} | train_loss={:.4} | val_loss={:.4} | start_acc={:.1}% | end_acc={:.1}%",
            epoch, cfg.epochs, avg_train_loss, avg_val_loss,
            start_acc * 100.0, end_acc * 100.0,
        );

        ckpt_manager.save_model(&model, epoch)?;
        tracing::info!("Checkpoint saved for epoch {}", epoch);
    }

    tracing::info!("Training complete!");
    Ok(())
}
