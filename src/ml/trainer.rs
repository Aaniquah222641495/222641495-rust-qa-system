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
    let model_cfg = TransformerQaConfig::new(
        cfg.vocab_size, cfg.max_seq_len, cfg.d_model,
        cfg.num_heads, cfg.num_layers, cfg.d_ff, cfg.dropout,
    );
    let mut model: TransformerQaModel<MyBackend> = model_cfg.init(&device);
    tracing::info!("Model ready: {} layers, d_model={}", cfg.num_layers, cfg.d_model);

    let mut optim = AdamConfig::new().with_epsilon(1e-8).init();

    let train_loader = DataLoaderBuilder::new(QaBatcher::<MyBackend>::new(device.clone()))
        .batch_size(cfg.batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(train_dataset);

    let val_loader = DataLoaderBuilder::new(QaBatcher::<MyInnerBackend>::new(device.clone()))
        .batch_size(cfg.batch_size)
        .num_workers(1)
        .build(val_dataset);

    for epoch in 1..=cfg.epochs {
        let mut train_loss_sum = 0.0f64;
        let mut train_batches  = 0usize;

        for batch in train_loader.iter() {
            let (loss, _) = model.forward_loss(
                batch.input_ids,
                batch.start_positions,
                batch.end_positions,
            );
            train_loss_sum += loss.clone().into_scalar().elem::<f64>();
            train_batches  += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(cfg.lr, model, grads);
        }

        let avg_train_loss = if train_batches > 0 {
            train_loss_sum / train_batches as f64
        } else { f64::NAN };

        // model.valid() → MyInnerBackend (dropout disabled, no autodiff overhead)
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
            let s_loss = ce.forward(output.start_logits.clone(), batch.start_positions.clone());
            let e_loss = ce.forward(output.end_logits.clone(),   batch.end_positions.clone());
            val_loss_sum += ((s_loss + e_loss) / 2.0_f64).into_scalar().elem::<f64>();
            val_batches  += 1;

            // argmax(1) → [batch, 1]; flatten to [batch] before comparing with targets
            let pred_start = output.start_logits.argmax(1).flatten::<1>(0, 1);
            let pred_end   = output.end_logits.argmax(1).flatten::<1>(0, 1);

            total_samples += batch.start_positions.dims()[0];
            correct_start += pred_start.equal(batch.start_positions).int().sum().into_scalar().elem::<i64>() as usize;
            correct_end   += pred_end.equal(batch.end_positions).int().sum().into_scalar().elem::<i64>() as usize;
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
