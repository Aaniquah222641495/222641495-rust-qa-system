use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
};
use crate::data::dataset::QaSample;

#[derive(Debug, Clone)]
pub struct QaBatch<B: Backend> {
    pub input_ids:       Tensor<B, 2, Int>,
    pub attention_mask:  Tensor<B, 2, Int>,
    pub start_positions: Tensor<B, 1, Int>,
    pub end_positions:   Tensor<B, 1, Int>,
}

#[derive(Clone, Debug)]
pub struct QaBatcher<B: Backend> {
    pub device: B::Device,
}

impl<B: Backend> QaBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

// Burn 0.20: batch() receives the device as a third argument
impl<B: Backend> Batcher<B, QaSample, QaBatch<B>> for QaBatcher<B> {
    fn batch(&self, items: Vec<QaSample>, device: &B::Device) -> QaBatch<B> {
        let batch_size = items.len();
        let seq_len    = items[0].input_ids.len();

        let input_flat: Vec<i32> = items
            .iter()
            .flat_map(|s| s.input_ids.iter().map(|&x| x as i32))
            .collect();

        let mask_flat: Vec<i32> = items
            .iter()
            .flat_map(|s| s.attention_mask.iter().map(|&x| x as i32))
            .collect();

        let starts: Vec<i32> = items.iter().map(|s| s.start_position as i32).collect();
        let ends:   Vec<i32> = items.iter().map(|s| s.end_position   as i32).collect();

        let input_ids = Tensor::<B, 1, Int>::from_ints(
            input_flat.as_slice(), device
        ).reshape([batch_size, seq_len]);

        let attention_mask = Tensor::<B, 1, Int>::from_ints(
            mask_flat.as_slice(), device
        ).reshape([batch_size, seq_len]);

        let start_positions = Tensor::<B, 1, Int>::from_ints(starts.as_slice(), device);
        let end_positions   = Tensor::<B, 1, Int>::from_ints(ends.as_slice(),   device);

        QaBatch { input_ids, attention_mask, start_positions, end_positions }
    }
}
