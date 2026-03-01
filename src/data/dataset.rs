use burn::data::dataset::Dataset;
use serde::{Deserialize, Serialize};

/// One fully tokenised and padded training sample.
/// Sequence format: [CLS] question [SEP] context [SEP] [PAD]...
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaSample {
    pub input_ids:      Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub start_position: usize,
    pub end_position:   usize,
}

impl QaSample {
    pub fn span_length(&self) -> usize {
        self.end_position.saturating_sub(self.start_position) + 1
    }

    pub fn answer_ids(&self) -> &[u32] {
        &self.input_ids[self.start_position..=self.end_position]
    }
}

pub struct QaDataset {
    samples: Vec<QaSample>,
}

impl QaDataset {
    pub fn new(samples: Vec<QaSample>) -> Self { Self { samples } }

    pub fn sample_count(&self) -> usize { self.samples.len() }
}

impl Dataset<QaSample> for QaDataset {
    fn get(&self, index: usize) -> Option<QaSample> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}
