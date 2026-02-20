// ============================================================
// Layer 4 — Train/Validation Splitter
// ============================================================
// Randomly shuffles samples and splits them into two sets:
//   - Training set:   used to update model weights
//   - Validation set: used to measure performance on unseen data
//
// Why do we need a validation set?
//   If we only train and test on the same data, the model
//   could memorise the answers without actually learning.
//   The validation set tells us if the model generalises
//   to data it has never seen before.
//
// Why shuffle before splitting?
//   Documents are often ordered (e.g. all calendar events
//   before all meeting notes). Without shuffling, the
//   validation set would only contain one type of document.
//   Shuffling ensures both sets have a representative mix.
//
// Split ratio: 80% training, 20% validation (configurable)
//
// Uses Fisher-Yates shuffle via rand::seq::SliceRandom
// which is the standard unbiased shuffle algorithm.
//
// Reference: Rust Book §8 (Vectors)
//            rand crate documentation

use rand::seq::SliceRandom;

/// Randomly shuffle `samples` and split into (train, validation).
///
/// # Arguments
/// * `samples`        - All available samples (consumed by this function)
/// * `train_fraction` - Proportion for training, e.g. 0.8 = 80%
///
/// # Returns
/// A tuple (train_samples, val_samples)
///
/// # Example
/// ```
/// let (train, val) = split_train_val(all_samples, 0.8);
/// // train has 80% of samples, val has 20%
/// ```
pub fn split_train_val<T>(mut samples: Vec<T>, train_fraction: f64) -> (Vec<T>, Vec<T>) {
    // Create a random number generator
    // thread_rng() gives a fast, seeded RNG per thread
    let mut rng = rand::thread_rng();

    // Fisher-Yates shuffle — every permutation is equally likely
    // This is the gold standard unbiased shuffle algorithm
    samples.shuffle(&mut rng);

    // Calculate the split index
    // e.g. 100 samples * 0.8 = 80 → first 80 are training
    let total    = samples.len();
    let split_at = ((total as f64) * train_fraction).round() as usize;

    // Clamp to valid range to avoid panics on tiny datasets
    let split_at = split_at.min(total);

    // split_off(n) removes elements [n..] from the Vec and returns them
    // After this: samples = [0..split_at], val = [split_at..total]
    let val = samples.split_off(split_at);

    tracing::debug!(
        "Dataset split: {} training, {} validation ({}% / {}%)",
        samples.len(),
        val.len(),
        (samples.len() * 100) / total.max(1),
        (val.len()     * 100) / total.max(1),
    );

    (samples, val)
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correct_split_sizes() {
        let items: Vec<usize> = (0..100).collect();
        let (train, val)      = split_train_val(items, 0.8);
        assert_eq!(train.len(), 80);
        assert_eq!(val.len(),   20);
    }

    #[test]
    fn test_all_items_preserved() {
        // No items should be lost in the split
        let items: Vec<usize> = (0..50).collect();
        let (train, val)      = split_train_val(items, 0.7);
        assert_eq!(train.len() + val.len(), 50);
    }

    #[test]
    fn test_empty_dataset() {
        let items: Vec<usize> = Vec::new();
        let (train, val)      = split_train_val(items, 0.8);
        assert!(train.is_empty());
        assert!(val.is_empty());
    }

    #[test]
    fn test_full_training_split() {
        // 1.0 fraction means everything goes to training
        let items: Vec<usize> = (0..10).collect();
        let (train, val)      = split_train_val(items, 1.0);
        assert_eq!(train.len(), 10);
        assert!(val.is_empty());
    }
}
