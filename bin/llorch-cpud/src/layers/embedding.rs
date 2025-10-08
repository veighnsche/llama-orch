//! Token and Position Embeddings
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: Phase 2 (Embeddings)

use ndarray::{Array1, Array2};

/// Token and Position Embeddings
pub struct Embedding {
    /// Token embedding table [vocab_size, dim]
    token_embeddings: Array2<f32>,
    /// Position embedding table [max_seq_len, dim]
    position_embeddings: Array2<f32>,
}

impl Embedding {
    /// Create new embedding layer
    pub fn new(token_embeddings: Array2<f32>, position_embeddings: Array2<f32>) -> Self {
        Self {
            token_embeddings,
            position_embeddings,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs [batch, seq]
    /// * `start_pos` - Starting position for position embeddings
    ///
    /// # Returns
    /// Embeddings [batch, seq, dim]
    pub fn forward(&self, token_ids: &Array1<usize>, start_pos: usize) -> Array2<f32> {
        // TODO: Implement embeddings
        // 1. Look up token embeddings
        // 2. Look up position embeddings (start_pos..start_pos+seq_len)
        // 3. Add them together

        // Placeholder
        let seq_len = token_ids.len();
        let dim = self.token_embeddings.shape()[1];
        Array2::zeros((1, seq_len, dim))
    }
}
