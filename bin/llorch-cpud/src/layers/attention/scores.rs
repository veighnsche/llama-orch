//! Attention Score Computation
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 4 (Attention Scores)

use ndarray::{Array3, Array4};

/// Attention Scores
///
/// Computes scaled dot-product attention scores: (Q @ K^T) / sqrt(head_dim)
pub struct AttentionScores {
    /// Scale factor: 1 / sqrt(head_dim)
    scale: f32,
}

impl AttentionScores {
    /// Create new attention scores layer
    ///
    /// # Arguments
    /// * `head_dim` - Dimension per head (e.g., 64)
    pub fn new(head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self { scale }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `q` - Query [batch, seq_q, n_heads, head_dim]
    /// * `k` - Key [batch, seq_k, n_heads, head_dim]
    /// * `mask` - Optional causal mask [batch, n_heads, seq_q, seq_k]
    ///
    /// # Returns
    /// Attention scores [batch, n_heads, seq_q, seq_k]
    pub fn forward(
        &self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        _mask: Option<&Array4<f32>>,
    ) -> Array4<f32> {
        // TODO: Implement attention scores (Checkpoint 4)
        // Placeholder
        let batch_seq = q.shape()[0];
        let n_heads = q.shape()[1];

        Array4::zeros((1, n_heads, batch_seq, batch_seq))
    }
}

/// Create causal mask
///
/// Returns mask [1, 1, seq, seq] where future positions are -inf
pub fn create_causal_mask(seq_len: usize) -> Array4<f32> {
    let mut mask = Array4::zeros((1, 1, seq_len, seq_len));

    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[[0, 0, i, j]] = f32::NEG_INFINITY;
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(3);

        // Diagonal and lower triangle should be 0
        assert_eq!(mask[[0, 0, 0, 0]], 0.0);
        assert_eq!(mask[[0, 0, 1, 0]], 0.0);
        assert_eq!(mask[[0, 0, 1, 1]], 0.0);

        // Upper triangle should be -inf
        assert_eq!(mask[[0, 0, 0, 1]], f32::NEG_INFINITY);
        assert_eq!(mask[[0, 0, 0, 2]], f32::NEG_INFINITY);
        assert_eq!(mask[[0, 0, 1, 2]], f32::NEG_INFINITY);
    }
}
