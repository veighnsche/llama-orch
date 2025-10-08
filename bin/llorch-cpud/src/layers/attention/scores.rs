//! Attention Score Computation
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 4 (Attention Scores)

use ndarray::{s, Array2, Array3, Array4, Axis};

/// Attention Scores
///
/// Computes scaled dot-product attention scores: (Q @ K^T) / sqrt(head_dim)
pub struct AttentionScores {
    /// Scale factor: 1 / sqrt(head_dim)
    scale: f32,
    head_dim: usize,
}

impl AttentionScores {
    /// Create new attention scores layer
    ///
    /// # Arguments
    /// * `head_dim` - Dimension per head (e.g., 64)
    pub fn new(head_dim: usize) -> Self {
        let scale = (head_dim as f32).sqrt();
        Self { scale, head_dim }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `q` - Query [seq_q, n_heads, head_dim]
    /// * `k` - Key [seq_k, n_heads, head_dim]
    /// * `mask` - Optional causal mask [1, 1, seq_q, seq_k]
    ///
    /// # Returns
    /// Attention scores [n_heads, seq_q, seq_k]
    pub fn forward(
        &self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        mask: Option<&Array2<f32>>,
    ) -> Array3<f32> {
        let seq_q = q.shape()[0];
        let seq_k = k.shape()[0];
        let n_heads = q.shape()[1];
        
        // Validate dimensions
        assert_eq!(q.shape()[2], self.head_dim, "Q head_dim mismatch");
        assert_eq!(k.shape()[2], self.head_dim, "K head_dim mismatch");
        assert_eq!(q.shape()[1], k.shape()[1], "Q and K must have same n_heads");
        
        // Compute attention scores for each head
        // Q: [seq_q, n_heads, head_dim]
        // K: [seq_k, n_heads, head_dim]
        // Result: [n_heads, seq_q, seq_k]
        let mut scores = Array3::zeros((n_heads, seq_q, seq_k));
        
        for h in 0..n_heads {
            // Extract Q and K for this head
            let q_head = q.slice(s![.., h, ..]);  // [seq_q, head_dim]
            let k_head = k.slice(s![.., h, ..]);  // [seq_k, head_dim]
            
            // Compute Q @ K.T for this head
            // q_head: [seq_q, head_dim] @ k_head.T: [head_dim, seq_k] -> [seq_q, seq_k]
            for i in 0..seq_q {
                for j in 0..seq_k {
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_head[[i, d]] * k_head[[j, d]];
                    }
                    scores[[h, i, j]] = dot / self.scale;
                }
            }
        }
        
        // Apply causal mask if provided
        if let Some(mask) = mask {
            for h in 0..n_heads {
                for i in 0..seq_q {
                    for j in 0..seq_k {
                        scores[[h, i, j]] += mask[[i, j]];
                    }
                }
            }
        }
        
        scores
    }
}

/// Create causal mask
///
/// Returns mask [1, 1, seq, seq] where future positions are -inf
pub fn _create_causal_mask(seq_len: usize) -> Array2<f32> {
    let mut mask = Array2::zeros((seq_len, seq_len));

    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[[i, j]] = f32::NEG_INFINITY;
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask() {
        let mask = _create_causal_mask(3);

        // Diagonal and lower triangle should be 0
        assert_eq!(mask[[0, 0]], 0.0);
        assert_eq!(mask[[1, 0]], 0.0);
        assert_eq!(mask[[1, 1]], 0.0);

        // Upper triangle should be -inf
        assert_eq!(mask[[0, 1]], f32::NEG_INFINITY);
        assert_eq!(mask[[0, 2]], f32::NEG_INFINITY);
        assert_eq!(mask[[1, 2]], f32::NEG_INFINITY);
    }
}
