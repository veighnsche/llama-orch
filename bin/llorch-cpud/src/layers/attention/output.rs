//! Attention Output Projection
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 5 (Attention Output)

use ndarray::{Array1, Array2, Array3, Array4};

/// Attention Output
///
/// Applies attention weights to values and projects back to model dimension.
pub struct AttentionOutput {
    /// Output projection weight [dim, dim]
    c_proj_weight: Array2<f32>,
    /// Output projection bias [dim]
    c_proj_bias: Array1<f32>,
}

impl AttentionOutput {
    /// Create new attention output layer
    pub fn new(weight: Array2<f32>, bias: Array1<f32>) -> Self {
        Self {
            c_proj_weight: weight,
            c_proj_bias: bias,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `attn_scores` - Attention scores [batch, n_heads, seq_q, seq_k]
    /// * `v` - Values [batch, seq_k, n_heads, head_dim]
    ///
    /// # Returns
    /// Attention output [batch, seq_q, dim]
    pub fn forward(&self, attn_scores: &Array4<f32>, v: &Array3<f32>) -> Array2<f32> {
        // TODO: Implement attention output (Checkpoint 5)
        // Placeholder
        let batch_seq = v.shape()[0];
        let dim = self.c_proj_weight.shape()[0];

        Array2::zeros((batch_seq, dim))
    }
}

/// Softmax along last dimension
fn softmax(x: &Array4<f32>) -> Array4<f32> {
    // TODO: Implement softmax
    // softmax(x) = exp(x) / sum(exp(x))
    // Apply along last dimension (seq_k)

    x.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_output_shape() {
        let dim = 1024;
        let weight = Array2::zeros((dim, dim));
        let bias = Array1::zeros(dim);

        let output_layer = AttentionOutput::new(weight, bias);

        let attn_scores = Array4::zeros((1, 16, 2, 2));
        let v = Array3::zeros((2, 16, 64)); // [batch*seq, n_heads, head_dim]

        let output = output_layer.forward(&attn_scores, &v);

        assert_eq!(output.shape()[0], 2); // batch*seq
        assert_eq!(output.shape()[1], dim);
    }
}
