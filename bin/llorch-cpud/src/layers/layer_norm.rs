//! Layer Normalization
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 1 (LayerNorm)

use ndarray::{Array1, Array2, Axis};

/// Layer Normalization
///
/// Normalizes across the last dimension (embedding dimension).
/// Uses biased variance (divide by N, not N-1).
pub struct LayerNorm {
    /// Scale parameter [dim]
    weight: Array1<f32>,
    /// Bias parameter [dim]
    bias: Array1<f32>,
    /// Epsilon for numerical stability
    eps: f32,
}

impl LayerNorm {
    /// Create new LayerNorm layer
    ///
    /// # Arguments
    /// * `weight` - Scale parameter [dim]
    /// * `bias` - Bias parameter [dim]
    /// * `eps` - Epsilon (typically 1e-5)
    pub fn new(weight: Array1<f32>, bias: Array1<f32>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input [batch, seq, dim]
    ///
    /// # Returns
    /// Normalized output [batch, seq, dim]
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // TODO: Implement LayerNorm
        // 1. Compute mean across last dimension
        // 2. Compute biased variance (divide by N)
        // 3. Normalize: (x - mean) / sqrt(variance + eps)
        // 4. Apply scale and bias: normalized * weight + bias

        // Placeholder
        x.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_shape() {
        let dim = 1024;
        let weight = Array1::ones(dim);
        let bias = Array1::zeros(dim);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = Array2::zeros((1, 2, dim));
        let output = ln.forward(&input);

        assert_eq!(output.shape(), input.shape());
    }
}
