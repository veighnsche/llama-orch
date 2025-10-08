//! Layer Normalization
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 1 (LayerNorm)
//!
//! Mathematical formula:
//! mean = sum(x) / N
//! variance = sum((x - mean)^2) / N  # Biased variance
//! normalized = (x - mean) / sqrt(variance + eps)
//! output = normalized * weight + bias

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
    /// * `x` - Input [batch*seq, dim] or [seq, dim]
    ///
    /// # Returns
    /// Normalized output [batch*seq, dim] or [seq, dim]
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let shape = x.shape();
        let _dim = shape[1];

        // 1. Compute mean across last dimension (axis=1)
        let mean = x.mean_axis(Axis(1)).unwrap();

        // 2. Center the input: x - mean
        // Need to broadcast mean to match x's shape
        let x_centered = x - &mean.insert_axis(Axis(1));

        // 3. Compute biased variance: mean((x - mean)^2)
        let variance = (&x_centered * &x_centered).mean_axis(Axis(1)).unwrap();

        // 4. Normalize: (x - mean) / sqrt(variance + eps)
        let std = (&variance + self.eps).mapv(f32::sqrt);
        let normalized = &x_centered / &std.insert_axis(Axis(1));

        // 5. Apply scale and bias: normalized * weight + bias
        // Broadcast weight and bias across batch dimension
        let output = &normalized * &self.weight.view().insert_axis(Axis(0))
            + self.bias.view().insert_axis(Axis(0));

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_layer_norm_shape() {
        let dim = 1024;
        let weight = Array1::ones(dim);
        let bias = Array1::zeros(dim);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = Array2::zeros((2, dim));
        let output = ln.forward(&input);

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_layer_norm_mean_variance() {
        let dim = 4;
        let weight = Array1::ones(dim);
        let bias = Array1::zeros(dim);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        // Create input with known mean and variance
        let input = Array::from_shape_vec((1, dim), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = ln.forward(&input);

        // After layer norm, mean should be ~0 and variance should be ~1
        let output_mean = output.mean_axis(Axis(1)).unwrap()[[0]];
        let output_centered = &output - output_mean;
        let output_var = (&output_centered * &output_centered).mean_axis(Axis(1)).unwrap()[[0]];

        assert!((output_mean.abs()) < 1e-5, "Mean should be ~0, got {}", output_mean);
        assert!((output_var - 1.0).abs() < 1e-4, "Variance should be ~1, got {}", output_var);
    }

    #[test]
    fn test_layer_norm_with_scale_bias() {
        let dim = 4;
        let weight = Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0]);
        let bias = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = Array::from_shape_vec((1, dim), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = ln.forward(&input);

        // With scale=2 and bias=1, output should be scaled and shifted
        // First normalize to mean=0, var=1, then scale by 2 and add 1
        assert_eq!(output.shape(), &[1, dim]);

        // Mean should be bias = 1.0 (after scaling and shifting)
        let output_mean = output.mean_axis(Axis(1)).unwrap()[[0]];
        assert!((output_mean - 1.0).abs() < 1e-4, "Mean should be ~1.0, got {}", output_mean);
    }

    #[test]
    fn test_layer_norm_batch() {
        let dim = 4;
        let batch_size = 3;
        let weight = Array1::ones(dim);
        let bias = Array1::zeros(dim);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = Array2::from_shape_fn((batch_size, dim), |(i, j)| (i * dim + j) as f32);
        let output = ln.forward(&input);

        assert_eq!(output.shape(), &[batch_size, dim]);

        // Each row should be normalized independently
        for i in 0..batch_size {
            let row = output.row(i);
            let row_mean = row.mean().unwrap();
            let row_var = row.mapv(|x| (x - row_mean).powi(2)).mean().unwrap();

            assert!((row_mean.abs()) < 1e-5, "Row {} mean should be ~0, got {}", i, row_mean);
            assert!(
                (row_var - 1.0).abs() < 1e-4,
                "Row {} variance should be ~1, got {}",
                i,
                row_var
            );
        }
    }
}
