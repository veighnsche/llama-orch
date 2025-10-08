//! Feedforward Network (MLP)
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 6 (FFN Output)

use ndarray::{Array1, Array2};

/// Feedforward Network
///
/// Two linear layers with GELU activation:
/// - Up projection (c_fc): dim → 4*dim
/// - Down projection (c_proj): 4*dim → dim
pub struct FFN {
    /// Up projection weight [dim, 4*dim]
    c_fc_weight: Array2<f32>,
    /// Up projection bias [4*dim]
    c_fc_bias: Array1<f32>,
    /// Down projection weight [4*dim, dim]
    c_proj_weight: Array2<f32>,
    /// Down projection bias [dim]
    c_proj_bias: Array1<f32>,
}

impl FFN {
    /// Create new FFN layer
    pub fn new(
        fc_weight: Array2<f32>,
        fc_bias: Array1<f32>,
        proj_weight: Array2<f32>,
        proj_bias: Array1<f32>,
    ) -> Self {
        Self {
            c_fc_weight: fc_weight,
            c_fc_bias: fc_bias,
            c_proj_weight: proj_weight,
            c_proj_bias: proj_bias,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input [batch, seq, dim]
    ///
    /// # Returns
    /// Output [batch, seq, dim]
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // TODO: Implement FFN (Checkpoint 6)
        x.clone()
    }
}

/// GELU activation (exact formula)
///
/// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
fn gelu(x: &Array2<f32>) -> Array2<f32> {
    // TODO: Implement GELU
    // Use exact formula, not tanh approximation

    x.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_shape() {
        let dim = 1024;
        let fc_weight = Array2::zeros((dim, 4 * dim));
        let fc_bias = Array1::zeros(4 * dim);
        let proj_weight = Array2::zeros((4 * dim, dim));
        let proj_bias = Array1::zeros(dim);

        let ffn = FFN::new(fc_weight, fc_bias, proj_weight, proj_bias);

        let input = Array2::zeros((2, dim)); // [batch*seq, dim]
        let output = ffn.forward(&input);

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_gelu() {
        let x = Array2::from_elem((1, 1), 0.0);
        let y = gelu(&x);

        // GELU(0) ≈ 0
        assert!((y[[0, 0]] - 0.0).abs() < 1e-6);
    }
}
