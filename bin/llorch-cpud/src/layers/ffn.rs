//! Feedforward Network (MLP)
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 6 (FFN Output)
//!
//! Created by: TEAM-002

use ndarray::{Array1, Array2};

/// Feedforward Network
/// Two linear layers with GELU activation:
/// - Up projection (c_fc): dim → 4*dim
/// - Down projection (c_proj): 4*dim → dim
/// TEAM-002: Implements 4x expansion MLP with exact GELU activation
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
    /// TEAM-002: Stores weights and biases for both projections
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
    /// TEAM-002: Complete FFN implementation matching HuggingFace/PyTorch
    ///
    /// # Arguments
    /// * `x` - Input [seq, dim] from previous layer
    ///
    /// # Returns
    /// Output [seq, dim] ready for residual connection
    ///
    /// # Process (matching PyTorch GPT-2)
    /// 1. Up projection: x @ c_fc_weight + c_fc_bias (dim → 4*dim)
    /// 2. GELU activation (exact formula)
    /// 3. Down projection: x @ c_proj_weight + c_proj_bias (4*dim → dim)
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // TEAM-002: Step 1 - Up projection (dim → 4*dim)
        // PyTorch: F.linear(x, c_fc_weight.T, c_fc_bias)
        // F.linear(x, w, b) computes: x @ w.T + b
        // So F.linear(x, w.T, b) = x @ (w.T).T + b = x @ w + b
        let hidden = x.dot(&self.c_fc_weight) + &self.c_fc_bias;
        
        // TEAM-002: Step 2 - GELU activation
        let hidden = gelu(&hidden);
        
        // TEAM-002: Step 3 - Down projection (4*dim → dim)
        let output = hidden.dot(&self.c_proj_weight) + &self.c_proj_bias;
        
        output
    }
}

/// GELU activation (exact formula)
///
/// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
/// TEAM-002: Implements exact GELU, not tanh approximation
fn gelu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| {
        // TEAM-002: Exact GELU formula matching PyTorch
        // Uses error function (erf) for mathematical correctness
        v * 0.5 * (1.0 + erf_approx(v / std::f32::consts::SQRT_2))
    })
}

/// Error function approximation
/// TEAM-002: High-precision erf approximation for GELU
fn erf_approx(x: f32) -> f32 {
    // Abramowitz and Stegun approximation
    // Maximum error: 1.5e-7
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
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
        // TEAM-002: Test GELU activation properties
        let x = Array2::from_elem((1, 1), 0.0);
        let y = gelu(&x);

        // GELU(0) ≈ 0
        assert!((y[[0, 0]] - 0.0).abs() < 1e-6);
        
        // GELU(1) ≈ 0.8413 (positive values mostly preserved)
        let x = Array2::from_elem((1, 1), 1.0);
        let y = gelu(&x);
        assert!((y[[0, 0]] - 0.8413).abs() < 0.01);
        
        // GELU(-1) ≈ -0.1587 (negative values dampened)
        let x = Array2::from_elem((1, 1), -1.0);
        let y = gelu(&x);
        assert!((y[[0, 0]] - (-0.1587)).abs() < 0.01);
    }
}
