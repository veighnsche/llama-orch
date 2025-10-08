//! CPU Tensor Operations
//!
//! IMPORTS: ndarray only
//! Helper functions for tensor operations

use ndarray::{Array1, Array2, Array4};

/// Matrix multiplication: [batch, seq, dim1] @ [dim1, dim2] → [batch, seq, dim2]
pub fn matmul_2d_3d(x: &Array2<f32>, weight: &Array2<f32>) -> Array2<f32> {
    // TODO: Implement batched matrix multiplication
    // For 2D @ 2D: use ndarray's dot
    // For 3D @ 2D: need to handle batch dimension

    x.dot(weight)
}

/// Softmax along specified axis
pub fn softmax(x: &Array4<f32>, _axis: usize) -> Array4<f32> {
    // TODO: Implement softmax
    x.clone()
}

/// Layer normalization
pub fn layer_norm(
    x: &Array2<f32>,
    _weight: &Array1<f32>,
    _bias: &Array1<f32>,
    _eps: f32,
) -> Array2<f32> {
    // TODO: Implement layer norm
    x.clone()
}

/// GELU activation
pub fn gelu(x: &Array2<f32>) -> Array2<f32> {
    // TODO: Implement GELU
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

    x.mapv(|v| {
        // Exact GELU formula
        v * 0.5 * (1.0 + erf_approx(v / std::f32::consts::SQRT_2))
    })
}

/// Error function approximation
fn erf_approx(x: f32) -> f32 {
    // TODO: Use proper erf implementation
    // For now, use tanh approximation
    // erf(x) ≈ tanh(sqrt(π) * x)

    let sqrt_pi = std::f32::consts::PI.sqrt();
    (sqrt_pi * x).tanh()
}

/// Transpose last two dimensions
pub fn transpose_last_two(x: &Array4<f32>) -> Array4<f32> {
    // TODO: Implement transpose
    // [batch, n_heads, seq, head_dim] → [batch, n_heads, head_dim, seq]

    x.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu() {
        let x = Array2::from_elem((1, 1), 0.0);
        let y = gelu(&x);

        // GELU(0) ≈ 0
        assert!((y[[0, 0]] - 0.0).abs() < 1e-1);
    }

    #[test]
    fn test_matmul() {
        let x = Array2::ones((2, 3));
        let w = Array2::ones((3, 4));
        let y = matmul_2d_3d(&x, &w);

        assert_eq!(y.shape(), &[2, 4]);
    }
}
