//! QKV Projection and Split
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 2 (QKV Projection)

use ndarray::{Array1, Array2, Array3, s};

/// QKV Projection
///
/// Projects input to combined QKV, then splits into Q, K, V.
pub struct QKVProjection {
    /// Combined QKV weight [dim, 3*dim]
    weight: Array2<f32>,
    /// Combined QKV bias [3*dim]
    bias: Array1<f32>,
    /// Number of attention heads
    n_heads: usize,
    /// Dimension per head
    head_dim: usize,
}

impl QKVProjection {
    /// Create new QKV projection
    ///
    /// # Arguments
    /// * `weight` - Weight matrix [dim, 3*dim] (may need transpose)
    /// * `bias` - Bias vector [3*dim]
    /// * `n_heads` - Number of attention heads
    pub fn new(weight: Array2<f32>, bias: Array1<f32>, n_heads: usize) -> Self {
        let dim = weight.shape()[0];
        let head_dim = dim / n_heads;

        Self {
            weight,
            bias,
            n_heads,
            head_dim,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input [batch, seq, dim]
    ///
    /// # Returns
    /// (Q, K, V) each [batch, seq, n_heads, head_dim]
    pub fn forward(&self, x: &Array2<f32>) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
        // TODO: Implement QKV projection
        // 1. Linear projection: x @ weight + bias → [batch, seq, 3*dim]
        // 2. Reshape to [batch, seq, 3, n_heads, head_dim]
        // 3. Split into Q, K, V along dim=2
        //    Q = qkv[:, :, 0, :, :]
        //    K = qkv[:, :, 1, :, :]
        //    V = qkv[:, :, 2, :, :]

        // IMPORTANT: Handle Conv1D weight transpose if needed!

        // Placeholder
        let batch = x.shape()[0];
        let seq = x.shape()[1];
        let q = Array3::zeros((batch, seq, self.n_heads, self.head_dim));
        let k = Array3::zeros((batch, seq, self.n_heads, self.head_dim));
        let v = Array3::zeros((batch, seq, self.n_heads, self.head_dim));

        (q, k, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qkv_shapes() {
        let dim = 1024;
        let n_heads = 16;
        let weight = Array2::zeros((dim, 3 * dim));
        let bias = Array1::zeros(3 * dim);

        let qkv = QKVProjection::new(weight, bias, n_heads);

        let input = Array2::zeros((1, 2, dim));
        let (q, k, v) = qkv.forward(&input);

        assert_eq!(q.shape(), &[1, 2, n_heads, dim / n_heads]);
        assert_eq!(k.shape(), &[1, 2, n_heads, dim / n_heads]);
        assert_eq!(v.shape(), &[1, 2, n_heads, dim / n_heads]);
    }
}
