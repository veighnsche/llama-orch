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
    /// * `x` - Input [batch*seq, dim] (flattened batch and sequence dimensions)
    ///
    /// # Returns
    /// (Q, K, V) each [batch*seq, n_heads, head_dim]
    pub fn forward(&self, x: &Array2<f32>) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
        // 1. Linear projection: x @ weight + bias → [batch*seq, 3*dim]
        let qkv_combined = x.dot(&self.weight) + &self.bias;
        
        // 2. Reshape to [batch*seq, 3*dim] → [batch*seq, 3, n_heads, head_dim]
        let batch_seq = qkv_combined.shape()[0];
        let total_dim = qkv_combined.shape()[1];
        
        // Verify dimensions
        assert_eq!(
            total_dim,
            3 * self.n_heads * self.head_dim,
            "QKV combined dimension mismatch"
        );
        
        // Reshape: [batch*seq, 3*dim] → [batch*seq, 3, n_heads, head_dim]
        let qkv_reshaped = qkv_combined
            .into_shape((batch_seq, 3, self.n_heads, self.head_dim))
            .expect("Failed to reshape QKV");
        
        // 3. Split into Q, K, V along dim=1 (the '3' dimension)
        let q = qkv_reshaped.slice(s![.., 0, .., ..]).to_owned();
        let k = qkv_reshaped.slice(s![.., 1, .., ..]).to_owned();
        let v = qkv_reshaped.slice(s![.., 2, .., ..]).to_owned();
        
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

        let input = Array2::zeros((2, dim)); // [batch*seq, dim]
        let (q, k, v) = qkv.forward(&input);

        // Placeholder returns flattened batch*seq dimension
        assert_eq!(q.shape()[0], 2); // batch*seq
        assert_eq!(q.shape()[1], n_heads);
        assert_eq!(q.shape()[2], dim / n_heads);
    }
}
