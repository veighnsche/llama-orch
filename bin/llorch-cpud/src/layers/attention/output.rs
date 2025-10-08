//! Attention Output Projection
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! CHECKPOINT: 5 (Attention Output)
//!
//! Modified by: TEAM-001

use ndarray::{Array1, Array2, Array3, Array4};

/// Attention Output
///
/// Applies attention weights to values and projects back to model dimension.
/// TEAM-001: Implements complete attention mechanism with softmax, weighted sum, and projection.
pub struct AttentionOutput {
    /// Output projection weight [dim, dim]
    c_proj_weight: Array2<f32>,
    /// Output projection bias [dim]
    c_proj_bias: Array1<f32>,
}

impl AttentionOutput {
    /// Create new attention output layer
    /// TEAM-001: Stores c_proj weights and bias for output projection
    pub fn new(weight: Array2<f32>, bias: Array1<f32>) -> Self {
        Self { c_proj_weight: weight, c_proj_bias: bias }
    }

    /// Forward pass
    ///
    /// TEAM-001: Complete attention mechanism implementation matching HuggingFace/PyTorch
    ///
    /// # Arguments
    /// * `attn_scores` - Attention scores [n_heads, seq_q, seq_k] from Checkpoint 4 (BEFORE softmax)
    /// * `v` - Values [seq, n_heads, head_dim] from Checkpoint 2
    ///
    /// # Returns
    /// Attention output [seq, dim] ready for residual connection
    ///
    /// # Process (matching PyTorch GPT-2)
    /// 1. Apply softmax to attention scores
    /// 2. Transpose V to [n_heads, seq, head_dim]
    /// 3. Compute weighted sum: attn_weights @ V_transposed
    /// 4. Transpose back to [seq, n_heads, head_dim]
    /// 5. Merge heads to [seq, dim]
    /// 6. Apply output projection (c_proj)
    pub fn forward(&self, attn_scores: &Array3<f32>, v: &Array3<f32>) -> Array2<f32> {
        // TEAM-001: Extract dimensions
        let (n_heads, seq_q, seq_k) = attn_scores.dim();
        let (seq, _n_heads_v, head_dim) = v.dim();
        
        assert_eq!(seq, seq_k, "V sequence length must match attention scores seq_k");
        assert_eq!(_n_heads_v, n_heads, "V must have same number of heads as attention scores");
        
        // TEAM-001: Step 1 - Apply softmax to attention scores (per head, per query position)
        let attn_weights = softmax_3d(attn_scores);
        
        // TEAM-001: Step 2 - Transpose V from [seq, n_heads, head_dim] to [n_heads, seq, head_dim]
        // This matches PyTorch: v.transpose(1, 2)
        let mut v_t = Array3::zeros((n_heads, seq, head_dim));
        for s in 0..seq {
            for h in 0..n_heads {
                for d in 0..head_dim {
                    v_t[[h, s, d]] = v[[s, h, d]];
                }
            }
        }
        
        // TEAM-001: Step 3 - Apply attention weights to transposed values
        // attn_weights[n_heads, seq_q, seq_k] @ v_t[n_heads, seq_k, head_dim]
        // Result: [n_heads, seq_q, head_dim]
        // This matches PyTorch: torch.matmul(attn_weights, v_t)
        let mut attn_output = Array3::zeros((n_heads, seq_q, head_dim));
        
        for h in 0..n_heads {
            for i in 0..seq_q {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..seq_k {
                        // attn_weights[h, i, j] * v_t[h, j, d]
                        sum += attn_weights[[h, i, j]] * v_t[[h, j, d]];
                    }
                    attn_output[[h, i, d]] = sum;
                }
            }
        }
        
        // TEAM-001: Step 4 - Transpose back from [n_heads, seq_q, head_dim] to [seq_q, n_heads, head_dim]
        // This matches PyTorch: attn_output.transpose(1, 2)
        let mut attn_output_t = Array3::zeros((seq_q, n_heads, head_dim));
        for h in 0..n_heads {
            for i in 0..seq_q {
                for d in 0..head_dim {
                    attn_output_t[[i, h, d]] = attn_output[[h, i, d]];
                }
            }
        }
        
        // TEAM-001: Step 5 - Merge heads: reshape from [seq_q, n_heads, head_dim] to [seq_q, n_heads * head_dim]
        // This matches PyTorch: attn_output.view(batch, seq, -1)
        let dim = n_heads * head_dim;
        let mut merged = Array2::zeros((seq_q, dim));
        
        for i in 0..seq_q {
            for h in 0..n_heads {
                for d in 0..head_dim {
                    merged[[i, h * head_dim + d]] = attn_output_t[[i, h, d]];
                }
            }
        }
        
        // TEAM-001: Step 6 - Apply output projection
        // PyTorch: F.linear(attn_output, c_proj_weight.T, c_proj_bias)
        // F.linear(x, w, b) computes: x @ w.T + b
        // So F.linear(x, w.T, b) = x @ (w.T).T + b = x @ w + b
        // Therefore we do: merged @ c_proj_weight (NO transpose!)
        let output = merged.dot(&self.c_proj_weight) + &self.c_proj_bias;
        
        output
    }
}

/// Softmax along last dimension for 3D array
/// TEAM-001: Applies softmax to attention scores per head, per query position
fn softmax_3d(x: &Array3<f32>) -> Array3<f32> {
    let (n_heads, seq_q, seq_k) = x.dim();
    let mut result = Array3::zeros((n_heads, seq_q, seq_k));
    
    for h in 0..n_heads {
        for i in 0..seq_q {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..seq_k {
                max_val = max_val.max(x[[h, i, j]]);
            }
            
            // Compute exp(x - max) and sum
            let mut sum = 0.0f32;
            for j in 0..seq_k {
                let exp_val = (x[[h, i, j]] - max_val).exp();
                result[[h, i, j]] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for j in 0..seq_k {
                result[[h, i, j]] /= sum;
            }
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_output_shape() {
        // TEAM-001: Test basic shape handling
        let dim = 768; // GPT-2 base
        let n_heads = 12;
        let head_dim = 64;
        let seq = 2;
        
        let weight = Array2::zeros((dim, dim));
        let bias = Array1::zeros(dim);

        let output_layer = AttentionOutput::new(weight, bias);

        let attn_scores = Array3::zeros((n_heads, seq, seq));
        let v = Array3::zeros((seq, n_heads, head_dim));

        let output = output_layer.forward(&attn_scores, &v);

        assert_eq!(output.shape()[0], seq);
        assert_eq!(output.shape()[1], dim);
    }
    
    #[test]
    fn test_softmax_3d_sums_to_one() {
        // TEAM-001: Verify softmax produces valid probability distribution
        let attn_scores = Array3::from_shape_fn((2, 3, 4), |(h, i, j)| {
            (h as f32 + i as f32 + j as f32) * 0.1
        });
        
        let result = softmax_3d(&attn_scores);
        
        // Check each row sums to 1.0
        for h in 0..2 {
            for i in 0..3 {
                let mut sum = 0.0f32;
                for j in 0..4 {
                    sum += result[[h, i, j]];
                }
                assert!((sum - 1.0).abs() < 1e-6, "Softmax row sum = {}, expected 1.0", sum);
            }
        }
    }
}
