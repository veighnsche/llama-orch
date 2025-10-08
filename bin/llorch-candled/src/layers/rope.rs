//! RoPE (Rotary Position Embedding)
//!
//! Llama-2 uses RoPE for position encoding (not learned embeddings like GPT-2)
//! - Applied to Q and K after projection
//! - NOT applied to V
//! - Position-dependent rotation
//!
//! Checkpoint 1B validation target
//!
//! Hybrid implementation using Candle:
//! - Uses Candle tensors for GPU acceleration
//! - Implements RoPE rotation formula
//! - Automatic CUDA kernel selection when available
//!
//! Created by: TEAM-000
//! Modified by: TEAM-003 (RoPE implementation)
//! Modified by: TEAM-005 (Optimized with candle_nn::rotary_emb)

use candle_core::{Tensor, Result as CandleResult, Device};
use candle_nn::rotary_emb::rope_i;

/// RoPE (Rotary Position Embedding) for Llama-2
///
/// Applies rotary position embeddings to Q and K tensors
/// Formula: rotate dimension pairs using position-dependent cos/sin
pub struct RoPE {
    cos_cache: Tensor,  // [max_seq_len, head_dim/2]
    sin_cache: Tensor,  // [max_seq_len, head_dim/2]
    head_dim: usize,
    device: Device,
}

impl RoPE {
    /// Create new RoPE layer with precomputed cos/sin cache
    ///
    /// # Arguments
    /// * `head_dim` - Dimension per attention head (128 for Llama-2 7B)
    /// * `max_seq_len` - Maximum sequence length (4096 for Llama-2)
    /// * `theta` - Base for frequency computation (10000.0 for Llama-2)
    /// * `device` - Device to place tensors on
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32, device: &Device) -> CandleResult<Self> {
        let dim_pairs = head_dim / 2;
        
        // Compute frequencies: θ_i = theta^(-2i/head_dim)
        let freqs: Vec<f32> = (0..dim_pairs)
            .map(|i| theta.powf(-2.0 * (i as f32) / (head_dim as f32)))
            .collect();
        
        // Precompute cos and sin for all positions
        let mut cos_values = Vec::with_capacity(max_seq_len * dim_pairs);
        let mut sin_values = Vec::with_capacity(max_seq_len * dim_pairs);
        
        for pos in 0..max_seq_len {
            for &freq in &freqs {
                let angle = (pos as f32) * freq;
                cos_values.push(angle.cos());
                sin_values.push(angle.sin());
            }
        }
        
        let cos_cache = Tensor::from_vec(cos_values, (max_seq_len, dim_pairs), device)?;
        let sin_cache = Tensor::from_vec(sin_values, (max_seq_len, dim_pairs), device)?;
        
        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
            device: device.clone(),
        })
    }

    /// Apply RoPE rotation to Q and K tensors using Candle's optimized implementation
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, seq_len, n_heads, head_dim]
    /// * `k` - Key tensor [batch, seq_len, n_heads, head_dim]
    /// * `position` - Starting position in sequence
    ///
    /// # Returns
    /// * `(q_rotated, k_rotated)` - Rotated Q and K tensors (same shapes)
    ///
    /// Note: V is NOT rotated (returned unchanged elsewhere)
    /// 
    /// TEAM-005: Now uses candle_nn::rotary_emb::rope_i for GPU-accelerated rotation
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        position: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        // Get sequence length from input
        let seq_len = q.dim(1)?;
        
        // Extract cos/sin for current position range
        let cos = self.cos_cache.narrow(0, position, seq_len)?;
        let sin = self.sin_cache.narrow(0, position, seq_len)?;
        
        // rope_i expects [batch, n_heads, seq_len, head_dim]
        // Our input is [batch, seq_len, n_heads, head_dim]
        // Transpose: (0, 1, 2, 3) -> (0, 2, 1, 3) and make contiguous
        let q_transposed = q.transpose(1, 2)?.contiguous()?;
        let k_transposed = k.transpose(1, 2)?.contiguous()?;
        
        // Apply Candle's optimized RoPE (GPU kernel on CUDA/Metal, parallel CPU)
        let q_rot = rope_i(&q_transposed, &cos, &sin)?;
        let k_rot = rope_i(&k_transposed, &cos, &sin)?;
        
        // Transpose back: (0, 2, 1, 3) -> (0, 1, 2, 3) and make contiguous
        let q_rot = q_rot.transpose(1, 2)?.contiguous()?;
        let k_rot = k_rot.transpose(1, 2)?.contiguous()?;
        
        Ok((q_rot, k_rot))
    }

    /// Get the device this layer is on
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rope_shape() -> CandleResult<()> {
        let device = Device::Cpu;
        let head_dim = 128;
        let max_seq_len = 4096;
        let rope = RoPE::new(head_dim, max_seq_len, 10000.0, &device)?;

        // Create Q and K tensors [batch=1, seq_len=2, n_heads=32, head_dim=128]
        let q = Tensor::randn(0f32, 1.0, (1, 2, 32, head_dim), &device)?;
        let k = Tensor::randn(0f32, 1.0, (1, 2, 32, head_dim), &device)?;

        let (q_rot, k_rot) = rope.forward(&q, &k, 0)?;

        // Validate shapes unchanged
        assert_eq!(q_rot.dims(), &[1, 2, 32, head_dim]);
        assert_eq!(k_rot.dims(), &[1, 2, 32, head_dim]);

        Ok(())
    }

    #[test]
    fn test_rope_no_nan() -> CandleResult<()> {
        let device = Device::Cpu;
        let rope = RoPE::new(128, 4096, 10000.0, &device)?;

        let q = Tensor::randn(0f32, 1.0, (1, 2, 32, 128), &device)?;
        let k = Tensor::randn(0f32, 1.0, (1, 2, 32, 128), &device)?;

        let (q_rot, k_rot) = rope.forward(&q, &k, 0)?;

        // Check no NaN values
        let q_vec = q_rot.flatten_all()?.to_vec1::<f32>()?;
        let k_vec = k_rot.flatten_all()?.to_vec1::<f32>()?;
        
        assert!(q_vec.iter().all(|&v| !v.is_nan()));
        assert!(k_vec.iter().all(|&v| !v.is_nan()));

        Ok(())
    }
}
