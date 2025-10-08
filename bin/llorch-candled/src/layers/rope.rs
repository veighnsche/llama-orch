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

use candle_core::{Tensor, Result as CandleResult, Device, DType};

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

    /// Apply RoPE rotation to Q and K tensors
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
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        position: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        let q_rot = self.apply_rotation(q, position)?;
        let k_rot = self.apply_rotation(k, position)?;
        Ok((q_rot, k_rot))
    }

    /// Apply rotation to a single tensor (Q or K)
    fn apply_rotation(&self, x: &Tensor, position: usize) -> CandleResult<Tensor> {
        // x shape: [batch, seq_len, n_heads, head_dim]
        let shape = x.dims();
        let batch = shape[0];
        let seq_len = shape[1];
        let n_heads = shape[2];
        
        // Reshape to [batch * seq_len * n_heads, head_dim]
        let x_flat = x.flatten(0, 2)?;
        let total_tokens = batch * seq_len * n_heads;
        
        // Split into even and odd dimensions by striding
        // We need to interleave: [0,1], [2,3], [4,5], ... -> separate into [0,2,4,...] and [1,3,5,...]
        let mut x_even_parts = Vec::new();
        let mut x_odd_parts = Vec::new();
        for i in 0..(self.head_dim / 2) {
            x_even_parts.push(x_flat.narrow(1, i * 2, 1)?);
            x_odd_parts.push(x_flat.narrow(1, i * 2 + 1, 1)?);
        }
        let x_even = Tensor::cat(&x_even_parts, 1)?;
        let x_odd = Tensor::cat(&x_odd_parts, 1)?;
        
        // Get cos/sin for positions [position..position+seq_len]
        let cos = self.cos_cache.narrow(0, position, seq_len)?;
        let sin = self.sin_cache.narrow(0, position, seq_len)?;
        
        // Repeat cos/sin for each head: [seq_len, dim/2] -> [batch * seq_len * n_heads, dim/2]
        // First repeat for batch
        let cos_batch = cos.repeat((batch, 1))?;
        let sin_batch = sin.repeat((batch, 1))?;
        
        // Then repeat for heads
        let cos_expanded = cos_batch.unsqueeze(1)?.repeat((1, n_heads, 1))?.flatten(0, 1)?;
        let sin_expanded = sin_batch.unsqueeze(1)?.repeat((1, n_heads, 1))?.flatten(0, 1)?;
        
        // Apply rotation:
        // x_even' = x_even * cos - x_odd * sin
        // x_odd'  = x_even * sin + x_odd * cos
        let x_even_rot = x_even.mul(&cos_expanded)?.sub(&x_odd.mul(&sin_expanded)?)?;
        let x_odd_rot = x_even.mul(&sin_expanded)?.add(&x_odd.mul(&cos_expanded)?)?;
        
        // Interleave back: [even[0], odd[0], even[1], odd[1], ...]
        let mut rotated_parts = Vec::new();
        for i in 0..(self.head_dim / 2) {
            rotated_parts.push(x_even_rot.narrow(1, i, 1)?);
            rotated_parts.push(x_odd_rot.narrow(1, i, 1)?);
        }
        
        let x_rot_flat = Tensor::cat(&rotated_parts, 1)?;
        
        // Reshape back to original shape
        x_rot_flat.reshape(shape)
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
