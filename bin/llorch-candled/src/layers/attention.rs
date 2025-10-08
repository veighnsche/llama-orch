//! Attention mechanism for Llama-2
//!
//! Multi-head self-attention with RoPE position encoding
//! - QKV projection
//! - RoPE application
//! - Attention computation
//! - Output projection
//!
//! Checkpoint 2 validation target
//!
//! Hybrid implementation using Candle:
//! - Uses Candle tensors for GPU acceleration
//! - Implements attention mechanism
//! - Automatic CUDA kernel selection when available
//!
//! Created by: TEAM-000
//! Modified by: TEAM-004 (QKV projection)

use candle_core::{Tensor, Result as CandleResult, Device};

/// QKV Projection for Llama-2 attention
///
/// Projects input to Query, Key, Value tensors
pub struct QKVProjection {
    q_proj: Tensor,  // [hidden_size, hidden_size]
    k_proj: Tensor,  // [hidden_size, hidden_size]
    v_proj: Tensor,  // [hidden_size, hidden_size]
    n_heads: usize,
    head_dim: usize,
    device: Device,
}

impl QKVProjection {
    /// Create new QKV projection layer
    ///
    /// # Arguments
    /// * `q_weight` - Query projection weight [hidden_size, hidden_size]
    /// * `k_weight` - Key projection weight [hidden_size, hidden_size]
    /// * `v_weight` - Value projection weight [hidden_size, hidden_size]
    /// * `n_heads` - Number of attention heads (32 for Llama-2 7B)
    /// * `device` - Device to place tensors on
    pub fn new(
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        n_heads: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let hidden_size = q_weight.dim(0)?;
        let head_dim = hidden_size / n_heads;
        
        Ok(Self {
            q_proj: q_weight,
            k_proj: k_weight,
            v_proj: v_weight,
            n_heads,
            head_dim,
            device: device.clone(),
        })
    }

    /// Create from raw f32 arrays (for testing)
    pub fn from_arrays(
        q_weight: &[f32],
        k_weight: &[f32],
        v_weight: &[f32],
        hidden_size: usize,
        n_heads: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let q_proj = Tensor::from_vec(q_weight.to_vec(), (hidden_size, hidden_size), device)?;
        let k_proj = Tensor::from_vec(k_weight.to_vec(), (hidden_size, hidden_size), device)?;
        let v_proj = Tensor::from_vec(v_weight.to_vec(), (hidden_size, hidden_size), device)?;
        
        Self::new(q_proj, k_proj, v_proj, n_heads, device)
    }

    /// Forward pass: project input to Q, K, V
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    ///
    /// # Returns
    /// * `(Q, K, V)` - Each [batch, seq_len, n_heads, head_dim]
    pub fn forward(&self, x: &Tensor) -> CandleResult<(Tensor, Tensor, Tensor)> {
        // x shape: [batch, seq_len, hidden_size]
        let shape = x.dims();
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];
        
        // Flatten to [batch * seq_len, hidden_size] for matmul
        let x_flat = x.reshape((batch * seq_len, hidden_size))?;
        
        // Linear projections: [batch * seq_len, hidden_size] @ [hidden_size, hidden_size]
        let q = x_flat.matmul(&self.q_proj)?;
        let k = x_flat.matmul(&self.k_proj)?;
        let v = x_flat.matmul(&self.v_proj)?;
        
        // Reshape to [batch, seq_len, n_heads, head_dim]
        let q = q.reshape((batch, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.n_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.n_heads, self.head_dim))?;
        
        Ok((q, k, v))
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
    fn test_qkv_projection_shape() -> CandleResult<()> {
        let device = Device::Cpu;
        let hidden_size = 4096;
        let n_heads = 32;
        let head_dim = hidden_size / n_heads;

        // Create identity-like weights for testing
        let q_weight = vec![1.0f32; hidden_size * hidden_size];
        let k_weight = vec![1.0f32; hidden_size * hidden_size];
        let v_weight = vec![1.0f32; hidden_size * hidden_size];

        let qkv = QKVProjection::from_arrays(
            &q_weight,
            &k_weight,
            &v_weight,
            hidden_size,
            n_heads,
            &device,
        )?;

        // Input: [batch=1, seq_len=2, hidden_size=4096]
        let input = Tensor::randn(0f32, 1.0, (1, 2, hidden_size), &device)?;
        let (q, k, v) = qkv.forward(&input)?;

        // Validate shapes
        assert_eq!(q.dims(), &[1, 2, n_heads, head_dim]);
        assert_eq!(k.dims(), &[1, 2, n_heads, head_dim]);
        assert_eq!(v.dims(), &[1, 2, n_heads, head_dim]);

        Ok(())
    }

    #[test]
    fn test_qkv_projection_no_nan() -> CandleResult<()> {
        let device = Device::Cpu;
        let hidden_size = 128;
        let n_heads = 4;

        let q_weight = vec![0.1f32; hidden_size * hidden_size];
        let k_weight = vec![0.1f32; hidden_size * hidden_size];
        let v_weight = vec![0.1f32; hidden_size * hidden_size];

        let qkv = QKVProjection::from_arrays(
            &q_weight,
            &k_weight,
            &v_weight,
            hidden_size,
            n_heads,
            &device,
        )?;

        let input = Tensor::randn(0f32, 1.0, (1, 2, hidden_size), &device)?;
        let (q, k, v) = qkv.forward(&input)?;

        // Check no NaN values
        let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
        let k_vec = k.flatten_all()?.to_vec1::<f32>()?;
        let v_vec = v.flatten_all()?.to_vec1::<f32>()?;

        assert!(q_vec.iter().all(|&x| !x.is_nan()));
        assert!(k_vec.iter().all(|&x| !x.is_nan()));
        assert!(v_vec.iter().all(|&x| !x.is_nan()));

        Ok(())
    }
}
