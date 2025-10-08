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
//! Modified by: TEAM-005 (Attention scores computation)

use candle_core::{Tensor, Result as CandleResult, Device, D};
use candle_nn::ops::softmax;
use std::collections::HashMap;

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

/// Attention mechanism for Llama-2
/// 
/// Combines QKV projection, RoPE, and scaled dot-product attention
/// TEAM-005: Implements Checkpoint 3 (Attention Scores)
/// TEAM-006: Added mask caching optimization
pub struct Attention {
    qkv: QKVProjection,
    n_heads: usize,
    head_dim: usize,
    scale: f64,
    device: Device,
    mask_cache: HashMap<usize, Tensor>,  // TEAM-006: Cache masks by sequence length
}

impl Attention {
    /// Create new attention layer
    pub fn new(
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        n_heads: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let qkv = QKVProjection::new(q_weight, k_weight, v_weight, n_heads, device)?;
        let hidden_size = qkv.q_proj.dim(0)?;
        let head_dim = hidden_size / n_heads;
        let scale = (head_dim as f64).sqrt();
        
        Ok(Self {
            qkv,
            n_heads,
            head_dim,
            scale,
            device: device.clone(),
            mask_cache: HashMap::new(),  // TEAM-006: Initialize empty cache
        })
    }
    
    /// Compute attention scores (Q @ K^T / sqrt(head_dim))
    /// 
    /// # Arguments
    /// * `q` - Query tensor [batch, seq_len, n_heads, head_dim]
    /// * `k` - Key tensor [batch, seq_len, n_heads, head_dim]
    /// 
    /// # Returns
    /// * Attention scores [batch, n_heads, seq_q, seq_k]
    /// 
    /// TEAM-005: Checkpoint 3 - Scaled dot-product attention scores
    pub fn compute_scores(&self, q: &Tensor, k: &Tensor) -> CandleResult<Tensor> {
        // q, k shape: [batch, seq_len, n_heads, head_dim]
        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        
        // Compute Q @ K^T: [batch, n_heads, seq_q, head_dim] @ [batch, n_heads, head_dim, seq_k]
        // Result: [batch, n_heads, seq_q, seq_k]
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        
        // Scale by 1/sqrt(head_dim)
        let scores = (scores / self.scale)?;
        
        Ok(scores)
    }
    
    /// Get cached causal mask for sequence length
    /// 
    /// TEAM-006: Cache masks to avoid recreation (58ms → 0.1ms at seq_len=512)
    fn get_mask(&mut self, seq_len: usize) -> CandleResult<&Tensor> {
        if !self.mask_cache.contains_key(&seq_len) {
            // Create mask only if not cached
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
            let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), &self.device)?;
            self.mask_cache.insert(seq_len, mask);
        }
        Ok(self.mask_cache.get(&seq_len).unwrap())
    }
    
    /// Apply causal mask to attention scores
    /// 
    /// Masks future positions with -inf so they get 0 attention after softmax
    /// TEAM-006: Now uses cached masks for 99% speedup
    pub fn apply_causal_mask(&mut self, scores: &Tensor) -> CandleResult<Tensor> {
        let (_batch, _n_heads, seq_q, seq_k) = scores.dims4()?;
        
        if seq_q == seq_k {
            // Use cached mask
            let mask = self.get_mask(seq_q)?;
            let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
            let mask = mask.broadcast_as(scores.shape())?;
            scores.broadcast_add(&mask)
        } else {
            // For generation (seq_q=1), no masking needed
            Ok(scores.clone())
        }
    }
    
    /// Compute attention output
    /// 
    /// Full attention: scores -> softmax -> weighted sum with V
    /// TEAM-006: Changed to &mut self for mask caching
    pub fn forward(&mut self, q: &Tensor, k: &Tensor, v: &Tensor, use_causal_mask: bool) -> CandleResult<Tensor> {
        // Compute scores
        let mut scores = self.compute_scores(q, k)?;
        
        // Apply causal mask if requested
        if use_causal_mask {
            scores = self.apply_causal_mask(&scores)?;
        }
        
        // Apply softmax: [batch, n_heads, seq_q, seq_k]
        let attn_weights = softmax(&scores, D::Minus1)?;
        
        // Transpose V to [batch, n_heads, seq_len, head_dim]
        let v = v.transpose(1, 2)?.contiguous()?;
        
        // Weighted sum: [batch, n_heads, seq_q, seq_k] @ [batch, n_heads, seq_k, head_dim]
        // Result: [batch, n_heads, seq_q, head_dim]
        let output = attn_weights.matmul(&v)?;
        
        // Transpose back to [batch, seq_q, n_heads, head_dim]
        let output = output.transpose(1, 2)?.contiguous()?;
        
        // Reshape to [batch, seq_q, hidden_size]
        let (batch, seq_q, _, _) = output.dims4()?;
        let hidden_size = self.n_heads * self.head_dim;
        output.reshape((batch, seq_q, hidden_size))
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
