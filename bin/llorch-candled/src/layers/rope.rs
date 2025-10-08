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
use candle_core::{Tensor, Result as CandleResult};
use candle_nn::rotary_emb::rope_i;
use candle_transformers::models::llama::Cache;

/// Apply RoPE rotation to Q and K tensors using Candle's cache
///
/// # Arguments
/// * `q` - Query tensor [batch, seq_len, n_heads, head_dim]
/// * `k` - Key tensor [batch, seq_len, n_heads, head_dim]
/// * `position` - Starting position in sequence
/// * `cache` - Candle's unified cache containing RoPE cos/sin
///
/// # Returns
/// * `(q_rotated, k_rotated)` - Rotated tensors
///
/// Note: V is NOT rotated (returned unchanged elsewhere)
/// 
/// TEAM-008: Now uses candle-transformers Cache
pub fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    position: usize,
    _cache: &Cache,  // TEAM-008: Not used yet (cos/sin are private)
) -> CandleResult<(Tensor, Tensor)> {
    // Get sequence length from input
    let seq_len = q.dim(1)?;
    
    // TEAM-008: Candle's Cache cos/sin fields are private
    // We need to compute RoPE values ourselves or use a different approach
    // For now, compute inline (TODO: optimize with cached values)
    let head_dim = q.dim(3)?;
    let dim_pairs = head_dim / 2;
    let theta = 10000.0f32;
    
    let freqs: Vec<f32> = (0..dim_pairs)
        .map(|i| theta.powf(-2.0 * (i as f32) / (head_dim as f32)))
        .collect();
    
    let mut cos_values = Vec::with_capacity(seq_len * dim_pairs);
    let mut sin_values = Vec::with_capacity(seq_len * dim_pairs);
    
    for pos in position..(position + seq_len) {
        for &freq in &freqs {
            let angle = (pos as f32) * freq;
            cos_values.push(angle.cos());
            sin_values.push(angle.sin());
        }
    }
    
    let cos = Tensor::from_vec(cos_values, (seq_len, dim_pairs), q.device())?;
    let sin = Tensor::from_vec(sin_values, (seq_len, dim_pairs), q.device())?;
    
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use crate::cache::Cache;

    // TEAM-008: Tests updated - need dummy Cache for signature
    #[test]
    fn test_rope_shape() -> CandleResult<()> {
        use candle_core::DType;
        use candle_transformers::models::llama::Config;
        
        let device = Device::Cpu;
        let head_dim = 128;
        
        // Create dummy config and cache
        let config = Config::config_7b_v2(false);
        let cache = Cache::new(false, DType::F32, &config, &device)?;

        // Create Q and K tensors [batch=1, seq_len=2, n_heads=32, head_dim=128]
        let q = Tensor::randn(0f32, 1.0, (1, 2, 32, head_dim), &device)?;
        let k = Tensor::randn(0f32, 1.0, (1, 2, 32, head_dim), &device)?;

        let (q_rot, k_rot) = apply_rope(&q, &k, 0, &cache)?;

        // Validate shapes unchanged
        assert_eq!(q_rot.dims(), &[1, 2, 32, head_dim]);
        assert_eq!(k_rot.dims(), &[1, 2, 32, head_dim]);

        Ok(())
    }

    #[test]
    fn test_rope_no_nan() -> CandleResult<()> {
        use candle_core::DType;
        use candle_transformers::models::llama::Config;
        
        let device = Device::Cpu;
        let config = Config::config_7b_v2(false);
        let cache = Cache::new(false, DType::F32, &config, &device)?;

        let q = Tensor::randn(0f32, 1.0, (1, 2, 32, 128), &device)?;
        let k = Tensor::randn(0f32, 1.0, (1, 2, 32, 128), &device)?;

        let (q_rot, k_rot) = apply_rope(&q, &k, 0, &cache)?;

        // Check no NaN values
        let q_vec = q_rot.flatten_all()?.to_vec1::<f32>()?;
        let k_vec = k_rot.flatten_all()?.to_vec1::<f32>()?;
        
        assert!(q_vec.iter().all(|&v| !v.is_nan()));
        assert!(k_vec.iter().all(|&v| !v.is_nan()));

        Ok(())
    }
}
