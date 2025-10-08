//! Integration tests for unified cache architecture
//!
//! Validates that RoPE and Attention work correctly with shared cache
//!
//! Created by: TEAM-008

use llorch_candled::cache::Cache;
use llorch_candled::layers::rope::apply_rope;
use llorch_candled::layers::attention::Attention;
use candle_core::{Tensor, Device, Result as CandleResult};

#[test]
fn test_rope_with_unified_cache() -> CandleResult<()> {
    let device = Device::Cpu;
    let cache = Cache::new(32, 128, 4096, 10000.0, &device)?;
    
    let q = Tensor::randn(0f32, 1.0, (1, 10, 32, 128), &device)?;
    let k = Tensor::randn(0f32, 1.0, (1, 10, 32, 128), &device)?;
    
    let (q_rot, k_rot) = apply_rope(&q, &k, 0, &cache)?;
    
    assert_eq!(q_rot.dims(), q.dims());
    assert_eq!(k_rot.dims(), k.dims());
    
    // Verify no NaN
    let q_vec = q_rot.flatten_all()?.to_vec1::<f32>()?;
    assert!(q_vec.iter().all(|&v| !v.is_nan()));
    
    Ok(())
}

#[test]
fn test_attention_with_unified_cache() -> CandleResult<()> {
    let device = Device::Cpu;
    let mut cache = Cache::new(32, 128, 4096, 10000.0, &device)?;
    
    let hidden_size = 4096;
    let n_heads = 32;
    
    // Create attention layer
    let q_weight = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), &device)?;
    let k_weight = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), &device)?;
    let v_weight = Tensor::randn(0f32, 1.0, (hidden_size, hidden_size), &device)?;
    
    let attention = Attention::new(q_weight, k_weight, v_weight, n_heads, &device)?;
    
    // Forward pass with cache
    let input = Tensor::randn(0f32, 1.0, (1, 10, hidden_size), &device)?;
    let (q, k, v) = attention.qkv().forward(&input)?;
    
    let output = attention.forward(&q, &k, &v, true)?;
    
    assert_eq!(output.dims(), &[1, 10, hidden_size]);
    
    // Verify no NaN
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_vec.iter().all(|&v| !v.is_nan()));
    
    Ok(())
}

#[test]
fn test_cache_reset() -> CandleResult<()> {
    let device = Device::Cpu;
    let mut cache = Cache::new(32, 128, 4096, 10000.0, &device)?;
    
    // Use cache
    let _ = cache.causal_mask(10)?;
    
    // Reset should clear KV but keep RoPE/masks
    cache.reset();
    
    // Should still work
    let (cos, sin) = cache.rope_values(0, 10)?;
    assert_eq!(cos.dims(), &[10, 64]);
    assert_eq!(sin.dims(), &[10, 64]);
    
    Ok(())
}

#[test]
fn test_rope_and_attention_together() -> CandleResult<()> {
    let device = Device::Cpu;
    
    let hidden_size = 512;
    let n_heads = 8;
    let head_dim = hidden_size / n_heads;  // 64
    let seq_len = 16;
    
    // Create cache with correct head_dim
    let mut cache = Cache::new(32, head_dim, 4096, 10000.0, &device)?;
    
    // Create attention layer
    let q_weight = Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device)?;
    let k_weight = Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device)?;
    let v_weight = Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device)?;
    
    let attention = Attention::new(q_weight, k_weight, v_weight, n_heads, &device)?;
    
    // Input
    let input = Tensor::randn(0f32, 1.0, (1, seq_len, hidden_size), &device)?;
    
    // QKV projection
    let (q, k, v) = attention.qkv().forward(&input)?;
    
    // Apply RoPE
    let (q_rot, k_rot) = apply_rope(&q, &k, 0, &cache)?;
    
    // Attention with rotated Q, K
    let output = attention.forward(&q_rot, &k_rot, &v, true)?;
    
    assert_eq!(output.dims(), &[1, seq_len, hidden_size]);
    
    // Verify no NaN
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_vec.iter().all(|&v| !v.is_nan()));
    
    Ok(())
}

#[test]
fn test_multiple_sequence_lengths() -> CandleResult<()> {
    let device = Device::Cpu;
    let mut cache = Cache::new(32, 128, 4096, 10000.0, &device)?;
    
    // Test different sequence lengths use cached masks correctly
    for seq_len in [8, 16, 32, 64] {
        let mask = cache.causal_mask(seq_len)?;
        assert_eq!(mask.dims(), &[seq_len, seq_len]);
    }
    
    // Second access should use cached values
    for seq_len in [8, 16, 32, 64] {
        let mask = cache.causal_mask(seq_len)?;
        assert_eq!(mask.dims(), &[seq_len, seq_len]);
    }
    
    Ok(())
}
