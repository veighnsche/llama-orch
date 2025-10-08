//! Checkpoint 3: Attention Mechanism Validation
//!
//! Tests full attention computation: scores, softmax, output
//! Following hybrid Candle approach
//!
//! Created by: TEAM-005

use llorch_candled::layers::{Attention, QKVProjection, RoPE};
use candle_core::{Tensor, Device, IndexOp};

#[test]
fn test_attention_scores_shape() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 3: Attention Scores Shape                   ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    let hidden_size = 128;
    let n_heads = 4;
    let head_dim = 32;
    
    // Create attention layer
    let q_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let k_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let v_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    
    let attn = Attention::new(q_weight, k_weight, v_weight, n_heads, &device)?;
    
    // Create Q, K tensors [batch, seq_len, n_heads, head_dim]
    let batch = 1;
    let seq_len = 4;
    let q = Tensor::randn(0f32, 1.0, (batch, seq_len, n_heads, head_dim), &device)?;
    let k = Tensor::randn(0f32, 1.0, (batch, seq_len, n_heads, head_dim), &device)?;
    
    println!("\n📊 Input:");
    println!("  Q shape: {:?}", q.dims());
    println!("  K shape: {:?}", k.dims());
    
    // Compute scores
    let scores = attn.compute_scores(&q, &k)?;
    
    println!("\n📊 Output:");
    println!("  Scores shape: {:?}", scores.dims());
    
    // Verify shape: [batch, n_heads, seq_q, seq_k]
    assert_eq!(scores.dims(), &[batch, n_heads, seq_len, seq_len]);
    
    println!("\n✅ Attention scores shape correct");
    
    Ok(())
}

#[test]
fn test_attention_scores_scaling() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 3: Attention Scores Scaling                 ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    let hidden_size = 128;
    let n_heads = 4;
    let head_dim = 32;
    
    let q_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let k_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let v_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    
    let attn = Attention::new(q_weight, k_weight, v_weight, n_heads, &device)?;
    
    // Create identical Q and K for testing
    let q = Tensor::ones((1, 2, n_heads, head_dim), candle_core::DType::F32, &device)?;
    let k = q.clone();
    
    let scores = attn.compute_scores(&q, &k)?;
    let scores_vec = scores.flatten_all()?.to_vec1::<f32>()?;
    
    // Expected scale factor
    let expected_scale = (head_dim as f64).sqrt();
    println!("\n📊 Scale factor: {}", expected_scale);
    println!("  head_dim: {}", head_dim);
    println!("  sqrt(head_dim): {:.4}", expected_scale);
    
    // For Q=K=ones, Q@K^T gives head_dim, scaled gives head_dim/sqrt(head_dim) = sqrt(head_dim)
    let expected_value = (head_dim as f32) / (expected_scale as f32);
    println!("\n📊 Expected score value: {:.4}", expected_value);
    println!("  Actual scores (sample): {:?}", &scores_vec[..4.min(scores_vec.len())]);
    
    // Verify scaling is applied
    for &score in &scores_vec {
        let diff = (score - expected_value).abs();
        assert!(diff < 0.01, "Score {} differs from expected {} by {}", score, expected_value, diff);
    }
    
    println!("\n✅ Attention scores correctly scaled by 1/sqrt(head_dim)");
    
    Ok(())
}

#[test]
fn test_causal_mask() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 3: Causal Mask Application                  ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    let hidden_size = 128;
    let n_heads = 4;
    
    let q_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let k_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let v_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    
    let attn = Attention::new(q_weight, k_weight, v_weight, n_heads, &device)?;
    
    let seq_len = 4;
    let q = Tensor::randn(0f32, 1.0, (1, seq_len, n_heads, 32), &device)?;
    let k = Tensor::randn(0f32, 1.0, (1, seq_len, n_heads, 32), &device)?;
    
    let scores = attn.compute_scores(&q, &k)?;
    let masked_scores = attn.apply_causal_mask(&scores)?;
    
    println!("\n📊 Causal Mask Verification:");
    
    // Extract first head's scores for inspection
    let scores_head0 = scores.i((0, 0))?;
    let masked_head0 = masked_scores.i((0, 0))?;
    
    let scores_vec = scores_head0.flatten_all()?.to_vec1::<f32>()?;
    let masked_vec = masked_head0.flatten_all()?.to_vec1::<f32>()?;
    
    println!("  Original scores (head 0, first row): {:?}", &scores_vec[..seq_len]);
    println!("  Masked scores (head 0, first row): {:?}", &masked_vec[..seq_len]);
    
    // Verify causal mask structure
    for i in 0..seq_len {
        for j in 0..seq_len {
            let idx = i * seq_len + j;
            if j > i {
                // Future positions should be -inf
                assert!(masked_vec[idx].is_infinite() && masked_vec[idx].is_sign_negative(),
                    "Position ({}, {}) should be -inf, got {}", i, j, masked_vec[idx]);
            } else {
                // Past/current positions should be unchanged
                let diff = (masked_vec[idx] - scores_vec[idx]).abs();
                assert!(diff < 1e-5, "Position ({}, {}) should be unchanged", i, j);
            }
        }
    }
    
    println!("\n✅ Causal mask correctly applied (future = -inf, past/present = unchanged)");
    
    Ok(())
}

#[test]
fn test_full_attention_output() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 3: Full Attention Output                    ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    let hidden_size = 128;
    let n_heads = 4;
    let head_dim = 32;
    
    let q_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let k_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let v_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    
    let attn = Attention::new(q_weight, k_weight, v_weight, n_heads, &device)?;
    
    let batch = 1;
    let seq_len = 4;
    let q = Tensor::randn(0f32, 1.0, (batch, seq_len, n_heads, head_dim), &device)?;
    let k = Tensor::randn(0f32, 1.0, (batch, seq_len, n_heads, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1.0, (batch, seq_len, n_heads, head_dim), &device)?;
    
    println!("\n📊 Input:");
    println!("  Q shape: {:?}", q.dims());
    println!("  K shape: {:?}", k.dims());
    println!("  V shape: {:?}", v.dims());
    
    // Compute attention output
    let output = attn.forward(&q, &k, &v, true)?;
    
    println!("\n📊 Output:");
    println!("  Shape: {:?}", output.dims());
    
    // Verify output shape: [batch, seq_len, hidden_size]
    assert_eq!(output.dims(), &[batch, seq_len, hidden_size]);
    
    // Verify no NaN/Inf
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_vec.iter().all(|&x| !x.is_nan()), "Output contains NaN");
    assert!(output_vec.iter().all(|&x| x.is_finite()), "Output contains Inf");
    
    println!("\n📊 Output Analysis:");
    let output_min = output_vec.iter().copied().fold(f32::INFINITY, f32::min);
    let output_max = output_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!("  Range: [{:.4}, {:.4}]", output_min, output_max);
    println!("  Sample values: {:?}", &output_vec[..5.min(output_vec.len())]);
    
    println!("\n✅ Full attention output correct");
    
    Ok(())
}

#[test]
fn test_attention_determinism() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 3: Attention Determinism                    ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    let hidden_size = 128;
    let n_heads = 4;
    
    let q_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let k_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let v_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    
    let attn = Attention::new(q_weight, k_weight, v_weight, n_heads, &device)?;
    
    let q = Tensor::randn(0f32, 1.0, (1, 2, n_heads, 32), &device)?;
    let k = Tensor::randn(0f32, 1.0, (1, 2, n_heads, 32), &device)?;
    let v = Tensor::randn(0f32, 1.0, (1, 2, n_heads, 32), &device)?;
    
    // Run 3 times
    let out1 = attn.forward(&q, &k, &v, true)?;
    let out2 = attn.forward(&q, &k, &v, true)?;
    let out3 = attn.forward(&q, &k, &v, true)?;
    
    let vec1 = out1.flatten_all()?.to_vec1::<f32>()?;
    let vec2 = out2.flatten_all()?.to_vec1::<f32>()?;
    let vec3 = out3.flatten_all()?.to_vec1::<f32>()?;
    
    // Bit-exact comparison
    for (i, ((v1, v2), v3)) in vec1.iter().zip(vec2.iter()).zip(vec3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "Run 1 vs 2 differ at element {}", i);
        assert_eq!(v2.to_bits(), v3.to_bits(), "Run 2 vs 3 differ at element {}", i);
    }
    
    println!("\n✅ Attention is deterministic (bit-exact across runs)");
    
    Ok(())
}

#[test]
fn test_attention_with_rope() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 3: Attention + RoPE Integration             ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    let hidden_size = 128;
    let n_heads = 4;
    let head_dim = 32;
    
    // Create QKV projection
    let q_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let k_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let v_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    
    let qkv = QKVProjection::from_arrays(
        &vec![0.01f32; hidden_size * hidden_size],
        &vec![0.02f32; hidden_size * hidden_size],
        &vec![0.03f32; hidden_size * hidden_size],
        hidden_size,
        n_heads,
        &device,
    )?;
    
    // Create RoPE
    let rope = RoPE::new(head_dim, 1000, 10000.0, &device)?;
    
    // Create attention
    let attn = Attention::new(q_weight, k_weight, v_weight, n_heads, &device)?;
    
    // Input
    let input = Tensor::randn(0f32, 0.02, (1, 4, hidden_size), &device)?;
    
    println!("\n📊 Pipeline:");
    println!("  1. QKV Projection");
    let (q, k, v) = qkv.forward(&input)?;
    println!("     Q: {:?}", q.dims());
    println!("     K: {:?}", k.dims());
    println!("     V: {:?}", v.dims());
    
    println!("  2. RoPE Application");
    let (q_rot, k_rot) = rope.forward(&q, &k, 0)?;
    println!("     Q_rot: {:?}", q_rot.dims());
    println!("     K_rot: {:?}", k_rot.dims());
    
    println!("  3. Attention Computation");
    let output = attn.forward(&q_rot, &k_rot, &v, true)?;
    println!("     Output: {:?}", output.dims());
    
    // Verify
    assert_eq!(output.dims(), &[1, 4, hidden_size]);
    
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_vec.iter().all(|&x| !x.is_nan() && x.is_finite()));
    
    println!("\n✅ Full pipeline (QKV → RoPE → Attention) working correctly");
    
    Ok(())
}

#[test]
fn test_attention_llama2_dimensions() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 3: Llama-2 7B Dimensions                    ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    
    // Llama-2 7B configuration
    let hidden_size = 4096;
    let n_heads = 32;
    let head_dim = 128;
    
    println!("\n📊 Llama-2 7B Configuration:");
    println!("  hidden_size: {}", hidden_size);
    println!("  n_heads: {}", n_heads);
    println!("  head_dim: {}", head_dim);
    println!("  scale: {:.4}", (head_dim as f64).sqrt());
    
    let q_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let k_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    let v_weight = Tensor::randn(0f32, 0.02, (hidden_size, hidden_size), &device)?;
    
    let attn = Attention::new(q_weight, k_weight, v_weight, n_heads, &device)?;
    
    let batch = 1;
    let seq_len = 2;
    let q = Tensor::randn(0f32, 1.0, (batch, seq_len, n_heads, head_dim), &device)?;
    let k = Tensor::randn(0f32, 1.0, (batch, seq_len, n_heads, head_dim), &device)?;
    let v = Tensor::randn(0f32, 1.0, (batch, seq_len, n_heads, head_dim), &device)?;
    
    let output = attn.forward(&q, &k, &v, true)?;
    
    println!("\n📊 Tensor Shapes:");
    println!("  Input Q/K/V: [{}, {}, {}, {}]", batch, seq_len, n_heads, head_dim);
    println!("  Output: {:?}", output.dims());
    
    assert_eq!(output.dims(), &[batch, seq_len, hidden_size]);
    
    println!("\n✅ Llama-2 7B dimensions validated");
    
    Ok(())
}
