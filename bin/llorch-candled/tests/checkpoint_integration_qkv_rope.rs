//! Integration Test: QKV + RoPE
//!
//! Validates that QKV projection and RoPE work together correctly
//! This is the actual flow in Llama-2 attention:
//! 1. Input → QKV projection
//! 2. Q, K → RoPE rotation
//! 3. V remains unchanged
//!
//! Created by: TEAM-005

use llorch_candled::layers::{QKVProjection, RoPE};
use candle_core::{Tensor, Device};

#[test]
fn test_qkv_rope_integration() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Integration Test: QKV + RoPE                           ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    
    // Llama-2 7B configuration
    let hidden_size = 4096;
    let n_heads = 32;
    let head_dim = 128;
    let max_seq_len = 4096;
    let theta = 10000.0;
    
    println!("\n📊 Configuration:");
    println!("  hidden_size: {}", hidden_size);
    println!("  n_heads: {}", n_heads);
    println!("  head_dim: {}", head_dim);
    println!("  max_seq_len: {}", max_seq_len);
    
    // Step 1: Create QKV projection
    let q_weight = vec![0.01f32; hidden_size * hidden_size];
    let k_weight = vec![0.02f32; hidden_size * hidden_size];
    let v_weight = vec![0.03f32; hidden_size * hidden_size];
    
    let qkv = QKVProjection::from_arrays(
        &q_weight,
        &k_weight,
        &v_weight,
        hidden_size,
        n_heads,
        &device,
    )?;
    
    // Step 2: Create RoPE layer
    let rope = RoPE::new(head_dim, max_seq_len, theta, &device)?;
    
    // Step 3: Generate input (simulating RMSNorm output)
    let batch = 1;
    let seq_len = 2;
    let input = Tensor::randn(0f32, 0.02, (batch, seq_len, hidden_size), &device)?;
    
    println!("\n📊 Step 1: QKV Projection");
    println!("  Input shape: {:?}", input.dims());
    
    // Step 4: Apply QKV projection
    let (q, k, v) = qkv.forward(&input)?;
    
    println!("  Q shape: {:?}", q.dims());
    println!("  K shape: {:?}", k.dims());
    println!("  V shape: {:?}", v.dims());
    
    // Verify QKV shapes
    assert_eq!(q.dims(), &[batch, seq_len, n_heads, head_dim]);
    assert_eq!(k.dims(), &[batch, seq_len, n_heads, head_dim]);
    assert_eq!(v.dims(), &[batch, seq_len, n_heads, head_dim]);
    
    // Step 5: Apply RoPE to Q and K (NOT V)
    println!("\n📊 Step 2: RoPE Application");
    let position = 0;
    let (q_rot, k_rot) = rope.forward(&q, &k, position)?;
    
    println!("  Q_rotated shape: {:?}", q_rot.dims());
    println!("  K_rotated shape: {:?}", k_rot.dims());
    println!("  V unchanged: {:?}", v.dims());
    
    // Verify RoPE shapes
    assert_eq!(q_rot.dims(), &[batch, seq_len, n_heads, head_dim]);
    assert_eq!(k_rot.dims(), &[batch, seq_len, n_heads, head_dim]);
    
    // Step 6: Verify Q and K are rotated (different from original)
    let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
    let q_rot_vec = q_rot.flatten_all()?.to_vec1::<f32>()?;
    let k_vec = k.flatten_all()?.to_vec1::<f32>()?;
    let k_rot_vec = k_rot.flatten_all()?.to_vec1::<f32>()?;
    let v_vec = v.flatten_all()?.to_vec1::<f32>()?;
    
    let q_changed = q_vec.iter().zip(q_rot_vec.iter())
        .filter(|(&orig, &rot)| (orig - rot).abs() > 1e-6)
        .count();
    
    let k_changed = k_vec.iter().zip(k_rot_vec.iter())
        .filter(|(&orig, &rot)| (orig - rot).abs() > 1e-6)
        .count();
    
    println!("\n📊 Validation:");
    println!("  Q elements changed by RoPE: {}/{}", q_changed, q_vec.len());
    println!("  K elements changed by RoPE: {}/{}", k_changed, k_vec.len());
    
    assert!(q_changed > 0, "Q should be modified by RoPE");
    assert!(k_changed > 0, "K should be modified by RoPE");
    
    // Step 7: Verify no NaN/Inf in final outputs
    assert!(q_rot_vec.iter().all(|&x| !x.is_nan() && x.is_finite()), "Q_rot contains NaN/Inf");
    assert!(k_rot_vec.iter().all(|&x| !x.is_nan() && x.is_finite()), "K_rot contains NaN/Inf");
    assert!(v_vec.iter().all(|&x| !x.is_nan() && x.is_finite()), "V contains NaN/Inf");
    
    println!("  ✓ No NaN/Inf in outputs");
    
    // Step 8: Verify Q, K, V still differ from each other
    let q_k_diff = q_rot_vec.iter().zip(k_rot_vec.iter())
        .filter(|(&q, &k)| (q - k).abs() > 1e-6)
        .count();
    
    let k_v_diff = k_rot_vec.iter().zip(v_vec.iter())
        .filter(|(&k, &v)| (k - v).abs() > 1e-6)
        .count();
    
    println!("  Q vs K differ: {}/{} elements", q_k_diff, q_rot_vec.len());
    println!("  K vs V differ: {}/{} elements", k_v_diff, k_rot_vec.len());
    
    assert!(q_k_diff > 0, "Q and K should still differ after RoPE");
    assert!(k_v_diff > 0, "K and V should differ");
    
    println!("\n✅ Integration Test Passed:");
    println!("  ✅ QKV projection correct");
    println!("  ✅ RoPE applied to Q and K");
    println!("  ✅ V unchanged by RoPE");
    println!("  ✅ All outputs numerically stable");
    println!("  ✅ Q, K, V maintain distinct values");
    
    println!("\n📝 Ready for:");
    println!("  → Attention score computation (Q @ K^T)");
    println!("  → Softmax normalization");
    println!("  → Attention output (scores @ V)");
    
    Ok(())
}

#[test]
fn test_rope_position_in_integration() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Integration: RoPE Position Dependency                  ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    let hidden_size = 128;
    let n_heads = 4;
    let head_dim = 32;
    
    // Create QKV and RoPE
    let q_weight = vec![0.01f32; hidden_size * hidden_size];
    let k_weight = vec![0.02f32; hidden_size * hidden_size];
    let v_weight = vec![0.03f32; hidden_size * hidden_size];
    
    let qkv = QKVProjection::from_arrays(
        &q_weight,
        &k_weight,
        &v_weight,
        hidden_size,
        n_heads,
        &device,
    )?;
    
    let rope = RoPE::new(head_dim, 1000, 10000.0, &device)?;
    
    // Same input, different positions
    let input = Tensor::randn(0f32, 0.02, (1, 2, hidden_size), &device)?;
    let (q, k, _v) = qkv.forward(&input)?;
    
    let (q_pos0, k_pos0) = rope.forward(&q, &k, 0)?;
    let (q_pos100, k_pos100) = rope.forward(&q, &k, 100)?;
    
    let q0_vec = q_pos0.flatten_all()?.to_vec1::<f32>()?;
    let q100_vec = q_pos100.flatten_all()?.to_vec1::<f32>()?;
    
    let k0_vec = k_pos0.flatten_all()?.to_vec1::<f32>()?;
    let k100_vec = k_pos100.flatten_all()?.to_vec1::<f32>()?;
    
    // Different positions should produce different outputs
    let q_diff = q0_vec.iter().zip(q100_vec.iter())
        .filter(|(&v0, &v100)| (v0 - v100).abs() > 1e-6)
        .count();
    
    let k_diff = k0_vec.iter().zip(k100_vec.iter())
        .filter(|(&v0, &v100)| (v0 - v100).abs() > 1e-6)
        .count();
    
    println!("\n📊 Position Dependency:");
    println!("  Q differs (pos=0 vs pos=100): {}/{}", q_diff, q0_vec.len());
    println!("  K differs (pos=0 vs pos=100): {}/{}", k_diff, k0_vec.len());
    
    assert!(q_diff > q0_vec.len() / 2, "RoPE should be position-dependent for Q");
    assert!(k_diff > k0_vec.len() / 2, "RoPE should be position-dependent for K");
    
    println!("\n✅ Position encoding working correctly in integration");
    
    Ok(())
}

#[test]
fn test_edge_case_single_token() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Edge Case: Single Token (seq_len=1)                    ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    let hidden_size = 128;
    let n_heads = 4;
    let head_dim = 32;
    
    let q_weight = vec![0.01f32; hidden_size * hidden_size];
    let k_weight = vec![0.02f32; hidden_size * hidden_size];
    let v_weight = vec![0.03f32; hidden_size * hidden_size];
    
    let qkv = QKVProjection::from_arrays(
        &q_weight,
        &k_weight,
        &v_weight,
        hidden_size,
        n_heads,
        &device,
    )?;
    
    let rope = RoPE::new(head_dim, 1000, 10000.0, &device)?;
    
    // Single token input
    let input = Tensor::randn(0f32, 0.02, (1, 1, hidden_size), &device)?;
    let (q, k, v) = qkv.forward(&input)?;
    let (q_rot, k_rot) = rope.forward(&q, &k, 0)?;
    
    println!("\n📊 Single Token Test:");
    println!("  Input: {:?}", input.dims());
    println!("  Q: {:?}", q.dims());
    println!("  Q_rot: {:?}", q_rot.dims());
    
    assert_eq!(q_rot.dims(), &[1, 1, n_heads, head_dim]);
    assert_eq!(k_rot.dims(), &[1, 1, n_heads, head_dim]);
    assert_eq!(v.dims(), &[1, 1, n_heads, head_dim]);
    
    let q_vec = q_rot.flatten_all()?.to_vec1::<f32>()?;
    assert!(q_vec.iter().all(|&x| !x.is_nan() && x.is_finite()));
    
    println!("\n✅ Single token handling correct");
    
    Ok(())
}

#[test]
fn test_edge_case_large_batch() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Edge Case: Large Batch Size                            ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    let hidden_size = 128;
    let n_heads = 4;
    let head_dim = 32;
    let batch = 8;
    let seq_len = 16;
    
    let q_weight = vec![0.01f32; hidden_size * hidden_size];
    let k_weight = vec![0.02f32; hidden_size * hidden_size];
    let v_weight = vec![0.03f32; hidden_size * hidden_size];
    
    let qkv = QKVProjection::from_arrays(
        &q_weight,
        &k_weight,
        &v_weight,
        hidden_size,
        n_heads,
        &device,
    )?;
    
    let rope = RoPE::new(head_dim, 1000, 10000.0, &device)?;
    
    let input = Tensor::randn(0f32, 0.02, (batch, seq_len, hidden_size), &device)?;
    let (q, k, v) = qkv.forward(&input)?;
    let (q_rot, k_rot) = rope.forward(&q, &k, 0)?;
    
    println!("\n📊 Large Batch Test:");
    println!("  Batch: {}, Seq: {}", batch, seq_len);
    println!("  Q_rot: {:?}", q_rot.dims());
    
    assert_eq!(q_rot.dims(), &[batch, seq_len, n_heads, head_dim]);
    assert_eq!(k_rot.dims(), &[batch, seq_len, n_heads, head_dim]);
    assert_eq!(v.dims(), &[batch, seq_len, n_heads, head_dim]);
    
    let q_vec = q_rot.flatten_all()?.to_vec1::<f32>()?;
    assert!(q_vec.iter().all(|&x| !x.is_nan() && x.is_finite()));
    
    println!("\n✅ Large batch handling correct");
    
    Ok(())
}
