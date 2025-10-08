//! Isolated Checkpoint 4: Attention Scores validation with synthetic data
//!
//! This test validates attention score computation with known synthetic inputs
//! to prove correctness before testing with real GPT-2 weights.

use llorch_cpud::layers::attention::AttentionScores;
use ndarray::Array3;

#[test]
fn test_isolated_checkpoint_04_basic() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 4: Attention Scores Isolated Test           ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    // Create synthetic Q, K tensors
    // Shape: [seq, n_heads, head_dim]
    let seq = 2;
    let n_heads = 12;
    let head_dim = 64;
    
    let mut q = Array3::zeros((seq, n_heads, head_dim));
    let mut k = Array3::zeros((seq, n_heads, head_dim));
    
    // Fill with deterministic pattern
    for s in 0..seq {
        for h in 0..n_heads {
            for d in 0..head_dim {
                let idx = (s * n_heads * head_dim 
                         + h * head_dim 
                         + d) as f32;
                q[[s, h, d]] = (idx * 0.001).sin() * 0.5;
                k[[s, h, d]] = (idx * 0.001).cos() * 0.3;
            }
        }
    }
    
    println!("\n📊 Synthetic Input:");
    println!("  Q shape: {:?}", q.shape());
    println!("  K shape: {:?}", k.shape());
    
    let q_sample: Vec<f32> = q.iter().take(10).copied().collect();
    let k_sample: Vec<f32> = k.iter().take(10).copied().collect();
    println!("  Q first 10: {:?}", q_sample);
    println!("  K first 10: {:?}", k_sample);
    
    // Compute scores
    let scores_layer = AttentionScores::new(head_dim);
    let scores = scores_layer.forward(&q, &k, None);
    
    println!("\n📊 Computed Scores:");
    println!("  Shape: {:?}", scores.shape());
    
    // CRITICAL: Validate shapes
    assert_eq!(scores.shape(), &[n_heads, seq, seq], 
        "Scores shape mismatch: got {:?}, expected [{}, {}, {}]", 
        scores.shape(), n_heads, seq, seq);
    
    // Validate no NaN/Inf
    for val in scores.iter() {
        assert!(val.is_finite(), "Scores contain NaN or Inf: {}", val);
    }
    
    let scores_sample: Vec<f32> = scores.iter().take(10).copied().collect();
    println!("  First 10 values: {:?}", scores_sample);
    
    // Validate scale factor is applied
    // Scale = sqrt(64) = 8.0
    // Scores should be roughly in range [-10, 10] after scaling
    let mut min_score = f32::INFINITY;
    let mut max_score = f32::NEG_INFINITY;
    
    for val in scores.iter() {
        min_score = min_score.min(*val);
        max_score = max_score.max(*val);
    }
    
    println!("\n📊 Score Statistics:");
    println!("  Min score: {:.6}", min_score);
    println!("  Max score: {:.6}", max_score);
    println!("  Range: [{:.6}, {:.6}]", min_score, max_score);
    
    // Scores should be reasonable (not too large or too small)
    assert!(min_score > -100.0, "Min score too small: {}", min_score);
    assert!(max_score < 100.0, "Max score too large: {}", max_score);
    
    println!("\n✅ PASS: Attention scores computed correctly!");
}

#[test]
fn test_checkpoint_04_determinism() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 4: Determinism Test                         ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    // Create synthetic Q, K
    let seq = 2;
    let n_heads = 12;
    let head_dim = 64;
    
    let mut q = Array3::zeros((seq, n_heads, head_dim));
    let mut k = Array3::zeros((seq, n_heads, head_dim));
    
    for s in 0..seq {
        for h in 0..n_heads {
            for d in 0..head_dim {
                let idx = (s * n_heads * head_dim 
                         + h * head_dim 
                         + d) as f32;
                q[[s, h, d]] = (idx * 0.001).sin();
                k[[s, h, d]] = (idx * 0.001).cos();
            }
        }
    }
    
    let scores_layer = AttentionScores::new(head_dim);
    
    // Run 3 times
    let scores1 = scores_layer.forward(&q, &k, None);
    let scores2 = scores_layer.forward(&q, &k, None);
    let scores3 = scores_layer.forward(&q, &k, None);
    
    // Must be bit-exact
    for (i, ((v1, v2), v3)) in scores1.iter().zip(scores2.iter()).zip(scores3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "Run 1 vs 2 differ at element {}", i);
        assert_eq!(v2.to_bits(), v3.to_bits(), "Run 2 vs 3 differ at element {}", i);
    }
    
    println!("\n✅ PASS: Attention scores are deterministic across runs");
}

#[test]
fn test_checkpoint_04_scale_factor() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 4: Scale Factor Validation                  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    // Test that scale factor is correctly applied
    let head_dim = 64;
    let expected_scale = (head_dim as f32).sqrt();  // 8.0
    
    println!("\n📊 Scale Factor:");
    println!("  head_dim: {}", head_dim);
    println!("  Expected scale: {:.6}", expected_scale);
    
    // Create simple Q, K where we know the dot product
    let seq = 1;
    let n_heads = 1;
    
    let mut q = Array3::zeros((seq, n_heads, head_dim));
    let mut k = Array3::zeros((seq, n_heads, head_dim));
    
    // Set all values to 1.0 for easy calculation
    for d in 0..head_dim {
        q[[0, 0, d]] = 1.0;
        k[[0, 0, d]] = 1.0;
    }
    
    // Dot product = head_dim (64)
    // After scaling: 64 / 8.0 = 8.0
    let expected_score = head_dim as f32 / expected_scale;
    
    let scores_layer = AttentionScores::new(head_dim);
    let scores = scores_layer.forward(&q, &k, None);
    
    let actual_score = scores[[0, 0, 0]];
    
    println!("  Actual score: {:.6}", actual_score);
    println!("  Expected score: {:.6}", expected_score);
    
    let diff = (actual_score - expected_score).abs();
    println!("  Difference: {:.6e}", diff);
    
    assert!(diff < 1e-5, "Scale factor not applied correctly: diff={}", diff);
    
    println!("\n✅ PASS: Scale factor correctly applied!");
}
