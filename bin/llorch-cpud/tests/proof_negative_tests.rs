//! Negative Tests: Prove validation catches errors
//!
//! These tests intentionally break the implementation to prove
//! that our validation is not giving false positives.

use llorch_cpud::layers::LayerNorm;
use llorch_cpud::layers::attention::QKVProjection;
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::PathBuf;

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.test-models/gpt2/extracted_weights")
}

/// NEGATIVE TEST 1: Wrong epsilon should fail
#[test]
#[should_panic(expected = "Max difference")]
fn test_wrong_epsilon_fails() {
    let dir = weights_dir();
    if !dir.exists() {
        panic!("GPT-2 weights not found - run extract_gpt2_weights.py first");
    }
    
    // Load real weights
    let mut emb_file = File::open(dir.join("embeddings.npy")).unwrap();
    let embeddings: Array2<f32> = Array2::read_npy(&mut emb_file).unwrap();
    
    let mut weight_file = File::open(dir.join("h0_ln_1_weight.npy")).unwrap();
    let ln_weight: Array1<f32> = Array1::read_npy(&mut weight_file).unwrap();
    
    let mut bias_file = File::open(dir.join("h0_ln_1_bias.npy")).unwrap();
    let ln_bias: Array1<f32> = Array1::read_npy(&mut bias_file).unwrap();
    
    // WRONG: Use epsilon = 1e-3 instead of 1e-5
    let layer_norm = LayerNorm::new(ln_weight, ln_bias, 1e-3);
    let output = layer_norm.forward(&embeddings);
    
    // Load reference
    let mut ref_file = File::open(dir.join("checkpoint_01_ln1_output.npy")).unwrap();
    let expected: Array2<f32> = Array2::read_npy(&mut ref_file).unwrap();
    
    // Compare - should fail!
    let mut max_diff = 0.0f32;
    for (our, exp) in output.iter().zip(expected.iter()) {
        max_diff = max_diff.max((our - exp).abs());
    }
    
    println!("Max diff with wrong epsilon: {:.6e}", max_diff);
    assert!(max_diff < 1e-4, "Max difference {} exceeds 1e-4", max_diff);
}

/// NEGATIVE TEST 2: Swapped weight and bias should fail
#[test]
#[should_panic]
fn test_swapped_weight_bias_fails() {
    let dir = weights_dir();
    if !dir.exists() {
        panic!("GPT-2 weights not found");
    }
    
    let mut emb_file = File::open(dir.join("embeddings.npy")).unwrap();
    let embeddings: Array2<f32> = Array2::read_npy(&mut emb_file).unwrap();
    
    let mut weight_file = File::open(dir.join("h0_ln_1_weight.npy")).unwrap();
    let ln_weight: Array1<f32> = Array1::read_npy(&mut weight_file).unwrap();
    
    let mut bias_file = File::open(dir.join("h0_ln_1_bias.npy")).unwrap();
    let ln_bias: Array1<f32> = Array1::read_npy(&mut bias_file).unwrap();
    
    // WRONG: Swap weight and bias
    let layer_norm = LayerNorm::new(ln_bias, ln_weight, 1e-5);
    let output = layer_norm.forward(&embeddings);
    
    let mut ref_file = File::open(dir.join("checkpoint_01_ln1_output.npy")).unwrap();
    let expected: Array2<f32> = Array2::read_npy(&mut ref_file).unwrap();
    
    let mut max_diff = 0.0f32;
    for (our, exp) in output.iter().zip(expected.iter()) {
        max_diff = max_diff.max((our - exp).abs());
    }
    
    println!("Max diff with swapped weight/bias: {:.6e}", max_diff);
    assert!(max_diff < 1e-4, "Max difference {} exceeds 1e-4", max_diff);
}

/// NEGATIVE TEST 3: Scaled weights should fail
#[test]
#[should_panic(expected = "Max difference")]
fn test_scaled_weights_fail() {
    let dir = weights_dir();
    if !dir.exists() {
        panic!("GPT-2 weights not found");
    }
    
    let mut emb_file = File::open(dir.join("embeddings.npy")).unwrap();
    let embeddings: Array2<f32> = Array2::read_npy(&mut emb_file).unwrap();
    
    let mut weight_file = File::open(dir.join("h0_ln_1_weight.npy")).unwrap();
    let mut ln_weight: Array1<f32> = Array1::read_npy(&mut weight_file).unwrap();
    
    let mut bias_file = File::open(dir.join("h0_ln_1_bias.npy")).unwrap();
    let ln_bias: Array1<f32> = Array1::read_npy(&mut bias_file).unwrap();
    
    // WRONG: Scale weights by 1.01
    ln_weight *= 1.01;
    
    let layer_norm = LayerNorm::new(ln_weight, ln_bias, 1e-5);
    let output = layer_norm.forward(&embeddings);
    
    let mut ref_file = File::open(dir.join("checkpoint_01_ln1_output.npy")).unwrap();
    let expected: Array2<f32> = Array2::read_npy(&mut ref_file).unwrap();
    
    let mut max_diff = 0.0f32;
    for (our, exp) in output.iter().zip(expected.iter()) {
        max_diff = max_diff.max((our - exp).abs());
    }
    
    println!("Max diff with scaled weights: {:.6e}", max_diff);
    assert!(max_diff < 1e-4, "Max difference {} exceeds 1e-4", max_diff);
}

/// NEGATIVE TEST 4: Wrong QKV weight shape should fail
#[test]
#[should_panic]
fn test_wrong_qkv_shape_fails() {
    let dir = weights_dir();
    if !dir.exists() {
        panic!("GPT-2 weights not found");
    }
    
    let mut ln_file = File::open(dir.join("checkpoint_01_ln1_output.npy")).unwrap();
    let ln_output: Array2<f32> = Array2::read_npy(&mut ln_file).unwrap();
    
    let mut weight_file = File::open(dir.join("h0_c_attn_weight.npy")).unwrap();
    let c_attn_weight: Array2<f32> = Array2::read_npy(&mut weight_file).unwrap();
    
    let mut bias_file = File::open(dir.join("h0_c_attn_bias.npy")).unwrap();
    let c_attn_bias: Array1<f32> = Array1::read_npy(&mut bias_file).unwrap();
    
    // WRONG: Transpose the weight (should cause dimension mismatch)
    let weight_t = c_attn_weight.t().to_owned();
    
    let qkv = QKVProjection::new(weight_t, c_attn_bias, 12);
    
    // This should panic with dimension mismatch
    let (_q, _k, _v) = qkv.forward(&ln_output);
}

/// NEGATIVE TEST 5: Wrong number of heads should produce wrong output
#[test]
#[should_panic]
fn test_wrong_heads_fails() {
    let dir = weights_dir();
    if !dir.exists() {
        panic!("GPT-2 weights not found");
    }
    
    let mut ln_file = File::open(dir.join("checkpoint_01_ln1_output.npy")).unwrap();
    let ln_output: Array2<f32> = Array2::read_npy(&mut ln_file).unwrap();
    
    let mut weight_file = File::open(dir.join("h0_c_attn_weight.npy")).unwrap();
    let c_attn_weight: Array2<f32> = Array2::read_npy(&mut weight_file).unwrap();
    
    let mut bias_file = File::open(dir.join("h0_c_attn_bias.npy")).unwrap();
    let c_attn_bias: Array1<f32> = Array1::read_npy(&mut bias_file).unwrap();
    
    // WRONG: Use 16 heads instead of 12
    let qkv = QKVProjection::new(c_attn_weight, c_attn_bias, 16);
    let (q, _k, _v) = qkv.forward(&ln_output);
    
    // Load reference
    let mut q_file = File::open(dir.join("checkpoint_02_q.npy")).unwrap();
    let ref_q: Array3<f32> = Array3::read_npy(&mut q_file).unwrap();
    
    // Shape will be wrong: [2, 16, 48] vs [2, 12, 64]
    println!("Our Q shape: {:?}", q.shape());
    println!("Ref Q shape: {:?}", ref_q.shape());
    
    assert_eq!(q.shape(), ref_q.shape(), "Shapes differ");
    
    let mut max_diff = 0.0f32;
    for (our, exp) in q.iter().zip(ref_q.iter()) {
        max_diff = max_diff.max((our - exp).abs());
    }
    
    assert!(max_diff < 1e-4, "Max difference {} exceeds 1e-4", max_diff);
}

/// NEGATIVE TEST 6: Zeroed bias should fail
#[test]
#[should_panic(expected = "Max difference")]
fn test_zeroed_bias_fails() {
    let dir = weights_dir();
    if !dir.exists() {
        panic!("GPT-2 weights not found");
    }
    
    let mut ln_file = File::open(dir.join("checkpoint_01_ln1_output.npy")).unwrap();
    let ln_output: Array2<f32> = Array2::read_npy(&mut ln_file).unwrap();
    
    let mut weight_file = File::open(dir.join("h0_c_attn_weight.npy")).unwrap();
    let c_attn_weight: Array2<f32> = Array2::read_npy(&mut weight_file).unwrap();
    
    // WRONG: Use zero bias instead of real bias
    let c_attn_bias = Array1::zeros(2304);
    
    let qkv = QKVProjection::new(c_attn_weight, c_attn_bias, 12);
    let (q, _k, _v) = qkv.forward(&ln_output);
    
    let mut q_file = File::open(dir.join("checkpoint_02_q.npy")).unwrap();
    let ref_q: Array3<f32> = Array3::read_npy(&mut q_file).unwrap();
    
    let mut max_diff = 0.0f32;
    for (our, exp) in q.iter().zip(ref_q.iter()) {
        max_diff = max_diff.max((our - exp).abs());
    }
    
    println!("Max diff with zero bias: {:.6e}", max_diff);
    assert!(max_diff < 1e-4, "Max difference {} exceeds 1e-4", max_diff);
}

/// NEGATIVE TEST 7: Wrong cache indexing should fail
#[test]
#[should_panic]
fn test_wrong_cache_start_pos_fails() {
    use llorch_cpud::cache::KVCache;
    
    let dir = weights_dir();
    if !dir.exists() {
        panic!("GPT-2 weights not found");
    }
    
    // Load K, V from checkpoint 2
    let mut k_file = File::open(dir.join("checkpoint_02_k.npy")).unwrap();
    let k: Array3<f32> = Array3::read_npy(&mut k_file).unwrap();
    
    let mut v_file = File::open(dir.join("checkpoint_02_v.npy")).unwrap();
    let v: Array3<f32> = Array3::read_npy(&mut v_file).unwrap();
    
    // WRONG: Use start_pos=1 instead of 0
    let mut cache = KVCache::new(2048, 12, 64);
    cache.update(&k, &v, 1);  // Should be 0!
    
    let seq_len = k.shape()[0];  // FIXED: Use correct dimension
    let (cached_k, _cached_v) = cache.get(seq_len);
    
    // Compare - should fail because cache has wrong indexing
    let mut max_diff = 0.0f32;
    for (our, exp) in cached_k.iter().zip(k.iter()) {
        max_diff = max_diff.max((our - exp).abs());
    }
    
    println!("Max diff with wrong start_pos: {:.6e}", max_diff);
    assert_eq!(max_diff, 0.0, "Cache should be bit-perfect but got diff {}", max_diff);
}

/// NEGATIVE TEST 8: Wrong retrieval end_pos should fail
#[test]
#[should_panic]
fn test_wrong_cache_end_pos_fails() {
    use llorch_cpud::cache::KVCache;
    
    let dir = weights_dir();
    if !dir.exists() {
        panic!("GPT-2 weights not found");
    }
    
    let mut k_file = File::open(dir.join("checkpoint_02_k.npy")).unwrap();
    let k: Array3<f32> = Array3::read_npy(&mut k_file).unwrap();
    
    let mut v_file = File::open(dir.join("checkpoint_02_v.npy")).unwrap();
    let v: Array3<f32> = Array3::read_npy(&mut v_file).unwrap();
    
    let mut cache = KVCache::new(2048, 12, 64);
    cache.update(&k, &v, 0);
    
    let seq_len = k.shape()[0];  // FIXED: Use correct dimension
    
    // WRONG: Retrieve with end_pos=1 instead of seq_len
    let (cached_k, _cached_v) = cache.get(1);
    
    // Shape will be wrong
    println!("Our shape: {:?}", cached_k.shape());
    println!("Expected shape: {:?}", k.shape());
    
    assert_eq!(cached_k.shape(), k.shape(), "Shapes should match");
}

/// NEGATIVE TEST 9: Uninitialized cache retrieval should return zeros
#[test]
fn test_uninitialized_cache_returns_zeros() {
    use llorch_cpud::cache::KVCache;
    
    let cache = KVCache::new(2048, 12, 64);
    
    // Get without update - should return zeros
    let (k, v) = cache.get(2);
    
    // All values should be zero
    for val in k.iter() {
        assert_eq!(*val, 0.0, "Uninitialized cache K should be zero");
    }
    
    for val in v.iter() {
        assert_eq!(*val, 0.0, "Uninitialized cache V should be zero");
    }
    
    println!("✅ PASS: Uninitialized cache correctly returns zeros");
}

/// NEGATIVE TEST 10: Wrong scale factor should produce wrong results
#[test]
fn test_wrong_scale_factor_produces_wrong_results() {
    use llorch_cpud::layers::attention::AttentionScores;
    
    // Create two sets of Q, K with different head_dims to test scale factor
    let seq = 2;
    let n_heads = 2;
    
    // Test 1: head_dim=64, scale=8.0
    let head_dim_1 = 64;
    let mut q1 = Array3::zeros((seq, n_heads, head_dim_1));
    let mut k1 = Array3::zeros((seq, n_heads, head_dim_1));
    
    for s in 0..seq {
        for h in 0..n_heads {
            for d in 0..head_dim_1 {
                q1[[s, h, d]] = 1.0;
                k1[[s, h, d]] = 1.0;
            }
        }
    }
    
    let layer1 = AttentionScores::new(head_dim_1);
    let scores1 = layer1.forward(&q1, &k1, None);
    
    // Test 2: head_dim=16, scale=4.0 (different scale)
    let head_dim_2 = 16;
    let mut q2 = Array3::zeros((seq, n_heads, head_dim_2));
    let mut k2 = Array3::zeros((seq, n_heads, head_dim_2));
    
    for s in 0..seq {
        for h in 0..n_heads {
            for d in 0..head_dim_2 {
                q2[[s, h, d]] = 1.0;
                k2[[s, h, d]] = 1.0;
            }
        }
    }
    
    let layer2 = AttentionScores::new(head_dim_2);
    let scores2 = layer2.forward(&q2, &k2, None);
    
    // Expected: scores1 = 64/8 = 8.0, scores2 = 16/4 = 4.0
    // They should differ by 2x
    let score1_val = scores1[[0, 0, 0]];
    let score2_val = scores2[[0, 0, 0]];
    
    println!("Score with head_dim=64: {:.6}", score1_val);
    println!("Score with head_dim=16: {:.6}", score2_val);
    println!("Ratio: {:.6}", score1_val / score2_val);
    
    // Verify they differ significantly (should be 2x)
    let ratio = score1_val / score2_val;
    assert!((ratio - 2.0).abs() < 0.01, "Scale factor should produce 2x difference, got {}", ratio);
    
    println!("✅ PASS: Different scale factors correctly produce different results");
}

/// NEGATIVE TEST 11: Mismatched Q/K dimensions should panic
#[test]
#[should_panic(expected = "Q and K must have same n_heads")]
fn test_mismatched_heads_fails() {
    use llorch_cpud::layers::attention::AttentionScores;
    
    // Q with 12 heads, K with 16 heads - should panic
    let q = Array3::zeros((2, 12, 64));
    let k = Array3::zeros((2, 16, 64));
    
    let scores_layer = AttentionScores::new(64);
    let _scores = scores_layer.forward(&q, &k, None);
}

/// NEGATIVE TEST 12: Wrong head_dim should panic
#[test]
#[should_panic(expected = "Q head_dim mismatch")]
fn test_wrong_head_dim_fails() {
    use llorch_cpud::layers::attention::AttentionScores;
    
    // Create Q with head_dim=64 but tell scores layer it's 32
    let q = Array3::zeros((2, 12, 64));
    let k = Array3::zeros((2, 12, 64));
    
    let scores_layer = AttentionScores::new(32);  // WRONG
    let _scores = scores_layer.forward(&q, &k, None);
}
