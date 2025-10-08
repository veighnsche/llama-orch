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
