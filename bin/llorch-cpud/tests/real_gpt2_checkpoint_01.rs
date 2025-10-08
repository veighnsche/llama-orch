//! Checkpoint 1: LayerNorm with REAL GPT-2 weights
//!
//! This test validates llorch-cpud LayerNorm against HuggingFace transformers
//! using REAL GPT-2 base (124M) model weights.

use llorch_cpud::layers::LayerNorm;
use ndarray::{Array1, Array2};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::PathBuf;

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.test-models/gpt2/extracted_weights")
}

#[test]
fn test_checkpoint_01_real_gpt2() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 1: LayerNorm with REAL GPT-2 Weights        ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    let dir = weights_dir();
    
    // Check if weights exist
    if !dir.exists() {
        eprintln!("\n❌ GPT-2 weights not found at: {}", dir.display());
        eprintln!("\nPlease run:");
        eprintln!("  cd .docs/testing");
        eprintln!("  python3 extract_gpt2_weights.py");
        eprintln!();
        panic!("GPT-2 weights not extracted");
    }
    
    // Load real embeddings
    let mut emb_file = File::open(dir.join("embeddings.npy"))
        .expect("Failed to open embeddings.npy");
    let embeddings: Array2<f32> = Array2::read_npy(&mut emb_file)
        .expect("Failed to read embeddings");
    
    // Load REAL ln_1 weights from GPT-2
    let mut weight_file = File::open(dir.join("h0_ln_1_weight.npy"))
        .expect("Failed to open h0_ln_1_weight.npy");
    let ln_weight: Array1<f32> = Array1::read_npy(&mut weight_file)
        .expect("Failed to read ln_1 weight");
    
    let mut bias_file = File::open(dir.join("h0_ln_1_bias.npy"))
        .expect("Failed to open h0_ln_1_bias.npy");
    let ln_bias: Array1<f32> = Array1::read_npy(&mut bias_file)
        .expect("Failed to read ln_1 bias");
    
    println!("\n📊 Real GPT-2 Weights:");
    println!("  ln_1.weight shape: {:?}", ln_weight.shape());
    println!("  ln_1.bias shape: {:?}", ln_bias.shape());
    println!("  embeddings shape: {:?}", embeddings.shape());
    
    // Create LayerNorm with REAL weights
    let layer_norm = LayerNorm::new(ln_weight, ln_bias, 1e-5);
    let output = layer_norm.forward(&embeddings);
    
    println!("\n📊 Our Output:");
    println!("  Shape: {:?}", output.shape());
    let sample: Vec<f32> = output.iter().take(10).copied().collect();
    println!("  First 10 values: {:?}", sample);
    
    // Load HuggingFace reference
    let mut ref_file = File::open(dir.join("checkpoint_01_ln1_output.npy"))
        .expect("Failed to open checkpoint_01_ln1_output.npy");
    let expected: Array2<f32> = Array2::read_npy(&mut ref_file)
        .expect("Failed to read reference output");
    
    println!("\n📊 HuggingFace Reference:");
    println!("  Shape: {:?}", expected.shape());
    let ref_sample: Vec<f32> = expected.iter().take(10).copied().collect();
    println!("  First 10 values: {:?}", ref_sample);
    
    // Compare
    let mut max_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    
    for (our, exp) in output.iter().zip(expected.iter()) {
        let abs_diff = (our - exp).abs();
        let rel_diff = if exp.abs() > 1e-10 {
            abs_diff / exp.abs()
        } else {
            abs_diff
        };
        
        max_diff = max_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
    }
    
    println!("\n📊 Comparison:");
    println!("  Max absolute difference: {:.6e}", max_diff);
    println!("  Max relative difference: {:.6e}", max_rel_diff);
    println!("  Tolerance: 1e-4");
    
    if max_diff < 1e-4 {
        println!("\n✅ PASS: LayerNorm matches HuggingFace with REAL GPT-2 weights!");
        println!("   This validates mathematical correctness with actual model weights.");
    } else {
        println!("\n❌ FAIL: Difference exceeds tolerance");
        panic!("Max difference {} exceeds 1e-4", max_diff);
    }
}

#[test]
#[ignore] // Run with: cargo test --test real_gpt2_checkpoint_01 -- --ignored
fn test_checkpoint_01_determinism() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 1: Determinism with Real Weights            ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    let dir = weights_dir();
    
    if !dir.exists() {
        eprintln!("\n❌ GPT-2 weights not found");
        panic!("Run extract_gpt2_weights.py first");
    }
    
    // Load weights
    let mut emb_file = File::open(dir.join("embeddings.npy"))
        .expect("Failed to open embeddings.npy");
    let embeddings: Array2<f32> = Array2::read_npy(&mut emb_file)
        .expect("Failed to load embeddings");
    
    let mut weight_file = File::open(dir.join("h0_ln_1_weight.npy"))
        .expect("Failed to open h0_ln_1_weight.npy");
    let ln_weight: Array1<f32> = Array1::read_npy(&mut weight_file)
        .expect("Failed to load weight");
    
    let mut bias_file = File::open(dir.join("h0_ln_1_bias.npy"))
        .expect("Failed to open h0_ln_1_bias.npy");
    let ln_bias: Array1<f32> = Array1::read_npy(&mut bias_file)
        .expect("Failed to load bias");
    
    let layer_norm = LayerNorm::new(ln_weight, ln_bias, 1e-5);
    
    // Run 3 times
    let output1 = layer_norm.forward(&embeddings);
    let output2 = layer_norm.forward(&embeddings);
    let output3 = layer_norm.forward(&embeddings);
    
    // Must be bit-exact
    for (i, ((v1, v2), v3)) in output1.iter().zip(output2.iter()).zip(output3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "Run 1 vs 2 differ at element {}", i);
        assert_eq!(v2.to_bits(), v3.to_bits(), "Run 2 vs 3 differ at element {}", i);
    }
    
    println!("\n✅ PASS: LayerNorm is deterministic with real weights");
}
