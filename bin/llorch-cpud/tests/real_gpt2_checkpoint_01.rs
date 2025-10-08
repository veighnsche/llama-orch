//! Checkpoint 1: LayerNorm with REAL GPT-2 weights
//!
//! This test validates llorch-cpud LayerNorm against HuggingFace transformers
//! using REAL GPT-2 base (124M) model weights.
//!
//! ## Python Virtual Environment Required
//!
//! **IMPORTANT FOR ENGINEERS:** This test requires Python dependencies to generate
//! reference data. A dedicated virtual environment is available at:
//!
//! ```bash
//! source ../../.venv-testing/bin/activate
//! ```
//!
//! To generate the required reference data:
//! ```bash
//! cd .docs/testing
//! source ../../.venv-testing/bin/activate
//! python3 extract_gpt2_weights.py
//! ```
//!
//! Modified by: TEAM-001

use llorch_cpud::layers::LayerNorm;
use ndarray::{Array1, Array2};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::PathBuf;

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.test-models/gpt2/extracted_weights")
}

// TEAM-003: Added multi-reference validation (PyTorch + Candle cross-validation)
#[test]
fn test_checkpoint_01_multi_reference() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 1: Multi-Reference Validation                ║");
    println!("║  PyTorch + Candle Cross-Validation                       ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    let dir = weights_dir();
    
    // Check if weights exist
    // TEAM-001: Added venv instructions for engineers
    if !dir.exists() {
        eprintln!("\n❌ GPT-2 weights not found at: {}", dir.display());
        eprintln!("\n⚠️  VENV REQUIRED: Activate the testing environment first:");
        eprintln!("  source ../../.venv-testing/bin/activate");
        eprintln!("\nThen run:");
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
    
    // Validate no NaN/Inf
    for val in output.iter() {
        assert!(val.is_finite(), "Output contains NaN or Inf: {}", val);
    }
    
    println!("\n📊 Comparison:");
    println!("  Max absolute difference: {:.6e}", max_diff);
    println!("  Max relative difference: {:.6e}", max_rel_diff);
    println!("  Tolerance: 1e-4");
    
    // TEAM-003: Validate against PyTorch
    if max_diff < 1e-4 {
        println!("\n✅ PYTORCH: LayerNorm matches HuggingFace (max diff {:.6e})", max_diff);
    } else {
        println!("\n❌ PYTORCH: Difference exceeds tolerance");
        panic!("PyTorch max difference {} exceeds 1e-4", max_diff);
    }
    
    // TEAM-003: Validate against Candle (if available)
    let candle_path = dir.join("checkpoint_01_ln1_output_candle.npy");
    if candle_path.exists() {
        let mut candle_file = File::open(&candle_path)
            .expect("Failed to open Candle reference");
        let candle_ref: Array2<f32> = Array2::read_npy(&mut candle_file)
            .expect("Failed to read Candle reference");
        
        let mut candle_diff = 0.0f32;
        for (our, candle) in output.iter().zip(candle_ref.iter()) {
            candle_diff = candle_diff.max((our - candle).abs());
        }
        
        println!("\n📊 Candle Comparison:");
        println!("  Max absolute difference: {:.6e}", candle_diff);
        
        if candle_diff < 1e-4 {
            println!("✅ CANDLE: Matches within tolerance");
        } else {
            println!("❌ CANDLE: Difference exceeds tolerance");
            panic!("Candle max difference {} exceeds 1e-4", candle_diff);
        }
        
        // TEAM-003: Cross-validate references against each other
        let mut cross_diff = 0.0f32;
        for (pytorch, candle) in expected.iter().zip(candle_ref.iter()) {
            cross_diff = cross_diff.max((pytorch - candle).abs());
        }
        
        println!("\n📊 Cross-Validation (PyTorch vs Candle):");
        println!("  Max difference: {:.6e}", cross_diff);
        
        if cross_diff < 1e-3 {
            println!("✅ CROSS-VALIDATION: References agree");
        } else {
            println!("⚠️  WARNING: References disagree by {:.6e}", cross_diff);
        }
        
        println!("\n🎉 MULTI-REFERENCE VALIDATION PASSED!");
        println!("   Our implementation matches BOTH PyTorch and Candle");
    } else {
        println!("\n⚠️  Candle reference not available");
        println!("   Run: cd .test_helpers/candle_gpt2_reference && cargo run --release");
        println!("   Single-reference validation only (PyTorch)");
    }
}

#[test]
fn test_checkpoint_01_determinism() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 1: Determinism with Real Weights            ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    let dir = weights_dir();
    
    // TEAM-001: Added venv instructions
    if !dir.exists() {
        eprintln!("\n❌ GPT-2 weights not found");
        eprintln!("⚠️  Activate venv: source ../../.venv-testing/bin/activate");
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
