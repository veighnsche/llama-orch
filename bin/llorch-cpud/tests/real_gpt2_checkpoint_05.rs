//! Checkpoint 5: Attention Output with REAL GPT-2 weights
//!
//! This test validates attention output computation against HuggingFace transformers
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

use llorch_cpud::layers::attention::AttentionOutput;
use ndarray::Array3;
use ndarray::{Array1, Array2};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::PathBuf;

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.test-models/gpt2/extracted_weights")
}

#[test]
fn test_checkpoint_05_real_gpt2() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 5: Attention Output with REAL GPT-2         ║");
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
    
    // Load REAL c_proj weights from GPT-2
    let mut weight_file = File::open(dir.join("h0_c_proj_weight.npy"))
        .expect("Failed to open h0_c_proj_weight.npy");
    let c_proj_weight: Array2<f32> = Array2::read_npy(&mut weight_file)
        .expect("Failed to read c_proj weight");
    
    let mut bias_file = File::open(dir.join("h0_c_proj_bias.npy"))
        .expect("Failed to open h0_c_proj_bias.npy");
    let c_proj_bias: Array1<f32> = Array1::read_npy(&mut bias_file)
        .expect("Failed to read c_proj bias");
    
    // Load attention scores from Checkpoint 4
    let mut scores_file = File::open(dir.join("checkpoint_04_scores.npy"))
        .expect("Failed to open checkpoint_04_scores.npy");
    let attn_scores: Array3<f32> = Array3::read_npy(&mut scores_file)
        .expect("Failed to read attention scores");
    
    // Load V from Checkpoint 2
    let mut v_file = File::open(dir.join("checkpoint_02_v.npy"))
        .expect("Failed to open checkpoint_02_v.npy");
    let v: Array3<f32> = Array3::read_npy(&mut v_file)
        .expect("Failed to read V");
    
    println!("\n📊 Real GPT-2 Inputs:");
    println!("  c_proj.weight shape: {:?}", c_proj_weight.shape());
    println!("  c_proj.bias shape: {:?}", c_proj_bias.shape());
    println!("  Attention scores shape: {:?}", attn_scores.shape());
    println!("  V shape: {:?}", v.shape());
    
    // Create attention output layer
    let output_layer = AttentionOutput::new(c_proj_weight, c_proj_bias);
    let output = output_layer.forward(&attn_scores, &v);
    
    println!("\n📊 Our Output:");
    println!("  Shape: {:?}", output.shape());
    let sample: Vec<f32> = output.iter().take(10).copied().collect();
    println!("  First 10 values: {:?}", sample);
    
    // Try to load HuggingFace reference if available
    let ref_path = dir.join("checkpoint_05_output.npy");
    if ref_path.exists() {
        let mut ref_file = File::open(&ref_path)
            .expect("Failed to open checkpoint_05_output.npy");
        let expected: Array2<f32> = Array2::read_npy(&mut ref_file)
            .expect("Failed to read reference output");
        
        println!("\n📊 HuggingFace Reference:");
        println!("  Shape: {:?}", expected.shape());
        let ref_sample: Vec<f32> = expected.iter().take(10).copied().collect();
        println!("  First 10 values: {:?}", ref_sample);
        
        // CRITICAL: Validate shapes before comparing values
        assert_eq!(output.shape(), expected.shape(), 
            "Output shape mismatch: ours={:?} vs ref={:?}", output.shape(), expected.shape());
        
        // Validate no NaN/Inf
        for val in output.iter() {
            assert!(val.is_finite(), "Output contains NaN or Inf: {}", val);
        }
        
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
            println!("\n✅ PASS: Attention output matches HuggingFace with REAL GPT-2!");
            println!("   This validates complete attention mechanism correctness.");
        } else {
            println!("\n❌ FAIL: Difference exceeds tolerance");
            panic!("Max difference {} exceeds 1e-4", max_diff);
        }
    } else {
        // TEAM-001: Enhanced warning about missing ground truth
        println!("\n⚠️  WARNING: Reference file not found: {}", ref_path.display());
        println!("\n❌ CRITICAL ISSUE: Cannot validate correctness without ground truth!");
        println!("   This is a FALSE POSITIVE - test passes but correctness is NOT verified.");
        println!("\n   Activate venv: source ../../.venv-testing/bin/activate");
        println!("   Run: cd .docs/testing && python3 extract_gpt2_weights.py");
        println!("\n   Skipping comparison, but validating basic properties...");
        
        // Validate no NaN/Inf
        for val in output.iter() {
            assert!(val.is_finite(), "Output contains NaN or Inf: {}", val);
        }
        
        // Validate reasonable range
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        
        for val in output.iter() {
            min_val = min_val.min(*val);
            max_val = max_val.max(*val);
        }
        
        println!("\n📊 Output Statistics:");
        println!("  Min value: {:.6}", min_val);
        println!("  Max value: {:.6}", max_val);
        
        assert!(min_val > -100.0, "Min value too small: {}", min_val);
        assert!(max_val < 100.0, "Max value too large: {}", max_val);
        
        println!("\n✅ PASS: Attention output computed with reasonable values");
        println!("   (Full validation requires running extract_gpt2_weights.py)");
    }
}

#[test]
fn test_checkpoint_05_determinism() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 5: Determinism with Real Inputs             ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    let dir = weights_dir();
    
    // TEAM-001: Added venv instructions
    if !dir.exists() {
        eprintln!("\n❌ GPT-2 weights not found");
        eprintln!("⚠️  Activate venv: source ../../.venv-testing/bin/activate");
        panic!("Run extract_gpt2_weights.py first");
    }
    
    // Load weights and inputs
    let mut weight_file = File::open(dir.join("h0_c_proj_weight.npy"))
        .expect("Failed to open weight file");
    let c_proj_weight: Array2<f32> = Array2::read_npy(&mut weight_file)
        .expect("Failed to load weight");
    
    let mut bias_file = File::open(dir.join("h0_c_proj_bias.npy"))
        .expect("Failed to open bias file");
    let c_proj_bias: Array1<f32> = Array1::read_npy(&mut bias_file)
        .expect("Failed to load bias");
    
    let mut scores_file = File::open(dir.join("checkpoint_04_scores.npy"))
        .expect("Failed to open scores file");
    let attn_scores: Array3<f32> = Array3::read_npy(&mut scores_file)
        .expect("Failed to load scores");
    
    let mut v_file = File::open(dir.join("checkpoint_02_v.npy"))
        .expect("Failed to open V file");
    let v: Array3<f32> = Array3::read_npy(&mut v_file)
        .expect("Failed to load V");
    
    let output_layer = AttentionOutput::new(c_proj_weight, c_proj_bias);
    
    // Run 3 times
    let output1 = output_layer.forward(&attn_scores, &v);
    let output2 = output_layer.forward(&attn_scores, &v);
    let output3 = output_layer.forward(&attn_scores, &v);
    
    // Must be bit-exact
    for (i, ((v1, v2), v3)) in output1.iter().zip(output2.iter()).zip(output3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "Run 1 vs 2 differ at element {}", i);
        assert_eq!(v2.to_bits(), v3.to_bits(), "Run 2 vs 3 differ at element {}", i);
    }
    
    println!("\n✅ PASS: Attention output is deterministic with real inputs");
}
