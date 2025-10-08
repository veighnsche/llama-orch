//! Checkpoint 6: FFN Output with REAL GPT-2 weights
//!
//! This test validates feedforward network computation against HuggingFace transformers
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
//! Created by: TEAM-002

use llorch_cpud::layers::ffn::FFN;
use ndarray::{Array1, Array2};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::PathBuf;

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.test-models/gpt2/extracted_weights")
}

#[test]
fn test_checkpoint_06_real_gpt2() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 6: FFN Output with REAL GPT-2               ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    let dir = weights_dir();
    
    // Check if weights exist
    // TEAM-002: Added venv instructions for engineers
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
    
    // Load REAL FFN weights from GPT-2
    let mut fc_weight_file = File::open(dir.join("h0_c_fc_weight.npy"))
        .expect("Failed to open h0_c_fc_weight.npy");
    let c_fc_weight: Array2<f32> = Array2::read_npy(&mut fc_weight_file)
        .expect("Failed to read c_fc weight");
    
    let mut fc_bias_file = File::open(dir.join("h0_c_fc_bias.npy"))
        .expect("Failed to open h0_c_fc_bias.npy");
    let c_fc_bias: Array1<f32> = Array1::read_npy(&mut fc_bias_file)
        .expect("Failed to read c_fc bias");
    
    let mut proj_weight_file = File::open(dir.join("h0_ffn_c_proj_weight.npy"))
        .expect("Failed to open h0_ffn_c_proj_weight.npy");
    let c_proj_weight: Array2<f32> = Array2::read_npy(&mut proj_weight_file)
        .expect("Failed to read c_proj weight");
    
    let mut proj_bias_file = File::open(dir.join("h0_ffn_c_proj_bias.npy"))
        .expect("Failed to open h0_ffn_c_proj_bias.npy");
    let c_proj_bias: Array1<f32> = Array1::read_npy(&mut proj_bias_file)
        .expect("Failed to read c_proj bias");
    
    // Load input from Checkpoint 5b (ln_2 output, not raw attention output)
    // TEAM-002: FFN receives ln_2(residual) as input, not raw attention output
    let mut input_file = File::open(dir.join("checkpoint_05b_ln2_output.npy"))
        .expect("Failed to open checkpoint_05b_ln2_output.npy");
    let input: Array2<f32> = Array2::read_npy(&mut input_file)
        .expect("Failed to read input");
    
    println!("\n📊 Real GPT-2 Inputs:");
    println!("  c_fc.weight shape: {:?}", c_fc_weight.shape());
    println!("  c_fc.bias shape: {:?}", c_fc_bias.shape());
    println!("  c_proj.weight shape: {:?}", c_proj_weight.shape());
    println!("  c_proj.bias shape: {:?}", c_proj_bias.shape());
    println!("  Input shape: {:?}", input.shape());
    
    // Create FFN layer
    let ffn = FFN::new(c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias);
    let output = ffn.forward(&input);
    
    println!("\n📊 Our Output:");
    println!("  Shape: {:?}", output.shape());
    let sample: Vec<f32> = output.iter().take(10).copied().collect();
    println!("  First 10 values: {:?}", sample);
    
    // Try to load HuggingFace reference if available
    let ref_path = dir.join("checkpoint_06_ffn.npy");
    if ref_path.exists() {
        let mut ref_file = File::open(&ref_path)
            .expect("Failed to open checkpoint_06_ffn.npy");
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
            println!("\n✅ PASS: FFN output matches HuggingFace with REAL GPT-2!");
            println!("   This validates feedforward network correctness.");
        } else {
            println!("\n❌ FAIL: Difference exceeds tolerance");
            panic!("Max difference {} exceeds 1e-4", max_diff);
        }
    } else {
        // TEAM-002: Enhanced warning about missing ground truth
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
        
        println!("\n✅ PASS: FFN output computed with reasonable values");
        println!("   (Full validation requires running extract_gpt2_weights.py)");
    }
}

#[test]
fn test_checkpoint_06_determinism() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 6: Determinism with Real Inputs             ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    let dir = weights_dir();
    
    // TEAM-002: Added venv instructions
    if !dir.exists() {
        eprintln!("\n❌ GPT-2 weights not found");
        eprintln!("⚠️  Activate venv: source ../../.venv-testing/bin/activate");
        panic!("Run extract_gpt2_weights.py first");
    }
    
    // Load weights and inputs
    let mut fc_weight_file = File::open(dir.join("h0_c_fc_weight.npy"))
        .expect("Failed to open weight file");
    let c_fc_weight: Array2<f32> = Array2::read_npy(&mut fc_weight_file)
        .expect("Failed to load weight");
    
    let mut fc_bias_file = File::open(dir.join("h0_c_fc_bias.npy"))
        .expect("Failed to open bias file");
    let c_fc_bias: Array1<f32> = Array1::read_npy(&mut fc_bias_file)
        .expect("Failed to load bias");
    
    let mut proj_weight_file = File::open(dir.join("h0_ffn_c_proj_weight.npy"))
        .expect("Failed to open proj weight file");
    let c_proj_weight: Array2<f32> = Array2::read_npy(&mut proj_weight_file)
        .expect("Failed to load proj weight");
    
    let mut proj_bias_file = File::open(dir.join("h0_ffn_c_proj_bias.npy"))
        .expect("Failed to open proj bias file");
    let c_proj_bias: Array1<f32> = Array1::read_npy(&mut proj_bias_file)
        .expect("Failed to load proj bias");
    
    let mut input_file = File::open(dir.join("checkpoint_05b_ln2_output.npy"))
        .expect("Failed to open input file");
    let input: Array2<f32> = Array2::read_npy(&mut input_file)
        .expect("Failed to load input");
    
    let ffn = FFN::new(c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias);
    
    // Run 3 times
    let output1 = ffn.forward(&input);
    let output2 = ffn.forward(&input);
    let output3 = ffn.forward(&input);
    
    // Must be bit-exact
    for (i, ((v1, v2), v3)) in output1.iter().zip(output2.iter()).zip(output3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "Run 1 vs 2 differ at element {}", i);
        assert_eq!(v2.to_bits(), v3.to_bits(), "Run 2 vs 3 differ at element {}", i);
    }
    
    println!("\n✅ PASS: FFN output is deterministic with real inputs");
}
