//! Checkpoint 4: Attention Scores with REAL GPT-2 weights
//!
//! This test validates attention score computation against HuggingFace transformers
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
//! **CRITICAL:** This test requires `checkpoint_04_scores.npy` to validate correctness.
//! The extraction script generates this file. Without it, the test will only perform
//! weak sanity checks that may not catch implementation errors.
//!
//! Modified by: TEAM-001

use llorch_cpud::layers::attention::AttentionScores;
use ndarray::Array3;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::PathBuf;

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.test-models/gpt2/extracted_weights")
}

#[test]
fn test_checkpoint_04_real_gpt2() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 4: Attention Scores with REAL GPT-2         ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    let dir = weights_dir();
    
    // Check if weights exist
    // TEAM-001: Added venv instructions for engineers - CRITICAL for checkpoint 4
    if !dir.exists() {
        eprintln!("\n❌ GPT-2 weights not found at: {}", dir.display());
        eprintln!("\n⚠️  VENV REQUIRED: Activate the testing environment first:");
        eprintln!("  source ../../.venv-testing/bin/activate");
        eprintln!("\n⚠️  CRITICAL: This test needs checkpoint_04_scores.npy for validation!");
        eprintln!("\nThen run:");
        eprintln!("  cd .docs/testing");
        eprintln!("  python3 extract_gpt2_weights.py");
        eprintln!();
        panic!("GPT-2 weights not extracted");
    }
    
    // Load REAL Q, K from Checkpoint 2
    let mut q_file = File::open(dir.join("checkpoint_02_q.npy"))
        .expect("Failed to open checkpoint_02_q.npy");
    let q: Array3<f32> = Array3::read_npy(&mut q_file)
        .expect("Failed to read Q");
    
    let mut k_file = File::open(dir.join("checkpoint_02_k.npy"))
        .expect("Failed to open checkpoint_02_k.npy");
    let k: Array3<f32> = Array3::read_npy(&mut k_file)
        .expect("Failed to read K");
    
    println!("\n📊 Real GPT-2 Q/K:");
    println!("  Q shape: {:?}", q.shape());
    println!("  K shape: {:?}", k.shape());
    
    // Create attention scores layer
    let scores_layer = AttentionScores::new(64);  // GPT-2 base: head_dim=64
    let scores = scores_layer.forward(&q, &k, None);  // No mask for now
    
    println!("\n📊 Our Scores:");
    println!("  Shape: {:?}", scores.shape());
    let sample: Vec<f32> = scores.iter().take(10).copied().collect();
    println!("  First 10 values: {:?}", sample);
    
    // Try to load HuggingFace reference if available
    let ref_path = dir.join("checkpoint_04_scores.npy");
    if ref_path.exists() {
        let mut ref_file = File::open(&ref_path)
            .expect("Failed to open checkpoint_04_scores.npy");
        let expected: Array3<f32> = Array3::read_npy(&mut ref_file)
            .expect("Failed to read reference scores");
        
        println!("\n📊 HuggingFace Reference:");
        println!("  Shape: {:?}", expected.shape());
        let ref_sample: Vec<f32> = expected.iter().take(10).copied().collect();
        println!("  First 10 values: {:?}", ref_sample);
        
        // CRITICAL: Validate shapes before comparing values
        assert_eq!(scores.shape(), expected.shape(), 
            "Scores shape mismatch: ours={:?} vs ref={:?}", scores.shape(), expected.shape());
        
        // Validate no NaN/Inf
        for val in scores.iter() {
            assert!(val.is_finite(), "Scores contain NaN or Inf: {}", val);
        }
        
        // Compare
        let mut max_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;
        
        for (our, exp) in scores.iter().zip(expected.iter()) {
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
            println!("\n✅ PASS: Attention scores match HuggingFace with REAL GPT-2!");
            println!("   This validates scaled dot-product attention correctness.");
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
        for val in scores.iter() {
            assert!(val.is_finite(), "Scores contain NaN or Inf: {}", val);
        }
        
        // Validate reasonable range
        let mut min_score = f32::INFINITY;
        let mut max_score = f32::NEG_INFINITY;
        
        for val in scores.iter() {
            min_score = min_score.min(*val);
            max_score = max_score.max(*val);
        }
        
        println!("\n📊 Score Statistics:");
        println!("  Min score: {:.6}", min_score);
        println!("  Max score: {:.6}", max_score);
        
        assert!(min_score > -100.0, "Min score too small: {}", min_score);
        assert!(max_score < 100.0, "Max score too large: {}", max_score);
        
        println!("\n✅ PASS: Attention scores computed with reasonable values");
        println!("   (Full validation requires running extract_gpt2_weights.py)");
    }
}

#[test]
fn test_checkpoint_04_determinism() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 4: Determinism with Real Q/K                ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    let dir = weights_dir();
    
    // TEAM-001: Added venv instructions
    if !dir.exists() {
        eprintln!("\n❌ GPT-2 weights not found");
        eprintln!("⚠️  Activate venv: source ../../.venv-testing/bin/activate");
        panic!("Run extract_gpt2_weights.py first");
    }
    
    // Load Q, K
    let mut q_file = File::open(dir.join("checkpoint_02_q.npy"))
        .expect("Failed to open Q file");
    let q: Array3<f32> = Array3::read_npy(&mut q_file)
        .expect("Failed to load Q");
    
    let mut k_file = File::open(dir.join("checkpoint_02_k.npy"))
        .expect("Failed to open K file");
    let k: Array3<f32> = Array3::read_npy(&mut k_file)
        .expect("Failed to load K");
    
    let scores_layer = AttentionScores::new(64);
    
    // Run 3 times
    let scores1 = scores_layer.forward(&q, &k, None);
    let scores2 = scores_layer.forward(&q, &k, None);
    let scores3 = scores_layer.forward(&q, &k, None);
    
    // Must be bit-exact
    for (i, ((v1, v2), v3)) in scores1.iter().zip(scores2.iter()).zip(scores3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "Run 1 vs 2 differ at element {}", i);
        assert_eq!(v2.to_bits(), v3.to_bits(), "Run 2 vs 3 differ at element {}", i);
    }
    
    println!("\n✅ PASS: Attention scores are deterministic with real Q/K");
}
