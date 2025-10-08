//! Checkpoint 2: QKV Projection with REAL GPT-2 weights
//!
//! This test validates llorch-cpud QKV projection against HuggingFace transformers
//! using REAL GPT-2 base (124M) model weights.

use llorch_cpud::layers::attention::QKVProjection;
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::PathBuf;

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.test-models/gpt2/extracted_weights")
}

#[test]
fn test_checkpoint_02_real_gpt2() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 2: QKV with REAL GPT-2 Weights              ║");
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
    
    // Load LayerNorm output (input to QKV)
    let mut ln_file = File::open(dir.join("checkpoint_01_ln1_output.npy"))
        .expect("Failed to open checkpoint_01_ln1_output.npy");
    let ln_output: Array2<f32> = Array2::read_npy(&mut ln_file)
        .expect("Failed to read ln output");
    
    // Load REAL c_attn weights from GPT-2
    let mut weight_file = File::open(dir.join("h0_c_attn_weight.npy"))
        .expect("Failed to open h0_c_attn_weight.npy");
    let c_attn_weight: Array2<f32> = Array2::read_npy(&mut weight_file)
        .expect("Failed to read c_attn weight");
    
    let mut bias_file = File::open(dir.join("h0_c_attn_bias.npy"))
        .expect("Failed to open h0_c_attn_bias.npy");
    let c_attn_bias: Array1<f32> = Array1::read_npy(&mut bias_file)
        .expect("Failed to read c_attn bias");
    
    println!("\n📊 Real GPT-2 Weights:");
    println!("  c_attn.weight shape: {:?}", c_attn_weight.shape());
    println!("  c_attn.bias shape: {:?}", c_attn_bias.shape());
    println!("  ln_output shape: {:?}", ln_output.shape());
    
    // PyTorch Conv1D stores as [out, in] = [2304, 768]
    // We need [in, out] = [768, 2304] for ndarray matmul
    let weight_t = c_attn_weight.t().to_owned();
    
    println!("\n📊 After transpose:");
    println!("  weight_t shape: {:?}", weight_t.shape());
    
    // GPT-2 base: 12 heads, 64 dim per head
    let qkv = QKVProjection::new(weight_t, c_attn_bias, 12);
    let (q, k, v) = qkv.forward(&ln_output);
    
    println!("\n📊 Our Outputs:");
    println!("  Q shape: {:?}", q.shape());
    println!("  K shape: {:?}", k.shape());
    println!("  V shape: {:?}", v.shape());
    
    let q_sample: Vec<f32> = q.iter().take(10).copied().collect();
    let k_sample: Vec<f32> = k.iter().take(10).copied().collect();
    let v_sample: Vec<f32> = v.iter().take(10).copied().collect();
    
    println!("  Q first 10: {:?}", q_sample);
    println!("  K first 10: {:?}", k_sample);
    println!("  V first 10: {:?}", v_sample);
    
    // Load HuggingFace references
    let mut q_file = File::open(dir.join("checkpoint_02_q.npy"))
        .expect("Failed to open checkpoint_02_q.npy");
    let ref_q: Array3<f32> = Array3::read_npy(&mut q_file)
        .expect("Failed to read Q reference");
    
    let mut k_file = File::open(dir.join("checkpoint_02_k.npy"))
        .expect("Failed to open checkpoint_02_k.npy");
    let ref_k: Array3<f32> = Array3::read_npy(&mut k_file)
        .expect("Failed to read K reference");
    
    let mut v_file = File::open(dir.join("checkpoint_02_v.npy"))
        .expect("Failed to open checkpoint_02_v.npy");
    let ref_v: Array3<f32> = Array3::read_npy(&mut v_file)
        .expect("Failed to read V reference");
    
    println!("\n📊 HuggingFace References:");
    println!("  Q shape: {:?}", ref_q.shape());
    println!("  K shape: {:?}", ref_k.shape());
    println!("  V shape: {:?}", ref_v.shape());
    
    let ref_q_sample: Vec<f32> = ref_q.iter().take(10).copied().collect();
    let ref_k_sample: Vec<f32> = ref_k.iter().take(10).copied().collect();
    let ref_v_sample: Vec<f32> = ref_v.iter().take(10).copied().collect();
    
    println!("  Q first 10: {:?}", ref_q_sample);
    println!("  K first 10: {:?}", ref_k_sample);
    println!("  V first 10: {:?}", ref_v_sample);
    
    // Compare
    let q_diff = compare(&q, &ref_q, "Q");
    let k_diff = compare(&k, &ref_k, "K");
    let v_diff = compare(&v, &ref_v, "V");
    
    let tolerance = 1e-4;
    let all_pass = q_diff < tolerance && k_diff < tolerance && v_diff < tolerance;
    
    println!("\n📊 Summary:");
    println!("  Tolerance: {:.6e}", tolerance);
    
    if all_pass {
        println!("\n✅ PASS: All QKV outputs match HuggingFace with REAL GPT-2 weights!");
        println!("   This validates QKV projection correctness with actual model weights.");
    } else {
        println!("\n❌ FAIL: Some outputs exceed tolerance");
        if q_diff >= tolerance {
            println!("  Q difference: {:.6e} (FAIL)", q_diff);
        }
        if k_diff >= tolerance {
            println!("  K difference: {:.6e} (FAIL)", k_diff);
        }
        if v_diff >= tolerance {
            println!("  V difference: {:.6e} (FAIL)", v_diff);
        }
        panic!("QKV validation failed");
    }
}

fn compare(ours: &Array3<f32>, reference: &Array3<f32>, name: &str) -> f32 {
    let mut max_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    
    for (our, exp) in ours.iter().zip(reference.iter()) {
        let abs_diff = (our - exp).abs();
        let rel_diff = if exp.abs() > 1e-10 {
            abs_diff / exp.abs()
        } else {
            abs_diff
        };
        
        max_diff = max_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
    }
    
    println!("\n  {} comparison:", name);
    println!("    Max absolute diff: {:.6e}", max_diff);
    println!("    Max relative diff: {:.6e}", max_rel_diff);
    println!("    Status: {}", if max_diff < 1e-4 { "✅ PASS" } else { "❌ FAIL" });
    
    max_diff
}

#[test]
#[ignore] // Run with: cargo test --test real_gpt2_checkpoint_02 -- --ignored
fn test_checkpoint_02_determinism() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 2: Determinism with Real Weights            ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    let dir = weights_dir();
    
    if !dir.exists() {
        eprintln!("\n❌ GPT-2 weights not found");
        panic!("Run extract_gpt2_weights.py first");
    }
    
    // Load inputs and weights
    let ln_output: Array2<f32> = File::open(dir.join("checkpoint_01_ln1_output.npy"))
        .and_then(|mut f| Array2::read_npy(&mut f))
        .expect("Failed to load ln output");
    
    let c_attn_weight: Array2<f32> = File::open(dir.join("h0_c_attn_weight.npy"))
        .and_then(|mut f| Array2::read_npy(&mut f))
        .expect("Failed to load weight");
    
    let c_attn_bias: Array1<f32> = File::open(dir.join("h0_c_attn_bias.npy"))
        .and_then(|mut f| Array1::read_npy(&mut f))
        .expect("Failed to load bias");
    
    let weight_t = c_attn_weight.t().to_owned();
    let qkv = QKVProjection::new(weight_t, c_attn_bias, 12);
    
    // Run 3 times
    let (q1, k1, v1) = qkv.forward(&ln_output);
    let (q2, k2, v2) = qkv.forward(&ln_output);
    let (q3, k3, v3) = qkv.forward(&ln_output);
    
    // Must be bit-exact
    for (i, ((v1, v2), v3)) in q1.iter().zip(q2.iter()).zip(q3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "Q: Run 1 vs 2 differ at element {}", i);
        assert_eq!(v2.to_bits(), v3.to_bits(), "Q: Run 2 vs 3 differ at element {}", i);
    }
    
    for (i, ((v1, v2), v3)) in k1.iter().zip(k2.iter()).zip(k3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "K: Run 1 vs 2 differ at element {}", i);
    }
    
    for (i, ((v1, v2), v3)) in v1.iter().zip(v2.iter()).zip(v3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "V: Run 1 vs 2 differ at element {}", i);
    }
    
    println!("\n✅ PASS: QKV projection is deterministic with real weights");
}
