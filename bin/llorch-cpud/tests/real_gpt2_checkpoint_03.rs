//! Checkpoint 3: KV Cache with REAL GPT-2 weights
//!
//! This test validates KV cache storage and retrieval using REAL GPT-2 K/V from Checkpoint 2.

use llorch_cpud::cache::KVCache;
use ndarray::Array3;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::PathBuf;

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.test-models/gpt2/extracted_weights")
}

#[test]
fn test_checkpoint_03_real_gpt2() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 3: KV Cache with REAL GPT-2                 ║");
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
    
    // Load REAL K, V from Checkpoint 2
    let mut k_file = File::open(dir.join("checkpoint_02_k.npy"))
        .expect("Failed to open checkpoint_02_k.npy");
    let k: Array3<f32> = Array3::read_npy(&mut k_file)
        .expect("Failed to read K");
    
    let mut v_file = File::open(dir.join("checkpoint_02_v.npy"))
        .expect("Failed to open checkpoint_02_v.npy");
    let v: Array3<f32> = Array3::read_npy(&mut v_file)
        .expect("Failed to read V");
    
    println!("\n📊 Real GPT-2 K/V:");
    println!("  K shape: {:?}", k.shape());
    println!("  V shape: {:?}", v.shape());
    
    // Create cache and update
    let mut cache = KVCache::new(2048, 12, 64);  // GPT-2 base: 12 heads, 64 dim
    cache.update(&k, &v, 0);
    
    // Retrieve - get up to seq_len (which is 2 for "Hello.")
    let seq_len = k.shape()[0];  // FIXED: Use first dimension (seq), not second (n_heads)
    let (cached_k, cached_v) = cache.get(seq_len);
    
    println!("\n📊 Retrieved from cache:");
    println!("  K shape: {:?}", cached_k.shape());
    println!("  V shape: {:?}", cached_v.shape());
    
    // CRITICAL: Validate shapes before comparing values
    assert_eq!(cached_k.shape(), k.shape(), 
        "K shape mismatch: cached={:?} vs input={:?}", cached_k.shape(), k.shape());
    assert_eq!(cached_v.shape(), v.shape(), 
        "V shape mismatch: cached={:?} vs input={:?}", cached_v.shape(), v.shape());
    
    // Validate no NaN/Inf
    for val in cached_k.iter().chain(cached_v.iter()) {
        assert!(val.is_finite(), "Cache contains NaN or Inf: {}", val);
    }
    
    // Compare with original (should be exact)
    let mut k_diff = 0.0f32;
    let mut v_diff = 0.0f32;
    
    for (our, exp) in cached_k.iter().zip(k.iter()) {
        k_diff = k_diff.max((our - exp).abs());
    }
    
    for (our, exp) in cached_v.iter().zip(v.iter()) {
        v_diff = v_diff.max((our - exp).abs());
    }
    
    println!("\n📊 Comparison:");
    println!("  K max diff: {:.6e}", k_diff);
    println!("  V max diff: {:.6e}", v_diff);
    println!("  Tolerance: EXACT (cache must be bit-perfect)");
    
    if k_diff == 0.0 && v_diff == 0.0 {
        println!("\n✅ PASS: KV cache stores and retrieves correctly!");
        println!("   Cache is bit-perfect with REAL GPT-2 K/V.");
    } else {
        println!("\n❌ FAIL: Cache has errors");
        panic!("K diff: {}, V diff: {}", k_diff, v_diff);
    }
}

#[test]
fn test_checkpoint_03_determinism() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 3: Determinism with Real K/V                ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    
    let dir = weights_dir();
    
    if !dir.exists() {
        eprintln!("\n❌ GPT-2 weights not found");
        panic!("Run extract_gpt2_weights.py first");
    }
    
    // Load K, V
    let mut k_file = File::open(dir.join("checkpoint_02_k.npy"))
        .expect("Failed to open K file");
    let k: Array3<f32> = Array3::read_npy(&mut k_file)
        .expect("Failed to load K");
    
    let mut v_file = File::open(dir.join("checkpoint_02_v.npy"))
        .expect("Failed to open V file");
    let v: Array3<f32> = Array3::read_npy(&mut v_file)
        .expect("Failed to load V");
    
    let seq_len = k.shape()[1];
    
    // Run 3 times
    let mut cache1 = KVCache::new(2048, 12, 64);
    cache1.update(&k, &v, 0);
    let (k1, v1) = cache1.get(seq_len);
    
    let mut cache2 = KVCache::new(2048, 12, 64);
    cache2.update(&k, &v, 0);
    let (k2, v2) = cache2.get(seq_len);
    
    let mut cache3 = KVCache::new(2048, 12, 64);
    cache3.update(&k, &v, 0);
    let (k3, v3) = cache3.get(seq_len);
    
    // Must be bit-exact
    for (i, ((v1, v2), v3)) in k1.iter().zip(k2.iter()).zip(k3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "K: Run 1 vs 2 differ at element {}", i);
        assert_eq!(v2.to_bits(), v3.to_bits(), "K: Run 2 vs 3 differ at element {}", i);
    }
    
    for (i, ((v1, v2), v3)) in v1.iter().zip(v2.iter()).zip(v3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "V: Run 1 vs 2 differ at element {}", i);
    }
    
    println!("\n✅ PASS: KV cache is deterministic with real K/V");
}
