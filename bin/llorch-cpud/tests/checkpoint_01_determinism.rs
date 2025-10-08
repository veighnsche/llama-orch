//! Checkpoint 1: LayerNorm Determinism Test
//!
//! PROOF OF DETERMINISM: Validates that LayerNorm produces identical outputs
//! across multiple runs with the same model and input.
//!
//! This test satisfies stakeholder requirement for determinism proof.

use llorch_cpud::layers::LayerNorm;
use ndarray::{Array1, Array2};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Compute a hash of the output tensor for comparison
fn hash_array(arr: &Array2<f32>) -> u64 {
    let mut hasher = DefaultHasher::new();
    
    // Hash shape
    arr.shape().hash(&mut hasher);
    
    // Hash values (convert to bits for exact comparison)
    for &val in arr.iter() {
        val.to_bits().hash(&mut hasher);
    }
    
    hasher.finish()
}

/// Serialize array to bytes for exact comparison
fn array_to_bytes(arr: &Array2<f32>) -> Vec<u8> {
    let mut bytes = Vec::new();
    
    // Add shape
    for &dim in arr.shape() {
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
    
    // Add values as exact bit patterns
    for &val in arr.iter() {
        bytes.extend_from_slice(&val.to_bits().to_le_bytes());
    }
    
    bytes
}

#[test]
fn test_layer_norm_determinism_synthetic() {
    // Test with synthetic data (no model loading required)
    let dim = 1024;
    let batch_size = 2;
    
    // Create LayerNorm with fixed parameters
    let weight = Array1::ones(dim);
    let bias = Array1::zeros(dim);
    let ln = LayerNorm::new(weight, bias, 1e-5);
    
    // Create deterministic input
    let input = Array2::from_shape_fn((batch_size, dim), |(i, j)| {
        ((i * dim + j) as f32 * 0.01).sin()
    });
    
    // Run 10 times and collect outputs
    let mut outputs = Vec::new();
    let mut hashes = Vec::new();
    let mut byte_representations = Vec::new();
    
    for run in 0..10 {
        let output = ln.forward(&input);
        
        let hash = hash_array(&output);
        let bytes = array_to_bytes(&output);
        
        outputs.push(output);
        hashes.push(hash);
        byte_representations.push(bytes);
        
        println!("Run {}: hash = {:#x}", run + 1, hash);
    }
    
    // Verify all hashes are identical
    let first_hash = hashes[0];
    for (i, &hash) in hashes.iter().enumerate() {
        assert_eq!(
            hash, first_hash,
            "Run {} hash mismatch: {:#x} != {:#x}",
            i + 1, hash, first_hash
        );
    }
    
    // Verify all byte representations are identical
    let first_bytes = &byte_representations[0];
    for (i, bytes) in byte_representations.iter().enumerate() {
        assert_eq!(
            bytes, first_bytes,
            "Run {} byte-level mismatch (length: {} vs {})",
            i + 1, bytes.len(), first_bytes.len()
        );
    }
    
    // Verify all outputs are element-wise identical
    let first_output = &outputs[0];
    for (i, output) in outputs.iter().enumerate() {
        assert_eq!(
            output.shape(), first_output.shape(),
            "Run {} shape mismatch",
            i + 1
        );
        
        for (idx, (&val, &expected)) in output.iter().zip(first_output.iter()).enumerate() {
            assert_eq!(
                val.to_bits(), expected.to_bits(),
                "Run {} element {} mismatch: {} != {} (bit pattern: {:#x} vs {:#x})",
                i + 1, idx, val, expected, val.to_bits(), expected.to_bits()
            );
        }
    }
    
    println!("✅ DETERMINISM PROOF: All 10 runs produced identical outputs");
    println!("   - Hash: {:#x}", first_hash);
    println!("   - Byte representation: {} bytes", first_bytes.len());
    println!("   - Element count: {}", first_output.len());
}

#[test]
fn test_layer_norm_determinism_different_inputs() {
    // Verify that different inputs produce different outputs (sanity check)
    let dim = 1024;
    let weight = Array1::ones(dim);
    let bias = Array1::zeros(dim);
    let ln = LayerNorm::new(weight, bias, 1e-5);
    
    // Input 1
    let input1 = Array2::from_shape_fn((1, dim), |(_, j)| (j as f32 * 0.01).sin());
    let output1 = ln.forward(&input1);
    let hash1 = hash_array(&output1);
    
    // Input 2 (different)
    let input2 = Array2::from_shape_fn((1, dim), |(_, j)| (j as f32 * 0.02).cos());
    let output2 = ln.forward(&input2);
    let hash2 = hash_array(&output2);
    
    // Also verify element-wise difference
    let mut has_difference = false;
    for (&v1, &v2) in output1.iter().zip(output2.iter()) {
        if v1.to_bits() != v2.to_bits() {
            has_difference = true;
            break;
        }
    }
    assert!(has_difference, "Different inputs should produce different outputs");
    
    // Hashes should be different
    assert_ne!(
        hash1, hash2,
        "Different inputs should produce different outputs"
    );
    
    println!("✅ SANITY CHECK: Different inputs produce different outputs");
    println!("   - Input 1 hash: {:#x}", hash1);
    println!("   - Input 2 hash: {:#x}", hash2);
}

#[test]
fn test_layer_norm_determinism_with_scale_bias() {
    // Test determinism with non-trivial scale and bias
    let dim = 768;
    
    // Create non-trivial parameters
    let weight = Array1::from_shape_fn(dim, |i| 1.0 + (i as f32 * 0.001).sin());
    let bias = Array1::from_shape_fn(dim, |i| (i as f32 * 0.002).cos());
    let ln = LayerNorm::new(weight, bias, 1e-5);
    
    // Create input
    let input = Array2::from_shape_fn((3, dim), |(i, j)| {
        ((i * dim + j) as f32 * 0.01).sin()
    });
    
    // Run 5 times
    let mut hashes = Vec::new();
    for run in 0..5 {
        let output = ln.forward(&input);
        let hash = hash_array(&output);
        hashes.push(hash);
        println!("Run {} (with scale/bias): hash = {:#x}", run + 1, hash);
    }
    
    // All hashes must be identical
    let first_hash = hashes[0];
    for (i, &hash) in hashes.iter().enumerate() {
        assert_eq!(
            hash, first_hash,
            "Run {} hash mismatch with scale/bias",
            i + 1
        );
    }
    
    println!("✅ DETERMINISM PROOF (with scale/bias): All 5 runs identical");
}

#[test]
fn test_layer_norm_determinism_batch_processing() {
    // Test determinism with various batch sizes
    let dim = 512;
    let weight = Array1::ones(dim);
    let bias = Array1::zeros(dim);
    let ln = LayerNorm::new(weight, bias, 1e-5);
    
    for batch_size in [1, 2, 4, 8, 16] {
        let input = Array2::from_shape_fn((batch_size, dim), |(i, j)| {
            ((i * dim + j) as f32 * 0.01).sin()
        });
        
        // Run 3 times per batch size
        let mut hashes = Vec::new();
        for _ in 0..3 {
            let output = ln.forward(&input);
            hashes.push(hash_array(&output));
        }
        
        // Verify determinism
        assert_eq!(hashes[0], hashes[1], "Batch {} run 1 vs 2", batch_size);
        assert_eq!(hashes[1], hashes[2], "Batch {} run 2 vs 3", batch_size);
        
        println!("✅ Batch size {}: deterministic (hash: {:#x})", batch_size, hashes[0]);
    }
    
    println!("✅ DETERMINISM PROOF: All batch sizes deterministic");
}

#[cfg(test)]
mod proof_bundle {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    
    /// Generate proof bundle for stakeholders
    #[test]
    #[ignore] // Run explicitly with: cargo test --test checkpoint_01_determinism proof_bundle -- --ignored
    fn generate_determinism_proof_bundle() {
        let proof_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".proof_bundle")
            .join("determinism")
            .join("checkpoint_01_layer_norm");
        
        fs::create_dir_all(&proof_dir).expect("Failed to create proof bundle directory");
        
        let dim = 1024;
        let batch_size = 2;
        
        let weight = Array1::ones(dim);
        let bias = Array1::zeros(dim);
        let ln = LayerNorm::new(weight, bias, 1e-5);
        
        let input = Array2::from_shape_fn((batch_size, dim), |(i, j)| {
            ((i * dim + j) as f32 * 0.01).sin()
        });
        
        // Run 100 times and record all outputs
        let mut proof_data = String::new();
        proof_data.push_str("# LayerNorm Determinism Proof Bundle\n\n");
        proof_data.push_str(&format!("**Generated:** {}\n", chrono::Utc::now().to_rfc3339()));
        proof_data.push_str(&format!("**Test:** checkpoint_01_layer_norm\n"));
        proof_data.push_str(&format!("**Runs:** 100\n\n"));
        proof_data.push_str("## Configuration\n\n");
        proof_data.push_str(&format!("- Input shape: [{}, {}]\n", batch_size, dim));
        proof_data.push_str(&format!("- Weight: ones({})\n", dim));
        proof_data.push_str(&format!("- Bias: zeros({})\n", dim));
        proof_data.push_str(&format!("- Epsilon: 1e-5\n\n"));
        proof_data.push_str("## Results\n\n");
        proof_data.push_str("| Run | Hash | First 5 Elements | Last 5 Elements |\n");
        proof_data.push_str("|-----|------|------------------|------------------|\n");
        
        let mut all_hashes = Vec::new();
        let mut all_outputs = Vec::new();
        
        for run in 0..100 {
            let output = ln.forward(&input);
            let hash = hash_array(&output);
            
            let first_5: Vec<f32> = output.iter().take(5).copied().collect();
            let total = output.len();
            let last_5: Vec<f32> = output.iter().skip(total.saturating_sub(5)).copied().collect();
            
            proof_data.push_str(&format!(
                "| {} | {:#018x} | {:?} | {:?} |\n",
                run + 1, hash, first_5, last_5
            ));
            
            all_hashes.push(hash);
            all_outputs.push(output);
        }
        
        // Verify all identical
        let first_hash = all_hashes[0];
        let all_identical = all_hashes.iter().all(|&h| h == first_hash);
        
        proof_data.push_str("\n## Verification\n\n");
        proof_data.push_str(&format!("- **All hashes identical:** {}\n", all_identical));
        proof_data.push_str(&format!("- **Reference hash:** {:#018x}\n", first_hash));
        proof_data.push_str(&format!("- **Total runs:** {}\n", all_hashes.len()));
        proof_data.push_str(&format!("- **Unique hashes:** {}\n", {
            let mut unique = all_hashes.clone();
            unique.sort();
            unique.dedup();
            unique.len()
        }));
        
        if all_identical {
            proof_data.push_str("\n✅ **PROOF COMPLETE:** LayerNorm is 100% deterministic across 100 runs.\n");
        } else {
            proof_data.push_str("\n❌ **PROOF FAILED:** Non-deterministic behavior detected.\n");
        }
        
        // Write proof bundle
        let proof_file = proof_dir.join("determinism_proof.md");
        fs::write(&proof_file, proof_data).expect("Failed to write proof bundle");
        
        println!("✅ Proof bundle generated: {}", proof_file.display());
        println!("   All hashes identical: {}", all_identical);
        
        assert!(all_identical, "Determinism proof failed");
    }
}
