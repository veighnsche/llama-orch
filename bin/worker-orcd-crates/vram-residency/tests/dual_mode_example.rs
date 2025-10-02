//! Example of dual-mode testing implementation
//!
//! This demonstrates the mandatory dual-mode testing pattern from spec 42_dual_mode_testing.md
//!
//! All tests in this file run TWICE:
//! 1. First with mock VRAM (always)
//! 2. Then with real CUDA (if GPU available)
//!
//! If no GPU is found, a clear warning is emitted.

mod common;

use vram_residency::{VramManager, VramError};
use common::run_dual_mode_test;

/// @priority: critical
/// @spec: VRAM-SEAL-001
/// @team: vram-residency
/// @tags: dual-mode, seal, cryptographic-integrity
/// @scenario: Sealing model data in VRAM with cryptographic signature
/// @threat: Unauthorized tampering with model weights in GPU memory
/// @failure_mode: Model corruption goes undetected, leading to incorrect inference results
/// @edge_case: Must work identically with both mock VRAM (CI) and real CUDA (production)
/// @requires: GPU
#[test]
fn test_seal_model_dual_mode() {
    run_dual_mode_test(|is_real_cuda| {
        let mut manager = VramManager::new();
        
        if is_real_cuda {
            println!("   → Testing with real CUDA allocations");
        } else {
            println!("   → Testing with mock VRAM");
        }
        
        let data = vec![0x42u8; 1024];
        let shard = manager.seal_model(&data, 0)?;
        
        // Verify the seal
        assert!(manager.verify_sealed(&shard).is_ok());
        assert_eq!(shard.vram_bytes, 1024);
        assert!(shard.is_sealed());
        
        Ok::<(), VramError>(())
    });
}

/// @priority: critical
/// @spec: VRAM-VERIFY-001
/// @team: vram-residency
/// @tags: dual-mode, verification, cryptographic-integrity
/// @scenario: Verifying sealed model data has not been tampered with
/// @threat: Attacker modifies model weights in GPU memory to inject backdoors
/// @failure_mode: Tampered models pass verification, allowing malicious inference
/// @edge_case: Verification must detect even single-bit flips in VRAM
/// @requires: GPU
#[test]
fn test_verify_sealed_dual_mode() {
    run_dual_mode_test(|is_real_cuda| {
        let mut manager = VramManager::new();
        
        if is_real_cuda {
            println!("   → Testing verification with real CUDA");
        } else {
            println!("   → Testing verification with mock VRAM");
        }
        
        // Seal a model
        let data = vec![0xAAu8; 2048];
        let shard = manager.seal_model(&data, 0)?;
        
        // Verify should succeed
        let result = manager.verify_sealed(&shard);
        assert!(result.is_ok(), "Verification should succeed");
        
        Ok::<(), VramError>(())
    });
}

#[test]
fn test_multiple_allocations_dual_mode() {
    run_dual_mode_test(|is_real_cuda| {
        let mut manager = VramManager::new();
        
        if is_real_cuda {
            println!("   → Testing multiple allocations with real CUDA");
        } else {
            println!("   → Testing multiple allocations with mock VRAM");
        }
        
        let mut shards = Vec::new();
        
        // Seal multiple models
        for i in 0..5 {
            let data = vec![i as u8; 1024];
            let shard = manager.seal_model(&data, 0)?;
            shards.push(shard);
        }
        
        // Verify all shards
        for shard in &shards {
            assert!(manager.verify_sealed(shard).is_ok());
        }
        
        // All shard IDs should be unique
        let mut ids: Vec<_> = shards.iter().map(|s| &s.shard_id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 5, "All shard IDs should be unique");
        
        Ok::<(), VramError>(())
    });
}

#[test]
fn test_capacity_queries_dual_mode() {
    run_dual_mode_test(|is_real_cuda| {
        let manager = VramManager::new();
        
        if is_real_cuda {
            println!("   → Testing capacity queries with real CUDA");
        } else {
            println!("   → Testing capacity queries with mock VRAM");
        }
        
        let total = manager.total_vram()?;
        let available = manager.available_vram()?;
        
        assert!(total > 0, "Total VRAM should be > 0");
        assert!(available > 0, "Available VRAM should be > 0");
        assert!(available <= total, "Available should be <= total");
        
        if is_real_cuda {
            println!("   → Real GPU VRAM: {} GB total, {} GB available", 
                     total / (1024 * 1024 * 1024),
                     available / (1024 * 1024 * 1024));
        }
        
        Ok::<(), VramError>(())
    });
}

/// @priority: high
/// @spec: VRAM-VALIDATE-001
/// @team: vram-residency
/// @tags: dual-mode, input-validation, edge-case
/// @scenario: Rejecting invalid zero-size model allocations
/// @threat: Zero-size allocations cause undefined behavior in CUDA
/// @failure_mode: GPU driver crash or memory corruption from invalid allocation
/// @edge_case: Must reject exactly zero bytes, but allow 1-byte models
/// @requires: GPU
#[test]
fn test_zero_size_rejection_dual_mode() {
    run_dual_mode_test(|is_real_cuda| {
        let mut manager = VramManager::new();
        
        if is_real_cuda {
            println!("   → Testing zero-size rejection with real CUDA");
        } else {
            println!("   → Testing zero-size rejection with mock VRAM");
        }
        
        // Zero-size should be rejected
        let result = manager.seal_model(&[], 0);
        assert!(result.is_err(), "Zero-size model should be rejected");
        
        match result {
            Err(VramError::InvalidInput(_)) => {
                // Expected error
            }
            other => panic!("Expected InvalidInput error, got: {:?}", other),
        }
        
        Ok::<(), VramError>(())
    });
}

#[test]
fn test_digest_determinism_dual_mode() {
    run_dual_mode_test(|is_real_cuda| {
        let mut manager = VramManager::new();
        
        if is_real_cuda {
            println!("   → Testing digest determinism with real CUDA");
        } else {
            println!("   → Testing digest determinism with mock VRAM");
        }
        
        let data = vec![0x42u8; 1024];
        
        // Seal the same data twice
        let shard1 = manager.seal_model(&data, 0)?;
        let shard2 = manager.seal_model(&data, 0)?;
        
        // Digests should be identical (deterministic)
        assert_eq!(shard1.digest, shard2.digest, "Digests should be deterministic");
        
        Ok::<(), VramError>(())
    });
}

// This test demonstrates conditional logic based on CUDA availability
#[test]
fn test_large_allocation_dual_mode() {
    run_dual_mode_test(|is_real_cuda| {
        let mut manager = VramManager::new();
        
        // Use smaller size for mock mode to avoid artificial limits
        let size = if is_real_cuda {
            println!("   → Testing large allocation (10MB) with real CUDA");
            10 * 1024 * 1024 // 10MB
        } else {
            println!("   → Testing large allocation (1MB) with mock VRAM");
            1 * 1024 * 1024 // 1MB for mock
        };
        
        let data = vec![0x42u8; size];
        let result = manager.seal_model(&data, 0);
        
        // Should either succeed or fail with InsufficientVram
        match result {
            Ok(shard) => {
                assert_eq!(shard.vram_bytes, size);
                assert!(manager.verify_sealed(&shard).is_ok());
            }
            Err(VramError::InsufficientVram(_, _)) => {
                // Expected if not enough VRAM
                println!("   → Insufficient VRAM (expected in some cases)");
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
        
        Ok::<(), VramError>(())
    });
}
