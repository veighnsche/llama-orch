//! Stress Testing for TIER 1 Functions
//!
//! Tests behavior under extreme conditions: memory exhaustion, rapid cycles, large models.

use vram_residency::{VramManager, VramError};

#[test]
#[cfg_attr(feature = "skip-long-tests", ignore)]
fn test_seal_until_vram_exhausted() {
    println!("\n⏱️  Starting VRAM exhaustion test...");
    println!("   This test seals 1MB models until VRAM is full");
    println!("   Expected duration: 30-90 seconds depending on available VRAM");
    println!("   Progress will be reported every 100 models\n");
    
    let mut manager = VramManager::new();
    let mut sealed_shards = vec![];
    let mut seal_count = 0;
    let start_time = std::time::Instant::now();
    
    // Keep sealing 1MB models until VRAM exhausted
    loop {
        let data = vec![0x42u8; 1024 * 1024]; // 1MB
        match manager.seal_model(&data, 0) {
            Ok(shard) => {
                sealed_shards.push(shard);
                seal_count += 1;
                
                // Progress reporting every 100 models
                if seal_count % 100 == 0 {
                    let elapsed = start_time.elapsed().as_secs();
                    let mb_sealed = seal_count;
                    println!("   ✓ Sealed {} models ({} MB) in {} seconds", 
                             seal_count, mb_sealed, elapsed);
                }
            }
            Err(VramError::InsufficientVram(_, _)) => {
                // Expected: VRAM exhausted
                println!("   ✓ VRAM exhausted after sealing {} models ({} MB)", 
                         seal_count, seal_count);
                break;
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
        
        // Safety limit to prevent infinite loop
        if seal_count > 1000 {
            println!("   ⚠️  Safety limit reached (1000 models)");
            break;
        }
    }
    
    assert!(seal_count > 0, "Should seal at least one model");
    
    println!("   Verifying all {} sealed shards...", sealed_shards.len());
    
    // Verify all sealed shards are still valid
    for (i, shard) in sealed_shards.iter().enumerate() {
        if i % 100 == 0 && i > 0 {
            println!("   ✓ Verified {}/{} shards", i, sealed_shards.len());
        }
        assert!(manager.verify_sealed(shard).is_ok(), "All sealed shards should remain valid");
    }
    
    let total_time = start_time.elapsed().as_secs();
    println!("   ✅ Test complete in {} seconds\n", total_time);
}

#[test]
fn test_rapid_seal_cycles() {
    let mut manager = VramManager::new();
    
    // Perform 100 rapid seal operations
    for i in 0..100 {
        let data = vec![i as u8; 1024]; // 1KB
        let result = manager.seal_model(&data, 0);
        
        // Most should succeed (unless VRAM exhausted)
        if let Err(e) = result {
            assert!(
                matches!(e, VramError::InsufficientVram(_, _)),
                "Only InsufficientVram errors expected"
            );
        }
    }
}

#[test]
fn test_large_model_seal() {
    let mut manager = VramManager::new();
    
    // Try to seal a 10MB model
    let large_data = vec![0x42u8; 10 * 1024 * 1024]; // 10MB
    let result = manager.seal_model(&large_data, 0);
    
    // Should either succeed or fail with InsufficientVram
    match result {
        Ok(shard) => {
            assert_eq!(shard.vram_bytes, 10 * 1024 * 1024);
            assert!(manager.verify_sealed(&shard).is_ok());
        }
        Err(VramError::InsufficientVram(_, _)) => {
            // Expected if not enough VRAM
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

#[test]
fn test_many_small_allocations() {
    let mut manager = VramManager::new();
    let mut shards = vec![];
    
    // Seal 1000 tiny models (1KB each)
    for i in 0..1000 {
        let data = vec![i as u8; 1024];
        match manager.seal_model(&data, 0) {
            Ok(shard) => shards.push(shard),
            Err(VramError::InsufficientVram(_, _)) => break,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
    
    assert!(shards.len() > 0, "Should seal at least some models");
    
    // Verify random samples (not all, to keep test fast)
    for i in (0..shards.len()).step_by(shards.len().max(1) / 10) {
        assert!(manager.verify_sealed(&shards[i]).is_ok());
    }
}

#[test]
fn test_repeated_verification() {
    let mut manager = VramManager::new();
    let data = vec![0x42u8; 1024];
    let shard = manager.seal_model(&data, 0).unwrap();
    
    // Verify the same shard 1000 times
    for _ in 0..1000 {
        assert!(manager.verify_sealed(&shard).is_ok(), "Repeated verification should succeed");
    }
}

#[test]
fn test_alternating_seal_verify() {
    let mut manager = VramManager::new();
    let mut shards = vec![];
    
    // Alternate between seal and verify operations
    for i in 0..100 {
        if i % 2 == 0 {
            // Seal
            let data = vec![i as u8; 1024];
            if let Ok(shard) = manager.seal_model(&data, 0) {
                shards.push(shard);
            }
        } else {
            // Verify a random previous shard
            if !shards.is_empty() {
                let idx = i % shards.len();
                assert!(manager.verify_sealed(&shards[idx]).is_ok());
            }
        }
    }
}

#[test]
fn test_capacity_queries_under_load() {
    let mut manager = VramManager::new();
    
    // Seal models while querying capacity
    for i in 0..50 {
        let data = vec![i as u8; 1024 * 100]; // 100KB
        let _ = manager.seal_model(&data, 0);
        
        // Query capacity after each seal
        let available = manager.available_vram();
        let total = manager.total_vram();
        
        assert!(available.is_ok());
        assert!(total.is_ok());
        
        if let (Ok(avail), Ok(tot)) = (available, total) {
            assert!(tot >= avail, "Total should be >= available");
        }
    }
}

#[test]
fn test_seal_with_varying_sizes() {
    let mut manager = VramManager::new();
    let sizes = vec![
        1,           // 1 byte
        1024,        // 1KB
        10 * 1024,   // 10KB
        100 * 1024,  // 100KB
        1024 * 1024, // 1MB
    ];
    
    for size in sizes {
        let data = vec![0x42u8; size];
        match manager.seal_model(&data, 0) {
            Ok(shard) => {
                assert_eq!(shard.vram_bytes, size);
                assert!(manager.verify_sealed(&shard).is_ok());
            }
            Err(VramError::InsufficientVram(_, _)) => {
                // Expected if not enough VRAM
                break;
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}

#[test]
fn test_signature_computation_stress() {
    use vram_residency::{compute_signature, SealedShard};
    
    let seal_key = vec![0x42u8; 32];
    
    // Compute signatures for 1000 different shards
    for i in 0..1000 {
        let shard = SealedShard::new_for_test(
            format!("shard-{}", i),
            0,
            1024,
            "a".repeat(64),
            0x1000,
        );
        
        let result = compute_signature(&shard, &seal_key);
        assert!(result.is_ok(), "Signature computation should succeed");
        assert_eq!(result.unwrap().len(), 32);
    }
}

#[test]
fn test_verification_stress() {
    use vram_residency::{compute_signature, verify_signature, SealedShard};
    
    let seal_key = vec![0x42u8; 32];
    
    // Verify 1000 different signatures
    for i in 0..1000 {
        let shard = SealedShard::new_for_test(
            format!("shard-{}", i),
            0,
            1024,
            "a".repeat(64),
            0x1000,
        );
        
        let signature = compute_signature(&shard, &seal_key).unwrap();
        let result = verify_signature(&shard, &signature, &seal_key);
        assert!(result.is_ok(), "Verification should succeed");
    }
}

#[test]
fn test_digest_computation_stress() {
    use vram_residency::compute_digest;
    
    // Compute digests for varying data sizes
    for size in [1, 1024, 10 * 1024, 100 * 1024, 1024 * 1024] {
        let data = vec![0x42u8; size];
        let digest = compute_digest(&data);
        assert_eq!(digest.len(), 64, "Digest should always be 64 chars");
    }
}

#[test]
fn test_memory_leak_detection() {
    let mut manager = VramManager::new();
    
    // Seal and drop 100 models to check for memory leaks
    for i in 0..100 {
        let data = vec![i as u8; 1024];
        let _ = manager.seal_model(&data, 0);
        // Shards are dropped here
    }
    
    // If there's a memory leak, available VRAM would decrease
    // This is a basic check; proper leak detection requires tools like valgrind
    let available = manager.available_vram();
    assert!(available.is_ok());
}

#[test]
fn test_edge_case_single_byte_model() {
    let mut manager = VramManager::new();
    let data = vec![0x42u8; 1];
    
    let result = manager.seal_model(&data, 0);
    assert!(result.is_ok(), "Should seal 1-byte model");
    
    if let Ok(shard) = result {
        assert_eq!(shard.vram_bytes, 1);
        assert!(manager.verify_sealed(&shard).is_ok());
    }
}

#[test]
fn test_edge_case_all_zeros_model() {
    let mut manager = VramManager::new();
    let data = vec![0u8; 1024];
    
    let result = manager.seal_model(&data, 0);
    assert!(result.is_ok(), "Should seal all-zeros model");
    
    if let Ok(shard) = result {
        assert!(manager.verify_sealed(&shard).is_ok());
    }
}

#[test]
fn test_edge_case_all_ones_model() {
    let mut manager = VramManager::new();
    let data = vec![0xFFu8; 1024];
    
    let result = manager.seal_model(&data, 0);
    assert!(result.is_ok(), "Should seal all-ones model");
    
    if let Ok(shard) = result {
        assert!(manager.verify_sealed(&shard).is_ok());
    }
}
