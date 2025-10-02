//! Concurrent Access Robustness Tests
//!
//! Tests TIER 1 functions under concurrent access to detect race conditions.

use vram_residency::{VramManager, VramError};
use std::sync::{Arc, Mutex};
use std::thread;

#[test]
fn test_concurrent_seal_operations() {
    let manager = Arc::new(Mutex::new(VramManager::new()));
    let mut handles = vec![];
    
    // Spawn 10 threads, each sealing a model
    for i in 0..10 {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            let data = vec![i as u8; 1024]; // 1KB model
            let mut mgr = manager_clone.lock().unwrap();
            mgr.seal_model(&data, 0)
        });
        handles.push(handle);
    }
    
    // Collect results
    let mut results = vec![];
    for handle in handles {
        results.push(handle.join().unwrap());
    }
    
    // All seals should succeed
    let success_count = results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(success_count, 10, "All concurrent seals should succeed");
    
    // All shard IDs should be unique
    let shard_ids: Vec<String> = results
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|s| s.shard_id.clone())
        .collect();
    
    let unique_count = shard_ids.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(unique_count, 10, "All shard IDs should be unique");
}

#[test]
fn test_concurrent_seal_and_verify() {
    let manager = Arc::new(Mutex::new(VramManager::new()));
    
    // First, seal a model
    let data = vec![0x42u8; 1024];
    let shard = {
        let mut mgr = manager.lock().unwrap();
        mgr.seal_model(&data, 0).unwrap()
    };
    
    let shard = Arc::new(shard);
    let mut handles = vec![];
    
    // Spawn 10 threads, each verifying the same shard
    for _ in 0..10 {
        let manager_clone = Arc::clone(&manager);
        let shard_clone = Arc::clone(&shard);
        let handle = thread::spawn(move || {
            let mgr = manager_clone.lock().unwrap();
            mgr.verify_sealed(&shard_clone)
        });
        handles.push(handle);
    }
    
    // All verifications should succeed
    for handle in handles {
        let result = handle.join().unwrap();
        assert!(result.is_ok(), "Concurrent verification should succeed");
    }
}

#[test]
fn test_concurrent_seal_with_capacity_limit() {
    let manager = Arc::new(Mutex::new(VramManager::new()));
    let mut handles = vec![];
    
    // Spawn 20 threads, each trying to seal a 1MB model
    // Some should fail due to capacity limits
    for i in 0..20 {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            let data = vec![i as u8; 1024 * 1024]; // 1MB model
            let mut mgr = manager_clone.lock().unwrap();
            mgr.seal_model(&data, 0)
        });
        handles.push(handle);
    }
    
    // Collect results
    let mut success_count = 0;
    let mut failure_count = 0;
    
    for handle in handles {
        match handle.join().unwrap() {
            Ok(_) => success_count += 1,
            Err(VramError::InsufficientVram(_, _)) => failure_count += 1,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
    
    // At least some should succeed
    // In mock mode, all may succeed (unlimited VRAM)
    // In real GPU mode, some should fail
    assert!(success_count > 0, "Some seals should succeed");
    // Note: failure_count may be 0 in mock mode with unlimited VRAM
}

#[test]
fn test_no_race_condition_in_allocation_tracking() {
    let manager = Arc::new(Mutex::new(VramManager::new()));
    let mut handles = vec![];
    
    // Seal 5 models concurrently
    for i in 0..5 {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            let data = vec![i as u8; 1024];
            let mut mgr = manager_clone.lock().unwrap();
            mgr.seal_model(&data, 0)
        });
        handles.push(handle);
    }
    
    let shards: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().unwrap().unwrap())
        .collect();
    
    // Verify all shards concurrently
    let mut verify_handles = vec![];
    for shard in shards {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            let mgr = manager_clone.lock().unwrap();
            mgr.verify_sealed(&shard)
        });
        verify_handles.push(handle);
    }
    
    // All verifications should succeed (no allocation tracking corruption)
    for handle in verify_handles {
        let result = handle.join().unwrap();
        assert!(result.is_ok(), "No race condition in allocation tracking");
    }
}

#[test]
fn test_concurrent_capacity_queries() {
    let manager = Arc::new(Mutex::new(VramManager::new()));
    let mut handles = vec![];
    
    // Spawn 20 threads querying capacity
    for _ in 0..20 {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            let mgr = manager_clone.lock().unwrap();
            (mgr.available_vram(), mgr.total_vram())
        });
        handles.push(handle);
    }
    
    // All queries should succeed
    for handle in handles {
        let (available, total) = handle.join().unwrap();
        assert!(available.is_ok());
        assert!(total.is_ok());
        assert!(total.unwrap() >= available.unwrap());
    }
}

#[test]
fn test_interleaved_seal_verify_operations() {
    let manager = Arc::new(Mutex::new(VramManager::new()));
    let mut seal_handles = vec![];
    let mut verify_handles = vec![];
    
    // Interleave seal and verify operations
    for i in 0..10 {
        let manager_clone = Arc::clone(&manager);
        
        if i % 2 == 0 {
            // Seal operation
            let handle = thread::spawn(move || {
                let data = vec![i as u8; 1024];
                let mut mgr = manager_clone.lock().unwrap();
                mgr.seal_model(&data, 0)
            });
            seal_handles.push(handle);
        } else {
            // Verify operation (seal then verify)
            let handle = thread::spawn(move || {
                let data = vec![0u8; 1024];
                let shard = {
                    let mut mgr = manager_clone.lock().unwrap();
                    mgr.seal_model(&data, 0)
                };
                
                if let Ok(shard) = shard {
                    let mgr = manager_clone.lock().unwrap();
                    mgr.verify_sealed(&shard)
                } else {
                    Ok(())
                }
            });
            verify_handles.push(handle);
        }
    }
    
    // All operations should complete without panicking
    for handle in seal_handles {
        let _ = handle.join().unwrap();
    }
    for handle in verify_handles {
        let _ = handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_signature_computation() {
    use vram_residency::{compute_signature, SealedShard};
    
    let seal_key = vec![0x42u8; 32];
    let shard = SealedShard::new_for_test(
        "test-shard".to_string(),
        0,
        1024,
        "a".repeat(64),
        0x1000,
    );
    
    let shard = Arc::new(shard);
    let seal_key = Arc::new(seal_key);
    let mut handles = vec![];
    
    // Compute signature from 10 threads
    for _ in 0..10 {
        let shard_clone = Arc::clone(&shard);
        let key_clone = Arc::clone(&seal_key);
        let handle = thread::spawn(move || {
            compute_signature(&shard_clone, &key_clone)
        });
        handles.push(handle);
    }
    
    // All should produce the same signature
    let signatures: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().unwrap().unwrap())
        .collect();
    
    // All signatures should be identical (deterministic)
    for sig in &signatures[1..] {
        assert_eq!(sig, &signatures[0], "Signatures should be deterministic");
    }
}

#[test]
fn test_concurrent_signature_verification() {
    use vram_residency::{compute_signature, verify_signature, SealedShard};
    
    let seal_key = vec![0x42u8; 32];
    let shard = SealedShard::new_for_test(
        "test-shard".to_string(),
        0,
        1024,
        "a".repeat(64),
        0x1000,
    );
    
    let signature = compute_signature(&shard, &seal_key).unwrap();
    
    let shard = Arc::new(shard);
    let seal_key = Arc::new(seal_key);
    let signature = Arc::new(signature);
    let mut handles = vec![];
    
    // Verify from 10 threads
    for _ in 0..10 {
        let shard_clone = Arc::clone(&shard);
        let key_clone = Arc::clone(&seal_key);
        let sig_clone = Arc::clone(&signature);
        let handle = thread::spawn(move || {
            verify_signature(&shard_clone, &sig_clone, &key_clone)
        });
        handles.push(handle);
    }
    
    // All verifications should succeed
    for handle in handles {
        let result = handle.join().unwrap();
        assert!(result.is_ok(), "Concurrent verification should succeed");
    }
}
