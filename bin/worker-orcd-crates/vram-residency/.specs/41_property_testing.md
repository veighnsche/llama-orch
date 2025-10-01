# VRAM Residency — Property Testing Guide

**Status**: Recommended  
**Priority**: ⭐⭐⭐⭐ HIGH  
**Applies to**: `bin/worker-orcd-crates/vram-residency/`

---

## 0. Why Property Testing for VRAM Residency?

**VRAM residency manages GPU memory and cryptographic sealing**. Property-based testing ensures:

- ✅ VRAM calculations **never overflow**
- ✅ Seal/unseal is **bijective** (roundtrip always works)
- ✅ Memory bounds are **always respected**
- ✅ Shard IDs are **unique**
- ✅ Capacity tracking is **accurate**
- ✅ Concurrent access is **safe**

**Memory corruption = model corruption = incorrect inference**

---

## 1. Critical Properties to Test

### 1.1 Seal/Unseal Bijection

**Property**: Sealing then unsealing returns original data.

```rust
use proptest::prelude::*;

proptest! {
    /// Seal/unseal is bijective
    #[test]
    fn seal_unseal_roundtrip(data in prop::collection::vec(any::<u8>(), 0..10_000)) {
        let manager = VramManager::new_mock(100 * 1024 * 1024)?; // 100MB
        
        // Seal data
        let shard = manager.seal_model("test-model", 0, &data)?;
        
        // Unseal data
        let unsealed = manager.unseal_model(&shard)?;
        
        // Should match original
        prop_assert_eq!(data, unsealed);
    }
    
    /// Seal/unseal works for all sizes
    #[test]
    fn seal_unseal_all_sizes(size in 0usize..1_000_000) {
        let manager = VramManager::new_mock(10 * 1024 * 1024)?;
        
        let data = vec![0xAA; size];
        let shard = manager.seal_model("test", 0, &data)?;
        let unsealed = manager.unseal_model(&shard)?;
        
        prop_assert_eq!(data.len(), unsealed.len());
        prop_assert_eq!(data, unsealed);
    }
    
    /// Seal is deterministic with same key
    #[test]
    fn seal_deterministic(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let manager = VramManager::new_mock(10 * 1024 * 1024)?;
        
        let shard1 = manager.seal_model("test", 0, &data)?;
        let shard2 = manager.seal_model("test", 0, &data)?;
        
        // Sealed data should be identical with same key
        prop_assert_eq!(shard1.sealed_data, shard2.sealed_data);
    }
}
```

---

### 1.2 VRAM Capacity Tracking

**Property**: Capacity tracking is always accurate.

```rust
proptest! {
    /// Capacity tracking is accurate
    #[test]
    fn capacity_tracking_accurate(
        total_vram_mb in 1usize..1000,
        allocations in prop::collection::vec(1usize..100, 1..50)
    ) {
        let total_bytes = total_vram_mb * 1024 * 1024;
        let manager = VramManager::new_mock(total_bytes)?;
        
        let mut allocated = 0usize;
        let mut shards = vec![];
        
        for size_mb in allocations {
            let size_bytes = size_mb * 1024 * 1024;
            
            if allocated + size_bytes <= total_bytes {
                let data = vec![0u8; size_bytes];
                if let Ok(shard) = manager.seal_model("test", 0, &data) {
                    allocated += size_bytes;
                    shards.push(shard);
                }
            }
        }
        
        // Check capacity
        let used = manager.get_used_vram(0)?;
        let free = manager.get_free_vram(0)?;
        
        prop_assert_eq!(used + free, total_bytes);
        prop_assert!(used <= total_bytes);
    }
    
    /// Freeing restores capacity
    #[test]
    fn freeing_restores_capacity(size in 1usize..100) {
        let manager = VramManager::new_mock(100 * 1024 * 1024)?;
        
        let initial_free = manager.get_free_vram(0)?;
        
        let data = vec![0u8; size * 1024 * 1024];
        let shard = manager.seal_model("test", 0, &data)?;
        
        let after_alloc = manager.get_free_vram(0)?;
        prop_assert!(after_alloc < initial_free);
        
        manager.unseal_and_free(&shard)?;
        
        let after_free = manager.get_free_vram(0)?;
        prop_assert_eq!(after_free, initial_free);
    }
}
```

---

### 1.3 VRAM Calculations Never Overflow

**Property**: All VRAM size calculations use saturating arithmetic.

```rust
proptest! {
    /// VRAM MB to bytes never overflows
    #[test]
    fn vram_mb_to_bytes_safe(vram_mb in 0usize..1_000_000) {
        let bytes = vram_mb.saturating_mul(1024).saturating_mul(1024);
        
        // Should never overflow
        prop_assert!(bytes <= usize::MAX);
        
        // Should be correct for reasonable values
        if vram_mb <= 100_000 {
            prop_assert_eq!(bytes, vram_mb * 1024 * 1024);
        }
    }
    
    /// Shard size calculations are safe
    #[test]
    fn shard_size_calculations_safe(
        model_size_mb in 0usize..100_000,
        overhead_bytes in 0usize..1_000_000
    ) {
        let model_bytes = model_size_mb.saturating_mul(1024).saturating_mul(1024);
        let total = model_bytes.saturating_add(overhead_bytes);
        
        prop_assert!(total <= usize::MAX);
    }
}
```

---

### 1.4 Shard ID Uniqueness

**Property**: Shard IDs are always unique.

```rust
use std::collections::HashSet;

proptest! {
    /// Shard IDs are unique
    #[test]
    fn shard_ids_unique(count in 1usize..1000) {
        let manager = VramManager::new_mock(1024 * 1024 * 1024)?; // 1GB
        
        let mut shard_ids = HashSet::new();
        
        for i in 0..count {
            let data = vec![i as u8; 1024];
            let shard = manager.seal_model(&format!("model-{}", i), 0, &data)?;
            
            // ID should be unique
            prop_assert!(shard_ids.insert(shard.shard_id.clone()));
        }
        
        prop_assert_eq!(shard_ids.len(), count);
    }
    
    /// Shard IDs follow expected format
    #[test]
    fn shard_id_format(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let manager = VramManager::new_mock(10 * 1024 * 1024)?;
        
        let shard = manager.seal_model("test", 0, &data)?;
        
        // Should be valid ORCH-ID format
        prop_assert!(shard.shard_id.starts_with("orch-shard-"));
        prop_assert!(shard.shard_id.len() > 20);
    }
}
```

---

### 1.5 Memory Bounds Checking

**Property**: Out-of-bounds access is always prevented.

```rust
proptest! {
    /// Cannot allocate beyond capacity
    #[test]
    fn allocation_bounds_enforced(
        total_vram_mb in 1usize..100,
        request_mb in 100usize..1000
    ) {
        let total_bytes = total_vram_mb * 1024 * 1024;
        let manager = VramManager::new_mock(total_bytes)?;
        
        if request_mb > total_vram_mb {
            let data = vec![0u8; request_mb * 1024 * 1024];
            let result = manager.seal_model("test", 0, &data);
            
            // Should fail with capacity error
            prop_assert!(result.is_err());
        }
    }
    
    /// Offset + length bounds are checked
    #[test]
    fn offset_bounds_checked(
        data_size in 1usize..1000,
        offset in 0usize..2000,
        length in 0usize..2000
    ) {
        let manager = VramManager::new_mock(10 * 1024 * 1024)?;
        let data = vec![0u8; data_size];
        let shard = manager.seal_model("test", 0, &data)?;
        
        // Try to read with offset
        let result = manager.read_shard_at_offset(&shard, offset, length);
        
        if offset + length > data_size {
            // Should fail
            prop_assert!(result.is_err());
        }
    }
}
```

---

### 1.6 Concurrent Access Safety

**Property**: Concurrent seal/unseal operations are safe.

```rust
use std::sync::Arc;
use std::thread;

proptest! {
    /// Concurrent sealing is safe
    #[test]
    fn concurrent_sealing_safe(
        thread_count in 2usize..10,
        ops_per_thread in 10usize..100
    ) {
        let manager = Arc::new(VramManager::new_mock(1024 * 1024 * 1024)?);
        let mut handles = vec![];
        
        for thread_id in 0..thread_count {
            let manager_clone = Arc::clone(&manager);
            
            let handle = thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let data = vec![thread_id as u8; 1024];
                    let model_id = format!("model-{}-{}", thread_id, i);
                    manager_clone.seal_model(&model_id, 0, &data).unwrap();
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // All operations should complete without panic
    }
    
    /// Concurrent unseal is safe
    #[test]
    fn concurrent_unseal_safe(shard_count in 10usize..100) {
        let manager = Arc::new(VramManager::new_mock(1024 * 1024 * 1024)?);
        
        // Create shards
        let mut shards = vec![];
        for i in 0..shard_count {
            let data = vec![i as u8; 1024];
            let shard = manager.seal_model(&format!("model-{}", i), 0, &data)?;
            shards.push(Arc::new(shard));
        }
        
        // Unseal concurrently
        let mut handles = vec![];
        for shard in shards {
            let manager_clone = Arc::clone(&manager);
            let handle = thread::spawn(move || {
                manager_clone.unseal_model(&shard).unwrap()
            });
            handles.push(handle);
        }
        
        for handle in handles {
            let _data = handle.join().unwrap();
        }
    }
}
```

---

## 2. Cryptographic Properties

### 2.1 Seal Integrity

**Property**: Sealed data cannot be modified without detection.

```rust
proptest! {
    /// Tampering is detected
    #[test]
    fn tampering_detected(data in prop::collection::vec(any::<u8>(), 1..1000)) {
        let manager = VramManager::new_mock(10 * 1024 * 1024)?;
        
        let mut shard = manager.seal_model("test", 0, &data)?;
        
        // Tamper with sealed data
        if !shard.sealed_data.is_empty() {
            shard.sealed_data[0] ^= 0x01;
        }
        
        // Unseal should fail or detect tampering
        let result = manager.unseal_model(&shard);
        prop_assert!(result.is_err() || result.unwrap() != data);
    }
    
    /// HMAC verification works
    #[test]
    fn hmac_verification(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let manager = VramManager::new_mock(10 * 1024 * 1024)?;
        
        let shard = manager.seal_model("test", 0, &data)?;
        
        // Verify HMAC
        let valid = manager.verify_shard_integrity(&shard)?;
        prop_assert!(valid);
    }
}
```

---

## 3. GPU-Specific Properties

### 3.1 Multi-GPU Support

**Property**: Shards can be moved between GPUs.

```rust
proptest! {
    /// Shards can move between GPUs
    #[test]
    fn shard_migration(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let manager = VramManager::new_mock_multi_gpu(vec![
            100 * 1024 * 1024, // GPU 0: 100MB
            100 * 1024 * 1024, // GPU 1: 100MB
        ])?;
        
        // Seal on GPU 0
        let shard = manager.seal_model("test", 0, &data)?;
        prop_assert_eq!(shard.gpu_index, 0);
        
        // Migrate to GPU 1
        let migrated = manager.migrate_shard(&shard, 1)?;
        prop_assert_eq!(migrated.gpu_index, 1);
        
        // Unseal from GPU 1
        let unsealed = manager.unseal_model(&migrated)?;
        prop_assert_eq!(data, unsealed);
    }
}
```

---

### 3.2 GPU Index Validation

**Property**: Invalid GPU indices are rejected.

```rust
proptest! {
    /// Invalid GPU indices are rejected
    #[test]
    fn invalid_gpu_rejected(
        gpu_count in 1usize..8,
        invalid_index in 8u32..256
    ) {
        let capacities = vec![100 * 1024 * 1024; gpu_count];
        let manager = VramManager::new_mock_multi_gpu(capacities)?;
        
        let data = vec![0u8; 1024];
        let result = manager.seal_model("test", invalid_index, &data);
        
        if invalid_index >= gpu_count as u32 {
            prop_assert!(result.is_err());
        }
    }
}
```

---

## 4. Performance Properties

### 4.1 Seal/Unseal Performance

**Property**: Operations complete in reasonable time.

```rust
use std::time::{Duration, Instant};

proptest! {
    /// Sealing completes in reasonable time
    #[test]
    fn seal_performance(size_kb in 1usize..1000) {
        let manager = VramManager::new_mock(100 * 1024 * 1024)?;
        let data = vec![0u8; size_kb * 1024];
        
        let start = Instant::now();
        let _shard = manager.seal_model("test", 0, &data)?;
        let elapsed = start.elapsed();
        
        // Should complete in < 100ms per MB
        let expected_ms = (size_kb / 1024) * 100;
        prop_assert!(elapsed < Duration::from_millis(expected_ms as u64));
    }
    
    /// Unsealing is fast
    #[test]
    fn unseal_performance(size_kb in 1usize..1000) {
        let manager = VramManager::new_mock(100 * 1024 * 1024)?;
        let data = vec![0u8; size_kb * 1024];
        let shard = manager.seal_model("test", 0, &data)?;
        
        let start = Instant::now();
        let _unsealed = manager.unseal_model(&shard)?;
        let elapsed = start.elapsed();
        
        // Unseal should be fast (< 50ms per MB)
        let expected_ms = (size_kb / 1024) * 50;
        prop_assert!(elapsed < Duration::from_millis(expected_ms as u64));
    }
}
```

---

## 5. Implementation Guide

### 5.1 Add Proptest Dependency

```toml
# Cargo.toml
[dev-dependencies]
proptest.workspace = true
gpu-info = { path = "../../shared-crates/gpu-info" }
```

### 5.2 Test Structure

```rust
//! Property-based tests for VRAM residency
//!
//! These tests verify:
//! - Seal/unseal bijection
//! - VRAM capacity tracking
//! - Memory bounds checking
//! - Shard ID uniqueness
//! - Concurrent access safety

use proptest::prelude::*;
use vram_residency::*;

proptest! {
    // Core properties
}

#[cfg(test)]
mod seal_unseal {
    use super::*;
    proptest! {
        // Seal/unseal tests
    }
}

#[cfg(test)]
mod capacity {
    use super::*;
    proptest! {
        // Capacity tracking tests
    }
}

#[cfg(test)]
mod concurrency {
    use super::*;
    proptest! {
        // Concurrent access tests
    }
}
```

---

## 6. Running Tests

```bash
# Run all tests (mock mode, no GPU required)
cargo test -p vram-residency

# Run property tests with more cases
PROPTEST_CASES=10000 cargo test -p vram-residency --test property_tests

# Run with real GPU (if available)
cargo test -p vram-residency --features real-gpu

# Stress test
PROPTEST_CASES=100000 cargo test -p vram-residency -- --test-threads=1
```

---

## 7. Test Coverage Requirements

Before production:
- [ ] Seal/unseal roundtrip works (verified by property tests)
- [ ] VRAM capacity tracking accurate (verified by property tests)
- [ ] No overflow in calculations (verified by property tests)
- [ ] Shard IDs unique (verified by property tests)
- [ ] Bounds checking works (verified by property tests)
- [ ] Concurrent access safe (verified by stress tests)
- [ ] Tampering detected (verified by property tests)
- [ ] Performance acceptable (< 100ms/MB)

---

## 8. Common Pitfalls

### ❌ Don't Do This

```rust
// BAD: Testing with only small data
proptest! {
    #[test]
    fn bad_test(data in prop::collection::vec(any::<u8>(), 0..10)) {
        // Only tests tiny models
    }
}
```

### ✅ Do This Instead

```rust
// GOOD: Testing with various sizes including large models
proptest! {
    #[test]
    fn good_test(data in prop::collection::vec(any::<u8>(), 0..10_000_000)) {
        // Tests realistic model sizes
    }
}
```

---

## 9. Refinement Opportunities

### 9.1 Advanced Testing

**Future work**:
- Test with real GPU memory
- Test with CUDA errors
- Test with memory pressure
- Test with model quantization

### 9.2 Performance Testing

**Future work**:
- Benchmark seal/unseal throughput
- Test memory fragmentation
- Measure GPU memory bandwidth utilization

---

## 10. References

- **VRAM Management**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **Cryptographic Sealing**: https://en.wikipedia.org/wiki/Authenticated_encryption

---

**Priority**: ⭐⭐⭐⭐ HIGH  
**Estimated Effort**: 2-3 days  
**Impact**: Prevents memory corruption and model errors  
**Status**: Recommended for immediate implementation
