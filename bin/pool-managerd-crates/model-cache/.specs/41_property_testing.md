# Model Cache — Property Testing Guide

**Status**: Recommended  
**Priority**: ⭐⭐⭐ MEDIUM  
**Applies to**: `bin/pool-managerd-crates/model-cache/`

---

## 0. Why Property Testing for Model Cache?

**Model cache manages expensive resources**. Property-based testing ensures:

- ✅ LRU eviction is **correct**
- ✅ Cache size limits are **respected**
- ✅ No **cache poisoning**
- ✅ Concurrent access is **safe**
- ✅ Hit/miss tracking is **accurate**

**Broken cache = performance degradation**

---

## 1. Critical Properties to Test

### 1.1 LRU Eviction Correctness

**Property**: Least recently used items are evicted first.

```rust
use proptest::prelude::*;

proptest! {
    /// LRU eviction is correct
    #[test]
    fn lru_eviction_correct(
        capacity in 2usize..20,
        access_sequence in prop::collection::vec(0usize..50, 10..100)
    ) {
        let cache = ModelCache::new_lru(capacity)?;
        
        // Track access order
        let mut access_order = vec![];
        
        for model_id in access_sequence {
            cache.get_or_insert(model_id, || format!("model-{}", model_id))?;
            
            // Update access order (remove if exists, add to end)
            access_order.retain(|&x| x != model_id);
            access_order.push(model_id);
            
            // Keep only last 'capacity' items
            if access_order.len() > capacity {
                access_order.remove(0);
            }
        }
        
        // Cache should contain exactly the last 'capacity' accessed items
        for &model_id in &access_order {
            prop_assert!(cache.contains(model_id));
        }
        
        prop_assert_eq!(cache.len(), access_order.len().min(capacity));
    }
}
```

---

### 1.2 Cache Size Limits

**Property**: Cache never exceeds configured capacity.

```rust
proptest! {
    /// Cache size is bounded
    #[test]
    fn cache_size_bounded(
        capacity in 1usize..100,
        insert_count in 1usize..1000
    ) {
        let cache = ModelCache::new(capacity)?;
        
        for i in 0..insert_count {
            cache.insert(i, format!("model-{}", i))?;
        }
        
        // Should never exceed capacity
        prop_assert!(cache.len() <= capacity);
        prop_assert_eq!(cache.len(), insert_count.min(capacity));
    }
    
    /// Eviction frees space
    #[test]
    fn eviction_frees_space(capacity in 10usize..100) {
        let cache = ModelCache::new(capacity)?;
        
        // Fill cache
        for i in 0..capacity {
            cache.insert(i, format!("model-{}", i))?;
        }
        
        prop_assert_eq!(cache.len(), capacity);
        
        // Insert one more (should evict)
        cache.insert(capacity, format!("model-{}", capacity))?;
        
        // Size should still be capacity
        prop_assert_eq!(cache.len(), capacity);
    }
}
```

---

### 1.3 Cache Coherency

**Property**: Cache always returns consistent data.

```rust
proptest! {
    /// Cache returns consistent data
    #[test]
    fn cache_coherency(
        inserts in prop::collection::vec((0usize..100, "\\PC{1,50}"), 1..100)
    ) {
        let cache = ModelCache::new(1000)?;
        
        // Insert data
        for (key, value) in &inserts {
            cache.insert(*key, value.clone())?;
        }
        
        // Verify all insertions
        for (key, expected_value) in &inserts {
            if let Some(actual_value) = cache.get(*key) {
                prop_assert_eq!(actual_value, expected_value);
            }
        }
    }
    
    /// Updates are reflected
    #[test]
    fn updates_reflected(
        key in 0usize..100,
        value1 in "\\PC{1,50}",
        value2 in "\\PC{1,50}"
    ) {
        let cache = ModelCache::new(100)?;
        
        cache.insert(key, value1.clone())?;
        prop_assert_eq!(cache.get(key), Some(&value1));
        
        cache.insert(key, value2.clone())?;
        prop_assert_eq!(cache.get(key), Some(&value2));
    }
}
```

---

### 1.4 No Cache Poisoning

**Property**: Invalid entries are rejected.

```rust
proptest! {
    /// Invalid entries are rejected
    #[test]
    fn invalid_entries_rejected(key in 0usize..100) {
        let cache = ModelCache::with_validation(100)?;
        
        // Try to insert invalid data
        let result = cache.insert_validated(key, "");
        
        // Empty models should be rejected
        prop_assert!(result.is_err());
        
        // Cache should not contain invalid entry
        prop_assert!(!cache.contains(key));
    }
    
    /// Corrupted entries are detected
    #[test]
    fn corruption_detected(
        key in 0usize..100,
        data in prop::collection::vec(any::<u8>(), 1..1000)
    ) {
        let cache = ModelCache::with_checksums(100)?;
        
        cache.insert_with_checksum(key, data.clone())?;
        
        // Verify checksum
        let valid = cache.verify_checksum(key)?;
        prop_assert!(valid);
        
        // Simulate corruption (if we could access internal data)
        // Verification should fail
    }
}
```

---

### 1.5 Concurrent Access Safety

**Property**: Cache is thread-safe.

```rust
use std::sync::Arc;
use std::thread;

proptest! {
    /// Concurrent access is safe
    #[test]
    fn concurrent_access_safe(
        capacity in 10usize..100,
        thread_count in 2usize..10,
        ops_per_thread in 10usize..100
    ) {
        let cache = Arc::new(ModelCache::new(capacity)?);
        let mut handles = vec![];
        
        for thread_id in 0..thread_count {
            let cache_clone = Arc::clone(&cache);
            
            let handle = thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let key = (thread_id * 1000 + i) % 50;
                    
                    // Mix of reads and writes
                    if i % 2 == 0 {
                        cache_clone.insert(key, format!("t{}-{}", thread_id, i)).unwrap();
                    } else {
                        let _ = cache_clone.get(key);
                    }
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Should complete without panic
        prop_assert!(cache.len() <= capacity);
    }
}
```

---

### 1.6 Hit/Miss Tracking

**Property**: Cache metrics are accurate.

```rust
proptest! {
    /// Hit/miss tracking is accurate
    #[test]
    fn hit_miss_tracking_accurate(
        capacity in 10usize..100,
        operations in prop::collection::vec((0usize..50, any::<bool>()), 10..200)
    ) {
        let cache = ModelCache::new(capacity)?;
        
        let mut expected_hits = 0;
        let mut expected_misses = 0;
        
        for (key, is_insert) in operations {
            if is_insert {
                cache.insert(key, format!("model-{}", key))?;
            } else {
                if cache.get(key).is_some() {
                    expected_hits += 1;
                } else {
                    expected_misses += 1;
                }
            }
        }
        
        let metrics = cache.metrics();
        prop_assert_eq!(metrics.hits, expected_hits);
        prop_assert_eq!(metrics.misses, expected_misses);
        
        if expected_hits + expected_misses > 0 {
            let expected_rate = (expected_hits as f64 / (expected_hits + expected_misses) as f64) * 100.0;
            prop_assert!((metrics.hit_rate - expected_rate).abs() < 0.1);
        }
    }
}
```

---

## 2. Specific Cache Strategies

### 2.1 LRU Cache

```rust
proptest! {
    /// LRU maintains access order
    #[test]
    fn lru_access_order(capacity in 5usize..20) {
        let cache = LruCache::new(capacity)?;
        
        // Fill cache
        for i in 0..capacity {
            cache.insert(i, i)?;
        }
        
        // Access first item (makes it most recent)
        let _ = cache.get(0);
        
        // Insert new item (should evict second item, not first)
        cache.insert(capacity, capacity)?;
        
        prop_assert!(cache.contains(0)); // First item still there
        prop_assert!(!cache.contains(1)); // Second item evicted
    }
}
```

---

### 2.2 LFU Cache

```rust
proptest! {
    /// LFU evicts least frequently used
    #[test]
    fn lfu_frequency_tracking(capacity in 5usize..20) {
        let cache = LfuCache::new(capacity)?;
        
        // Insert items
        for i in 0..capacity {
            cache.insert(i, i)?;
        }
        
        // Access some items multiple times
        for _ in 0..10 {
            let _ = cache.get(0);
        }
        
        // Insert new item (should evict least frequent, not most recent)
        cache.insert(capacity, capacity)?;
        
        prop_assert!(cache.contains(0)); // Most frequent still there
    }
}
```

---

## 3. Implementation Guide

### 3.1 Add Proptest Dependency

```toml
# Cargo.toml
[dev-dependencies]
proptest.workspace = true
```

### 3.2 Test Structure

```rust
//! Property-based tests for model cache
//!
//! These tests verify:
//! - LRU eviction correctness
//! - Cache size limits
//! - Cache coherency
//! - Concurrent access safety

use proptest::prelude::*;
use model_cache::*;

proptest! {
    // Core properties
}

#[cfg(test)]
mod lru {
    use super::*;
    proptest! {
        // LRU tests
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

## 4. Running Tests

```bash
# Run all tests
cargo test -p model-cache

# Run property tests with more cases
PROPTEST_CASES=10000 cargo test -p model-cache --test property_tests

# Test with thread sanitizer
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test -p model-cache
```

---

## 5. Refinement Opportunities

### 5.1 Advanced Testing

**Future work**:
- Test with TTL expiration
- Test with size-based eviction
- Test with distributed cache
- Test with cache warming

---

**Priority**: ⭐⭐⭐ MEDIUM  
**Estimated Effort**: 1-2 days  
**Impact**: Ensures cache correctness  
**Status**: Recommended for implementation
