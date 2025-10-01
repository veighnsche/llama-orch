# Backpressure — Property Testing Guide

**Status**: Recommended  
**Priority**: ⭐⭐⭐ MEDIUM  
**Applies to**: `bin/orchestratord-crates/backpressure/`

---

## 0. Why Property Testing for Backpressure?

**Backpressure prevents system overload**. Property-based testing ensures:

- ✅ Queue bounds are **always enforced**
- ✅ Admission control **works correctly**
- ✅ No **deadlocks** occur
- ✅ Fair **scheduling** is maintained
- ✅ Shedding decisions are **correct**

**Broken backpressure = system collapse**

---

## 1. Critical Properties to Test

### 1.1 Queue Bounds Enforcement

**Property**: Queue never exceeds configured capacity.

```rust
use proptest::prelude::*;

proptest! {
    /// Queue bounds are enforced
    #[test]
    fn queue_bounds_enforced(
        capacity in 1usize..1000,
        enqueue_count in 1usize..10000
    ) {
        let bp = Backpressure::new(capacity)?;
        
        let mut enqueued = 0;
        for _ in 0..enqueue_count {
            if bp.try_enqueue("request").is_ok() {
                enqueued += 1;
            }
        }
        
        // Should never exceed capacity
        prop_assert!(enqueued <= capacity);
        prop_assert_eq!(bp.queue_size(), enqueued.min(capacity));
    }
    
    /// Dequeue frees space
    #[test]
    fn dequeue_frees_space(capacity in 10usize..100) {
        let bp = Backpressure::new(capacity)?;
        
        // Fill queue
        for i in 0..capacity {
            bp.try_enqueue(i)?;
        }
        
        prop_assert_eq!(bp.queue_size(), capacity);
        prop_assert!(bp.is_full());
        
        // Dequeue one
        let _ = bp.dequeue()?;
        
        // Should have space
        prop_assert!(!bp.is_full());
        prop_assert_eq!(bp.queue_size(), capacity - 1);
    }
}
```

---

### 1.2 Admission Control

**Property**: Admission decisions are consistent with load.

```rust
proptest! {
    /// Admission control works
    #[test]
    fn admission_control_works(
        capacity in 10usize..100,
        load_factor in 0.0f64..2.0
    ) {
        let bp = Backpressure::with_threshold(capacity, 0.8)?; // 80% threshold
        
        // Fill to load factor
        let target_size = (capacity as f64 * load_factor) as usize;
        let mut admitted = 0;
        
        for _ in 0..target_size {
            if bp.try_enqueue("request").is_ok() {
                admitted += 1;
            }
        }
        
        // Check admission decision
        let should_admit = bp.should_admit()?;
        
        if admitted < (capacity as f64 * 0.8) as usize {
            prop_assert!(should_admit); // Below threshold
        } else if admitted >= capacity {
            prop_assert!(!should_admit); // At capacity
        }
    }
}
```

---

### 1.3 No Deadlocks

**Property**: System never deadlocks under concurrent load.

```rust
use std::sync::Arc;
use std::thread;
use std::time::Duration;

proptest! {
    /// No deadlocks under concurrent load
    #[test]
    fn no_deadlocks(
        capacity in 10usize..100,
        producer_count in 2usize..10,
        consumer_count in 2usize..10
    ) {
        let bp = Arc::new(Backpressure::new(capacity)?);
        let mut handles = vec![];
        
        // Start producers
        for i in 0..producer_count {
            let bp_clone = Arc::clone(&bp);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let _ = bp_clone.enqueue_with_timeout(
                        format!("p{}-{}", i, j),
                        Duration::from_millis(100)
                    );
                }
            });
            handles.push(handle);
        }
        
        // Start consumers
        for _ in 0..consumer_count {
            let bp_clone = Arc::clone(&bp);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let _ = bp_clone.dequeue_with_timeout(Duration::from_millis(100));
                }
            });
            handles.push(handle);
        }
        
        // All threads should complete (no deadlock)
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
```

---

### 1.4 Fair Scheduling

**Property**: Requests are processed in fair order (FIFO or priority).

```rust
proptest! {
    /// FIFO ordering is maintained
    #[test]
    fn fifo_ordering(count in 10usize..100) {
        let bp = Backpressure::new_fifo(1000)?;
        
        // Enqueue in order
        for i in 0..count {
            bp.enqueue(i)?;
        }
        
        // Dequeue should be in same order
        for expected in 0..count {
            let actual = bp.dequeue()?;
            prop_assert_eq!(actual, expected);
        }
    }
    
    /// Priority ordering works
    #[test]
    fn priority_ordering(count in 10usize..100) {
        let bp = Backpressure::new_priority(1000)?;
        
        // Enqueue with random priorities
        let mut priorities = vec![];
        for i in 0..count {
            let priority = (i * 7) % 10; // Pseudo-random
            bp.enqueue_with_priority(i, priority)?;
            priorities.push((i, priority));
        }
        
        // Sort by priority (descending)
        priorities.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Dequeue should be in priority order
        for (expected_val, _) in priorities {
            let actual = bp.dequeue()?;
            prop_assert_eq!(actual, expected_val);
        }
    }
}
```

---

### 1.5 Load Shedding

**Property**: Load shedding drops lowest priority requests first.

```rust
proptest! {
    /// Load shedding drops low priority
    #[test]
    fn load_shedding_priority(
        capacity in 10usize..50,
        request_count in 50usize..200
    ) {
        let bp = Backpressure::with_shedding(capacity)?;
        
        // Enqueue requests with priorities
        for i in 0..request_count {
            let priority = (i % 10) as u8; // 0-9
            let _ = bp.enqueue_with_priority(i, priority);
        }
        
        // Check what's in queue
        let queued = bp.get_all_queued()?;
        
        // Should have dropped lowest priorities
        prop_assert!(queued.len() <= capacity);
        
        if queued.len() == capacity {
            // All queued items should have higher priority than dropped
            let min_queued_priority = queued.iter()
                .map(|(_, p)| p)
                .min()
                .unwrap();
            
            prop_assert!(*min_queued_priority >= 5); // Dropped 0-4
        }
    }
}
```

---

### 1.6 Metrics Accuracy

**Property**: Metrics accurately reflect queue state.

```rust
proptest! {
    /// Metrics are accurate
    #[test]
    fn metrics_accurate(
        enqueue_count in 0usize..100,
        dequeue_count in 0usize..100
    ) {
        let bp = Backpressure::new(1000)?;
        
        for i in 0..enqueue_count {
            bp.enqueue(i)?;
        }
        
        for _ in 0..dequeue_count.min(enqueue_count) {
            bp.dequeue()?;
        }
        
        let metrics = bp.metrics();
        
        prop_assert_eq!(metrics.total_enqueued, enqueue_count);
        prop_assert_eq!(metrics.total_dequeued, dequeue_count.min(enqueue_count));
        
        let expected_size = enqueue_count.saturating_sub(dequeue_count);
        prop_assert_eq!(metrics.current_size, expected_size);
    }
}
```

---

## 2. Implementation Guide

### 2.1 Add Proptest Dependency

```toml
# Cargo.toml
[dev-dependencies]
proptest.workspace = true
```

### 2.2 Test Structure

```rust
//! Property-based tests for backpressure
//!
//! These tests verify:
//! - Queue bounds are enforced
//! - Admission control works
//! - No deadlocks occur
//! - Fair scheduling is maintained

use proptest::prelude::*;
use backpressure::*;

proptest! {
    // Core properties
}

#[cfg(test)]
mod queue_bounds {
    use super::*;
    proptest! {
        // Queue bound tests
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

## 3. Running Tests

```bash
# Run all tests
cargo test -p backpressure

# Run property tests with more cases
PROPTEST_CASES=10000 cargo test -p backpressure --test property_tests

# Test with thread sanitizer
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test -p backpressure
```

---

## 4. Refinement Opportunities

### 4.1 Advanced Testing

**Future work**:
- Test with adaptive capacity
- Test with multiple priority levels
- Test with deadline-based scheduling
- Test with distributed queues

---

**Priority**: ⭐⭐⭐ MEDIUM  
**Estimated Effort**: 2 days  
**Impact**: Prevents system overload  
**Status**: Recommended for implementation
