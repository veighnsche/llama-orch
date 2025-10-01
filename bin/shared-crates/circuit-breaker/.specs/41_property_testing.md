# Circuit Breaker — Property Testing Guide

**Status**: Recommended  
**Priority**: ⭐⭐⭐ MEDIUM  
**Applies to**: `bin/shared-crates/circuit-breaker/`

---

## 0. Why Property Testing for Circuit Breaker?

**Circuit breakers prevent cascading failures**. Property-based testing ensures:

- ✅ State transitions are **always valid**
- ✅ Thresholds are **never violated**
- ✅ Timeouts work **correctly**
- ✅ No **race conditions** in state changes
- ✅ Half-open state **probes correctly**

**Broken circuit breaker = cascading failures**

---

## 1. Critical Properties to Test

### 1.1 State Transition Validity

**Property**: State transitions follow valid state machine.

```rust
use proptest::prelude::*;

proptest! {
    /// State transitions are valid
    #[test]
    fn state_transitions_valid(
        failure_count in 0usize..100,
        threshold in 1usize..50
    ) {
        let cb = CircuitBreaker::new(threshold, Duration::from_secs(10))?;
        
        let initial_state = cb.state();
        prop_assert_eq!(initial_state, State::Closed);
        
        // Record failures
        for _ in 0..failure_count {
            cb.record_failure();
        }
        
        let state = cb.state();
        
        // Valid states based on failures
        if failure_count >= threshold {
            prop_assert_eq!(state, State::Open);
        } else {
            prop_assert_eq!(state, State::Closed);
        }
    }
    
    /// Cannot transition from Open to Closed directly
    #[test]
    fn no_direct_open_to_closed(threshold in 1usize..10) {
        let cb = CircuitBreaker::new(threshold, Duration::from_millis(100))?;
        
        // Open the circuit
        for _ in 0..threshold {
            cb.record_failure();
        }
        prop_assert_eq!(cb.state(), State::Open);
        
        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));
        
        // Should transition to HalfOpen, not Closed
        let state = cb.state();
        prop_assert_eq!(state, State::HalfOpen);
    }
}
```

---

### 1.2 Threshold Enforcement

**Property**: Circuit opens exactly at threshold.

```rust
proptest! {
    /// Threshold is enforced
    #[test]
    fn threshold_enforced(threshold in 1usize..100) {
        let cb = CircuitBreaker::new(threshold, Duration::from_secs(10))?;
        
        // Record threshold-1 failures
        for _ in 0..(threshold - 1) {
            cb.record_failure();
        }
        
        // Should still be closed
        prop_assert_eq!(cb.state(), State::Closed);
        prop_assert!(cb.allow_request());
        
        // One more failure should open
        cb.record_failure();
        prop_assert_eq!(cb.state(), State::Open);
        prop_assert!(!cb.allow_request());
    }
    
    /// Success resets failure count
    #[test]
    fn success_resets_count(
        threshold in 2usize..100,
        failures_before_success in 1usize..50
    ) {
        let cb = CircuitBreaker::new(threshold, Duration::from_secs(10))?;
        
        // Record some failures (less than threshold)
        let failures = failures_before_success.min(threshold - 1);
        for _ in 0..failures {
            cb.record_failure();
        }
        
        // Record success
        cb.record_success();
        
        // Should reset counter, so we can add threshold-1 more failures
        for _ in 0..(threshold - 1) {
            cb.record_failure();
        }
        
        // Should still be closed
        prop_assert_eq!(cb.state(), State::Closed);
    }
}
```

---

### 1.3 Timeout Behavior

**Property**: Circuit transitions to half-open after timeout.

```rust
proptest! {
    /// Timeout triggers half-open
    #[test]
    fn timeout_triggers_halfopen(
        threshold in 1usize..10,
        timeout_ms in 100u64..1000
    ) {
        let cb = CircuitBreaker::new(threshold, Duration::from_millis(timeout_ms))?;
        
        // Open the circuit
        for _ in 0..threshold {
            cb.record_failure();
        }
        prop_assert_eq!(cb.state(), State::Open);
        
        // Wait for timeout
        std::thread::sleep(Duration::from_millis(timeout_ms + 50));
        
        // Should be half-open
        prop_assert_eq!(cb.state(), State::HalfOpen);
        
        // Should allow one probe request
        prop_assert!(cb.allow_request());
    }
}
```

---

### 1.4 Half-Open Probing

**Property**: Half-open state allows limited probes.

```rust
proptest! {
    /// Half-open allows limited probes
    #[test]
    fn halfopen_limited_probes(
        threshold in 1usize..10,
        probe_count in 1usize..10
    ) {
        let cb = CircuitBreaker::with_probes(
            threshold,
            Duration::from_millis(100),
            probe_count
        )?;
        
        // Open the circuit
        for _ in 0..threshold {
            cb.record_failure();
        }
        
        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));
        prop_assert_eq!(cb.state(), State::HalfOpen);
        
        // Should allow exactly probe_count requests
        let mut allowed = 0;
        for _ in 0..(probe_count * 2) {
            if cb.allow_request() {
                allowed += 1;
            }
        }
        
        prop_assert_eq!(allowed, probe_count);
    }
    
    /// Successful probe closes circuit
    #[test]
    fn successful_probe_closes(threshold in 1usize..10) {
        let cb = CircuitBreaker::new(threshold, Duration::from_millis(100))?;
        
        // Open the circuit
        for _ in 0..threshold {
            cb.record_failure();
        }
        
        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));
        prop_assert_eq!(cb.state(), State::HalfOpen);
        
        // Successful probe
        cb.record_success();
        
        // Should close
        prop_assert_eq!(cb.state(), State::Closed);
    }
    
    /// Failed probe reopens circuit
    #[test]
    fn failed_probe_reopens(threshold in 1usize..10) {
        let cb = CircuitBreaker::new(threshold, Duration::from_millis(100))?;
        
        // Open the circuit
        for _ in 0..threshold {
            cb.record_failure();
        }
        
        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));
        prop_assert_eq!(cb.state(), State::HalfOpen);
        
        // Failed probe
        cb.record_failure();
        
        // Should reopen
        prop_assert_eq!(cb.state(), State::Open);
    }
}
```

---

### 1.5 Concurrent Access Safety

**Property**: Circuit breaker is thread-safe.

```rust
use std::sync::Arc;
use std::thread;

proptest! {
    /// Concurrent access is safe
    #[test]
    fn concurrent_access_safe(
        thread_count in 2usize..20,
        ops_per_thread in 10usize..100,
        threshold in 5usize..50
    ) {
        let cb = Arc::new(CircuitBreaker::new(threshold, Duration::from_secs(10))?);
        let mut handles = vec![];
        
        for _ in 0..thread_count {
            let cb_clone = Arc::clone(&cb);
            
            let handle = thread::spawn(move || {
                for i in 0..ops_per_thread {
                    if cb_clone.allow_request() {
                        // Simulate success/failure
                        if i % 3 == 0 {
                            cb_clone.record_failure();
                        } else {
                            cb_clone.record_success();
                        }
                    }
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Should complete without panic
        // State should be valid
        let state = cb.state();
        prop_assert!(
            state == State::Closed ||
            state == State::Open ||
            state == State::HalfOpen
        );
    }
}
```

---

### 1.6 Metrics Accuracy

**Property**: Metrics accurately reflect circuit state.

```rust
proptest! {
    /// Metrics are accurate
    #[test]
    fn metrics_accurate(
        success_count in 0usize..100,
        failure_count in 0usize..100
    ) {
        let cb = CircuitBreaker::new(100, Duration::from_secs(10))?;
        
        for _ in 0..success_count {
            cb.record_success();
        }
        
        for _ in 0..failure_count {
            cb.record_failure();
        }
        
        let metrics = cb.metrics();
        
        prop_assert_eq!(metrics.success_count, success_count);
        prop_assert_eq!(metrics.failure_count, failure_count);
        
        let total = success_count + failure_count;
        if total > 0 {
            let expected_rate = (failure_count as f64 / total as f64) * 100.0;
            prop_assert!((metrics.failure_rate - expected_rate).abs() < 0.1);
        }
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
//! Property-based tests for circuit breaker
//!
//! These tests verify:
//! - State transitions are valid
//! - Thresholds are enforced
//! - Timeouts work correctly
//! - Half-open probing works
//! - Concurrent access is safe

use proptest::prelude::*;
use circuit_breaker::*;

proptest! {
    // Core properties
}

#[cfg(test)]
mod state_machine {
    use super::*;
    proptest! {
        // State transition tests
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
cargo test -p circuit-breaker

# Run property tests with more cases
PROPTEST_CASES=10000 cargo test -p circuit-breaker --test property_tests

# Test with thread sanitizer
RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test -p circuit-breaker
```

---

## 4. Refinement Opportunities

### 4.1 Advanced Testing

**Future work**:
- Test with distributed circuit breakers
- Test with adaptive thresholds
- Test with multiple failure types
- Test with custom health checks

---

**Priority**: ⭐⭐⭐ MEDIUM  
**Estimated Effort**: 1-2 days  
**Impact**: Prevents cascading failures  
**Status**: Recommended for implementation
