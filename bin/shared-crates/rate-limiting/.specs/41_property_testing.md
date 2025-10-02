# Rate Limiting — Property Testing Guide

**Status**: Recommended  
**Priority**: ⭐⭐⭐ MEDIUM  
**Applies to**: `bin/shared-crates/rate-limiting/`

---

## 0. Why Property Testing for Rate Limiting?

**Rate limiting prevents DoS attacks**. Property-based testing ensures:

- ✅ Rate limits are **never exceeded**
- ✅ Token bucket math is **correct**
- ✅ Concurrent requests are **handled safely**
- ✅ Time calculations don't **overflow**
- ✅ Burst allowances work **correctly**

**Broken rate limiting = service unavailable**

---

## 1. Critical Properties to Test

### 1.1 Rate Limits Never Exceeded

**Property**: Number of allowed requests never exceeds configured rate.

```rust
use proptest::prelude::*;
use std::time::{Duration, Instant};

proptest! {
    /// Rate limit is never exceeded
    #[test]
    fn rate_limit_enforced(
        requests_per_sec in 1usize..1000,
        total_requests in 1usize..10000
    ) {
        let limiter = RateLimiter::new(requests_per_sec, Duration::from_secs(1))?;
        
        let start = Instant::now();
        let mut allowed = 0;
        
        for _ in 0..total_requests {
            if limiter.check_rate_limit("test-key")? {
                allowed += 1;
            }
        }
        
        let elapsed = start.elapsed();
        let seconds = elapsed.as_secs_f64();
        
        // Allowed requests should not exceed rate * time
        let max_allowed = (requests_per_sec as f64 * seconds).ceil() as usize;
        prop_assert!(allowed <= max_allowed + requests_per_sec); // +burst allowance
    }
    
    /// Sustained load respects rate
    #[test]
    fn sustained_load_rate(rate in 10usize..100) {
        let limiter = RateLimiter::new(rate, Duration::from_secs(1))?;
        
        let mut allowed_count = 0;
        let start = Instant::now();
        
        // Run for 5 seconds
        while start.elapsed() < Duration::from_secs(5) {
            if limiter.check_rate_limit("test")? {
                allowed_count += 1;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        
        let elapsed_secs = start.elapsed().as_secs_f64();
        let expected = (rate as f64 * elapsed_secs) as usize;
        
        // Should be within 10% of expected
        let tolerance = (expected as f64 * 0.1) as usize;
        prop_assert!((allowed_count as i64 - expected as i64).abs() <= tolerance as i64);
    }
}
```

---

### 1.2 Token Bucket Math Correctness

**Property**: Token bucket refill calculations are correct.

```rust
proptest! {
    /// Token refill is accurate
    #[test]
    fn token_refill_accurate(
        capacity in 1usize..1000,
        refill_rate in 1usize..100,
        elapsed_ms in 0u64..10000
    ) {
        let limiter = TokenBucket::new(capacity, refill_rate)?;
        
        // Drain bucket
        for _ in 0..capacity {
            limiter.consume(1)?;
        }
        
        // Wait and refill
        std::thread::sleep(Duration::from_millis(elapsed_ms));
        limiter.refill()?;
        
        let available = limiter.available_tokens()?;
        
        // Calculate expected tokens
        let elapsed_secs = elapsed_ms as f64 / 1000.0;
        let expected = ((refill_rate as f64 * elapsed_secs) as usize).min(capacity);
        
        // Should be close (within 1 token due to timing)
        prop_assert!((available as i64 - expected as i64).abs() <= 1);
    }
    
    /// Token consumption is accurate
    #[test]
    fn token_consumption_accurate(
        capacity in 10usize..1000,
        consume_count in 1usize..100
    ) {
        let limiter = TokenBucket::new(capacity, 10)?;
        
        let initial = limiter.available_tokens()?;
        
        let mut consumed = 0;
        for _ in 0..consume_count {
            if limiter.try_consume(1)? {
                consumed += 1;
            }
        }
        
        let remaining = limiter.available_tokens()?;
        
        prop_assert_eq!(initial - consumed, remaining);
    }
}
```

---

### 1.3 Concurrent Access Safety

**Property**: Rate limiter is thread-safe.

```rust
use std::sync::Arc;
use std::thread;

proptest! {
    /// Concurrent access is safe
    #[test]
    fn concurrent_access_safe(
        thread_count in 2usize..20,
        requests_per_thread in 10usize..100
    ) {
        let limiter = Arc::new(RateLimiter::new(1000, Duration::from_secs(1))?);
        let mut handles = vec![];
        
        for _ in 0..thread_count {
            let limiter_clone = Arc::clone(&limiter);
            
            let handle = thread::spawn(move || {
                let mut allowed = 0;
                for _ in 0..requests_per_thread {
                    if limiter_clone.check_rate_limit("test").unwrap() {
                        allowed += 1;
                    }
                }
                allowed
            });
            handles.push(handle);
        }
        
        let mut total_allowed = 0;
        for handle in handles {
            total_allowed += handle.join().unwrap();
        }
        
        // All threads should complete without panic
        // Total allowed should be reasonable
        prop_assert!(total_allowed > 0);
        prop_assert!(total_allowed <= thread_count * requests_per_thread);
    }
}
```

---

### 1.4 Time Calculations Don't Overflow

**Property**: Time arithmetic is safe from overflow.

```rust
proptest! {
    /// Time calculations are safe
    #[test]
    fn time_calculations_safe(
        rate in 1usize..1_000_000,
        window_secs in 1u64..86400 // Up to 1 day
    ) {
        let window = Duration::from_secs(window_secs);
        let limiter = RateLimiter::new(rate, window)?;
        
        // Should not panic or overflow
        let _ = limiter.check_rate_limit("test")?;
    }
    
    /// Large elapsed times don't overflow
    #[test]
    fn large_elapsed_safe(capacity in 1usize..1000) {
        let limiter = TokenBucket::new(capacity, 100)?;
        
        // Simulate very long elapsed time
        let large_duration = Duration::from_secs(u32::MAX as u64);
        
        // Should not overflow when calculating refill
        let result = limiter.refill_with_duration(large_duration);
        prop_assert!(result.is_ok());
    }
}
```

---

### 1.5 Burst Allowance Works Correctly

**Property**: Burst allowance allows temporary spikes.

```rust
proptest! {
    /// Burst allowance works
    #[test]
    fn burst_allowance_works(
        rate in 10usize..100,
        burst_multiplier in 2usize..10
    ) {
        let burst_capacity = rate * burst_multiplier;
        let limiter = RateLimiter::with_burst(rate, Duration::from_secs(1), burst_capacity)?;
        
        // Should allow burst
        let mut allowed = 0;
        for _ in 0..burst_capacity {
            if limiter.check_rate_limit("test")? {
                allowed += 1;
            }
        }
        
        // Should allow at least the burst capacity
        prop_assert!(allowed >= burst_capacity);
        
        // After burst, should be rate-limited
        std::thread::sleep(Duration::from_millis(100));
        
        let mut post_burst_allowed = 0;
        for _ in 0..rate {
            if limiter.check_rate_limit("test")? {
                post_burst_allowed += 1;
            }
        }
        
        // Should be limited after burst
        prop_assert!(post_burst_allowed < rate);
    }
}
```

---

### 1.6 Per-Key Isolation

**Property**: Rate limits are isolated per key.

```rust
proptest! {
    /// Keys are isolated
    #[test]
    fn keys_isolated(
        rate in 10usize..100,
        key_count in 2usize..20
    ) {
        let limiter = RateLimiter::new(rate, Duration::from_secs(1))?;
        
        let keys: Vec<String> = (0..key_count)
            .map(|i| format!("key-{}", i))
            .collect();
        
        // Each key should get its own rate limit
        for key in &keys {
            let mut allowed = 0;
            for _ in 0..rate * 2 {
                if limiter.check_rate_limit(key)? {
                    allowed += 1;
                }
            }
            
            // Each key should allow at least 'rate' requests
            prop_assert!(allowed >= rate);
        }
    }
}
```

---

## 2. Specific Rate Limiting Strategies

### 2.1 Fixed Window

```rust
proptest! {
    /// Fixed window resets correctly
    #[test]
    fn fixed_window_resets(
        rate in 10usize..100,
        window_ms in 100u64..1000
    ) {
        let limiter = FixedWindowLimiter::new(rate, Duration::from_millis(window_ms))?;
        
        // Fill window
        for _ in 0..rate {
            limiter.check_rate_limit("test")?;
        }
        
        // Should be limited
        prop_assert!(!limiter.check_rate_limit("test")?);
        
        // Wait for window to reset
        std::thread::sleep(Duration::from_millis(window_ms + 10));
        
        // Should allow again
        prop_assert!(limiter.check_rate_limit("test")?);
    }
}
```

---

### 2.2 Sliding Window

```rust
proptest! {
    /// Sliding window is smooth
    #[test]
    fn sliding_window_smooth(rate in 10usize..100) {
        let limiter = SlidingWindowLimiter::new(rate, Duration::from_secs(1))?;
        
        let mut allowed_counts = vec![];
        
        for _ in 0..10 {
            let mut allowed = 0;
            let start = Instant::now();
            
            while start.elapsed() < Duration::from_millis(100) {
                if limiter.check_rate_limit("test")? {
                    allowed += 1;
                }
            }
            
            allowed_counts.push(allowed);
            std::thread::sleep(Duration::from_millis(100));
        }
        
        // Counts should be relatively consistent (no sharp resets)
        let max = *allowed_counts.iter().max().unwrap();
        let min = *allowed_counts.iter().min().unwrap();
        
        prop_assert!(max - min <= rate / 5); // Within 20%
    }
}
```

---

### 2.3 Token Bucket

```rust
proptest! {
    /// Token bucket allows bursts
    #[test]
    fn token_bucket_bursts(
        capacity in 10usize..100,
        refill_rate in 1usize..50
    ) {
        let limiter = TokenBucket::new(capacity, refill_rate)?;
        
        // Should allow burst up to capacity
        let mut burst_allowed = 0;
        for _ in 0..capacity * 2 {
            if limiter.try_consume(1)? {
                burst_allowed += 1;
            }
        }
        
        prop_assert_eq!(burst_allowed, capacity);
        
        // Wait for refill
        std::thread::sleep(Duration::from_secs(1));
        
        // Should have refilled
        let refilled = limiter.available_tokens()?;
        prop_assert!(refilled > 0);
        prop_assert!(refilled <= capacity);
    }
}
```

---

## 3. Error Handling Properties

### 3.1 Invalid Configuration Rejected

```rust
proptest! {
    /// Invalid rates are rejected
    #[test]
    fn invalid_rates_rejected(rate in 0usize..1) {
        let result = RateLimiter::new(rate, Duration::from_secs(1));
        
        if rate == 0 {
            prop_assert!(result.is_err());
        }
    }
    
    /// Invalid windows are rejected
    #[test]
    fn invalid_windows_rejected(window_ms in 0u64..1) {
        let result = RateLimiter::new(100, Duration::from_millis(window_ms));
        
        if window_ms == 0 {
            prop_assert!(result.is_err());
        }
    }
}
```

---

## 4. Performance Properties

### 4.1 Check Performance

```rust
use std::time::Instant;

proptest! {
    /// Rate limit check is fast
    #[test]
    fn check_performance(rate in 1usize..1000) {
        let limiter = RateLimiter::new(rate, Duration::from_secs(1))?;
        
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = limiter.check_rate_limit("test")?;
        }
        let elapsed = start.elapsed();
        
        // Should handle > 10k checks/sec
        prop_assert!(elapsed < Duration::from_millis(100));
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
```

### 5.2 Test Structure

```rust
//! Property-based tests for rate limiting
//!
//! These tests verify:
//! - Rate limits are never exceeded
//! - Token bucket math is correct
//! - Concurrent access is safe
//! - Time calculations don't overflow

use proptest::prelude::*;
use rate_limiting::*;

proptest! {
    // Core properties
}

#[cfg(test)]
mod token_bucket {
    use super::*;
    proptest! {
        // Token bucket tests
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
# Run all tests
cargo test -p rate-limiting

# Run property tests with more cases
PROPTEST_CASES=10000 cargo test -p rate-limiting --test property_tests

# Stress test
PROPTEST_CASES=100000 cargo test -p rate-limiting -- --test-threads=1
```

---

## 7. Refinement Opportunities

### 7.1 Advanced Testing

**Future work**:
- Test with distributed rate limiting (Redis)
- Test with clock skew
- Test with time travel (mocked time)
- Test with different time sources

### 7.2 Performance Testing

**Future work**:
- Benchmark different strategies
- Test memory usage under load
- Measure lock contention

---

**Priority**: ⭐⭐⭐ MEDIUM  
**Estimated Effort**: 1-2 days  
**Impact**: Prevents DoS attacks  
**Status**: Recommended for implementation
