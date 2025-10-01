//! retry-policy — Standardized retry policies
//!
//! Configurable retry logic with exponential backoff, jitter, circuit breaking.
//!
//! TODO(ARCH-CHANGE): This crate is minimal. Needs:
//! - Add jitter to prevent thundering herd
//! - Implement circuit breaker integration
//! - Add retry budget tracking (prevent infinite retries)
//! - Implement per-error-type retry policies
//! - Add metrics for retry attempts
//! - Integrate with backpressure crate

// Medium-importance crate: TIER 3 Clippy configuration
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::missing_errors_doc)]

use std::time::Duration;

pub struct RetryPolicy {
    max_attempts: u32,
    base_delay: Duration,
}

impl RetryPolicy {
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            base_delay: Duration::from_millis(100),
        }
    }
    
    pub fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.max_attempts
    }
    
    pub fn delay(&self, attempt: u32) -> Duration {
        // TODO(ARCH-CHANGE): Add jitter to prevent thundering herd:
        // - Use rand::thread_rng() for jitter
        // - Add ±20% randomness to delay
        // - Cap maximum delay (e.g., 30 seconds)
        let multiplier = 2u64.saturating_pow(attempt);
        self.base_delay.saturating_mul(multiplier as u32)
    }
}
