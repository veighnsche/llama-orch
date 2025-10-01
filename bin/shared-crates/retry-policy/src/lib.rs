//! retry-policy — Standardized retry policies
//!
//! Configurable retry logic with exponential backoff, jitter, circuit breaking.

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
        let multiplier = 2u64.saturating_pow(attempt);
        self.base_delay.saturating_mul(multiplier as u32)
    }
}
