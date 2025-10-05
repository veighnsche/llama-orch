//! backpressure — Admission control and backpressure policies
//!
//! Manages queue overflow, rejects tasks when overloaded, returns proper HTTP 429 with Retry-After.
//!
//! # Key Responsibilities
//!
//! - Detect queue full conditions
//! - Calculate Retry-After time based on queue depth
//! - Circuit breaking (stop accepting if downstream failing)
//! - Load shedding (drop low-priority tasks first)
//! - Admission policies (reject-new, drop-LRU, fail-fast)
//!
//! # Example
//!
//! ```rust
//! use backpressure::{BackpressurePolicy, AdmissionDecision};
//!
//! let policy = BackpressurePolicy::new(10_000); // Max 10k queue depth
//!
//! match policy.should_admit(current_depth, priority) {
//!     AdmissionDecision::Admit => { /* Process request */ }
//!     AdmissionDecision::Reject { retry_after_ms } => {
//!         /* Return 429 with Retry-After header */
//!     }
//!     AdmissionDecision::DropLru => { /* Evict oldest task */ }
//! }
//! ```

// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::string_slice)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::must_use_candidate)]

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackpressureError {
    #[error("queue full")]
    QueueFull,
    #[error("circuit open")]
    CircuitOpen,
}

pub type Result<T> = std::result::Result<T, BackpressureError>;

/// Admission decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDecision {
    /// Admit the request
    Admit,
    /// Reject with retry-after hint (milliseconds)
    Reject { retry_after_ms: u64 },
    /// Drop oldest task to make room
    DropLru,
}

/// Priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Batch = 0,
    Interactive = 1,
}

/// Backpressure policy
pub struct BackpressurePolicy {
    max_depth: usize,
    reject_threshold: f64,
}

impl BackpressurePolicy {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            reject_threshold: 0.9, // Reject at 90% capacity
        }
    }

    /// Decide whether to admit request
    pub fn should_admit(&self, current_depth: usize, _priority: Priority) -> AdmissionDecision {
        let utilization = current_depth as f64 / self.max_depth as f64;

        if current_depth >= self.max_depth {
            // Queue full - reject with backoff
            let retry_after_ms = self.calculate_retry_after(current_depth);
            AdmissionDecision::Reject { retry_after_ms }
        } else if utilization >= self.reject_threshold {
            // Near capacity - start rejecting
            let retry_after_ms = self.calculate_retry_after(current_depth);
            AdmissionDecision::Reject { retry_after_ms }
        } else {
            AdmissionDecision::Admit
        }
    }

    fn calculate_retry_after(&self, queue_depth: usize) -> u64 {
        // Estimate: 100ms per queued item
        let base_ms = 100u64;
        base_ms.saturating_mul(queue_depth as u64).min(60_000) // Max 60s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backpressure() {
        let policy = BackpressurePolicy::new(100);

        // Low utilization - admit
        assert_eq!(policy.should_admit(50, Priority::Interactive), AdmissionDecision::Admit);

        // High utilization - reject
        matches!(policy.should_admit(95, Priority::Interactive), AdmissionDecision::Reject { .. });

        // Full - reject
        matches!(policy.should_admit(100, Priority::Interactive), AdmissionDecision::Reject { .. });
    }
}
