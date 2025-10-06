//! deadline-propagation — Request deadline propagation and enforcement
//!
//! Propagates client-specified deadlines through orchestrator → pool-manager → worker.
//!
//! # Key Responsibilities
//!
//! - Parse client deadline (X-Deadline header or deadline_ms field)
//! - Calculate remaining time at each hop
//! - Abort work if deadline already exceeded
//! - Return 504 Gateway Timeout if deadline missed
//! - Cancel downstream requests when deadline hit
//!
//! # Example
//!
//! ```rust
//! use deadline_propagation::{Deadline, DeadlineContext};
//!
//! // Parse from request
//! let deadline = Deadline::from_header("2025-10-01T17:30:00Z")?;
//!
//! // Check if expired
//! if deadline.is_expired() {
//!     return Err("Deadline exceeded");
//! }
//!
//! // Get remaining time
//! let remaining_ms = deadline.remaining_ms();
//! ```

// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::missing_errors_doc)]

use std::time::{Duration, SystemTime};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DeadlineError {
    #[error("deadline exceeded")]
    Exceeded,
    #[error("invalid deadline format: {0}")]
    InvalidFormat(String),
}

pub type Result<T> = std::result::Result<T, DeadlineError>;

/// Request deadline
#[derive(Debug, Clone, Copy)]
pub struct Deadline {
    deadline_ms: u64,
}

impl Deadline {
    /// Create deadline from milliseconds since epoch
    pub fn from_ms(deadline_ms: u64) -> Self {
        Self { deadline_ms }
    }

    /// Create deadline from duration from now
    pub fn from_duration(duration: Duration) -> Result<Self> {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| DeadlineError::InvalidFormat(format!("Time error: {}", e)))?
            .as_millis() as u64;

        Ok(Self { deadline_ms: now_ms.saturating_add(duration.as_millis() as u64) })
    }

    /// Check if deadline expired
    pub fn is_expired(&self) -> bool {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        now_ms >= self.deadline_ms
    }

    /// Get remaining time in milliseconds
    pub fn remaining_ms(&self) -> u64 {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.deadline_ms.saturating_sub(now_ms)
    }

    /// Convert to HTTP header value
    pub fn to_header_value(&self) -> String {
        format!("{}", self.deadline_ms)
    }

    // TODO(ARCH-CHANGE): Add deadline propagation helpers:
    // - pub fn from_header(header: &str) -> Result<Self> (parse X-Deadline header)
    // - pub fn to_tokio_timeout(&self) -> Duration (convert to tokio timeout)
    // - pub fn propagate_to_request(&self, req: &mut Request) (add header)
    // - pub fn with_buffer(&self, buffer_ms: u64) -> Self (add safety margin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deadline() {
        let deadline = Deadline::from_duration(Duration::from_secs(10)).ok();
        assert!(deadline.is_some());

        let deadline = deadline.unwrap();
        assert!(!deadline.is_expired());
        assert!(deadline.remaining_ms() > 0);
    }
}
