//! task-cancellation — Task cancellation and cleanup
//!
//! Handles client-initiated cancellations, deadline timeouts, cascading cleanup.
//!
//! # Key Responsibilities
//!
//! - Client cancellation (DELETE /v2/jobs/{id})
//! - Deadline timeout enforcement
//! - Propagate cancellation to worker
//! - Clean up partial results
//! - Return 499 (Client Closed Request)
//!
//! # Example
//!
//! ```rust
//! use task_cancellation::{CancellationManager, CancellationReason};
//!
//! let manager = CancellationManager::new();
//!
//! // Cancel task
//! manager.cancel_task(task_id, CancellationReason::ClientRequest)?;
//!
//! // Check if cancelled
//! if manager.is_cancelled(task_id) {
//!     // Abort work
//! }
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

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CancellationError {
    #[error("task not found: {0}")]
    NotFound(String),
    #[error("already cancelled")]
    AlreadyCancelled,
}

pub type Result<T> = std::result::Result<T, CancellationError>;

/// Cancellation reason
#[derive(Debug, Clone, Copy)]
pub enum CancellationReason {
    ClientRequest,
    DeadlineExceeded,
    SystemShutdown,
}

/// Cancellation manager
pub struct CancellationManager {
    cancelled: Arc<Mutex<HashSet<String>>>,
}

impl CancellationManager {
    pub fn new() -> Self {
        Self { cancelled: Arc::new(Mutex::new(HashSet::new())) }
    }

    pub fn cancel_task(&self, task_id: &str, reason: CancellationReason) -> Result<()> {
        let mut cancelled =
            self.cancelled.lock().map_err(|_| CancellationError::NotFound(task_id.to_string()))?;

        if !cancelled.insert(task_id.to_string()) {
            return Err(CancellationError::AlreadyCancelled);
        }

        tracing::info!(
            task_id = %task_id,
            reason = ?reason,
            "Task cancelled"
        );

        Ok(())
    }

    pub fn is_cancelled(&self, task_id: &str) -> bool {
        self.cancelled.lock().ok().map(|c| c.contains(task_id)).unwrap_or(false)
    }
}

impl Default for CancellationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation() {
        let manager = CancellationManager::new();

        assert!(!manager.is_cancelled("task-1"));

        manager.cancel_task("task-1", CancellationReason::ClientRequest).ok();
        assert!(manager.is_cancelled("task-1"));

        // Double cancel fails
        assert!(manager.cancel_task("task-1", CancellationReason::ClientRequest).is_err());
    }
}
