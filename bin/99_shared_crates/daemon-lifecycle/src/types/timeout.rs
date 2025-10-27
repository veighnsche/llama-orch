//! Timeout configuration types
//!
//! TEAM-329: Extracted from src/utils/timeout.rs

use std::time::Duration;

/// Configuration for timeout enforcement
///
/// TEAM-276: Wrapper around TimeoutEnforcer for daemon operations
/// TEAM-329: Moved from src/utils/timeout.rs to types/timeout.rs
pub struct TimeoutConfig {
    /// Operation name for narration
    pub operation_name: String,

    /// Timeout duration
    pub timeout: Duration,

    /// Optional job_id for narration routing
    pub job_id: Option<String>,
}

impl TimeoutConfig {
    /// Create a new timeout config
    pub fn new(operation_name: impl Into<String>, timeout: Duration) -> Self {
        Self { operation_name: operation_name.into(), timeout, job_id: None }
    }

    /// Set the job_id for narration routing
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }

    /// Set timeout in seconds (convenience method)
    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout = Duration::from_secs(secs);
        self
    }
}
