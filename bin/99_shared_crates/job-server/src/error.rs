//! Job error types
//!
//! TEAM-305-FIX: Type-safe error handling
//! TEAM-312: Extracted to separate module

use std::time::Duration;

/// Job execution error types
///
/// TEAM-305-FIX: Type-safe error handling instead of string matching
#[derive(Debug, Clone)]
pub enum JobError {
    /// Job was cancelled by user
    Cancelled,
    /// Job timed out after specified duration
    Timeout(Duration),
    /// Job execution failed with error message
    ExecutionFailed(String),
}

impl std::fmt::Display for JobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobError::Cancelled => write!(f, "Job cancelled by user"),
            JobError::Timeout(d) => write!(f, "Job timed out after {:?}", d),
            JobError::ExecutionFailed(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for JobError {}

impl From<anyhow::Error> for JobError {
    fn from(err: anyhow::Error) -> Self {
        JobError::ExecutionFailed(err.to_string())
    }
}
