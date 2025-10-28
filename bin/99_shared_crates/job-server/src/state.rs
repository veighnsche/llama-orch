//! Job state definitions
//!
//! TEAM-312: Extracted to separate module

/// Job state in the registry
#[derive(Debug, Clone)]
pub enum JobState {
    /// Job is queued, waiting for processing
    Queued,
    /// Job is currently being processed
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed with error message
    Failed(String),
    /// Job was cancelled by user
    Cancelled, // TEAM-305: New state for cancelled jobs
}
