//! Jobs Contract
//!
//! TEAM-305: Interface trait for breaking circular dependency between
//! job-server and narration-core.
//! TEAM-312: Renamed from job-registry-interface to jobs-contract and moved to contracts/
//!
//! ## Problem
//! - job-server depends on narration-core (for narration events)
//! - narration-core test binaries need job-server (for JobRegistry)
//! - This creates a circular dependency
//!
//! ## Solution
//! - Extract JobRegistry interface to this contract
//! - job-server implements the trait
//! - narration-core test binaries depend on contract (not job-server)
//! - No circular dependency!

use tokio::sync::mpsc::UnboundedReceiver;

/// Job state in the registry
///
/// TEAM-305: Shared between interface and implementation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    /// Job is queued, waiting for processing
    Queued,
    /// Job is currently being processed
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed with error message
    Failed(String),
    /// Job was cancelled by user (TEAM-305: Added for cancellation support)
    Cancelled,
}

/// Job registry trait
///
/// TEAM-305: Interface allows narration-core test binaries to use
/// real JobRegistry without circular dependency
///
/// This trait defines the minimal interface needed by test binaries.
/// The full implementation lives in job-server crate.
pub trait JobRegistryInterface<T>: Send + Sync {
    /// Create a new job and return job_id
    fn create_job(&self) -> String;

    /// Set payload for a job (for deferred execution)
    fn set_payload(&self, job_id: &str, payload: serde_json::Value);

    /// Take payload from a job (consumes it)
    fn take_payload(&self, job_id: &str) -> Option<serde_json::Value>;

    /// Check if job exists
    fn has_job(&self, job_id: &str) -> bool;

    /// Get job state
    fn get_job_state(&self, job_id: &str) -> Option<JobState>;

    /// Update job state
    fn update_state(&self, job_id: &str, state: JobState);

    /// Set token receiver for streaming
    fn set_token_receiver(&self, job_id: &str, receiver: UnboundedReceiver<T>);

    /// Take the token receiver for a job (consumes it)
    fn take_token_receiver(&self, job_id: &str) -> Option<UnboundedReceiver<T>>;

    /// Remove a job from the registry
    fn remove_job(&self, job_id: &str);

    /// Get count of jobs in registry
    fn job_count(&self) -> usize;

    /// Get all job IDs
    fn job_ids(&self) -> Vec<String>;

    /// Cancel a job (TEAM-305: Added for cancellation support)
    fn cancel_job(&self, job_id: &str) -> bool;
}
