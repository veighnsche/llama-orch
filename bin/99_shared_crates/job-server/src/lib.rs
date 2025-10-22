//! Shared job registry for managing inference job state
//!
//! **Category:** State Management
//! **Pattern:** Registry Pattern
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! TEAM-154: Created by TEAM-154
//! TEAM-154 FIX: Fixed dual-call pattern by storing receiver instead of sender
//! TEAM-197: Migrated to narration-core v0.5.0 pattern
//!
//! This crate provides in-memory job state management for the dual-call pattern:
//! 1. POST creates job, returns job_id + sse_url
//! 2. GET streams results via SSE
//!
//! # Interface
//!
//! ## Registry Operations
//! ```rust,no_run
//! # use job_server::{JobRegistry, JobState};
//! # let registry: JobRegistry<String> = JobRegistry::new();
//! # let job_id = "job-123";
//! // CREATE
//! let job_id = registry.create_job();
//!
//! // READ
//! let exists = registry.has_job(&job_id);
//! let state = registry.get_job_state(&job_id);
//!
//! // UPDATE
//! registry.update_state(&job_id, JobState::Running);
//! // registry.set_token_receiver(&job_id, receiver);
//!
//! // DELETE
//! // let removed = registry.remove_job(&job_id);
//!
//! // UTILITY
//! let count = registry.job_count();
//! let ids = registry.job_ids();
//! ```
//!
//! # Pattern
//!
//! 1. POST /v1/inference → creates job, returns job_id + sse_url
//! 2. GET /v1/inference/{job_id}/stream → streams results via SSE
//!
//! # Usage
//!
//! ```rust
//! use job_server::{JobRegistry, JobState};
//! use tokio::sync::mpsc;
//!
//! let registry: JobRegistry<String> = JobRegistry::new();
//!
//! // Create a job (server generates ID)
//! let job_id = registry.create_job();
//!
//! // Store token receiver for streaming
//! let (tx, rx) = mpsc::unbounded_channel();
//! registry.set_token_receiver(&job_id, rx);
//!
//! // Later: retrieve job state
//! let state = registry.get_job_state(&job_id).unwrap();
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::UnboundedSender;

// TEAM-186: For execute_and_stream helper
use futures::stream::{self, Stream};
use observability_narration_core::NarrationFactory;

// TEAM-197: Migrated to narration-core v0.5.0 pattern
// Actor: "job-exec" (8 chars, ≤10 limit)
const NARRATE: NarrationFactory = NarrationFactory::new("job-exec");

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
}

/// Generic token response type
///
/// TEAM-154: Generic type parameter allows worker/queen/hive to use their own token types
pub type TokenSender<T> = UnboundedSender<T>;
pub type TokenReceiver<T> = tokio::sync::mpsc::UnboundedReceiver<T>;

/// Job information stored in registry
///
/// TEAM-154: Generic over token type T to support different token formats
/// - Worker: TokenResponse (Token/Error/Done)
/// - Queen: Could be different format
/// - Hive: Could be different format
///
/// TEAM-154 FIX: Store receiver, not sender!
/// - POST creates channel, stores receiver, passes sender to generation engine
/// - GET retrieves receiver and streams tokens
///
/// TEAM-186: Added payload field for deferred execution
/// - POST stores payload, GET retrieves and executes
pub struct Job<T> {
    pub job_id: String,
    pub state: JobState,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub token_receiver: Option<TokenReceiver<T>>,
    /// TEAM-186: Store operation payload for deferred execution
    pub payload: Option<serde_json::Value>,
}

// TEAM-154: Job cannot be cloned because UnboundedReceiver is not Clone
// We don't need Clone anyway - jobs are retrieved by reference

/// Job registry - tracks all active jobs
///
/// TEAM-154: Generic over token type T
/// - Central registry for dual-call pattern
/// - POST creates job and stores in registry
/// - GET retrieves job and subscribes to token stream
pub struct JobRegistry<T> {
    jobs: Arc<Mutex<HashMap<String, Job<T>>>>,
}

impl<T> JobRegistry<T>
where
    T: Send + 'static,
{
    /// Create a new job registry
    pub fn new() -> Self {
        Self { jobs: Arc::new(Mutex::new(HashMap::new())) }
    }

    /// Create a new job and return job_id
    ///
    /// TEAM-154: Server generates job_id (client doesn't provide it)
    pub fn create_job(&self) -> String {
        let job_id = format!("job-{}", uuid::Uuid::new_v4());

        let job = Job {
            job_id: job_id.clone(),
            state: JobState::Queued,
            created_at: chrono::Utc::now(),
            token_receiver: None,
            payload: None, // TEAM-186: Initialize as None
        };

        self.jobs.lock().unwrap().insert(job_id.clone(), job);
        job_id
    }

    /// Set payload for a job (for deferred execution)
    ///
    /// TEAM-186: Store operation payload to execute later when client connects
    pub fn set_payload(&self, job_id: &str, payload: serde_json::Value) {
        if let Some(job) = self.jobs.lock().unwrap().get_mut(job_id) {
            job.payload = Some(payload);
        }
    }

    /// Take payload from a job (consumes it)
    ///
    /// TEAM-186: Retrieve and remove payload for execution
    /// This can only be called once per job!
    pub fn take_payload(&self, job_id: &str) -> Option<serde_json::Value> {
        self.jobs.lock().unwrap().get_mut(job_id).and_then(|job| job.payload.take())
    }

    /// Check if job exists
    pub fn has_job(&self, job_id: &str) -> bool {
        self.jobs.lock().unwrap().contains_key(job_id)
    }

    /// Get job state
    pub fn get_job_state(&self, job_id: &str) -> Option<JobState> {
        self.jobs.lock().unwrap().get(job_id).map(|j| j.state.clone())
    }

    /// Update job state
    pub fn update_state(&self, job_id: &str, state: JobState) {
        if let Some(job) = self.jobs.lock().unwrap().get_mut(job_id) {
            job.state = state;
        }
    }

    /// Set token receiver for streaming
    ///
    /// TEAM-154 FIX: Store receiver, not sender!
    /// Called after job is queued, before GET stream connects
    pub fn set_token_receiver(&self, job_id: &str, receiver: TokenReceiver<T>) {
        if let Some(job) = self.jobs.lock().unwrap().get_mut(job_id) {
            job.token_receiver = Some(receiver);
        }
    }

    /// Take the token receiver for a job (consumes it)
    ///
    /// TEAM-154 FIX: GET stream endpoint takes ownership of receiver
    /// This can only be called once per job!
    pub fn take_token_receiver(&self, job_id: &str) -> Option<TokenReceiver<T>> {
        self.jobs.lock().unwrap().get_mut(job_id).and_then(|job| job.token_receiver.take())
    }

    /// Remove a job from the registry
    ///
    /// TEAM-154: For cleanup after job completes
    pub fn remove_job(&self, job_id: &str) -> Option<Job<T>> {
        self.jobs.lock().unwrap().remove(job_id)
    }

    /// Get count of jobs in registry
    pub fn job_count(&self) -> usize {
        self.jobs.lock().unwrap().len()
    }

    /// Get all job IDs
    pub fn job_ids(&self) -> Vec<String> {
        self.jobs.lock().unwrap().keys().cloned().collect()
    }
}

impl<T> Default for JobRegistry<T>
where
    T: Send + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for JobRegistry<T> {
    fn clone(&self) -> Self {
        Self { jobs: Arc::clone(&self.jobs) }
    }
}

// ============================================================================
// TEAM-186: Execute and Stream Helper
// ============================================================================

/// Execute a job and stream its results
///
/// TEAM-186: Reusable helper for deferred execution pattern
///
/// This function:
/// 1. Retrieves the job payload from the registry
/// 2. Spawns async execution in background
/// 3. Returns a stream of results for SSE
///
/// # Type Parameters
/// - `T`: Token type for streaming (must implement ToString)
/// - `F`: Future that executes the job
/// - `Exec`: Function that creates the execution future
///
/// # Arguments
/// - `job_id`: The job ID to execute
/// - `registry`: Job registry containing the job
/// - `executor`: Function that takes (job_id, payload) and returns a Future
///
/// # Returns
/// A stream of string tokens suitable for SSE streaming
///
/// # Example
/// ```rust,ignore
/// use job_server::execute_and_stream;
///
/// let stream = execute_and_stream(
///     job_id,
///     registry,
///     |job_id, payload| async move {
///         // Execute job logic here
///         route_job(state, payload).await
///     }
/// ).await;
/// ```
pub async fn execute_and_stream<T, F, Exec>(
    job_id: String,
    registry: Arc<JobRegistry<T>>,
    executor: Exec,
) -> impl Stream<Item = String>
where
    T: ToString + Send + 'static,
    F: std::future::Future<Output = Result<(), anyhow::Error>> + Send + 'static,
    Exec: FnOnce(String, serde_json::Value) -> F + Send + 'static,
{
    // TEAM-186: Retrieve payload and spawn execution
    let payload = registry.take_payload(&job_id);

    if let Some(payload) = payload {
        let job_id_clone = job_id.clone();

        tokio::spawn(async move {
            // TEAM-197: Use narration v0.5.0 pattern
            NARRATE
                .action("execute")
                .job_id(&job_id_clone)
                .context(job_id_clone.clone())
                .human("Executing job {}")
                .emit();

            // Execute the job
            if let Err(e) = executor(job_id_clone.clone(), payload).await {
                NARRATE
                    .action("failed")
                    .job_id(&job_id_clone)
                    .context(job_id_clone.clone())
                    .context(e.to_string())
                    .human("Job {} failed: {}")
                    .error_kind("job_execution_failed")
                    .emit_error();
            }
        });
    } else {
        NARRATE
            .action("no_payload")
            .job_id(&job_id)
            .context(job_id.clone())
            .human("Warning: No payload found for job {}")
            .emit();
    }

    // TEAM-186: Stream results
    let receiver = registry.take_token_receiver(&job_id);

    stream::unfold(receiver, |rx_opt| async move {
        match rx_opt {
            Some(mut rx) => match rx.recv().await {
                Some(token) => {
                    let data = token.to_string();
                    Some((data, Some(rx)))
                }
                None => None,
            },
            None => None,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_job() {
        let registry: JobRegistry<String> = JobRegistry::new();
        let job_id = registry.create_job();

        assert!(job_id.starts_with("job-"));
        assert_eq!(registry.job_count(), 1);
    }

    #[test]
    fn test_get_job_state() {
        let registry: JobRegistry<String> = JobRegistry::new();
        let job_id = registry.create_job();

        let state = registry.get_job_state(&job_id).unwrap();
        assert!(matches!(state, JobState::Queued));
    }

    #[test]
    fn test_update_state() {
        let registry: JobRegistry<String> = JobRegistry::new();
        let job_id = registry.create_job();

        registry.update_state(&job_id, JobState::Running);

        let state = registry.get_job_state(&job_id).unwrap();
        assert!(matches!(state, JobState::Running));
    }

    #[tokio::test]
    async fn test_token_receiver() {
        let registry: JobRegistry<String> = JobRegistry::new();
        let job_id = registry.create_job();

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        registry.set_token_receiver(&job_id, rx);

        // Send a token
        tx.send("test".to_string()).unwrap();

        // Take receiver and read token
        let mut receiver = registry.take_token_receiver(&job_id).unwrap();
        let received = receiver.recv().await.unwrap();
        assert_eq!(received, "test");
    }

    #[test]
    fn test_remove_job() {
        let registry: JobRegistry<String> = JobRegistry::new();
        let job_id = registry.create_job();

        assert_eq!(registry.job_count(), 1);

        let removed = registry.remove_job(&job_id);
        assert!(removed.is_some());
        assert_eq!(registry.job_count(), 0);
    }

    #[test]
    fn test_job_ids() {
        let registry: JobRegistry<String> = JobRegistry::new();
        let job_id1 = registry.create_job();
        let job_id2 = registry.create_job();

        let ids = registry.job_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&job_id1));
        assert!(ids.contains(&job_id2));
    }
}
