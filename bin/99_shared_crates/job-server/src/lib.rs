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
use std::time::Duration;  // TEAM-305: For timeout
use tokio::sync::mpsc::UnboundedSender;

// TEAM-186: For execute_and_stream helper
use futures::stream::{self, Stream};
use observability_narration_core::NarrationFactory;

// TEAM-305: For job cancellation
use tokio_util::sync::CancellationToken;

// TEAM-197: Migrated to narration-core v0.5.0 pattern
// Actor: "job-exec" (8 chars, ≤10 limit)
const NARRATE: NarrationFactory = NarrationFactory::new("job-exec");

// ============================================================================
// TEAM-305-FIX: Job Error Types
// ============================================================================

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
    Cancelled,  // TEAM-305: New state for cancelled jobs
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
///
/// TEAM-305: Added cancellation_token for job cancellation
pub struct Job<T> {
    pub job_id: String,
    pub state: JobState,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub token_receiver: Option<TokenReceiver<T>>,
    /// TEAM-186: Store operation payload for deferred execution
    pub payload: Option<serde_json::Value>,
    /// TEAM-305: Cancellation token for graceful job cancellation
    pub cancellation_token: CancellationToken,
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
    /// TEAM-305: Initialize with cancellation token
    pub fn create_job(&self) -> String {
        let job_id = format!("job-{}", uuid::Uuid::new_v4());

        let job = Job {
            job_id: job_id.clone(),
            state: JobState::Queued,
            created_at: chrono::Utc::now(),
            token_receiver: None,
            payload: None, // TEAM-186: Initialize as None
            cancellation_token: CancellationToken::new(),  // TEAM-305: Initialize cancellation token
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

    /// Cancel a job
    ///
    /// TEAM-305: Gracefully cancel a running job
    /// This signals the executor to stop processing and updates the job state
    pub fn cancel_job(&self, job_id: &str) -> bool {
        let mut jobs = self.jobs.lock().unwrap();
        if let Some(job) = jobs.get_mut(job_id) {
            // Only cancel if job is Queued or Running
            match job.state {
                JobState::Queued | JobState::Running => {
                    job.cancellation_token.cancel();
                    job.state = JobState::Cancelled;
                    true
                }
                _ => false,  // Already completed, failed, or cancelled
            }
        } else {
            false  // Job not found
        }
    }

    /// Get cancellation token for a job
    ///
    /// TEAM-305: Retrieve cancellation token for executor
    pub fn get_cancellation_token(&self, job_id: &str) -> Option<CancellationToken> {
        self.jobs.lock().unwrap().get(job_id).map(|j| j.cancellation_token.clone())
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
// TEAM-305: Implement JobRegistryInterface Trait
// ============================================================================

impl<T> job_registry_interface::JobRegistryInterface<T> for JobRegistry<T>
where
    T: Send + 'static,
{
    fn create_job(&self) -> String {
        self.create_job()
    }
    
    fn set_payload(&self, job_id: &str, payload: serde_json::Value) {
        self.set_payload(job_id, payload)
    }
    
    fn take_payload(&self, job_id: &str) -> Option<serde_json::Value> {
        self.take_payload(job_id)
    }
    
    fn has_job(&self, job_id: &str) -> bool {
        self.has_job(job_id)
    }
    
    fn get_job_state(&self, job_id: &str) -> Option<job_registry_interface::JobState> {
        self.get_job_state(job_id).map(|state| match state {
            JobState::Queued => job_registry_interface::JobState::Queued,
            JobState::Running => job_registry_interface::JobState::Running,
            JobState::Completed => job_registry_interface::JobState::Completed,
            JobState::Failed(msg) => job_registry_interface::JobState::Failed(msg),
            JobState::Cancelled => job_registry_interface::JobState::Cancelled,
        })
    }
    
    fn update_state(&self, job_id: &str, state: job_registry_interface::JobState) {
        let local_state = match state {
            job_registry_interface::JobState::Queued => JobState::Queued,
            job_registry_interface::JobState::Running => JobState::Running,
            job_registry_interface::JobState::Completed => JobState::Completed,
            job_registry_interface::JobState::Failed(msg) => JobState::Failed(msg),
            job_registry_interface::JobState::Cancelled => JobState::Cancelled,
        };
        self.update_state(job_id, local_state)
    }
    
    fn set_token_receiver(&self, job_id: &str, receiver: tokio::sync::mpsc::UnboundedReceiver<T>) {
        self.set_token_receiver(job_id, receiver)
    }
    
    fn take_token_receiver(&self, job_id: &str) -> Option<tokio::sync::mpsc::UnboundedReceiver<T>> {
        self.take_token_receiver(job_id)
    }
    
    fn remove_job(&self, job_id: &str) {
        self.remove_job(job_id);
    }
    
    fn job_count(&self) -> usize {
        self.job_count()
    }
    
    fn job_ids(&self) -> Vec<String> {
        self.job_ids()
    }
    
    fn cancel_job(&self, job_id: &str) -> bool {
        self.cancel_job(job_id)
    }
}

// ============================================================================
// TEAM-186: Execute and Stream Helper
// TEAM-305: Added timeout and cancellation support
// ============================================================================

/// Execute a job and stream its results (without timeout)
///
/// TEAM-186: Reusable helper for deferred execution pattern
/// TEAM-305: For backward compatibility, use execute_and_stream_with_timeout for new code
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

        let registry_clone = registry.clone();
        
        tokio::spawn(async move {
            // TEAM-197: Use narration v0.5.0 pattern
            NARRATE
                .action("execute")
                .job_id(&job_id_clone)
                .context(job_id_clone.clone())
                .human("Executing job {}")
                .emit();

            // TEAM-304: Execute the job and update state based on result
            let result = executor(job_id_clone.clone(), payload).await;
            
            match result {
                Ok(_) => {
                    registry_clone.update_state(&job_id_clone, JobState::Completed);
                }
                Err(e) => {
                    registry_clone.update_state(&job_id_clone, JobState::Failed(e.to_string()));
                    NARRATE
                        .action("failed")
                        .job_id(&job_id_clone)
                        .context(job_id_clone.clone())
                        .context(e.to_string())
                        .human("Job {} failed: {}")
                        .error_kind("job_execution_failed")
                        .emit_error();
                }
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

    // TEAM-304: Stream results and send [DONE] or [ERROR]
    let receiver = registry.take_token_receiver(&job_id);
    let registry_clone = registry.clone();
    let job_id_clone = job_id.clone();

    stream::unfold((receiver, false, job_id_clone, registry_clone), 
        |(rx_opt, done_sent, job_id, registry)| async move {
            if done_sent {
                return None;
            }

            match rx_opt {
                Some(mut rx) => match rx.recv().await {
                    Some(token) => {
                        let data = token.to_string();
                        Some((data, (Some(rx), false, job_id, registry)))
                    }
                    None => {
                        // TEAM-304: Channel closed - check job state and send appropriate signal
                        let state = registry.get_job_state(&job_id);
                        let signal = match state {
                            Some(JobState::Failed(err)) => format!("[ERROR] {}", err),
                            _ => "[DONE]".to_string(),
                        };
                        Some((signal, (None, true, job_id, registry)))
                    }
                },
                None => {
                    // TEAM-304: No receiver - send [DONE] immediately
                    Some(("[DONE]".to_string(), (None, true, job_id, registry)))
                }
            }
        }
    )
}

/// Execute a job and stream its results with timeout and cancellation support
///
/// TEAM-305: Enhanced version with timeout and cancellation
///
/// This function:
/// 1. Retrieves the job payload from the registry
/// 2. Spawns async execution in background with timeout and cancellation support
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
/// - `timeout`: Maximum duration for job execution (None for no timeout)
///
/// # Returns
/// A stream of string tokens suitable for SSE streaming
///
/// # Example
/// ```rust,ignore
/// use job_server::execute_and_stream_with_timeout;
/// use std::time::Duration;
///
/// let stream = execute_and_stream_with_timeout(
///     job_id,
///     registry,
///     |job_id, payload| async move {
///         // Execute job logic here
///         route_job(state, payload).await
///     },
///     Some(Duration::from_secs(300))  // 5 minute timeout
/// ).await;
/// ```
pub async fn execute_and_stream_with_timeout<T, F, Exec>(
    job_id: String,
    registry: Arc<JobRegistry<T>>,
    executor: Exec,
    timeout: Option<Duration>,
) -> impl Stream<Item = String>
where
    T: ToString + Send + 'static,
    F: std::future::Future<Output = Result<(), anyhow::Error>> + Send + 'static,
    Exec: FnOnce(String, serde_json::Value) -> F + Send + 'static,
{
    // TEAM-305: Retrieve payload and cancellation token
    let payload = registry.take_payload(&job_id);
    let cancellation_token = registry.get_cancellation_token(&job_id);

    if let Some(payload) = payload {
        let job_id_clone = job_id.clone();
        let registry_clone = registry.clone();
        
        tokio::spawn(async move {
            // TEAM-305: Use narration v0.5.0 pattern
            NARRATE
                .action("execute")
                .job_id(&job_id_clone)
                .context(job_id_clone.clone())
                .human("Executing job {}")
                .emit();

            // TEAM-305-FIX: Execute with timeout and cancellation support using JobError
            let execution_future = executor(job_id_clone.clone(), payload);
            
            let result: Result<(), JobError> = if let Some(cancellation_token) = cancellation_token {
                // With cancellation support
                if let Some(timeout_duration) = timeout {
                    // With both timeout and cancellation
                    tokio::select! {
                        result = execution_future => result.map_err(JobError::from),
                        _ = cancellation_token.cancelled() => {
                            Err(JobError::Cancelled)
                        }
                        _ = tokio::time::sleep(timeout_duration) => {
                            Err(JobError::Timeout(timeout_duration))
                        }
                    }
                } else {
                    // With cancellation only
                    tokio::select! {
                        result = execution_future => result.map_err(JobError::from),
                        _ = cancellation_token.cancelled() => {
                            Err(JobError::Cancelled)
                        }
                    }
                }
            } else if let Some(timeout_duration) = timeout {
                // With timeout only
                match tokio::time::timeout(timeout_duration, execution_future).await {
                    Ok(result) => result.map_err(JobError::from),
                    Err(_) => Err(JobError::Timeout(timeout_duration)),
                }
            } else {
                // No timeout or cancellation
                execution_future.await.map_err(JobError::from)
            };
            
            // TEAM-305-FIX: Update state based on JobError type
            match result {
                Ok(_) => {
                    registry_clone.update_state(&job_id_clone, JobState::Completed);
                }
                Err(JobError::Cancelled) => {
                    registry_clone.update_state(&job_id_clone, JobState::Cancelled);
                    NARRATE
                        .action("cancelled")
                        .job_id(&job_id_clone)
                        .context(job_id_clone.clone())
                        .human("Job {} cancelled")
                        .emit();
                }
                Err(JobError::Timeout(duration)) => {
                    let error_msg = format!("Timeout after {:?}", duration);
                    registry_clone.update_state(&job_id_clone, JobState::Failed(error_msg.clone()));
                    NARRATE
                        .action("timeout")
                        .job_id(&job_id_clone)
                        .context(job_id_clone.clone())
                        .context(error_msg.clone())
                        .human("Job {} timed out: {}")
                        .error_kind("job_timeout")
                        .emit_error();
                }
                Err(JobError::ExecutionFailed(error_msg)) => {
                    registry_clone.update_state(&job_id_clone, JobState::Failed(error_msg.clone()));
                    NARRATE
                        .action("failed")
                        .job_id(&job_id_clone)
                        .context(job_id_clone.clone())
                        .context(error_msg.clone())
                        .human("Job {} failed: {}")
                        .error_kind("job_execution_failed")
                        .emit_error();
                }
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

    // TEAM-305: Stream results and send [DONE], [ERROR], or [CANCELLED]
    let receiver = registry.take_token_receiver(&job_id);
    let registry_clone = registry.clone();
    let job_id_clone = job_id.clone();

    stream::unfold((receiver, false, job_id_clone, registry_clone), 
        |(rx_opt, done_sent, job_id, registry)| async move {
            if done_sent {
                return None;
            }

            match rx_opt {
                Some(mut rx) => match rx.recv().await {
                    Some(token) => {
                        let data = token.to_string();
                        Some((data, (Some(rx), false, job_id, registry)))
                    }
                    None => {
                        // TEAM-305: Channel closed - check job state and send appropriate signal
                        let state = registry.get_job_state(&job_id);
                        let signal = match state {
                            Some(JobState::Failed(err)) => format!("[ERROR] {}", err),
                            Some(JobState::Cancelled) => "[CANCELLED]".to_string(),
                            _ => "[DONE]".to_string(),
                        };
                        Some((signal, (None, true, job_id, registry)))
                    }
                },
                None => {
                    // TEAM-305: No receiver - check state and send appropriate signal
                    let state = registry.get_job_state(&job_id);
                    let signal = match state {
                        Some(JobState::Cancelled) => "[CANCELLED]".to_string(),
                        _ => "[DONE]".to_string(),
                    };
                    Some((signal, (None, true, job_id, registry)))
                }
            }
        }
    )
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
