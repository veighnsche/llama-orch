//! Job registry implementation
//!
//! TEAM-312: Extracted to separate module

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

use crate::state::JobState;

/// Generic token response type
///
/// TEAM-154: Generic type parameter allows worker/queen/hive to use their own token types
pub type TokenSender<T> = tokio::sync::mpsc::UnboundedSender<T>;
pub type TokenReceiver<T> = tokio::sync::mpsc::UnboundedReceiver<T>;

/// Job information stored in registry
///
/// TEAM-154: Generic over token type T
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

/// Job registry - tracks all active jobs
///
/// TEAM-154: Generic over token type T
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
    /// TEAM-154: Server generates job_id
    /// TEAM-305: Initialize with cancellation token
    pub fn create_job(&self) -> String {
        let job_id = format!("job-{}", uuid::Uuid::new_v4());

        let job = Job {
            job_id: job_id.clone(),
            state: JobState::Queued,
            created_at: chrono::Utc::now(),
            token_receiver: None,
            payload: None,
            cancellation_token: CancellationToken::new(),
        };

        self.jobs.lock().unwrap().insert(job_id.clone(), job);
        job_id
    }

    /// Set payload for a job (for deferred execution)
    pub fn set_payload(&self, job_id: &str, payload: serde_json::Value) {
        if let Some(job) = self.jobs.lock().unwrap().get_mut(job_id) {
            job.payload = Some(payload);
        }
    }

    /// Take payload from a job (consumes it)
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
    pub fn set_token_receiver(&self, job_id: &str, receiver: TokenReceiver<T>) {
        if let Some(job) = self.jobs.lock().unwrap().get_mut(job_id) {
            job.token_receiver = Some(receiver);
        }
    }

    /// Take the token receiver for a job (consumes it)
    pub fn take_token_receiver(&self, job_id: &str) -> Option<TokenReceiver<T>> {
        self.jobs.lock().unwrap().get_mut(job_id).and_then(|job| job.token_receiver.take())
    }

    /// Remove a job from the registry
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
    pub fn cancel_job(&self, job_id: &str) -> bool {
        let mut jobs = self.jobs.lock().unwrap();
        if let Some(job) = jobs.get_mut(job_id) {
            match job.state {
                JobState::Queued | JobState::Running => {
                    job.cancellation_token.cancel();
                    job.state = JobState::Cancelled;
                    true
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Get cancellation token for a job
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
