//! Shared job registry for managing inference job state
//!
//! **Category:** State Management
//! **Pattern:** Registry Pattern
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! TEAM-154: Created by TEAM-154
//! TEAM-154 FIX: Fixed dual-call pattern by storing receiver instead of sender
//! TEAM-197: Migrated to narration-core v0.5.0 pattern
//! TEAM-312: Split into modules, migrated to n!() macro
//!
//! This crate provides in-memory job state management for the dual-call pattern:
//! 1. POST creates job, returns job_id + sse_url
//! 2. GET streams results via SSE
//!
//! # Modules
//!
//! - `error` - Job error types
//! - `state` - Job state enum
//! - `registry` - Job registry implementation
//! - `execution` - Job execution and streaming

// TEAM-312: Module organization
mod error;
mod state;
mod registry;
mod execution;

// Re-exports
pub use error::JobError;
pub use state::JobState;
pub use registry::{Job, JobRegistry, TokenReceiver, TokenSender};
pub use execution::execute_and_stream; // TEAM-312: Deleted execute_and_stream_with_timeout (backwards compat trap)

// ============================================================================
// TEAM-305/312: Trait Implementation for jobs-contract
// ============================================================================

impl<T> jobs_contract::JobRegistryInterface<T> for JobRegistry<T>
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
    
    fn get_job_state(&self, job_id: &str) -> Option<jobs_contract::JobState> {
        self.get_job_state(job_id).map(|state| match state {
            JobState::Queued => jobs_contract::JobState::Queued,
            JobState::Running => jobs_contract::JobState::Running,
            JobState::Completed => jobs_contract::JobState::Completed,
            JobState::Failed(msg) => jobs_contract::JobState::Failed(msg),
            JobState::Cancelled => jobs_contract::JobState::Cancelled,
        })
    }
    
    fn update_state(&self, job_id: &str, state: jobs_contract::JobState) {
        let local_state = match state {
            jobs_contract::JobState::Queued => JobState::Queued,
            jobs_contract::JobState::Running => JobState::Running,
            jobs_contract::JobState::Completed => JobState::Completed,
            jobs_contract::JobState::Failed(msg) => JobState::Failed(msg),
            jobs_contract::JobState::Cancelled => JobState::Cancelled,
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
// TESTS
// ============================================================================

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
