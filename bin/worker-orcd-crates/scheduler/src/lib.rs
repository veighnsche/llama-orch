//! scheduler — Single-slot job scheduler for M0
//!
//! M0 scope: Simple single-slot scheduler (one job at a time per worker).
//! Post-M0: Continuous batching, dynamic batch size optimization, multi-request scheduling.
//!
//! # M0 Responsibilities
//!
//! - Accept single inference job
//! - Validate job fits in VRAM (via vram-residency)
//! - Execute job (one at a time)
//! - Track job state (pending, executing, completed)
//!
//! # Post-M0 Enhancements
//!
//! - Continuous batching support
//! - Dynamic batch size optimization
//! - KV cache allocation planning
//! - Prefill/decode phase scheduling
//! - Multi-request batch scheduling
//!
//! See: ARCHITECTURE_CHANGE_PLAN.md Phase 5 (production hardening)

// Medium-importance crate: TIER 3 Clippy configuration
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::missing_errors_doc)]

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("worker busy: slot occupied")]
    WorkerBusy,
    #[error("job not found: {0}")]
    JobNotFound(String),
}

pub type Result<T> = std::result::Result<T, SchedulerError>;

/// Job state for M0 single-slot scheduler
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    Pending,
    Executing,
    Completed,
    Failed,
}

/// Single-slot scheduler for M0
pub struct Scheduler {
    current_job: Option<String>, // job_id
    state: JobState,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            current_job: None,
            state: JobState::Pending,
        }
    }
    
    /// Check if scheduler can accept a new job
    pub fn can_accept(&self) -> bool {
        self.current_job.is_none() || self.state == JobState::Completed || self.state == JobState::Failed
    }
    
    /// Schedule a job (M0: single slot only)
    pub fn schedule(&mut self, job_id: String) -> Result<()> {
        if !self.can_accept() {
            return Err(SchedulerError::WorkerBusy);
        }
        
        self.current_job = Some(job_id);
        self.state = JobState::Pending;
        Ok(())
    }
    
    /// Mark job as executing
    pub fn mark_executing(&mut self, job_id: &str) -> Result<()> {
        if self.current_job.as_deref() != Some(job_id) {
            return Err(SchedulerError::JobNotFound(job_id.to_string()));
        }
        self.state = JobState::Executing;
        Ok(())
    }
    
    /// Mark job as completed
    pub fn mark_completed(&mut self, job_id: &str) -> Result<()> {
        if self.current_job.as_deref() != Some(job_id) {
            return Err(SchedulerError::JobNotFound(job_id.to_string()));
        }
        self.state = JobState::Completed;
        Ok(())
    }
    
    /// Mark job as failed
    pub fn mark_failed(&mut self, job_id: &str) -> Result<()> {
        if self.current_job.as_deref() != Some(job_id) {
            return Err(SchedulerError::JobNotFound(job_id.to_string()));
        }
        self.state = JobState::Failed;
        Ok(())
    }
    
    /// Get current job state
    pub fn get_state(&self, job_id: &str) -> Result<JobState> {
        if self.current_job.as_deref() != Some(job_id) {
            return Err(SchedulerError::JobNotFound(job_id.to_string()));
        }
        Ok(self.state.clone())
    }
    
    /// Clear completed/failed job to accept new one
    pub fn clear(&mut self) {
        if self.state == JobState::Completed || self.state == JobState::Failed {
            self.current_job = None;
            self.state = JobState::Pending;
        }
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_single_slot_scheduling() {
        let mut scheduler = Scheduler::new();
        
        // Can accept first job
        assert!(scheduler.can_accept());
        assert!(scheduler.schedule("job-1".to_string()).is_ok());
        
        // Cannot accept second job while first is pending
        assert!(!scheduler.can_accept());
        assert!(scheduler.schedule("job-2".to_string()).is_err());
        
        // Mark executing
        assert!(scheduler.mark_executing("job-1").is_ok());
        assert_eq!(scheduler.get_state("job-1").unwrap(), JobState::Executing);
        
        // Mark completed
        assert!(scheduler.mark_completed("job-1").is_ok());
        
        // Can accept new job after completion
        scheduler.clear();
        assert!(scheduler.can_accept());
        assert!(scheduler.schedule("job-2".to_string()).is_ok());
    }
}
