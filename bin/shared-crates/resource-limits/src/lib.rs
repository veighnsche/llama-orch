//! resource-limits — Resource limit enforcement
//!
//! Enforces VRAM, memory, CPU, and time limits per job.
//!
//! TODO(ARCH-CHANGE): This crate is minimal. Per SECURITY_AUDIT Issue #17:
//! - Add enforcement logic (not just struct definition)
//! - Implement VRAM tracking per job
//! - Add execution timeout enforcement with cancellation
//! - Implement concurrent job limiting
//! - Add resource exhaustion detection
//! - Integrate with worker-orcd and pool-managerd
//! See: SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #17 (no resource limits)

// Medium-importance crate: TIER 3 Clippy configuration
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::missing_errors_doc)]

use std::time::Duration;

pub struct ResourceLimits {
    pub max_vram_per_job: u64,
    pub max_execution_time: Duration,
    pub max_concurrent_jobs: u32,
}

impl ResourceLimits {
    pub fn default_limits() -> Self {
        Self {
            max_vram_per_job: 20_000_000_000, // 20GB
            max_execution_time: Duration::from_secs(300), // 5 min
            max_concurrent_jobs: 4,
        }
    }
    
    // TODO(ARCH-CHANGE): Add enforcement methods:
    // - pub fn check_vram(&self, requested: u64) -> Result<()>
    // - pub fn check_execution_time(&self, elapsed: Duration) -> Result<()>
    // - pub fn check_concurrent_jobs(&self, current: u32) -> Result<()>
    // - pub fn track_job(&mut self, job_id: &str, vram: u64) -> Result<()>
    // - pub fn release_job(&mut self, job_id: &str)
}
