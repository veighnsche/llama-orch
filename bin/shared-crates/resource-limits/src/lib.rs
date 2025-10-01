//! resource-limits — Resource limit enforcement
//!
//! Enforces VRAM, memory, CPU, and time limits per job.

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
}
