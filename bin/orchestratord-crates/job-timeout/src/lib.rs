//! job-timeout — Job timeout enforcement

// Medium-importance crate: TIER 3 Clippy configuration
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::missing_errors_doc)]

use std::time::Duration;

pub struct TimeoutManager {
    default_timeout: Duration,
}

impl TimeoutManager {
    pub fn new(timeout: Duration) -> Self {
        Self {
            default_timeout: timeout,
        }
    }
}
