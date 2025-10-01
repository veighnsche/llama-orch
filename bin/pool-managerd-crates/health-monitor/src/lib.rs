//! health-monitor — Worker and pool health monitoring
//!
//! Continuously monitors worker health, detects failures, triggers recovery.

// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::integer_arithmetic)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::missing_errors_doc)]

use std::time::{Duration, Instant};

pub struct HealthMonitor {
    last_heartbeat: Option<Instant>,
    timeout: Duration,
}

impl HealthMonitor {
    pub fn new(timeout: Duration) -> Self {
        Self {
            last_heartbeat: None,
            timeout,
        }
    }
    
    pub fn record_heartbeat(&mut self) {
        self.last_heartbeat = Some(Instant::now());
    }
    
    pub fn is_healthy(&self) -> bool {
        self.last_heartbeat
            .map(|last| last.elapsed() < self.timeout)
            .unwrap_or(false)
    }
}
