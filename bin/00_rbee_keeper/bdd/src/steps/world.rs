// Created by: TEAM-135 (BDD scaffolding)
// Updated by: TEAM-151 (2025-10-20) - Added health check state
//! BDD World for rbee-keeper integration tests

use cucumber::World;
use std::process::Child;

#[derive(Debug, World)]
pub struct BddWorld {
    /// Last validation result
    pub last_result: Option<Result<(), String>>,

    /// Queen URL for health checks
    pub queen_url: String,

    /// Queen process handle (if started by test)
    pub queen_process: Option<Child>,

    /// Health check result
    pub health_check_result: Option<Result<bool, String>>,

    /// Expected message for validation
    pub expected_message: Option<String>,
}

impl Default for BddWorld {
    fn default() -> Self {
        Self {
            last_result: None,
            queen_url: "http://localhost:8500".to_string(),
            queen_process: None,
            health_check_result: None,
            expected_message: None,
        }
    }
}

impl Drop for BddWorld {
    fn drop(&mut self) {
        // Clean up: kill queen process if we started it
        if let Some(mut child) = self.queen_process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl BddWorld {
    /// Store validation result
    pub fn store_result(&mut self, result: Result<(), String>) {
        self.last_result = Some(result);
    }

    /// Check if last validation succeeded
    pub fn last_succeeded(&self) -> bool {
        matches!(self.last_result, Some(Ok(())))
    }

    /// Check if last validation failed
    pub fn last_failed(&self) -> bool {
        matches!(self.last_result, Some(Err(_)))
    }
}
