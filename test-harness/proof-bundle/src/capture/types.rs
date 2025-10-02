//! Types for test capture results

use serde::{Deserialize, Serialize};

/// Summary of test run results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    /// Total number of tests run
    pub total: usize,
    
    /// Number of tests that passed
    pub passed: usize,
    
    /// Number of tests that failed
    pub failed: usize,
    
    /// Number of tests that were ignored
    pub ignored: usize,
    
    /// Total duration in seconds
    pub duration_secs: f64,
    
    /// Pass rate (0.0 to 100.0)
    pub pass_rate: f64,
    
    /// Individual test results
    pub tests: Vec<TestResult>,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name (e.g., "vram_residency::tests::test_seal_model")
    pub name: String,
    
    /// Test status
    pub status: TestStatus,
    
    /// Duration in seconds
    pub duration_secs: f64,
    
    /// Standard output (if captured)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout: Option<String>,
    
    /// Standard error (if captured)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr: Option<String>,
    
    /// Error message (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

/// Test status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TestStatus {
    Passed,
    Failed,
    Ignored,
    Timeout,
}

impl TestSummary {
    /// Calculate pass rate
    pub fn calculate_pass_rate(passed: usize, total: usize) -> f64 {
        if total == 0 {
            0.0
        } else {
            (passed as f64 / total as f64) * 100.0
        }
    }
}
