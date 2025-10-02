//! Individual test result

use serde::{Deserialize, Serialize};
use super::{TestStatus, TestMetadata};

/// Result of a single test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name (e.g., "my_module::test_foo")
    pub name: String,
    
    /// Test status
    pub status: TestStatus,
    
    /// Duration in seconds
    pub duration_secs: f64,
    
    /// Captured stdout (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout: Option<String>,
    
    /// Captured stderr (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr: Option<String>,
    
    /// Error message (for failed tests)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    
    /// Test metadata (extracted from source)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<TestMetadata>,
}

impl TestResult {
    /// Create a new test result
    pub fn new(name: String, status: TestStatus) -> Self {
        Self {
            name,
            status,
            duration_secs: 0.0,
            stdout: None,
            stderr: None,
            error_message: None,
            metadata: None,
        }
    }
    
    /// Set duration
    pub fn with_duration(mut self, secs: f64) -> Self {
        self.duration_secs = secs;
        self
    }
    
    /// Set metadata
    pub fn with_metadata(mut self, metadata: TestMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }
    
    /// Set error message
    pub fn with_error(mut self, message: String) -> Self {
        self.error_message = Some(message);
        self
    }
}
