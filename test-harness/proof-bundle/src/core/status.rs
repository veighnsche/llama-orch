//! Test status enumeration

use serde::{Deserialize, Serialize};

/// Status of a test execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TestStatus {
    /// Test passed
    Passed,
    
    /// Test failed
    Failed,
    
    /// Test was ignored/skipped
    Ignored,
}

impl TestStatus {
    /// Check if this is a passing status
    pub fn is_passing(self) -> bool {
        matches!(self, TestStatus::Passed)
    }
    
    /// Check if this is a failing status
    pub fn is_failing(self) -> bool {
        matches!(self, TestStatus::Failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_is_passing() {
        assert!(TestStatus::Passed.is_passing());
        assert!(!TestStatus::Failed.is_passing());
        assert!(!TestStatus::Ignored.is_passing());
    }
    
    #[test]
    fn test_is_failing() {
        assert!(TestStatus::Failed.is_failing());
        assert!(!TestStatus::Passed.is_failing());
        assert!(!TestStatus::Ignored.is_failing());
    }
    
    #[test]
    fn test_serialization() {
        assert_eq!(serde_json::to_string(&TestStatus::Passed).unwrap(), r#""passed""#);
        assert_eq!(serde_json::to_string(&TestStatus::Failed).unwrap(), r#""failed""#);
        assert_eq!(serde_json::to_string(&TestStatus::Ignored).unwrap(), r#""ignored""#);
    }
}
