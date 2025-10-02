//! Test metadata - copied from src/metadata/types.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Test metadata extracted from source code annotations
///
/// Captures metadata about a test for proof bundle reporting.
///
/// # Example Annotations
///
/// ```rust,ignore
/// /// @priority: critical
/// /// @spec: ORCH-3250
/// /// @team: orchestrator
/// /// @owner: alice@example.com
/// #[test]
/// fn test_something() { }
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TestMetadata {
    /// Test priority level (critical, high, medium, low)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<String>,
    
    /// Spec or requirement ID (e.g., ORCH-3250)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec: Option<String>,
    
    /// Owning team
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team: Option<String>,
    
    /// Owner email
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    
    /// Related issue (e.g., #1234)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issue: Option<String>,
    
    /// Flakiness description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flaky: Option<String>,
    
    /// Expected timeout
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<String>,
    
    /// Required resources (e.g., GPU, CUDA)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub requires: Vec<String>,
    
    /// Tags for categorization
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub tags: Vec<String>,
    
    /// Custom key-value fields
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    #[serde(default)]
    pub custom: HashMap<String, String>,
}

impl TestMetadata {
    /// Check if this test is marked as critical
    pub fn is_critical(&self) -> bool {
        self.priority.as_deref() == Some("critical")
    }
    
    /// Check if this test is high priority (critical or high)
    pub fn is_high_priority(&self) -> bool {
        matches!(self.priority.as_deref(), Some("critical") | Some("high"))
    }
    
    /// Check if this test is marked as flaky
    pub fn is_flaky(&self) -> bool {
        self.flaky.is_some()
    }
    
    /// Get priority level as number for sorting (4=critical, 0=none)
    pub fn priority_level(&self) -> u8 {
        match self.priority.as_deref() {
            Some("critical") => 4,
            Some("high") => 3,
            Some("medium") => 2,
            Some("low") => 1,
            _ => 0,
        }
    }
}
