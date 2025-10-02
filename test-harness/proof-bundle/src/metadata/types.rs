//! Core metadata types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Test metadata
///
/// Captures metadata about a test for proof bundle reporting.
///
/// # Fields
///
/// - **priority**: Test priority level (`critical`, `high`, `medium`, `low`)
/// - **spec**: Spec or requirement ID (e.g., `ORCH-3250`)
/// - **team**: Owning team name
/// - **owner**: Owner email address
/// - **issue**: Related issue tracker ID (e.g., `#1234`)
/// - **flaky**: Flakiness description
/// - **timeout**: Expected timeout duration
/// - **requires**: Required resources (e.g., `["GPU", "CUDA"]`)
/// - **tags**: Test tags (e.g., `["integration", "slow"]`)
/// - **custom**: Custom key-value pairs
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TestMetadata {
    /// Test priority level
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<String>,
    
    /// Spec or requirement ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec: Option<String>,
    
    /// Owning team
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team: Option<String>,
    
    /// Owner email
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    
    /// Related issue
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issue: Option<String>,
    
    /// Flakiness description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flaky: Option<String>,
    
    /// Expected timeout
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<String>,
    
    /// Required resources
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub requires: Vec<String>,
    
    /// Tags
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub tags: Vec<String>,
    
    /// Custom fields
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    #[serde(default)]
    pub custom: HashMap<String, String>,
}
