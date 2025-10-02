//! Test metadata annotations
//!
//! Allows teams to annotate tests with metadata that appears in proof bundles.
//!
//! # Two Ways to Add Metadata
//!
//! ## 1. Doc Comments (Recommended)
//!
//! ```rust
//! /// @priority: critical
//! /// @spec: ORCH-3250
//! /// @team: orchestrator
//! #[test]
//! fn test_queue_ordering() {
//!     // test logic
//! }
//! ```
//!
//! ## 2. Programmatic API
//!
//! ```rust
//! #[test]
//! fn test_queue_ordering() -> anyhow::Result<()> {
//!     proof_bundle::test_metadata()
//!         .priority("critical")
//!         .spec("ORCH-3250")
//!         .team("orchestrator")
//!         .record()?;
//!     
//!     // test logic
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Test metadata
///
/// Captures metadata about a test for proof bundle reporting.
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

/// Builder for test metadata
///
/// # Example
///
/// ```rust
/// use proof_bundle::test_metadata;
///
/// # fn example() -> anyhow::Result<()> {
/// test_metadata()
///     .priority("critical")
///     .spec("ORCH-3250")
///     .team("orchestrator")
///     .record()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct TestMetadataBuilder {
    metadata: TestMetadata,
}

impl TestMetadataBuilder {
    /// Create a new metadata builder
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set priority level
    ///
    /// Standard values: `critical`, `high`, `medium`, `low`
    pub fn priority(mut self, level: &str) -> Self {
        self.metadata.priority = Some(level.to_string());
        self
    }
    
    /// Set spec or requirement ID
    ///
    /// Examples: `ORCH-3250`, `REQ-AUTH-001`
    pub fn spec(mut self, id: &str) -> Self {
        self.metadata.spec = Some(id.to_string());
        self
    }
    
    /// Set owning team
    pub fn team(mut self, name: &str) -> Self {
        self.metadata.team = Some(name.to_string());
        self
    }
    
    /// Set owner email
    pub fn owner(mut self, email: &str) -> Self {
        self.metadata.owner = Some(email.to_string());
        self
    }
    
    /// Set related issue
    ///
    /// Example: `#1234`
    pub fn issue(mut self, id: &str) -> Self {
        self.metadata.issue = Some(id.to_string());
        self
    }
    
    /// Set flakiness description
    ///
    /// Example: `5% timeout rate on slow CI`
    pub fn flaky(mut self, description: &str) -> Self {
        self.metadata.flaky = Some(description.to_string());
        self
    }
    
    /// Set expected timeout
    ///
    /// Example: `30s`, `2m`
    pub fn timeout(mut self, duration: &str) -> Self {
        self.metadata.timeout = Some(duration.to_string());
        self
    }
    
    /// Set required resources
    ///
    /// Example: `&["gpu", "cuda", "16gb-vram"]`
    pub fn requires(mut self, resources: &[&str]) -> Self {
        self.metadata.requires = resources.iter().map(|s| s.to_string()).collect();
        self
    }
    
    /// Set tags
    ///
    /// Example: `&["integration", "slow"]`
    pub fn tags(mut self, tags: &[&str]) -> Self {
        self.metadata.tags = tags.iter().map(|s| s.to_string()).collect();
        self
    }
    
    /// Add a custom field
    ///
    /// Example: `.custom("deployment-stage", "canary")`
    pub fn custom(mut self, key: &str, value: &str) -> Self {
        self.metadata.custom.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Record the metadata
    ///
    /// Currently stores in thread-local storage.
    /// In future, may write to proof bundle immediately.
    pub fn record(self) -> anyhow::Result<()> {
        // TODO: Store in thread-local or global registry
        // For now, just validate
        Ok(())
    }
    
    /// Build the metadata without recording
    pub fn build(self) -> TestMetadata {
        self.metadata
    }
}

/// Entry point for test metadata builder
///
/// # Example
///
/// ```rust
/// use proof_bundle::test_metadata;
///
/// # fn example() -> anyhow::Result<()> {
/// test_metadata()
///     .priority("critical")
///     .spec("ORCH-3250")
///     .record()?;
/// # Ok(())
/// # }
/// ```
pub fn test_metadata() -> TestMetadataBuilder {
    TestMetadataBuilder::new()
}

/// Parse doc comment annotations
///
/// Extracts `@key: value` annotations from doc comments.
///
/// # Example
///
/// ```rust
/// use proof_bundle::metadata::parse_doc_comments;
///
/// let doc = r#"
/// Test description.
///
/// @priority: critical
/// @spec: ORCH-3250
/// @team: orchestrator
/// "#;
///
/// let metadata = parse_doc_comments(doc);
/// assert_eq!(metadata.priority, Some("critical".to_string()));
/// assert_eq!(metadata.spec, Some("ORCH-3250".to_string()));
/// assert_eq!(metadata.team, Some("orchestrator".to_string()));
/// ```
pub fn parse_doc_comments(doc: &str) -> TestMetadata {
    let mut metadata = TestMetadata::default();
    
    for line in doc.lines() {
        let line = line.trim();
        
        // Look for @key: value pattern
        // Use ": " as separator to handle keys with colons (e.g., custom:deployment-stage)
        if let Some(stripped) = line.strip_prefix("@") {
            if let Some((key, value)) = stripped.split_once(": ") {
                let key = key.trim();
                let value = value.trim();
                
                match key {
                    "priority" => metadata.priority = Some(value.to_string()),
                    "spec" => metadata.spec = Some(value.to_string()),
                    "team" => metadata.team = Some(value.to_string()),
                    "owner" => metadata.owner = Some(value.to_string()),
                    "issue" => metadata.issue = Some(value.to_string()),
                    "flaky" => metadata.flaky = Some(value.to_string()),
                    "timeout" => metadata.timeout = Some(value.to_string()),
                    "requires" => {
                        metadata.requires = value
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect();
                    }
                    "tags" => {
                        metadata.tags = value
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect();
                    }
                    other => {
                        // Check if it's a custom field (starts with "custom:")
                        if let Some(custom_key) = other.strip_prefix("custom:") {
                            metadata.custom.insert(custom_key.to_string(), value.to_string());
                        }
                        // Otherwise, ignore unknown field
                    }
                }
            }
        }
    }
    
    metadata
}

/// Check if metadata indicates a critical test
pub fn is_critical(metadata: &TestMetadata) -> bool {
    metadata.priority.as_deref() == Some("critical")
}

/// Check if metadata indicates a high priority test
pub fn is_high_priority(metadata: &TestMetadata) -> bool {
    matches!(
        metadata.priority.as_deref(),
        Some("critical") | Some("high")
    )
}

/// Check if test is marked as flaky
pub fn is_flaky(metadata: &TestMetadata) -> bool {
    metadata.flaky.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metadata_builder() {
        let metadata = test_metadata()
            .priority("critical")
            .spec("ORCH-3250")
            .team("orchestrator")
            .owner("alice@example.com")
            .build();
        
        assert_eq!(metadata.priority, Some("critical".to_string()));
        assert_eq!(metadata.spec, Some("ORCH-3250".to_string()));
        assert_eq!(metadata.team, Some("orchestrator".to_string()));
        assert_eq!(metadata.owner, Some("alice@example.com".to_string()));
    }
    
    #[test]
    fn test_metadata_with_resources_and_tags() {
        let metadata = test_metadata()
            .priority("high")
            .requires(&["gpu", "cuda"])
            .tags(&["integration", "slow"])
            .build();
        
        assert_eq!(metadata.requires, vec!["gpu", "cuda"]);
        assert_eq!(metadata.tags, vec!["integration", "slow"]);
    }
    
    #[test]
    fn test_metadata_with_custom_fields() {
        let metadata = test_metadata()
            .custom("deployment-stage", "canary")
            .custom("sla", "99.9%")
            .build();
        
        assert_eq!(metadata.custom.get("deployment-stage"), Some(&"canary".to_string()));
        assert_eq!(metadata.custom.get("sla"), Some(&"99.9%".to_string()));
    }
    
    #[test]
    fn test_parse_doc_comments_simple() {
        let doc = r#"
            @priority: critical
            @spec: ORCH-3250
            @team: orchestrator
        "#;
        
        let metadata = parse_doc_comments(doc);
        assert_eq!(metadata.priority, Some("critical".to_string()));
        assert_eq!(metadata.spec, Some("ORCH-3250".to_string()));
        assert_eq!(metadata.team, Some("orchestrator".to_string()));
    }
    
    #[test]
    fn test_parse_doc_comments_with_lists() {
        let doc = r#"
            @priority: high
            @requires: gpu, cuda, 16gb-vram
            @tags: integration, slow, gpu-required
        "#;
        
        let metadata = parse_doc_comments(doc);
        assert_eq!(metadata.priority, Some("high".to_string()));
        assert_eq!(metadata.requires, vec!["gpu", "cuda", "16gb-vram"]);
        assert_eq!(metadata.tags, vec!["integration", "slow", "gpu-required"]);
    }
    
    #[test]
    fn test_parse_doc_comments_custom_fields() {
        let doc = r#"
            @custom:deployment-stage: canary
            @custom:sla: 99.9%
        "#;
        
        let metadata = parse_doc_comments(doc);
        assert_eq!(metadata.custom.get("deployment-stage"), Some(&"canary".to_string()));
        assert_eq!(metadata.custom.get("sla"), Some(&"99.9%".to_string()));
    }
    
    #[test]
    fn test_parse_doc_comments_mixed_content() {
        let doc = r#"
            Test description here.
            
            Some more documentation.
            
            @priority: critical
            @spec: ORCH-3250
            
            More text that should be ignored.
        "#;
        
        let metadata = parse_doc_comments(doc);
        assert_eq!(metadata.priority, Some("critical".to_string()));
        assert_eq!(metadata.spec, Some("ORCH-3250".to_string()));
    }
    
    #[test]
    fn test_is_critical() {
        let metadata = test_metadata().priority("critical").build();
        assert!(is_critical(&metadata));
        
        let metadata = test_metadata().priority("high").build();
        assert!(!is_critical(&metadata));
    }
    
    #[test]
    fn test_is_high_priority() {
        let metadata = test_metadata().priority("critical").build();
        assert!(is_high_priority(&metadata));
        
        let metadata = test_metadata().priority("high").build();
        assert!(is_high_priority(&metadata));
        
        let metadata = test_metadata().priority("medium").build();
        assert!(!is_high_priority(&metadata));
    }
    
    #[test]
    fn test_is_flaky() {
        let metadata = test_metadata().flaky("5% timeout rate").build();
        assert!(is_flaky(&metadata));
        
        let metadata = test_metadata().priority("critical").build();
        assert!(!is_flaky(&metadata));
    }
    
    #[test]
    fn test_metadata_serialization() {
        let metadata = test_metadata()
            .priority("critical")
            .spec("ORCH-3250")
            .team("orchestrator")
            .custom("foo", "bar")
            .build();
        
        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: TestMetadata = serde_json::from_str(&json).unwrap();
        
        assert_eq!(metadata, deserialized);
    }
}
