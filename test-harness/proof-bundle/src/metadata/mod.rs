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
//!
//! # Module Organization
//!
//! - **types**: Core `TestMetadata` struct
//! - **builder**: Fluent builder API (`TestMetadataBuilder`)
//! - **parser**: Doc comment annotation parser
//! - **helpers**: Query functions (`is_critical`, `is_high_priority`, etc.)

mod builder;
mod helpers;
mod parser;
mod types;

// Re-export public API
pub use builder::{test_metadata, TestMetadataBuilder};
pub use helpers::{has_tag, is_critical, is_flaky, is_high_priority, priority_level, requires_resource};
pub use parser::parse_doc_comments;
pub use types::TestMetadata;

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
