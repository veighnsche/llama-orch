//! Helper functions for metadata queries

use super::types::TestMetadata;

/// Check if metadata indicates a critical test
///
/// Returns `true` if priority is exactly "critical".
///
/// # Example
///
/// ```rust
/// use proof_bundle::{test_metadata, metadata::is_critical};
///
/// let metadata = test_metadata().priority("critical").build();
/// assert!(is_critical(&metadata));
///
/// let metadata = test_metadata().priority("high").build();
/// assert!(!is_critical(&metadata));
/// ```
pub fn is_critical(metadata: &TestMetadata) -> bool {
    metadata.priority.as_deref() == Some("critical")
}

/// Check if metadata indicates a high priority test
///
/// Returns `true` if priority is "critical" or "high".
///
/// # Example
///
/// ```rust
/// use proof_bundle::{test_metadata, metadata::is_high_priority};
///
/// let metadata = test_metadata().priority("critical").build();
/// assert!(is_high_priority(&metadata));
///
/// let metadata = test_metadata().priority("high").build();
/// assert!(is_high_priority(&metadata));
///
/// let metadata = test_metadata().priority("medium").build();
/// assert!(!is_high_priority(&metadata));
/// ```
pub fn is_high_priority(metadata: &TestMetadata) -> bool {
    matches!(
        metadata.priority.as_deref(),
        Some("critical") | Some("high")
    )
}

/// Check if test is marked as flaky
///
/// Returns `true` if the flaky field is set (regardless of value).
///
/// # Example
///
/// ```rust
/// use proof_bundle::{test_metadata, metadata::is_flaky};
///
/// let metadata = test_metadata().flaky("5% timeout rate").build();
/// assert!(is_flaky(&metadata));
///
/// let metadata = test_metadata().priority("critical").build();
/// assert!(!is_flaky(&metadata));
/// ```
pub fn is_flaky(metadata: &TestMetadata) -> bool {
    metadata.flaky.is_some()
}

/// Get priority level as a number for sorting
///
/// Returns:
/// - 4 for "critical"
/// - 3 for "high"
/// - 2 for "medium"
/// - 1 for "low"
/// - 0 for unset
///
/// # Example
///
/// ```rust
/// use proof_bundle::{test_metadata, metadata::priority_level};
///
/// let critical = test_metadata().priority("critical").build();
/// let high = test_metadata().priority("high").build();
/// let medium = test_metadata().priority("medium").build();
///
/// assert!(priority_level(&critical) > priority_level(&high));
/// assert!(priority_level(&high) > priority_level(&medium));
/// ```
pub fn priority_level(metadata: &TestMetadata) -> u8 {
    match metadata.priority.as_deref() {
        Some("critical") => 4,
        Some("high") => 3,
        Some("medium") => 2,
        Some("low") => 1,
        _ => 0,
    }
}

/// Check if test requires a specific resource
///
/// # Example
///
/// ```rust
/// use proof_bundle::{test_metadata, metadata::requires_resource};
///
/// let metadata = test_metadata()
///     .requires(&["GPU", "CUDA"])
///     .build();
///
/// assert!(requires_resource(&metadata, "GPU"));
/// assert!(requires_resource(&metadata, "CUDA"));
/// assert!(!requires_resource(&metadata, "TPU"));
/// ```
pub fn requires_resource(metadata: &TestMetadata, resource: &str) -> bool {
    metadata.requires.iter().any(|r| r == resource)
}

/// Check if test has a specific tag
///
/// # Example
///
/// ```rust
/// use proof_bundle::{test_metadata, metadata::has_tag};
///
/// let metadata = test_metadata()
///     .tags(&["integration", "slow"])
///     .build();
///
/// assert!(has_tag(&metadata, "integration"));
/// assert!(has_tag(&metadata, "slow"));
/// assert!(!has_tag(&metadata, "unit"));
/// ```
pub fn has_tag(metadata: &TestMetadata, tag: &str) -> bool {
    metadata.tags.iter().any(|t| t == tag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_metadata;
    
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
    fn test_priority_level() {
        assert_eq!(priority_level(&test_metadata().priority("critical").build()), 4);
        assert_eq!(priority_level(&test_metadata().priority("high").build()), 3);
        assert_eq!(priority_level(&test_metadata().priority("medium").build()), 2);
        assert_eq!(priority_level(&test_metadata().priority("low").build()), 1);
        assert_eq!(priority_level(&test_metadata().build()), 0);
    }
    
    #[test]
    fn test_requires_resource() {
        let metadata = test_metadata().requires(&["GPU", "CUDA"]).build();
        assert!(requires_resource(&metadata, "GPU"));
        assert!(requires_resource(&metadata, "CUDA"));
        assert!(!requires_resource(&metadata, "TPU"));
    }
    
    #[test]
    fn test_has_tag() {
        let metadata = test_metadata().tags(&["integration", "slow"]).build();
        assert!(has_tag(&metadata, "integration"));
        assert!(has_tag(&metadata, "slow"));
        assert!(!has_tag(&metadata, "unit"));
    }
}
