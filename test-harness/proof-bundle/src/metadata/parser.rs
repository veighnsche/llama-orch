//! Doc comment parser for metadata annotations

use super::types::TestMetadata;

/// Parse doc comment annotations
///
/// Extracts `@key: value` annotations from doc comments.
///
/// # Supported Annotations
///
/// - `@priority: <level>` - Test priority (critical, high, medium, low)
/// - `@spec: <id>` - Spec or requirement ID
/// - `@team: <name>` - Owning team
/// - `@owner: <email>` - Owner email
/// - `@issue: <id>` - Related issue
/// - `@flaky: <description>` - Flakiness description
/// - `@timeout: <duration>` - Expected timeout
/// - `@requires: <resource>` - Required resource (can be repeated or comma-separated)
/// - `@tags: <tag>` - Tag (can be repeated or comma-separated)
/// - `@custom:<key>: <value>` - Custom field
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
///
/// # Format
///
/// Annotations must follow the format `@key: value` where:
/// - The line starts with `@`
/// - The key and value are separated by `: ` (colon + space)
/// - Values are trimmed
///
/// For lists (requires, tags), values can be:
/// - Comma-separated: `@requires: GPU, CUDA, 16GB VRAM`
/// - Repeated: Multiple `@requires:` lines
///
/// For custom fields, use `@custom:<key>: <value>` format.
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
                        // Support both comma-separated and repeated annotations
                        let resources: Vec<String> = value
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect();
                        metadata.requires.extend(resources);
                    }
                    "tags" => {
                        // Support both comma-separated and repeated annotations
                        let tags: Vec<String> = value
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect();
                        metadata.tags.extend(tags);
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_simple_annotations() {
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
    fn test_parse_comma_separated_lists() {
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
    fn test_parse_repeated_annotations() {
        let doc = r#"
            @requires: GPU
            @requires: CUDA
            @requires: 16GB VRAM
            @tags: integration
            @tags: slow
        "#;
        
        let metadata = parse_doc_comments(doc);
        assert_eq!(metadata.requires, vec!["GPU", "CUDA", "16GB VRAM"]);
        assert_eq!(metadata.tags, vec!["integration", "slow"]);
    }
    
    #[test]
    fn test_parse_custom_fields() {
        let doc = r#"
            @custom:deployment-stage: canary
            @custom:sla: 99.9%
        "#;
        
        let metadata = parse_doc_comments(doc);
        assert_eq!(metadata.custom.get("deployment-stage"), Some(&"canary".to_string()));
        assert_eq!(metadata.custom.get("sla"), Some(&"99.9%".to_string()));
    }
    
    #[test]
    fn test_parse_mixed_content() {
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
    fn test_parse_ignores_unknown_fields() {
        let doc = r#"
            @priority: critical
            @unknown-field: some value
            @spec: ORCH-3250
        "#;
        
        let metadata = parse_doc_comments(doc);
        assert_eq!(metadata.priority, Some("critical".to_string()));
        assert_eq!(metadata.spec, Some("ORCH-3250".to_string()));
        // unknown-field should be ignored
    }
}
