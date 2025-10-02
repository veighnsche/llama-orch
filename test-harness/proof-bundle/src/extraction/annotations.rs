//! Annotation parsing from doc comments
//!
//! Parses @key: value syntax from doc comments on test functions.

use crate::core::TestMetadata;

/// Parse annotations from a doc comment
///
/// Recognizes the following formats:
/// - `@priority: critical`
/// - `@spec: ORCH-1234`
/// - `@team: orchestrator`
/// - `@owner: alice@example.com`
/// - `@tags: integration, slow`
/// - `@requires: GPU, CUDA`
/// - `@custom:key: value`
pub fn parse_annotations(doc: &str, mut metadata: TestMetadata) -> TestMetadata {
    for line in doc.lines() {
        let line = line.trim();
        
        // Skip lines that don't start with @
        if !line.starts_with('@') {
            continue;
        }
        
        // Parse @key: value
        if let Some((key, value)) = parse_annotation_line(line) {
            match key.as_str() {
                "priority" => metadata.priority = Some(value.to_string()),
                "spec" => metadata.spec = Some(value.to_string()),
                "team" => metadata.team = Some(value.to_string()),
                "owner" => metadata.owner = Some(value.to_string()),
                "issue" => metadata.issue = Some(value.to_string()),
                "flaky" => metadata.flaky = Some(value.to_string()),
                "timeout" => metadata.timeout = Some(value.to_string()),
                "tags" => {
                    // Parse comma-separated tags
                    metadata.tags.extend(
                        value.split(',').map(|s| s.trim().to_string())
                    );
                }
                "requires" => {
                    // Parse comma-separated requirements
                    metadata.requires.extend(
                        value.split(',').map(|s| s.trim().to_string())
                    );
                }
                _ if key.starts_with("custom:") => {
                    // Custom field: @custom:key: value
                    let custom_key = key.strip_prefix("custom:").unwrap();
                    metadata.custom.insert(custom_key.to_string(), value.to_string());
                }
                _ => {
                    // Unknown annotation, ignore
                }
            }
        }
    }
    
    metadata
}

/// Parse a single annotation line
///
/// Format: `@key: value` or `@custom:key: value`
fn parse_annotation_line(line: &str) -> Option<(String, String)> {
    let line = line.strip_prefix('@')?;
    
    // Handle @custom:key: value format (has two colons)
    if line.starts_with("custom:") {
        // Split into "custom:key" and "value"
        let mut parts = line.splitn(2, ':');
        let first = parts.next()?.trim(); // "custom"
        let rest = parts.next()?.trim();   // "key: value"
        
        // Now split "key: value"
        let mut rest_parts = rest.splitn(2, ':');
        let key_part = rest_parts.next()?.trim();
        let value = rest_parts.next()?.trim();
        
        // Reconstruct as "custom:key"
        return Some((format!("custom:{}", key_part), value.to_string()));
    }
    
    // Normal format: @key: value
    let mut parts = line.splitn(2, ':');
    let key = parts.next()?.trim();
    let value = parts.next()?.trim();
    
    Some((key.to_string(), value.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_priority() {
        let doc = "@priority: critical";
        let metadata = parse_annotations(doc, TestMetadata::default());
        assert_eq!(metadata.priority, Some("critical".to_string()));
    }
    
    #[test]
    fn test_parse_spec() {
        let doc = "@spec: ORCH-1234";
        let metadata = parse_annotations(doc, TestMetadata::default());
        assert_eq!(metadata.spec, Some("ORCH-1234".to_string()));
    }
    
    #[test]
    fn test_parse_tags() {
        let doc = "@tags: integration, slow, gpu-required";
        let metadata = parse_annotations(doc, TestMetadata::default());
        assert_eq!(metadata.tags, vec!["integration", "slow", "gpu-required"]);
    }
    
    #[test]
    fn test_parse_requires() {
        let doc = "@requires: GPU, CUDA, 16GB VRAM";
        let metadata = parse_annotations(doc, TestMetadata::default());
        assert_eq!(metadata.requires, vec!["GPU", "CUDA", "16GB VRAM"]);
    }
    
    #[test]
    fn test_parse_custom() {
        let doc = "@custom:environment: staging";
        let metadata = parse_annotations(doc, TestMetadata::default());
        assert_eq!(metadata.custom.get("environment"), Some(&"staging".to_string()));
    }
    
    #[test]
    fn test_parse_multiple() {
        let doc = r#"
            @priority: critical
            @spec: ORCH-1234
            @team: orchestrator
            @owner: alice@example.com
        "#;
        let metadata = parse_annotations(doc, TestMetadata::default());
        assert_eq!(metadata.priority, Some("critical".to_string()));
        assert_eq!(metadata.spec, Some("ORCH-1234".to_string()));
        assert_eq!(metadata.team, Some("orchestrator".to_string()));
        assert_eq!(metadata.owner, Some("alice@example.com".to_string()));
    }
    
    #[test]
    fn test_ignores_non_annotations() {
        let doc = r#"
            This is a regular comment
            @priority: critical
            More regular text
        "#;
        let metadata = parse_annotations(doc, TestMetadata::default());
        assert_eq!(metadata.priority, Some("critical".to_string()));
    }
}
