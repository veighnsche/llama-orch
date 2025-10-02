//! Metadata report formatter
//!
//! Generates compliance-focused reports grouped by metadata.

use crate::core::{TestSummary, TestStatus};
use std::collections::HashMap;

/// Generate metadata report
///
/// Groups tests by priority, spec, team, etc.
pub fn generate_metadata_report(summary: &TestSummary) -> Result<String, String> {
    // Validate input
    if summary.total == 0 {
        return Err("Cannot generate metadata report: no tests in summary".to_string());
    }
    
    let mut md = String::new();
    
    // Header
    md.push_str("# Metadata Report\n\n");
    
    // Group by priority
    let mut by_priority: HashMap<String, Vec<&crate::core::TestResult>> = HashMap::new();
    for test in &summary.tests {
        if let Some(ref metadata) = test.metadata {
            if let Some(ref priority) = metadata.priority {
                by_priority.entry(priority.clone()).or_default().push(test);
            }
        }
    }
    
    if !by_priority.is_empty() {
        md.push_str("## By Priority\n\n");
        
        // Critical first
        if let Some(tests) = by_priority.get("critical") {
            md.push_str(&format!("### 🚨 Critical ({})\n\n", tests.len()));
            for test in tests {
                format_test_line(&mut md, test);
            }
            md.push_str("\n");
        }
        
        // High
        if let Some(tests) = by_priority.get("high") {
            md.push_str(&format!("### ⚠️ High ({})\n\n", tests.len()));
            for test in tests {
                format_test_line(&mut md, test);
            }
            md.push_str("\n");
        }
        
        // Medium
        if let Some(tests) = by_priority.get("medium") {
            md.push_str(&format!("### 📋 Medium ({})\n\n", tests.len()));
            for test in tests {
                format_test_line(&mut md, test);
            }
            md.push_str("\n");
        }
        
        // Low
        if let Some(tests) = by_priority.get("low") {
            md.push_str(&format!("### 📝 Low ({})\n\n", tests.len()));
            for test in tests {
                format_test_line(&mut md, test);
            }
            md.push_str("\n");
        }
    }
    
    // Group by spec
    let mut by_spec: HashMap<String, Vec<&crate::core::TestResult>> = HashMap::new();
    for test in &summary.tests {
        if let Some(ref metadata) = test.metadata {
            if let Some(ref spec) = metadata.spec {
                by_spec.entry(spec.clone()).or_default().push(test);
            }
        }
    }
    
    if !by_spec.is_empty() {
        md.push_str("## By Specification\n\n");
        let mut specs: Vec<_> = by_spec.keys().collect();
        specs.sort();
        
        for spec in specs {
            let tests = &by_spec[spec];
            md.push_str(&format!("### {} ({})\n\n", spec, tests.len()));
            for test in tests {
                format_test_line(&mut md, test);
            }
            md.push_str("\n");
        }
    }
    
    // Flaky tests
    let flaky: Vec<_> = summary.tests.iter()
        .filter(|t| t.metadata.as_ref().map_or(false, |m| m.is_flaky()))
        .collect();
    
    if !flaky.is_empty() {
        md.push_str("## ⚠️ Known Flaky Tests\n\n");
        for test in flaky {
            md.push_str(&format!("- {}", test.name));
            if let Some(ref metadata) = test.metadata {
                if let Some(ref flaky_desc) = metadata.flaky {
                    md.push_str(&format!(" — {}", flaky_desc));
                }
            }
            md.push_str("\n");
        }
        md.push_str("\n");
    }
    
    Ok(md)
}

fn format_test_line(md: &mut String, test: &crate::core::TestResult) {
    let status_emoji = match test.status {
        TestStatus::Passed => "✅",
        TestStatus::Failed => "❌",
        TestStatus::Ignored => "⏭️",
    };
    
    md.push_str(&format!("- {} **{}**", status_emoji, test.name));
    
    if let Some(ref metadata) = test.metadata {
        if let Some(ref spec) = metadata.spec {
            md.push_str(&format!(" `{}`", spec));
        }
        
        md.push_str("\n");
        
        // Show scenario (WHAT is being tested)
        if let Some(ref scenario) = metadata.scenario {
            md.push_str(&format!("  - **Scenario**: {}\n", scenario));
        }
        
        // Show threat (security/risk addressed)
        if let Some(ref threat) = metadata.threat {
            md.push_str(&format!("  - **Threat**: {}\n", threat));
        }
        
        // Show failure mode (what failure is prevented)
        if let Some(ref failure_mode) = metadata.failure_mode {
            md.push_str(&format!("  - **Prevents**: {}\n", failure_mode));
        }
        
        // Show edge case (boundary conditions)
        if let Some(ref edge_case) = metadata.edge_case {
            md.push_str(&format!("  - **Edge Case**: {}\n", edge_case));
        }
        
        // Show team and tags on same line
        let mut details = Vec::new();
        if let Some(ref team) = metadata.team {
            details.push(format!("@{}", team));
        }
        if !metadata.tags.is_empty() {
            details.push(metadata.tags.join(", "));
        }
        if !details.is_empty() {
            md.push_str(&format!("  - *{}*\n", details.join(" · ")));
        }
    } else {
        md.push_str("\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{TestResult, TestStatus, TestMetadata};
    
    /// @priority: critical
    /// @spec: PB-V3-VALIDATION
    /// @team: proof-bundle
    /// @tags: unit, formatter, metadata, zero-tests-bug-fix
    #[test]
    fn test_rejects_empty_summary() {
        let summary = TestSummary::default();
        let result = generate_metadata_report(&summary);
        assert!(result.is_err());
    }
    
    /// @priority: critical
    /// @spec: PB-V3-FORMATTER
    /// @team: proof-bundle
    /// @tags: unit, formatter, metadata, grouping
    #[test]
    fn test_groups_by_priority() {
        let mut meta1 = TestMetadata::default();
        meta1.priority = Some("critical".to_string());
        
        let mut meta2 = TestMetadata::default();
        meta2.priority = Some("high".to_string());
        
        let tests = vec![
            TestResult::new("test1".to_string(), TestStatus::Passed).with_metadata(meta1),
            TestResult::new("test2".to_string(), TestStatus::Passed).with_metadata(meta2),
        ];
        let summary = TestSummary::new(tests);
        
        let result = generate_metadata_report(&summary);
        assert!(result.is_ok());
        
        let report = result.unwrap();
        assert!(report.contains("Critical"));
        assert!(report.contains("High"));
    }
}
