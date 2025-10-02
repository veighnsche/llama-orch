//! Failure report formatter
//!
//! Generates focused reports on test failures for debugging.

use crate::core::{TestSummary, TestStatus};

/// Generate failure report
///
/// # Validation
///
/// Returns error if summary has 0 tests.
/// Returns empty report if no failures (not an error).
pub fn generate_failure_report(summary: &TestSummary) -> Result<String, String> {
    // Validate input
    if summary.total == 0 {
        return Err("Cannot generate failure report: no tests in summary".to_string());
    }
    
    let mut md = String::new();
    
    // Header
    md.push_str("# Failure Report\n\n");
    
    let failed_tests: Vec<_> = summary.tests.iter()
        .filter(|t| t.status == TestStatus::Failed)
        .collect();
    
    if failed_tests.is_empty() {
        md.push_str("✅ **No failures** — All tests passed!\n\n");
        return Ok(md);
    }
    
    md.push_str(&format!("**{} test(s) failed**\n\n", failed_tests.len()));
    
    // Group by priority if metadata available
    let critical: Vec<_> = failed_tests.iter()
        .filter(|t| t.metadata.as_ref().map_or(false, |m| m.is_critical()))
        .collect();
    
    let high: Vec<_> = failed_tests.iter()
        .filter(|t| t.metadata.as_ref().map_or(false, |m| 
            m.is_high_priority() && !m.is_critical()))
        .collect();
    
    let other: Vec<_> = failed_tests.iter()
        .filter(|t| !t.metadata.as_ref().map_or(false, |m| m.is_high_priority()))
        .collect();
    
    // Critical failures first
    if !critical.is_empty() {
        md.push_str("## 🚨 Critical Failures\n\n");
        for test in critical {
            format_failure(&mut md, test);
        }
    }
    
    // High priority failures
    if !high.is_empty() {
        md.push_str("## ⚠️ High Priority Failures\n\n");
        for test in high {
            format_failure(&mut md, test);
        }
    }
    
    // Other failures
    if !other.is_empty() {
        md.push_str("## Other Failures\n\n");
        for test in other {
            format_failure(&mut md, test);
        }
    }
    
    Ok(md)
}

fn format_failure(md: &mut String, test: &crate::core::TestResult) {
    md.push_str(&format!("### {}\n\n", test.name));
    
    if let Some(ref metadata) = test.metadata {
        if let Some(ref spec) = metadata.spec {
            md.push_str(&format!("- **Spec**: {}\n", spec));
        }
        if let Some(ref team) = metadata.team {
            md.push_str(&format!("- **Team**: {}\n", team));
        }
        if let Some(ref owner) = metadata.owner {
            md.push_str(&format!("- **Owner**: {}\n", owner));
        }
        md.push_str("\n");
    }
    
    if let Some(ref error) = test.error_message {
        md.push_str("**Error**:\n```\n");
        md.push_str(error);
        md.push_str("\n```\n\n");
    }
    
    md.push_str("---\n\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{TestResult, TestStatus};
    
    /// @priority: critical
    /// @spec: PB-V3-VALIDATION
    /// @team: proof-bundle
    /// @tags: unit, formatter, failure, zero-tests-bug-fix
    #[test]
    fn test_rejects_empty_summary() {
        let summary = TestSummary::default();
        let result = generate_failure_report(&summary);
        assert!(result.is_err());
    }
    
    /// @priority: high
    /// @spec: PB-V3-FORMATTER
    /// @team: proof-bundle
    /// @tags: unit, formatter, failure, edge-case
    #[test]
    fn test_handles_no_failures() {
        let tests = vec![
            TestResult::new("test1".to_string(), TestStatus::Passed),
        ];
        let summary = TestSummary::new(tests);
        
        let result = generate_failure_report(&summary);
        assert!(result.is_ok());
        
        let report = result.unwrap();
        assert!(report.contains("No failures"));
    }
}
