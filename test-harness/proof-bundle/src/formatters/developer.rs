//! Developer report formatter
//!
//! Generates technical reports for developers with test details.

use crate::core::{TestSummary, TestStatus};

/// Generate developer report
///
/// # Validation
///
/// Returns error if summary has 0 tests.
pub fn generate_developer_report(summary: &TestSummary) -> Result<String, String> {
    // Validate input
    if summary.total == 0 {
        return Err("Cannot generate developer report: no tests in summary".to_string());
    }
    
    let mut md = String::new();
    
    // Header
    md.push_str("# Test Report\n\n");
    
    // Summary
    md.push_str("## Summary\n\n");
    md.push_str(&format!("- Total: {} tests\n", summary.total));
    md.push_str(&format!("- Passed: {} ({:.1}%)\n", summary.passed, summary.pass_rate));
    md.push_str(&format!("- Failed: {} ({:.1}%)\n", summary.failed, 
                         if summary.total > 0 { (summary.failed as f64 / summary.total as f64) * 100.0 } else { 0.0 }));
    md.push_str(&format!("- Ignored: {}\n", summary.ignored));
    md.push_str(&format!("- Duration: {:.2}s\n\n", summary.duration_secs));
    
    // Failed tests (if any)
    let failed_tests: Vec<_> = summary.tests.iter()
        .filter(|t| t.status == TestStatus::Failed)
        .collect();
    
    if !failed_tests.is_empty() {
        md.push_str("## Failed Tests\n\n");
        for test in failed_tests {
            md.push_str(&format!("### ❌ {}\n\n", test.name));
            
            if let Some(ref error) = test.error_message {
                md.push_str("**Error**:\n```\n");
                md.push_str(error);
                md.push_str("\n```\n\n");
            }
            
            if let Some(ref metadata) = test.metadata {
                if let Some(ref spec) = metadata.spec {
                    md.push_str(&format!("- Spec: {}\n", spec));
                }
                if let Some(ref team) = metadata.team {
                    md.push_str(&format!("- Team: {}\n", team));
                }
                if let Some(ref owner) = metadata.owner {
                    md.push_str(&format!("- Owner: {}\n", owner));
                }
                md.push_str("\n");
            }
        }
    }
    
    // Test breakdown by status
    md.push_str("## Test Breakdown\n\n");
    
    if summary.passed > 0 {
        md.push_str(&format!("### ✅ Passed ({})\n\n", summary.passed));
        for test in summary.tests.iter().filter(|t| t.status == TestStatus::Passed).take(10) {
            md.push_str(&format!("- {}\n", test.name));
        }
        if summary.passed > 10 {
            md.push_str(&format!("\n... and {} more\n", summary.passed - 10));
        }
        md.push_str("\n");
    }
    
    if summary.ignored > 0 {
        md.push_str(&format!("### ⏭️ Ignored ({})\n\n", summary.ignored));
        for test in summary.tests.iter().filter(|t| t.status == TestStatus::Ignored) {
            md.push_str(&format!("- {}\n", test.name));
        }
        md.push_str("\n");
    }
    
    Ok(md)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{TestResult, TestStatus};
    
    /// @priority: critical
    /// @spec: PB-V3-VALIDATION
    /// @team: proof-bundle
    /// @tags: unit, formatter, developer, zero-tests-bug-fix
    #[test]
    fn test_rejects_empty_summary() {
        let summary = TestSummary::default();
        let result = generate_developer_report(&summary);
        assert!(result.is_err());
    }
    
    /// @priority: critical
    /// @spec: PB-V3-FORMATTER
    /// @team: proof-bundle
    /// @tags: unit, formatter, developer, technical-report
    #[test]
    fn test_generates_for_valid_summary() {
        let tests = vec![
            TestResult::new("test1".to_string(), TestStatus::Passed),
            TestResult::new("test2".to_string(), TestStatus::Failed)
                .with_error("assertion failed".to_string()),
        ];
        let summary = TestSummary::new(tests);
        
        let result = generate_developer_report(&summary);
        assert!(result.is_ok());
        
        let report = result.unwrap();
        assert!(report.contains("Test Report"));
        assert!(report.contains("Failed Tests"));
        assert!(report.contains("assertion failed"));
    }
}
