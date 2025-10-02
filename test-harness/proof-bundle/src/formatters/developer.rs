//! Developer report formatter with technical details

use crate::{TestResult, TestStatus, TestSummary};
use crate::metadata::{is_critical, is_high_priority, is_flaky};
use super::helpers::categorize_tests;

/// Generate detailed developer report
///
/// This produces a technical report with test breakdowns, failure details,
/// and performance metrics suitable for developers and technical reviewers.
///
/// # Format
///
/// - Summary statistics
/// - Test breakdown by type
/// - Failed tests with metadata and errors
/// - Performance metrics
/// - Full technical details
///
/// # Example
///
/// ```rust
/// use proof_bundle::{TestSummary, formatters};
///
/// let summary = TestSummary::default();
/// let report = formatters::generate_test_report(&summary);
/// assert!(report.contains("Test Report"));
/// ```
pub fn generate_test_report(summary: &TestSummary) -> String {
    let mut md = String::new();
    
    // Header
    md.push_str("# Test Report\n\n");
    
    // Summary
    md.push_str("## Summary\n\n");
    md.push_str(&format!("- Total: {} tests\n", summary.total));
    md.push_str(&format!("- Passed: {} ({:.1}%)\n", summary.passed, summary.pass_rate));
    md.push_str(&format!("- Failed: {} ({:.1}%)\n", summary.failed, 
                         (summary.failed as f64 / summary.total as f64) * 100.0));
    md.push_str(&format!("- Ignored: {}\n", summary.ignored));
    md.push_str(&format!("- Duration: {:.2}s\n\n", summary.duration_secs));
    
    // Test Breakdown by Type
    md.push_str("## Test Breakdown\n\n");
    
    let breakdown = categorize_tests(&summary.tests);
    
    for (category, tests) in &breakdown {
        let passed = tests.iter().filter(|t| t.status == TestStatus::Passed).count();
        let failed = tests.iter().filter(|t| t.status == TestStatus::Failed).count();
        let ignored = tests.iter().filter(|t| t.status == TestStatus::Ignored).count();
        let duration: f64 = tests.iter().map(|t| t.duration_secs).sum();
        
        md.push_str(&format!("### {} ({} tests)\n", category, tests.len()));
        
        if passed > 0 {
            md.push_str(&format!("- ✅ {} passed\n", passed));
        }
        if failed > 0 {
            md.push_str(&format!("- ❌ {} failed\n", failed));
        }
        if ignored > 0 {
            md.push_str(&format!("- ⏭️ {} ignored\n", ignored));
        }
        
        md.push_str(&format!("- Duration: {:.2}s\n\n", duration));
    }
    
    // Failed Tests
    if summary.failed > 0 {
        md.push_str("## Failed Tests\n\n");
        
        let failed_tests: Vec<&TestResult> = summary.tests.iter()
            .filter(|t| t.status == TestStatus::Failed)
            .collect();
        
        for test in failed_tests {
            // Add priority badge
            let badge = if let Some(ref metadata) = test.metadata {
                if is_critical(metadata) { " 🚨" }
                else if is_high_priority(metadata) { " ⚠️" }
                else { "" }
            } else { "" };
            
            md.push_str(&format!("### {}{}\n\n", test.name, badge));
            
            // Show metadata
            if let Some(ref metadata) = test.metadata {
                if let Some(ref priority) = metadata.priority {
                    md.push_str(&format!("**Priority**: {}\n", priority));
                }
                if let Some(ref spec) = metadata.spec {
                    md.push_str(&format!("**Spec**: {}\n", spec));
                }
                if let Some(ref team) = metadata.team {
                    md.push_str(&format!("**Team**: {}\n", team));
                }
                if let Some(ref owner) = metadata.owner {
                    md.push_str(&format!("**Owner**: {}\n", owner));
                }
                if let Some(ref issue) = metadata.issue {
                    md.push_str(&format!("**Issue**: {}\n", issue));
                }
                if is_flaky(metadata) {
                    if let Some(ref flaky) = metadata.flaky {
                        md.push_str(&format!("⚠️ **Known Flaky**: {}\n", flaky));
                    }
                }
                if !metadata.tags.is_empty() {
                    md.push_str(&format!("**Tags**: {}\n", metadata.tags.join(", ")));
                }
                md.push_str("\n");
            }
            
            if test.duration_secs > 0.0 {
                md.push_str(&format!("**Duration**: {:.2}s\n\n", test.duration_secs));
            }
            
            if let Some(ref err) = test.error_message {
                md.push_str("**Error**:\n```\n");
                md.push_str(err);
                md.push_str("\n```\n\n");
            } else {
                md.push_str("**Error**: No error message captured\n\n");
            }
        }
    }
    
    // Performance
    md.push_str("## Performance\n\n");
    
    let mut timed_tests: Vec<&TestResult> = summary.tests.iter()
        .filter(|t| t.duration_secs > 0.0)
        .collect();
    
    timed_tests.sort_by(|a, b| {
        b.duration_secs.partial_cmp(&a.duration_secs).unwrap()
    });
    
    md.push_str("**Slowest tests**:\n");
    for (i, test) in timed_tests.iter().take(10).enumerate() {
        md.push_str(&format!("{}. {} — {:.2}s\n", i + 1, test.name, test.duration_secs));
    }
    
    if timed_tests.len() > 10 {
        md.push_str(&format!("\n*(Showing top 10 of {} timed tests)*\n", timed_tests.len()));
    }
    
    md
}
