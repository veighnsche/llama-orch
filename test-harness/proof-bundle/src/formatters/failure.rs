//! Failure report formatter for debugging

use crate::{TestResult, TestStatus, TestSummary};

/// Generate detailed failure report for debugging
///
/// This produces a debugging-focused report with stack traces, context,
/// and reproduction steps for failed tests.
///
/// # Format
///
/// - Only failed tests
/// - Full stack traces
/// - Test context
/// - Reproduction steps
///
/// # Example
///
/// ```rust
/// use proof_bundle::{TestSummary, formatters};
///
/// let summary = TestSummary::default();
/// let report = formatters::generate_failure_report(&summary);
/// ```
pub fn generate_failure_report(summary: &TestSummary) -> String {
    let mut md = String::new();
    
    // Header
    md.push_str("# Failure Report\n\n");
    
    let failed_tests: Vec<&TestResult> = summary.tests.iter()
        .filter(|t| t.status == TestStatus::Failed)
        .collect();
    
    if failed_tests.is_empty() {
        md.push_str("**Status**: ✅ NO FAILURES\n\n");
        md.push_str("All tests passed successfully. This report is empty.\n");
        return md;
    }
    
    md.push_str(&format!("**Failed Tests**: {} of {} ({:.1}%)\n\n", 
                         failed_tests.len(), 
                         summary.total,
                         (failed_tests.len() as f64 / summary.total as f64) * 100.0));
    
    md.push_str("---\n\n");
    
    // Detailed failure information
    for (i, test) in failed_tests.iter().enumerate() {
        md.push_str(&format!("## Failure {}: {}\n\n", i + 1, test.name));
        
        // Test location (extract from name if available)
        if test.name.contains("::") {
            let parts: Vec<&str> = test.name.split("::").collect();
            if parts.len() >= 2 {
                md.push_str(&format!("**Module**: `{}`\n", parts[..parts.len()-1].join("::")));
                md.push_str(&format!("**Test**: `{}`\n\n", parts[parts.len()-1]));
            }
        }
        
        // Duration
        if test.duration_secs > 0.0 {
            md.push_str(&format!("**Duration**: {:.2}s\n\n", test.duration_secs));
        }
        
        // Error message
        if let Some(ref err) = test.error_message {
            md.push_str("### Error Message\n\n");
            md.push_str("```\n");
            md.push_str(err);
            md.push_str("\n```\n\n");
            
            // Try to extract assertion details
            if err.contains("assertion") || err.contains("expected") {
                md.push_str("### Context\n\n");
                md.push_str("This appears to be an assertion failure. Review the expected vs. actual values above.\n\n");
            }
            
            // Try to extract panic details
            if err.contains("panicked") {
                md.push_str("### Panic Details\n\n");
                md.push_str("This test panicked. Check for:\n");
                md.push_str("- Unwrap on None/Err\n");
                md.push_str("- Index out of bounds\n");
                md.push_str("- Divide by zero\n\n");
            }
        }
        
        // Reproduction
        md.push_str("### Reproduction\n\n");
        md.push_str("```bash\n");
        md.push_str(&format!("cargo test {} -- --exact --nocapture\n", test.name));
        md.push_str("```\n\n");
        
        md.push_str("---\n\n");
    }
    
    // Summary recommendations
    md.push_str("## Recommendations\n\n");
    md.push_str("1. Run each test individually with `--nocapture` to see full output\n");
    md.push_str("2. Check if failures are consistent or intermittent\n");
    md.push_str("3. Review recent changes that may have affected these tests\n");
    md.push_str("4. Add additional logging/assertions if error context is unclear\n");
    
    md
}
