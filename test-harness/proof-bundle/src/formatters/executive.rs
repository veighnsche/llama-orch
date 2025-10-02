//! Executive summary formatter
//!
//! Generates management-friendly, non-technical reports.
//! Enhanced with validation to prevent garbage output.

use crate::core::{TestSummary, TestStatus};

/// Generate executive summary for management
///
/// # Validation
///
/// Returns error if summary has 0 tests (prevents garbage output).
pub fn generate_executive_summary(summary: &TestSummary) -> Result<String, String> {
    // Validate input
    if summary.total == 0 {
        return Err("Cannot generate executive summary: no tests in summary".to_string());
    }
    
    let mut md = String::new();
    
    // Header
    md.push_str("# Test Results Summary\n\n");
    md.push_str(&format!("**Date**: {}\n", chrono::Utc::now().format("%Y-%m-%d")));
    
    // Status line with emoji
    let status_emoji = if summary.failed == 0 { "✅" } else { "⚠️" };
    md.push_str(&format!("**Status**: {} {:.1}% PASS RATE\n", status_emoji, summary.pass_rate));
    
    // Confidence level
    let confidence = if summary.pass_rate >= 98.0 { "HIGH" }
                     else if summary.pass_rate >= 95.0 { "MEDIUM" }
                     else { "LOW" };
    md.push_str(&format!("**Confidence**: {}\n\n", confidence));
    
    // Quick Facts
    md.push_str("## Quick Facts\n\n");
    md.push_str(&format!("- **{} tests** executed\n", summary.total));
    md.push_str(&format!("- **{} passed** ({:.1}%)\n", summary.passed, summary.pass_rate));
    
    // Safe division for failed percentage
    let failed_pct = if summary.total > 0 {
        (summary.failed as f64 / summary.total as f64) * 100.0
    } else {
        0.0
    };
    md.push_str(&format!("- **{} failed** ({:.1}%)\n", summary.failed, failed_pct));
    md.push_str(&format!("- **{} skipped**\n", summary.ignored));
    md.push_str(&format!("- **Duration**: {:.1} seconds\n\n", summary.duration_secs));
    
    // Check for critical test failures
    let critical_failures: Vec<_> = summary.tests.iter()
        .filter(|t| t.status == TestStatus::Failed)
        .filter(|t| t.metadata.as_ref().map_or(false, |m| m.is_critical()))
        .collect();
    
    // CRITICAL ALERT (if any critical tests failed)
    if !critical_failures.is_empty() {
        md.push_str("## 🚨 CRITICAL ALERT\n\n");
        md.push_str(&format!("**{} CRITICAL TEST FAILURE(S)**\n\n", critical_failures.len()));
        
        for test in &critical_failures {
            let metadata = test.metadata.as_ref().unwrap();
            md.push_str(&format!("- **{}**", test.name));
            if let Some(spec) = &metadata.spec {
                md.push_str(&format!(" ({})", spec));
            }
            md.push_str("\n");
            if let Some(team) = &metadata.team {
                md.push_str(&format!("  - Team: {}\n", team));
            }
            if let Some(owner) = &metadata.owner {
                md.push_str(&format!("  - Owner: {}\n", owner));
            }
        }
        md.push_str("\n**Action Required**: Critical tests must pass before deployment.\n\n");
    }
    
    // Risk Assessment
    md.push_str("## Risk Assessment\n\n");
    let risk = if !critical_failures.is_empty() { "CRITICAL" }
               else if summary.failed == 0 { "LOW" }
               else if summary.failed <= 2 { "MEDIUM" }
               else { "HIGH" };
    let risk_emoji = match risk {
        "CRITICAL" => "🚨",
        "LOW" => "✅",
        "MEDIUM" => "⚠️",
        _ => "❌",
    };
    
    md.push_str(&format!("{} **{} RISK**", risk_emoji, risk));
    
    if summary.failed == 0 {
        md.push_str(" — All tests passing\n\n");
    } else {
        md.push_str(&format!(" — {} test(s) failing\n\n", summary.failed));
    }
    
    // Recommendation
    md.push_str("## Recommendation\n\n");
    if !critical_failures.is_empty() {
        md.push_str("**❌ NOT APPROVED** — Critical test failures require immediate resolution\n\n");
    } else if summary.pass_rate >= 98.0 {
        md.push_str("**✅ APPROVED** — High confidence for deployment\n\n");
    } else if summary.pass_rate >= 95.0 {
        md.push_str("**⚠️ CONDITIONAL APPROVAL** — Review failures before deployment\n\n");
    } else {
        md.push_str("**❌ NOT APPROVED** — Significant test failures require resolution\n\n");
    }
    
    Ok(md)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{TestResult, TestStatus};
    
    #[test]
    fn test_rejects_empty_summary() {
        let summary = TestSummary::default();
        let result = generate_executive_summary(&summary);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no tests"));
    }
    
    #[test]
    fn test_generates_for_valid_summary() {
        let tests = vec![
            TestResult::new("test1".to_string(), TestStatus::Passed),
            TestResult::new("test2".to_string(), TestStatus::Passed),
        ];
        let summary = TestSummary::new(tests);
        
        let result = generate_executive_summary(&summary);
        assert!(result.is_ok());
        
        let report = result.unwrap();
        assert!(report.contains("Test Results Summary"));
        assert!(report.contains("100.0% PASS RATE"));
        assert!(report.contains("APPROVED"));
    }
    
    #[test]
    fn test_no_division_by_zero() {
        let tests = vec![
            TestResult::new("test1".to_string(), TestStatus::Passed),
        ];
        let summary = TestSummary::new(tests);
        
        let result = generate_executive_summary(&summary);
        assert!(result.is_ok());
        
        let report = result.unwrap();
        // Should not contain NaN
        assert!(!report.contains("NaN"));
    }
}
