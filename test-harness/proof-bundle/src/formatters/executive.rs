//! Executive summary formatter for management audit

use crate::{TestResult, TestStatus, TestSummary};
use crate::metadata::{is_critical, is_high_priority};
use super::helpers::simplify_error;

/// Generate executive summary for management audit
///
/// This produces a non-technical, business-focused report suitable for
/// stakeholders who need to understand test results without technical details.
///
/// # Format
///
/// - Pass rate and confidence level
/// - Risk assessment (LOW/MEDIUM/HIGH/CRITICAL)
/// - Critical test failure alerts
/// - Failed test impact analysis
/// - Deployment recommendation
/// - Non-technical language
///
/// # Example
///
/// ```rust
/// use proof_bundle::{TestSummary, formatters};
///
/// let summary = TestSummary::default();
/// let report = formatters::generate_executive_summary(&summary);
/// assert!(report.contains("Test Results Summary"));
/// ```
pub fn generate_executive_summary(summary: &TestSummary) -> String {
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
    md.push_str(&format!("- **{} failed** ({:.1}%)\n", summary.failed, 
                         (summary.failed as f64 / summary.total as f64) * 100.0));
    md.push_str(&format!("- **{} skipped**\n", summary.ignored));
    md.push_str(&format!("- **Duration**: {:.1} seconds\n\n", summary.duration_secs));
    
    // Check for critical test failures
    let critical_failures: Vec<_> = summary.tests.iter()
        .filter(|t| t.status == TestStatus::Failed)
        .filter(|t| t.metadata.as_ref().map_or(false, |m| is_critical(m)))
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
    } else if !critical_failures.is_empty() {
        md.push_str(&format!(" — {} critical failure(s), {} total failures\n\n", 
                             critical_failures.len(), summary.failed));
    } else {
        md.push_str(&format!(" — {} test failure(s)\n\n", summary.failed));
    }
    
    // Failed Tests (if any)
    if summary.failed > 0 {
        md.push_str("## Failed Tests\n\n");
        
        let failed_tests: Vec<&TestResult> = summary.tests.iter()
            .filter(|t| t.status == TestStatus::Failed)
            .collect();
        
        for (i, test) in failed_tests.iter().enumerate() {
            // Show priority badge if available
            let priority_badge = if let Some(ref metadata) = test.metadata {
                if is_critical(metadata) {
                    " 🚨 **CRITICAL**"
                } else if is_high_priority(metadata) {
                    " ⚠️ **HIGH**"
                } else {
                    ""
                }
            } else {
                ""
            };
            
            md.push_str(&format!("{}. **{}**{}\n", i + 1, test.name, priority_badge));
            
            // Show metadata if available
            if let Some(ref metadata) = test.metadata {
                if let Some(ref spec) = metadata.spec {
                    md.push_str(&format!("   - Spec: {}\n", spec));
                }
                if let Some(ref team) = metadata.team {
                    md.push_str(&format!("   - Team: {}\n", team));
                }
                if let Some(ref owner) = metadata.owner {
                    md.push_str(&format!("   - Owner: {}\n", owner));
                }
            }
            
            if let Some(ref err) = test.error_message {
                // Simplify error for management
                let simplified = simplify_error(err);
                md.push_str(&format!("   - Issue: {}\n", simplified));
            }
            
            let impact = if let Some(ref metadata) = test.metadata {
                if is_critical(metadata) { "**CRITICAL** — Blocks deployment" }
                else if is_high_priority(metadata) { "**HIGH** — Review required" }
                else { "Review required" }
            } else {
                "Review required"
            };
            md.push_str(&format!("   - Impact: {}\n", impact));
            md.push_str("   - Action: Engineering review in progress\n\n");
        }
    }
    
    // Recommendation
    md.push_str("## Recommendation\n\n");
    
    if !critical_failures.is_empty() {
        md.push_str("**❌ NOT APPROVED** — Critical test failures must be resolved before deployment\n");
        md.push_str(&format!("\n{} critical test(s) failed. These tests are marked as deployment blockers.\n", critical_failures.len()));
    } else if summary.failed == 0 && summary.pass_rate >= 98.0 {
        md.push_str("**✅ APPROVED FOR DEPLOYMENT** — All tests passing, high confidence\n");
    } else if summary.failed <= 2 && summary.pass_rate >= 95.0 {
        md.push_str("**⚠️ APPROVED FOR STAGING** — Minor issues tracked, review recommended\n");
    } else {
        md.push_str("**❌ NOT APPROVED** — Significant test failures require resolution\n");
    }
    
    md
}
