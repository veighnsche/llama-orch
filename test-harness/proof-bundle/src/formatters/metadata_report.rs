//! Metadata report formatter for compliance tracking

use crate::{TestResult, TestStatus, TestSummary};
use crate::metadata::is_flaky;
use std::collections::HashMap;

/// Generate metadata-focused report
///
/// This produces a report organized by test metadata (priority, spec, team)
/// for management and compliance tracking.
///
/// # Format
///
/// - Tests grouped by priority
/// - Tests grouped by spec/requirement
/// - Tests grouped by team
/// - Known flaky tests
///
/// # Example
///
/// ```rust
/// use proof_bundle::{TestSummary, formatters};
///
/// let summary = TestSummary::default();
/// let report = formatters::generate_metadata_report(&summary);
/// ```
pub fn generate_metadata_report(summary: &TestSummary) -> String {
    let mut md = String::new();
    
    // Header
    md.push_str("# Test Metadata Report\n\n");
    md.push_str(&format!("**Generated**: {}\n\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
    
    // Count tests with metadata
    let tests_with_metadata = summary.tests.iter()
        .filter(|t| t.metadata.is_some())
        .count();
    
    if tests_with_metadata == 0 {
        md.push_str("**Status**: No tests have metadata annotations\n\n");
        md.push_str("To add metadata to tests, use doc comment annotations:\n\n");
        md.push_str("```rust\n");
        md.push_str("/// @priority: critical\n");
        md.push_str("/// @spec: ORCH-3250\n");
        md.push_str("/// @team: orchestrator\n");
        md.push_str("#[test]\n");
        md.push_str("fn test_something() { }\n");
        md.push_str("```\n");
        return md;
    }
    
    md.push_str(&format!("**Tests with metadata**: {} of {} ({:.1}%)\n\n", 
                         tests_with_metadata, 
                         summary.total,
                         (tests_with_metadata as f64 / summary.total as f64) * 100.0));
    
    md.push_str("---\n\n");
    
    // Group by priority
    md.push_str("## By Priority\n\n");
    
    let priorities = vec!["critical", "high", "medium", "low"];
    for priority in &priorities {
        let tests: Vec<_> = summary.tests.iter()
            .filter(|t| t.metadata.as_ref()
                .and_then(|m| m.priority.as_ref())
                .map_or(false, |p| p.to_lowercase() == *priority))
            .collect();
        
        if !tests.is_empty() {
            let passed = tests.iter().filter(|t| t.status == TestStatus::Passed).count();
            let failed = tests.iter().filter(|t| t.status == TestStatus::Failed).count();
            
            let emoji = match *priority {
                "critical" => "🚨",
                "high" => "⚠️",
                "medium" => "ℹ️",
                _ => "📝",
            };
            
            md.push_str(&format!("### {} {} ({} tests)\n\n", emoji, priority.to_uppercase(), tests.len()));
            
            if failed > 0 {
                md.push_str(&format!("**❌ {} FAILED**, {} passed\n\n", failed, passed));
                
                for test in tests.iter().filter(|t| t.status == TestStatus::Failed) {
                    md.push_str(&format!("- ❌ **{}**", test.name));
                    if let Some(ref metadata) = test.metadata {
                        if let Some(ref spec) = metadata.spec {
                            md.push_str(&format!(" ({})", spec));
                        }
                    }
                    md.push_str("\n");
                }
                md.push_str("\n");
            } else {
                md.push_str(&format!("**✅ All {} tests passing**\n\n", tests.len()));
            }
        }
    }
    
    // Group by spec
    md.push_str("## By Spec/Requirement\n\n");
    
    let mut specs: HashMap<String, Vec<&TestResult>> = HashMap::new();
    for test in &summary.tests {
        if let Some(ref metadata) = test.metadata {
            if let Some(ref spec) = metadata.spec {
                specs.entry(spec.clone()).or_insert_with(Vec::new).push(test);
            }
        }
    }
    
    if specs.is_empty() {
        md.push_str("*No tests have spec annotations*\n\n");
    } else {
        let mut spec_list: Vec<_> = specs.iter().collect();
        spec_list.sort_by_key(|(spec, _)| *spec);
        
        for (spec, tests) in spec_list {
            let passed = tests.iter().filter(|t| t.status == TestStatus::Passed).count();
            let failed = tests.iter().filter(|t| t.status == TestStatus::Failed).count();
            
            let status = if failed > 0 { "❌" } else { "✅" };
            
            md.push_str(&format!("### {} {}\n\n", status, spec));
            md.push_str(&format!("**{} tests**: {} passed", tests.len(), passed));
            if failed > 0 {
                md.push_str(&format!(", **{} failed**", failed));
            }
            md.push_str("\n\n");
            
            for test in tests {
                let status_icon = match test.status {
                    TestStatus::Passed => "✅",
                    TestStatus::Failed => "❌",
                    TestStatus::Ignored => "⏭️",
                    _ => "❓",
                };
                md.push_str(&format!("- {} {}\n", status_icon, test.name));
            }
            md.push_str("\n");
        }
    }
    
    // Group by team
    md.push_str("## By Team\n\n");
    
    let mut teams: HashMap<String, Vec<&TestResult>> = HashMap::new();
    for test in &summary.tests {
        if let Some(ref metadata) = test.metadata {
            if let Some(ref team) = metadata.team {
                teams.entry(team.clone()).or_insert_with(Vec::new).push(test);
            }
        }
    }
    
    if teams.is_empty() {
        md.push_str("*No tests have team annotations*\n\n");
    } else {
        let mut team_list: Vec<_> = teams.iter().collect();
        team_list.sort_by_key(|(team, _)| *team);
        
        for (team, tests) in team_list {
            let passed = tests.iter().filter(|t| t.status == TestStatus::Passed).count();
            let failed = tests.iter().filter(|t| t.status == TestStatus::Failed).count();
            
            let status = if failed > 0 { "❌" } else { "✅" };
            
            md.push_str(&format!("### {} {}\n\n", status, team));
            md.push_str(&format!("**{} tests**: {} passed", tests.len(), passed));
            if failed > 0 {
                md.push_str(&format!(", **{} failed**", failed));
            }
            md.push_str("\n\n");
        }
    }
    
    // Flaky tests
    let flaky_tests: Vec<_> = summary.tests.iter()
        .filter(|t| t.metadata.as_ref().map_or(false, |m| is_flaky(m)))
        .collect();
    
    if !flaky_tests.is_empty() {
        md.push_str("## Known Flaky Tests\n\n");
        
        for test in flaky_tests {
            let status_icon = match test.status {
                TestStatus::Passed => "✅",
                TestStatus::Failed => "❌",
                _ => "❓",
            };
            
            md.push_str(&format!("- {} **{}**", status_icon, test.name));
            if let Some(ref metadata) = test.metadata {
                if let Some(ref flaky) = metadata.flaky {
                    md.push_str(&format!(" — {}", flaky));
                }
                if let Some(ref issue) = metadata.issue {
                    md.push_str(&format!(" ({})", issue));
                }
            }
            md.push_str("\n");
        }
        md.push_str("\n");
    }
    
    md
}
