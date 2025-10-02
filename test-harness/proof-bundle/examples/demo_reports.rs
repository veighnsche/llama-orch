//! Demo: Generate beautiful V2 reports with mock data
//!
//! Run with: cargo run --example demo_reports

use proof_bundle::{formatters, metadata, ProofBundle, TestResult, TestStatus, TestSummary, LegacyTestType};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    println!("\n🎯 Generating demo V2 proof bundle with realistic data...\n");

    // Create proof bundle
    let pb = ProofBundle::for_type(LegacyTestType::Unit)?;

    // Create realistic test summary with metadata
    let summary = create_realistic_summary();

    // Write all files
    for result in &summary.tests {
        pb.append_ndjson("test_results", result)?;
    }

    pb.write_json("summary", &summary)?;

    // Generate all 4 V2 reports
    let executive = formatters::generate_executive_summary(&summary);
    pb.write_markdown("executive_summary", &executive)?;

    let developer = formatters::generate_test_report(&summary);
    pb.write_markdown("test_report", &developer)?;

    let failure = formatters::generate_failure_report(&summary);
    pb.write_markdown("failure_report", &failure)?;

    let metadata_report = formatters::generate_metadata_report(&summary);
    pb.write_markdown("metadata_report", &metadata_report)?;

    println!("✅ Demo proof bundle generated!\n");
    println!("📊 Summary:");
    println!("   Total tests: {}", summary.total);
    println!("   Passed: {} ({:.1}%)", summary.passed, summary.pass_rate);
    println!("   Failed: {}", summary.failed);
    println!("   Duration: {:.2}s\n", summary.duration_secs);

    println!("📁 Location: {}\n", pb.root().display());
    println!("📄 Generated reports:");
    println!("   ✓ executive_summary.md     - Open this for management view!");
    println!("   ✓ test_report.md           - Open this for developer view!");
    println!("   ✓ failure_report.md        - Open this for debugging view!");
    println!("   ✓ metadata_report.md       - Open this for compliance view!\n");

    println!("🎨 These are BEAUTIFUL V2 reports with:");
    println!("   • Priority badges (🚨 critical, ⚠️ high)");
    println!("   • Risk assessment");
    println!("   • Metadata grouping");
    println!("   • Deployment recommendations\n");

    Ok(())
}

fn create_realistic_summary() -> TestSummary {
    let mut tests = vec![];

    // Critical test that FAILED
    tests.push(TestResult {
        name: "orchestrator::admission::test_critical_path_validation".to_string(),
        status: TestStatus::Failed,
        duration_secs: 0.523,
        stdout: None,
        stderr: Some("thread 'test_critical_path_validation' panicked at 'assertion failed'".to_string()),
        error_message: Some("Expected admission to succeed but got rejection: InvalidModel".to_string()),
        metadata: Some(metadata::TestMetadata {
            priority: Some("critical".to_string()),
            spec: Some("ORCH-1234".to_string()),
            team: Some("orchestrator".to_string()),
            owner: Some("alice@example.com".to_string()),
            issue: None,
            flaky: None,
            timeout: None,
            requires: vec!["GPU".to_string()],
            tags: vec!["admission".to_string(), "critical-path".to_string()],
            custom: HashMap::new(),
        }),
    });

    // High priority test that passed
    tests.push(TestResult {
        name: "pool_manager::lifecycle::test_engine_restart".to_string(),
        status: TestStatus::Passed,
        duration_secs: 1.234,
        stdout: None,
        stderr: None,
        error_message: None,
        metadata: Some(metadata::TestMetadata {
            priority: Some("high".to_string()),
            spec: Some("ORCH-5678".to_string()),
            team: Some("pool-manager".to_string()),
            owner: Some("bob@example.com".to_string()),
            issue: None,
            flaky: None,
            timeout: Some("30s".to_string()),
            requires: vec![],
            tags: vec!["lifecycle".to_string()],
            custom: HashMap::new(),
        }),
    });

    // Flaky test that passed this time
    tests.push(TestResult {
        name: "streaming::test_token_delivery_timing".to_string(),
        status: TestStatus::Passed,
        duration_secs: 2.156,
        stdout: None,
        stderr: None,
        error_message: None,
        metadata: Some(metadata::TestMetadata {
            priority: Some("medium".to_string()),
            spec: Some("ORCH-9999".to_string()),
            team: Some("streaming".to_string()),
            owner: None,
            issue: Some("ORCH-ISSUE-42".to_string()),
            flaky: Some("Intermittent timing issues on CI".to_string()),
            timeout: None,
            requires: vec![],
            tags: vec!["streaming".to_string(), "timing".to_string()],
            custom: HashMap::new(),
        }),
    });

    // Regular tests
    for i in 1..=15 {
        tests.push(TestResult {
            name: format!("utils::test_helper_function_{}", i),
            status: TestStatus::Passed,
            duration_secs: 0.001 * i as f64,
            stdout: None,
            stderr: None,
            error_message: None,
            metadata: None,
        });
    }

    // Another failed test (low priority)
    tests.push(TestResult {
        name: "logging::test_debug_output_format".to_string(),
        status: TestStatus::Failed,
        duration_secs: 0.045,
        stdout: None,
        stderr: None,
        error_message: Some("Output format mismatch: expected JSON, got plain text".to_string()),
        metadata: Some(metadata::TestMetadata {
            priority: Some("low".to_string()),
            spec: Some("ORCH-LOG-1".to_string()),
            team: Some("observability".to_string()),
            owner: None,
            issue: None,
            flaky: None,
            timeout: None,
            requires: vec![],
            tags: vec!["logging".to_string()],
            custom: HashMap::new(),
        }),
    });

    let total = tests.len();
    let passed = tests.iter().filter(|t| t.status == TestStatus::Passed).count();
    let failed = tests.iter().filter(|t| t.status == TestStatus::Failed).count();
    let duration_secs: f64 = tests.iter().map(|t| t.duration_secs).sum();
    let pass_rate = (passed as f64 / total as f64) * 100.0;

    TestSummary {
        total,
        passed,
        failed,
        ignored: 0,
        duration_secs,
        pass_rate,
        tests,
    }
}
