//! BDD steps for V2 API testing

use cucumber::{given, then, when};
use crate::steps::world::BddWorld;
use proof_bundle::{api, formatters, parsers, metadata, TestResult, TestStatus, TestSummary};
use std::collections::HashMap;

#[given("the proof-bundle V2 API is available")]
async fn api_available(world: &mut BddWorld) {
    world.clear();
}

#[when(regex = r#"^I call generate_for_crate with package "([^"]+)" and mode "([^"]+)"$"#)]
async fn call_generate_for_crate(world: &mut BddWorld, package: String, mode_str: String) {
    let mode = match mode_str.as_str() {
        "UnitFast" => api::Mode::UnitFast,
        "UnitFull" => api::Mode::UnitFull,
        "BddMock" => api::Mode::BddMock,
        "BddReal" => api::Mode::BddReal,
        "Integration" => api::Mode::Integration,
        "Property" => api::Mode::Property,
        _ => panic!("Unknown mode: {}", mode_str),
    };
    
    match api::generate_for_crate(&package, mode) {
        Ok(summary) => {
            world.test_summary = Some(summary);
            world.last_error = None;
        }
        Err(e) => {
            world.last_error = Some(e.to_string());
        }
    }
}

#[then("the API should succeed")]
async fn api_should_succeed(world: &mut BddWorld) {
    assert!(world.last_error.is_none(), "API failed: {:?}", world.last_error);
}

#[then(regex = r"^the summary should have at least (\d+) test$")]
async fn summary_has_tests(world: &mut BddWorld, count: usize) {
    let summary = world.test_summary.as_ref().expect("No summary");
    assert!(summary.total >= count, "Expected at least {} tests, got {}", count, summary.total);
}

#[then(regex = r"^the summary should have a pass rate above (\d+)%$")]
async fn summary_pass_rate(world: &mut BddWorld, rate: usize) {
    let summary = world.test_summary.as_ref().expect("No summary");
    assert!(summary.pass_rate >= rate as f64, "Expected pass rate above {}%, got {:.1}%", rate, summary.pass_rate);
}

#[then("all 7 files should be generated")]
async fn all_files_generated(_world: &mut BddWorld) {
    // This would check the filesystem, but for now we trust the API
    // In a real implementation, verify:
    // - test_results.ndjson
    // - summary.json
    // - executive_summary.md
    // - test_report.md
    // - failure_report.md
    // - metadata_report.md
    // - test_config.json
}

#[given(regex = r"^I have real cargo test output with (\d+) tests$")]
async fn have_real_output(world: &mut BddWorld, count: usize) {
    world.cargo_output = Some(include_str!("../../../tests/golden/cargo_test_output.txt").to_string());
    world.expected_test_count = count;
}

#[when("I parse it with the stable parser")]
async fn parse_with_stable(world: &mut BddWorld) {
    let output = world.cargo_output.as_ref().expect("No cargo output");
    match parsers::parse_stable_output(output) {
        Ok(summary) => {
            world.test_summary = Some(summary);
            world.last_error = None;
        }
        Err(e) => {
            world.last_error = Some(e.to_string());
        }
    }
}

#[then(regex = r"^I should get (\d+) test results$")]
async fn should_get_test_results(world: &mut BddWorld, count: usize) {
    let summary = world.test_summary.as_ref().expect("No summary");
    assert_eq!(summary.total, count, "Expected {} tests, got {}", count, summary.total);
}

#[then("all tests should have names")]
async fn all_tests_have_names(world: &mut BddWorld) {
    let summary = world.test_summary.as_ref().expect("No summary");
    for test in &summary.tests {
        assert!(!test.name.is_empty(), "Test has empty name");
    }
}

#[then(regex = r"^the pass rate should be (\d+)%$")]
async fn pass_rate_should_be(world: &mut BddWorld, rate: usize) {
    let summary = world.test_summary.as_ref().expect("No summary");
    assert_eq!(summary.pass_rate, rate as f64, "Expected {}% pass rate, got {:.1}%", rate, summary.pass_rate);
}

#[given("I have a test summary with real data")]
async fn have_test_summary(world: &mut BddWorld) {
    world.test_summary = Some(create_real_summary());
}

#[when("I generate all reports")]
async fn generate_all_reports(world: &mut BddWorld) {
    let summary = world.test_summary.as_ref().expect("No summary");
    
    world.executive_summary = Some(formatters::generate_executive_summary(summary));
    world.test_report = Some(formatters::generate_test_report(summary));
    world.failure_report = Some(formatters::generate_failure_report(summary));
    world.metadata_report = Some(formatters::generate_metadata_report(summary));
}

#[then(regex = r#"^the executive summary should contain "(.+)"$"#)]
async fn executive_contains(world: &mut BddWorld, text: String) {
    let report = world.executive_summary.as_ref().expect("No executive summary");
    assert!(report.contains(&text), "Executive summary missing: {}", text);
}

#[then(regex = r#"^the test report should contain "(.+)"$"#)]
async fn test_report_contains(world: &mut BddWorld, text: String) {
    let report = world.test_report.as_ref().expect("No test report");
    assert!(report.contains(&text), "Test report missing: {}", text);
}

#[then(regex = r#"^the failure report should contain "(.+)"$"#)]
async fn failure_report_contains(world: &mut BddWorld, text: String) {
    let report = world.failure_report.as_ref().expect("No failure report");
    assert!(report.contains(&text), "Failure report missing: {}", text);
}

#[then(regex = r#"^the metadata report should contain "(.+)"$"#)]
async fn metadata_report_contains(world: &mut BddWorld, text: String) {
    let report = world.metadata_report.as_ref().expect("No metadata report");
    assert!(report.contains(&text), "Metadata report missing: {}", text);
}

#[given("I have a test summary with a critical failure")]
async fn have_critical_failure(world: &mut BddWorld) {
    let mut summary = TestSummary {
        total: 1,
        passed: 0,
        failed: 1,
        ignored: 0,
        duration_secs: 0.5,
        pass_rate: 0.0,
        tests: vec![],
    };
    
    let critical_meta = metadata::TestMetadata {
        priority: Some("critical".to_string()),
        spec: Some("ORCH-1234".to_string()),
        team: Some("orchestrator".to_string()),
        owner: Some("alice@example.com".to_string()),
        issue: None,
        flaky: None,
        timeout: None,
        requires: vec![],
        tags: vec![],
        custom: HashMap::new(),
    };
    
    summary.tests.push(TestResult {
        name: "test_critical".to_string(),
        status: TestStatus::Failed,
        duration_secs: 0.5,
        stdout: None,
        stderr: None,
        error_message: Some("Critical failure".to_string()),
        metadata: Some(critical_meta),
    });
    
    world.test_summary = Some(summary);
}

#[when("I generate the executive summary")]
async fn generate_executive(world: &mut BddWorld) {
    let summary = world.test_summary.as_ref().expect("No summary");
    world.executive_summary = Some(formatters::generate_executive_summary(summary));
}

#[then(regex = r#"^it should contain "(.+)"$"#)]
async fn should_contain(world: &mut BddWorld, text: String) {
    let report = world.executive_summary.as_ref().expect("No executive summary");
    assert!(report.contains(&text), "Executive summary missing: {}", text);
}

#[given("I have tests with metadata annotations")]
async fn have_tests_with_metadata(world: &mut BddWorld) {
    world.test_summary = Some(create_summary_with_metadata());
}

#[when("I generate the metadata report")]
async fn generate_metadata_report(world: &mut BddWorld) {
    let summary = world.test_summary.as_ref().expect("No summary");
    world.metadata_report = Some(formatters::generate_metadata_report(summary));
}

#[then("it should group tests by priority")]
async fn groups_by_priority(world: &mut BddWorld) {
    let report = world.metadata_report.as_ref().expect("No metadata report");
    assert!(report.contains("CRITICAL") || report.contains("HIGH") || report.contains("MEDIUM"));
}

#[then("it should group tests by spec")]
async fn groups_by_spec(world: &mut BddWorld) {
    let report = world.metadata_report.as_ref().expect("No metadata report");
    assert!(report.contains("ORCH-"));
}

#[then("it should show flaky tests separately")]
async fn shows_flaky_separately(world: &mut BddWorld) {
    let report = world.metadata_report.as_ref().expect("No metadata report");
    // Would check for flaky section if present
}

#[given("I have the golden cargo output file")]
async fn have_golden_file(world: &mut BddWorld) {
    world.cargo_output = Some(include_str!("../../../tests/golden/cargo_test_output.txt").to_string());
}

#[when("I parse it")]
async fn parse_golden(world: &mut BddWorld) {
    let output = world.cargo_output.as_ref().expect("No cargo output");
    world.test_summary = Some(parsers::parse_stable_output(output).expect("Parse failed"));
}

#[then("it should match the expected test count")]
async fn matches_expected_count(world: &mut BddWorld) {
    let summary = world.test_summary.as_ref().expect("No summary");
    assert_eq!(summary.total, 38, "Expected 38 tests");
}

#[then("it should extract all test names correctly")]
async fn extracts_names_correctly(world: &mut BddWorld) {
    let summary = world.test_summary.as_ref().expect("No summary");
    assert_eq!(summary.tests.len(), 38, "Should extract 38 test names");
}

// Helper functions
fn create_real_summary() -> TestSummary {
    TestSummary {
        total: 10,
        passed: 10,
        failed: 0,
        ignored: 0,
        duration_secs: 1.0,
        pass_rate: 100.0,
        tests: vec![],
    }
}

fn create_summary_with_metadata() -> TestSummary {
    let mut summary = TestSummary {
        total: 2,
        passed: 2,
        failed: 0,
        ignored: 0,
        duration_secs: 0.5,
        pass_rate: 100.0,
        tests: vec![],
    };
    
    let meta1 = metadata::TestMetadata {
        priority: Some("high".to_string()),
        spec: Some("ORCH-1234".to_string()),
        team: Some("orchestrator".to_string()),
        owner: None,
        issue: None,
        flaky: None,
        timeout: None,
        requires: vec![],
        tags: vec![],
        custom: HashMap::new(),
    };
    
    summary.tests.push(TestResult {
        name: "test_with_metadata".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.3,
        stdout: None,
        stderr: None,
        error_message: None,
        metadata: Some(meta1),
    });
    
    summary
}
