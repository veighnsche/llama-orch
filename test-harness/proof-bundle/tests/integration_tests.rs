//! Integration tests for proof-bundle V2
//!
//! These tests verify that all modules work together correctly.

use proof_bundle::{
    api, formatters, metadata, parsers, templates, LegacyTestType, ProofBundle, TestResult,
    TestStatus, TestSummary,
};

#[test]
fn test_end_to_end_workflow() {
    // Create a mock test summary
    let summary = create_test_summary_with_metadata();

    // Test all formatters work
    let executive = formatters::generate_executive_summary(&summary);
    assert!(executive.contains("Test Results Summary"));
    assert!(executive.contains("CRITICAL ALERT")); // Has critical failure

    let developer = formatters::generate_test_report(&summary);
    assert!(developer.contains("Test Report"));
    assert!(developer.contains("🚨")); // Critical badge

    let failure = formatters::generate_failure_report(&summary);
    assert!(failure.contains("Failure Report"));

    let metadata_report = formatters::generate_metadata_report(&summary);
    assert!(metadata_report.contains("Test Metadata Report"));
    assert!(metadata_report.contains("CRITICAL"));
}

#[test]
fn test_parser_integration() {
    // Test JSON parser
    let json_output = r#"{"type":"test","event":"ok","name":"test_pass","exec_time":0.001}
{"type":"test","event":"failed","name":"test_fail","exec_time":0.5,"stderr":"error"}"#;

    let summary = parsers::parse_json_output(json_output).unwrap();
    assert_eq!(summary.total, 2);
    assert_eq!(summary.passed, 1);
    assert_eq!(summary.failed, 1);

    // Test stable parser
    let stable_output = r#"running 2 tests
test test_pass ... ok
test test_fail ... FAILED

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"#;

    let summary = parsers::parse_stable_output(stable_output).unwrap();
    assert_eq!(summary.total, 2);
    assert_eq!(summary.passed, 1);
    assert_eq!(summary.failed, 1);
}

#[test]
fn test_template_integration() {
    // Test all templates exist and have correct properties
    let unit_fast = templates::unit_test_fast();
    assert_eq!(unit_fast.name, "unit-fast");
    assert!(unit_fast.skip_long_tests);
    assert!(unit_fast.mock_external_services);

    let bdd_real = templates::bdd_test_real();
    assert_eq!(bdd_real.name, "bdd-real");
    assert!(bdd_real.requires_gpu);
    assert!(!bdd_real.mock_external_services);

    // Test find_by_name
    let found = templates::find_by_name("unit-fast").unwrap();
    assert_eq!(found.name, "unit-fast");

    assert!(templates::find_by_name("nonexistent").is_none());
}

#[test]
fn test_metadata_integration() {
    // Test metadata builder
    let meta = metadata::test_metadata()
        .priority("critical")
        .spec("ORCH-1234")
        .team("orchestrator")
        .owner("alice@example.com")
        .record();

    assert_eq!(meta.priority, Some("critical".to_string()));
    assert_eq!(meta.spec, Some("ORCH-1234".to_string()));

    // Test helper functions
    assert!(metadata::is_critical(&meta));
    assert!(metadata::is_high_priority(&meta));

    // Test doc comment parsing
    let doc = r#"/// @priority: high
/// @spec: ORCH-5678
/// @team: pool-manager"#;

    let parsed = metadata::parse_doc_comments(doc);
    assert_eq!(parsed.priority, Some("high".to_string()));
    assert_eq!(parsed.spec, Some("ORCH-5678".to_string()));
    assert_eq!(parsed.team, Some("pool-manager".to_string()));
}

#[test]
fn test_api_mode_integration() {
    // Test Mode enum
    let mode = api::Mode::UnitFast;
    let template = mode.template();
    assert_eq!(template.name, "unit-fast");

    let mode = api::Mode::BddReal;
    let template = mode.template();
    assert_eq!(template.name, "bdd-real");
    assert!(template.requires_gpu);
}

#[test]
fn test_proof_bundle_creation() {
    // Test proof bundle can be created for all types
    let pb = ProofBundle::for_type(LegacyTestType::Unit).unwrap();
    assert!(pb.root().exists());

    let pb = ProofBundle::for_type(LegacyTestType::Bdd).unwrap();
    assert!(pb.root().exists());

    let pb = ProofBundle::for_type(LegacyTestType::Integration).unwrap();
    assert!(pb.root().exists());
}

#[test]
fn test_critical_failure_detection() {
    let summary = create_test_summary_with_metadata();

    // Executive summary should detect critical failures
    let executive = formatters::generate_executive_summary(&summary);
    assert!(executive.contains("CRITICAL ALERT"));
    assert!(executive.contains("NOT APPROVED"));
    assert!(executive.contains("CRITICAL RISK"));

    // Metadata report should group by priority
    let metadata_report = formatters::generate_metadata_report(&summary);
    assert!(metadata_report.contains("🚨 CRITICAL"));
}

#[test]
fn test_flaky_test_detection() {
    let mut summary = TestSummary {
        total: 1,
        passed: 1,
        failed: 0,
        ignored: 0,
        duration_secs: 0.1,
        pass_rate: 100.0,
        tests: vec![],
    };

    let flaky_meta = metadata::TestMetadata {
        priority: None,
        spec: None,
        team: None,
        owner: None,
        issue: Some("ORCH-999".to_string()),
        flaky: Some("Intermittent network timeout".to_string()),
        timeout: None,
        requires: vec![],
        tags: vec![],
        custom: std::collections::HashMap::new(),
    };

    summary.tests.push(TestResult {
        name: "test_flaky".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.1,
        stdout: None,
        stderr: None,
        error_message: None,
        metadata: Some(flaky_meta),
    });

    let metadata_report = formatters::generate_metadata_report(&summary);
    assert!(metadata_report.contains("Known Flaky Tests"));
    assert!(metadata_report.contains("Intermittent network timeout"));
}

#[test]
fn test_multiple_priorities() {
    let mut summary = TestSummary {
        total: 3,
        passed: 2,
        failed: 1,
        ignored: 0,
        duration_secs: 1.0,
        pass_rate: 66.7,
        tests: vec![],
    };

    // Critical test (failed)
    let critical_meta = metadata::test_metadata()
        .priority("critical")
        .spec("ORCH-100")
        .record();

    summary.tests.push(TestResult {
        name: "test_critical".to_string(),
        status: TestStatus::Failed,
        duration_secs: 0.5,
        stdout: None,
        stderr: None,
        error_message: Some("Critical failure".to_string()),
        metadata: Some(critical_meta),
    });

    // High priority test (passed)
    let high_meta = metadata::test_metadata()
        .priority("high")
        .spec("ORCH-200")
        .record();

    summary.tests.push(TestResult {
        name: "test_high".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.3,
        stdout: None,
        stderr: None,
        error_message: None,
        metadata: Some(high_meta),
    });

    // Low priority test (passed)
    let low_meta = metadata::TestMetadata {
        priority: Some("low".to_string()),
        spec: Some("ORCH-300".to_string()),
        team: None,
        owner: None,
        issue: None,
        flaky: None,
        timeout: None,
        requires: vec![],
        tags: vec![],
        custom: std::collections::HashMap::new(),
    };

    summary.tests.push(TestResult {
        name: "test_low".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.2,
        stdout: None,
        stderr: None,
        error_message: None,
        metadata: Some(low_meta),
    });

    // Test executive summary
    let executive = formatters::generate_executive_summary(&summary);
    assert!(executive.contains("CRITICAL ALERT"));
    assert!(executive.contains("CRITICAL RISK"));

    // Test metadata report groups correctly
    let metadata_report = formatters::generate_metadata_report(&summary);
    assert!(metadata_report.contains("🚨 CRITICAL"));
    assert!(metadata_report.contains("⚠️ HIGH"));
    assert!(metadata_report.contains("📝 LOW"));
}

#[test]
fn test_spec_grouping() {
    let mut summary = TestSummary {
        total: 3,
        passed: 3,
        failed: 0,
        ignored: 0,
        duration_secs: 0.6,
        pass_rate: 100.0,
        tests: vec![],
    };

    // Multiple tests for same spec
    for i in 1..=2 {
        let meta = metadata::TestMetadata {
            priority: None,
            spec: Some("ORCH-1234".to_string()),
            team: Some("orchestrator".to_string()),
            owner: None,
            issue: None,
            flaky: None,
            timeout: None,
            requires: vec![],
            tags: vec![],
            custom: std::collections::HashMap::new(),
        };

        summary.tests.push(TestResult {
            name: format!("test_orch_1234_{}", i),
            status: TestStatus::Passed,
            duration_secs: 0.2,
            stdout: None,
            stderr: None,
            error_message: None,
            metadata: Some(meta),
        });
    }

    // Test for different spec
    let meta = metadata::TestMetadata {
        priority: None,
        spec: Some("ORCH-5678".to_string()),
        team: Some("pool-manager".to_string()),
        owner: None,
        issue: None,
        flaky: None,
        timeout: None,
        requires: vec![],
        tags: vec![],
        custom: std::collections::HashMap::new(),
    };

    summary.tests.push(TestResult {
        name: "test_orch_5678".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.2,
        stdout: None,
        stderr: None,
        error_message: None,
        metadata: Some(meta),
    });

    let metadata_report = formatters::generate_metadata_report(&summary);
    assert!(metadata_report.contains("ORCH-1234"));
    assert!(metadata_report.contains("ORCH-5678"));
    assert!(metadata_report.contains("2 tests")); // ORCH-1234 has 2 tests
}

#[test]
fn test_team_grouping() {
    let mut summary = TestSummary {
        total: 2,
        passed: 2,
        failed: 0,
        ignored: 0,
        duration_secs: 0.4,
        pass_rate: 100.0,
        tests: vec![],
    };

    let meta1 = metadata::TestMetadata {
        priority: None,
        spec: None,
        team: Some("orchestrator".to_string()),
        owner: None,
        issue: None,
        flaky: None,
        timeout: None,
        requires: vec![],
        tags: vec![],
        custom: std::collections::HashMap::new(),
    };

    summary.tests.push(TestResult {
        name: "test_orchestrator".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.2,
        stdout: None,
        stderr: None,
        error_message: None,
        metadata: Some(meta1),
    });

    let meta2 = metadata::TestMetadata {
        priority: None,
        spec: None,
        team: Some("pool-manager".to_string()),
        owner: None,
        issue: None,
        flaky: None,
        timeout: None,
        requires: vec![],
        tags: vec![],
        custom: std::collections::HashMap::new(),
    };

    summary.tests.push(TestResult {
        name: "test_pool_manager".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.2,
        stdout: None,
        stderr: None,
        error_message: None,
        metadata: Some(meta2),
    });

    let metadata_report = formatters::generate_metadata_report(&summary);
    assert!(metadata_report.contains("orchestrator"));
    assert!(metadata_report.contains("pool-manager"));
}

// Helper function to create a test summary with metadata
fn create_test_summary_with_metadata() -> TestSummary {
    let mut summary = TestSummary {
        total: 3,
        passed: 2,
        failed: 1,
        ignored: 0,
        duration_secs: 1.5,
        pass_rate: 66.7,
        tests: vec![],
    };

    // Critical test that failed
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
        custom: std::collections::HashMap::new(),
    };

    summary.tests.push(TestResult {
        name: "test_critical_feature".to_string(),
        status: TestStatus::Failed,
        duration_secs: 0.5,
        stdout: None,
        stderr: None,
        error_message: Some("Critical assertion failed".to_string()),
        metadata: Some(critical_meta),
    });

    // High priority test that passed
    let high_meta = metadata::TestMetadata {
        priority: Some("high".to_string()),
        spec: Some("ORCH-5678".to_string()),
        team: Some("pool-manager".to_string()),
        owner: None,
        issue: None,
        flaky: None,
        timeout: None,
        requires: vec![],
        tags: vec![],
        custom: std::collections::HashMap::new(),
    };

    summary.tests.push(TestResult {
        name: "test_high_priority".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.5,
        stdout: None,
        stderr: None,
        error_message: None,
        metadata: Some(high_meta),
    });

    // Regular test that passed
    summary.tests.push(TestResult {
        name: "test_regular".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.5,
        stdout: None,
        stderr: None,
        error_message: None,
        metadata: None,
    });

    summary
}
