//! Tests for formatters

use super::*;
use crate::{TestResult, TestStatus, TestSummary};

fn sample_summary() -> TestSummary {
    TestSummary {
        total: 100,
        passed: 98,
        failed: 2,
        ignored: 0,
        duration_secs: 5.2,
        pass_rate: 98.0,
        tests: vec![
            TestResult {
                name: "unit::test_success".to_string(),
                status: TestStatus::Passed,
                duration_secs: 0.01,
                stdout: None,
                stderr: None,
                error_message: None,
                metadata: None,
            },
            TestResult {
                name: "integration::test_failure".to_string(),
                status: TestStatus::Failed,
                duration_secs: 0.5,
                stdout: None,
                stderr: None,
                error_message: Some("assertion failed: expected 42, got 43".to_string()),
                metadata: None,
            },
            TestResult {
                name: "property::test_edge_case".to_string(),
                status: TestStatus::Failed,
                duration_secs: 1.2,
                stdout: None,
                stderr: None,
                error_message: Some("thread panicked at 'index out of bounds'".to_string()),
                metadata: None,
            },
        ],
    }
}

#[test]
fn test_generate_executive_summary() {
    let summary = sample_summary();
    let report = generate_executive_summary(&summary);
    
    assert!(report.contains("Test Results Summary"));
    assert!(report.contains("98.0% PASS RATE"));
    assert!(report.contains("Quick Facts"));
    assert!(report.contains("Risk Assessment"));
    assert!(report.contains("Failed Tests"));
    assert!(report.contains("Recommendation"));
}

#[test]
fn test_generate_test_report() {
    let summary = sample_summary();
    let report = generate_test_report(&summary);
    
    assert!(report.contains("Test Report"));
    assert!(report.contains("Summary"));
    assert!(report.contains("Test Breakdown"));
    assert!(report.contains("Failed Tests"));
    assert!(report.contains("Performance"));
}

#[test]
fn test_generate_failure_report() {
    let summary = sample_summary();
    let report = generate_failure_report(&summary);
    
    assert!(report.contains("Failure Report"));
    assert!(report.contains("Failure 1:"));
    assert!(report.contains("Failure 2:"));
    assert!(report.contains("Reproduction"));
    assert!(report.contains("Recommendations"));
}

#[test]
fn test_generate_failure_report_no_failures() {
    let summary = TestSummary {
        total: 10,
        passed: 10,
        failed: 0,
        ignored: 0,
        duration_secs: 1.0,
        pass_rate: 100.0,
        tests: vec![],
    };
    
    let report = generate_failure_report(&summary);
    assert!(report.contains("NO FAILURES"));
    assert!(report.contains("All tests passed"));
}

#[test]
fn test_categorize_tests() {
    let tests = vec![
        TestResult {
            name: "test_unit".to_string(),
            status: TestStatus::Passed,
            duration_secs: 0.0,
            stdout: None,
            stderr: None,
            error_message: None,
            metadata: None,
        },
        TestResult {
            name: "property_test_bounds".to_string(),
            status: TestStatus::Passed,
            duration_secs: 0.0,
            stdout: None,
            stderr: None,
            error_message: None,
            metadata: None,
        },
        TestResult {
            name: "integration_test_api".to_string(),
            status: TestStatus::Passed,
            duration_secs: 0.0,
            stdout: None,
            stderr: None,
            error_message: None,
            metadata: None,
        },
    ];
    
    let categories = helpers::categorize_tests(&tests);
    
    assert!(categories.contains_key("Unit Tests"));
    assert!(categories.contains_key("Property Tests"));
    assert!(categories.contains_key("Integration Tests"));
}

#[test]
fn test_simplify_error() {
    assert_eq!(helpers::simplify_error("assertion failed"), "Test expectation not met");
    assert_eq!(helpers::simplify_error("thread panicked"), "Test encountered unexpected condition");
    assert_eq!(helpers::simplify_error("timeout exceeded"), "Test exceeded time limit");
}
