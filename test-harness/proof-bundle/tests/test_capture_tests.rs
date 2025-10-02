//! Unit tests for capture_tests() functionality

use proof_bundle::{ProofBundle, TestType, TestStatus};
use std::path::PathBuf;

#[test]
fn test_capture_builder_creation() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    let builder = pb.capture_tests("proof-bundle");
    
    // Builder should be created successfully
    // (We can't inspect private fields, but creation should not panic)
}

#[test]
fn test_capture_builder_chaining() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    
    // Test that builder methods chain correctly
    let builder = pb.capture_tests("proof-bundle")
        .lib()
        .tests()
        .benches()
        .doc()
        .features(&["test-feature"])
        .no_fail_fast()
        .test_threads(4);
    
    // Should not panic
}

#[test]
fn test_capture_builder_all() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    
    // Test .all() convenience method
    let builder = pb.capture_tests("proof-bundle")
        .all();
    
    // Should not panic
}

#[test]
#[ignore] // Requires nightly Rust
fn test_capture_tests_run() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    
    // Run actual test capture
    let result = pb.capture_tests("proof-bundle")
        .lib()
        .run();
    
    // Should either succeed or fail with clear error
    match result {
        Ok(summary) => {
            // Verify summary structure
            assert!(summary.total > 0, "Should capture at least some tests");
            assert_eq!(
                summary.total,
                summary.passed + summary.failed + summary.ignored,
                "Total should equal passed + failed + ignored"
            );
            
            // Verify pass rate calculation
            let expected_rate = if summary.total == 0 {
                0.0
            } else {
                (summary.passed as f64 / summary.total as f64) * 100.0
            };
            assert!(
                (summary.pass_rate - expected_rate).abs() < 0.01,
                "Pass rate should be calculated correctly"
            );
            
            // Verify test results
            assert_eq!(
                summary.tests.len(),
                summary.total,
                "Should have one TestResult per test"
            );
            
            // Verify files were created
            let root = pb.root();
            assert!(root.join("test_results.ndjson").exists(), "Should create test_results.ndjson");
            assert!(root.join("summary.json").exists(), "Should create summary.json");
            assert!(root.join("test_report.md").exists(), "Should create test_report.md");
        }
        Err(e) => {
            // Should fail with clear error about nightly requirement
            let err_msg = format!("{}", e);
            assert!(
                err_msg.contains("nightly") || err_msg.contains("unstable"),
                "Error should mention nightly requirement: {}",
                err_msg
            );
        }
    }
}

#[test]
fn test_test_status_serialization() {
    use serde_json;
    
    // Test that TestStatus serializes correctly
    let passed = serde_json::to_string(&TestStatus::Passed).unwrap();
    assert_eq!(passed, r#""passed""#);
    
    let failed = serde_json::to_string(&TestStatus::Failed).unwrap();
    assert_eq!(failed, r#""failed""#);
    
    let ignored = serde_json::to_string(&TestStatus::Ignored).unwrap();
    assert_eq!(ignored, r#""ignored""#);
    
    let timeout = serde_json::to_string(&TestStatus::Timeout).unwrap();
    assert_eq!(timeout, r#""timeout""#);
}

#[test]
fn test_test_status_equality() {
    assert_eq!(TestStatus::Passed, TestStatus::Passed);
    assert_eq!(TestStatus::Failed, TestStatus::Failed);
    assert_ne!(TestStatus::Passed, TestStatus::Failed);
}

#[test]
fn test_test_summary_pass_rate_calculation() {
    use proof_bundle::TestSummary;
    
    // Test pass rate calculation
    assert_eq!(TestSummary::calculate_pass_rate(0, 0), 0.0);
    assert_eq!(TestSummary::calculate_pass_rate(10, 10), 100.0);
    assert_eq!(TestSummary::calculate_pass_rate(5, 10), 50.0);
    
    // Use approximate comparison for floating point
    let rate = TestSummary::calculate_pass_rate(1, 3);
    assert!((rate - 33.333333333333336).abs() < 0.0001, "Expected ~33.33%, got {}", rate);
}

#[test]
fn test_test_result_serialization() {
    use proof_bundle::TestResult;
    use serde_json;
    
    let result = TestResult {
        name: "test_example".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.123,
        stdout: Some("output".to_string()),
        stderr: None,
        error_message: None,
    };
    
    let json = serde_json::to_string(&result).unwrap();
    
    // Should serialize successfully
    assert!(json.contains("test_example"));
    assert!(json.contains("passed"));
    assert!(json.contains("0.123"));
    assert!(json.contains("output"));
    
    // stderr and error_message should be omitted (skip_serializing_if)
    assert!(!json.contains("stderr"));
    assert!(!json.contains("error_message"));
}

#[test]
fn test_test_result_with_error() {
    use proof_bundle::TestResult;
    use serde_json;
    
    let result = TestResult {
        name: "test_failure".to_string(),
        status: TestStatus::Failed,
        duration_secs: 0.456,
        stdout: None,
        stderr: Some("error output".to_string()),
        error_message: Some("assertion failed".to_string()),
    };
    
    let json = serde_json::to_string(&result).unwrap();
    
    // Should include error details
    assert!(json.contains("test_failure"));
    assert!(json.contains("failed"));
    assert!(json.contains("error output"));
    assert!(json.contains("assertion failed"));
}

#[test]
fn test_test_summary_serialization() {
    use proof_bundle::{TestSummary, TestResult};
    use serde_json;
    
    let summary = TestSummary {
        total: 10,
        passed: 8,
        failed: 1,
        ignored: 1,
        duration_secs: 5.5,
        pass_rate: 80.0,
        tests: vec![
            TestResult {
                name: "test1".to_string(),
                status: TestStatus::Passed,
                duration_secs: 0.1,
                stdout: None,
                stderr: None,
                error_message: None,
            },
        ],
    };
    
    let json = serde_json::to_string(&summary).unwrap();
    
    // Should serialize all fields
    assert!(json.contains(r#""total":10"#));
    assert!(json.contains(r#""passed":8"#));
    assert!(json.contains(r#""failed":1"#));
    assert!(json.contains(r#""ignored":1"#));
    assert!(json.contains(r#""duration_secs":5.5"#));
    assert!(json.contains(r#""pass_rate":80"#));
    assert!(json.contains("test1"));
}

#[test]
fn test_test_summary_deserialization() {
    use proof_bundle::TestSummary;
    use serde_json;
    
    let json = r#"{
        "total": 5,
        "passed": 4,
        "failed": 1,
        "ignored": 0,
        "duration_secs": 2.5,
        "pass_rate": 80.0,
        "tests": []
    }"#;
    
    let summary: TestSummary = serde_json::from_str(json).unwrap();
    
    assert_eq!(summary.total, 5);
    assert_eq!(summary.passed, 4);
    assert_eq!(summary.failed, 1);
    assert_eq!(summary.ignored, 0);
    assert_eq!(summary.duration_secs, 2.5);
    assert_eq!(summary.pass_rate, 80.0);
    assert_eq!(summary.tests.len(), 0);
}

#[test]
fn test_capture_tests_creates_proof_bundle_in_crate_dir() {
    // Set custom proof dir for this test
    let temp_dir = std::env::temp_dir().join("proof-bundle-test");
    std::fs::create_dir_all(&temp_dir).unwrap();
    std::env::set_var("LLORCH_PROOF_DIR", temp_dir.to_str().unwrap());
    
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    
    // Verify proof bundle is in custom dir
    assert!(pb.root().starts_with(&temp_dir));
    
    // Clean up
    std::env::remove_var("LLORCH_PROOF_DIR");
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_multiple_capture_builders_independent() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    
    // Create multiple builders
    let builder1 = pb.capture_tests("crate1").lib();
    let builder2 = pb.capture_tests("crate2").tests();
    
    // Should be independent (can't verify without running, but shouldn't panic)
}

#[test]
fn test_builder_with_empty_features() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    
    // Empty features array should work
    let builder = pb.capture_tests("proof-bundle")
        .features(&[]);
    
    // Should not panic
}

#[test]
fn test_builder_with_multiple_features() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    
    // Multiple features should work
    let builder = pb.capture_tests("proof-bundle")
        .features(&["feature1", "feature2", "feature3"]);
    
    // Should not panic
}

#[test]
fn test_test_threads_zero() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    
    // Zero threads should be allowed (cargo test will handle it)
    let builder = pb.capture_tests("proof-bundle")
        .test_threads(0);
    
    // Should not panic
}

#[test]
fn test_test_threads_large() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    
    // Large thread count should be allowed
    let builder = pb.capture_tests("proof-bundle")
        .test_threads(1000);
    
    // Should not panic
}

#[test]
fn test_capture_tests_method_exists() {
    let pb = ProofBundle::for_type(TestType::Unit).unwrap();
    
    // Verify capture_tests method exists and returns builder
    let builder = pb.capture_tests("test-package");
    
    // Type should be TestCaptureBuilder (verified by compilation)
}

#[test]
fn test_exported_types_accessible() {
    // Verify all types are exported and accessible
    use proof_bundle::{TestCaptureBuilder, TestResult, TestStatus, TestSummary};
    
    // Should compile without errors
}

#[test]
fn test_test_status_debug() {
    // Verify Debug trait is implemented
    let status = TestStatus::Passed;
    let debug_str = format!("{:?}", status);
    assert_eq!(debug_str, "Passed");
}

#[test]
fn test_test_result_debug() {
    use proof_bundle::TestResult;
    
    let result = TestResult {
        name: "test".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.1,
        stdout: None,
        stderr: None,
        error_message: None,
    };
    
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("test"));
    assert!(debug_str.contains("Passed"));
}

#[test]
fn test_test_summary_debug() {
    use proof_bundle::TestSummary;
    
    let summary = TestSummary {
        total: 1,
        passed: 1,
        failed: 0,
        ignored: 0,
        duration_secs: 0.1,
        pass_rate: 100.0,
        tests: vec![],
    };
    
    let debug_str = format!("{:?}", summary);
    assert!(debug_str.contains("total"));
    assert!(debug_str.contains("passed"));
}

#[test]
fn test_test_summary_clone() {
    use proof_bundle::TestSummary;
    
    let summary = TestSummary {
        total: 1,
        passed: 1,
        failed: 0,
        ignored: 0,
        duration_secs: 0.1,
        pass_rate: 100.0,
        tests: vec![],
    };
    
    let cloned = summary.clone();
    assert_eq!(cloned.total, summary.total);
    assert_eq!(cloned.passed, summary.passed);
}

#[test]
fn test_test_result_clone() {
    use proof_bundle::TestResult;
    
    let result = TestResult {
        name: "test".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.1,
        stdout: None,
        stderr: None,
        error_message: None,
    };
    
    let cloned = result.clone();
    assert_eq!(cloned.name, result.name);
    assert_eq!(cloned.status, result.status);
}

#[test]
fn test_test_status_copy() {
    let status1 = TestStatus::Passed;
    let status2 = status1; // Copy, not move
    
    // Both should be usable
    assert_eq!(status1, TestStatus::Passed);
    assert_eq!(status2, TestStatus::Passed);
}
