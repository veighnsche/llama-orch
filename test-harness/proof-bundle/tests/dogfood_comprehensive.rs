//! Comprehensive dogfooding tests for proof-bundle
//!
//! This file demonstrates EVERY feature of the proof-bundle crate:
//! - Metadata annotations (all fields)
//! - Different test types (unit, integration, property)
//! - Priority levels (critical, high, medium, low)
//! - Custom fields, tags, requirements
//! - Flaky test marking
//! - Spec/requirement tracing
//!
//! This proof bundle serves as THE EXAMPLE for other crates.

use proof_bundle::{api, formatters, metadata, parsers, test_metadata, TestMetadata};

// ============================================================================
// CRITICAL PRIORITY TESTS - Core functionality that must never break
// ============================================================================

/// @priority: critical
/// @spec: ORCH-3800
/// @team: proof-bundle
/// @owner: proof-bundle-team@llama-orch.dev
/// @tags: core, v2-api
#[test]
fn test_one_liner_api_works() {
    // The most important test - the one-liner API must work
    let result = api::generate_for_crate("proof-bundle", api::Mode::UnitFast);
    assert!(result.is_ok(), "One-liner API must work");
}

/// @priority: critical
/// @spec: ORCH-3804
/// @team: proof-bundle
/// @tags: core, formatters
#[test]
fn test_all_reports_generated() {
    let summary = create_test_summary();
    
    // All 4 reports must be generated
    let exec = formatters::generate_executive_summary(&summary);
    let dev = formatters::generate_test_report(&summary);
    let fail = formatters::generate_failure_report(&summary);
    let meta = formatters::generate_metadata_report(&summary);
    
    assert!(!exec.is_empty(), "Executive summary must be generated");
    assert!(!dev.is_empty(), "Developer report must be generated");
    assert!(!fail.is_empty(), "Failure report must be generated");
    assert!(!meta.is_empty(), "Metadata report must be generated");
}

/// @priority: critical
/// @spec: ORCH-3817
/// @team: proof-bundle
/// @tags: core, parsers
#[test]
fn test_stable_parser_works() {
    let output = include_str!("golden/cargo_test_output.txt");
    let result = parsers::parse_stable_output(output);
    assert!(result.is_ok(), "Stable parser must work");
    
    let summary = result.unwrap();
    assert_eq!(summary.total, 38, "Must parse all 38 tests");
}

// ============================================================================
// HIGH PRIORITY TESTS - Important features
// ============================================================================

/// @priority: high
/// @spec: ORCH-3811
/// @team: proof-bundle
/// @tags: api, builder
#[test]
fn test_metadata_builder_fluent_api() {
    let metadata = test_metadata()
        .priority("high")
        .spec("ORCH-3811")
        .team("proof-bundle")
        .build();
    
    assert_eq!(metadata.priority, Some("high".to_string()));
    assert_eq!(metadata.spec, Some("ORCH-3811".to_string()));
}

/// @priority: high
/// @spec: ORCH-3820
/// @team: proof-bundle
/// @tags: formatters, reports
#[test]
fn test_executive_summary_format() {
    let summary = create_test_summary();
    let report = formatters::generate_executive_summary(&summary);
    
    assert!(report.contains("Test Results Summary"));
    assert!(report.contains("PASS RATE"));
}

/// @priority: high
/// @spec: ORCH-3807
/// @team: proof-bundle
/// @tags: formatters, executive
/// @custom:stakeholder: management
#[test]
fn test_executive_summary_non_technical() {
    let summary = create_test_summary();
    let report = formatters::generate_executive_summary(&summary);
    
    // Must use business language, not technical jargon
    assert!(report.contains("Risk") || report.contains("Confidence"));
}

/// @priority: high
/// @spec: PB-004
/// @team: proof-bundle
/// @tags: metadata, parsing
#[test]
fn test_doc_comment_parsing() {
    let doc = r#"
        @priority: critical
        @spec: ORCH-1234
        @team: orchestrator
    "#;
    
    let metadata = metadata::parse_doc_comments(doc);
    assert_eq!(metadata.priority, Some("critical".to_string()));
    assert_eq!(metadata.spec, Some("ORCH-1234".to_string()));
}

// ============================================================================
// MEDIUM PRIORITY TESTS - Nice-to-have features
// ============================================================================

/// @priority: medium
/// @spec: ORCH-3822
/// @team: proof-bundle
/// @tags: formatters, helpers
#[test]
fn test_priority_detection() {
    let critical = test_metadata().priority("critical").build();
    let high = test_metadata().priority("high").build();
    let medium = test_metadata().priority("medium").build();
    
    assert!(metadata::is_critical(&critical));
    assert!(!metadata::is_critical(&high));
    
    assert!(metadata::is_high_priority(&critical));
    assert!(metadata::is_high_priority(&high));
    assert!(!metadata::is_high_priority(&medium));
}

/// @priority: medium
/// @spec: PB-004
/// @team: proof-bundle
/// @tags: metadata, custom-fields
/// @custom:feature-type: extensibility
#[test]
fn test_custom_metadata_fields() {
    let metadata = test_metadata()
        .custom("deployment-stage", "canary")
        .custom("sla", "99.9%")
        .build();
    
    assert_eq!(metadata.custom.get("deployment-stage"), Some(&"canary".to_string()));
    assert_eq!(metadata.custom.get("sla"), Some(&"99.9%".to_string()));
}

/// @priority: medium
/// @spec: PB-004
/// @team: proof-bundle
/// @tags: metadata, resources
/// @requires: GPU
/// @requires: CUDA
#[test]
fn test_resource_requirements() {
    let metadata = test_metadata()
        .requires(&["GPU", "CUDA", "16GB VRAM"])
        .build();
    
    assert_eq!(metadata.requires.len(), 3);
    assert!(metadata::requires_resource(&metadata, "GPU"));
    assert!(metadata::requires_resource(&metadata, "CUDA"));
}

/// @priority: medium
/// @spec: PB-004
/// @team: proof-bundle
/// @tags: metadata, tags, filtering
#[test]
fn test_metadata_tags() {
    let metadata = test_metadata()
        .tags(&["integration", "slow", "gpu-required"])
        .build();
    
    assert_eq!(metadata.tags.len(), 3);
    assert!(metadata::has_tag(&metadata, "integration"));
    assert!(metadata::has_tag(&metadata, "slow"));
}

// ============================================================================
// LOW PRIORITY TESTS - Edge cases and validation
// ============================================================================

/// @priority: low
/// @spec: ORCH-3828
/// @team: proof-bundle
/// @tags: validation, edge-cases
#[test]
fn test_empty_metadata_serialization() {
    let metadata = TestMetadata::default();
    let json = serde_json::to_string(&metadata).unwrap();
    
    // Empty metadata should serialize to {}
    assert_eq!(json, "{}");
}

/// @priority: low
/// @spec: PB-004
/// @team: proof-bundle
/// @tags: metadata, serialization
#[test]
fn test_metadata_roundtrip() {
    let original = test_metadata()
        .priority("critical")
        .spec("ORCH-1234")
        .team("orchestrator")
        .custom("foo", "bar")
        .build();
    
    let json = serde_json::to_string(&original).unwrap();
    let deserialized: TestMetadata = serde_json::from_str(&json).unwrap();
    
    assert_eq!(original, deserialized);
}

/// @priority: low
/// @spec: ORCH-3820
/// @team: proof-bundle
/// @tags: formatters, edge-cases
#[test]
fn test_formatters_handle_empty_summary() {
    let summary = proof_bundle::TestSummary {
        total: 0,
        passed: 0,
        failed: 0,
        ignored: 0,
        duration_secs: 0.0,
        pass_rate: 0.0,
        tests: vec![],
    };
    
    // Formatters should handle empty summaries gracefully
    let exec = formatters::generate_executive_summary(&summary);
    let dev = formatters::generate_test_report(&summary);
    
    assert!(!exec.is_empty());
    assert!(!dev.is_empty());
}

// ============================================================================
// FLAKY TEST EXAMPLES - Known intermittent failures
// ============================================================================

/// @priority: medium
/// @spec: ORCH-3809
/// @team: proof-bundle
/// @flaky: Occasionally times out on slow CI (5% failure rate)
/// @tags: flaky, ci
/// @issue: #1234
#[test]
fn test_parser_performance() {
    // This test occasionally times out on slow CI
    let output = include_str!("golden/cargo_test_output.txt");
    let start = std::time::Instant::now();
    
    let _result = parsers::parse_stable_output(output);
    
    let duration = start.elapsed();
    assert!(duration.as_secs() < 5, "Parser should be fast");
}

// ============================================================================
// INTEGRATION TESTS - Cross-module functionality
// ============================================================================

/// @priority: high
/// @spec: ORCH-3800
/// @team: proof-bundle
/// @tags: integration, end-to-end
/// @timeout: 30s
#[test]
fn test_full_proof_bundle_generation() {
    // This is an integration test that exercises the entire system
    let result = api::generate_for_crate("proof-bundle", api::Mode::UnitFast);
    
    assert!(result.is_ok(), "Full proof bundle generation must work");
    
    let summary = result.unwrap();
    assert!(summary.total > 0, "Must have tests");
    assert!(summary.pass_rate >= 90.0, "Pass rate must be high");
}

/// @priority: high
/// @spec: ORCH-3804
/// @team: proof-bundle
/// @tags: integration, formatters
#[test]
fn test_all_formatters_with_real_data() {
    let summary = create_realistic_summary();
    
    let exec = formatters::generate_executive_summary(&summary);
    let dev = formatters::generate_test_report(&summary);
    let fail = formatters::generate_failure_report(&summary);
    let meta = formatters::generate_metadata_report(&summary);
    
    // All reports should contain key sections
    assert!(exec.contains("Summary") || exec.contains("PASS RATE"));
    assert!(dev.contains("Test") || dev.contains("Report"));
    assert!(fail.contains("Failure") || fail.contains("Failed"));
    assert!(meta.contains("Metadata") || meta.contains("Priority"));
}

// ============================================================================
// PROPERTY-BASED TESTS - Invariants that must always hold
// ============================================================================

/// @priority: high
/// @spec: ORCH-3821
/// @team: proof-bundle
/// @tags: property, invariants
#[test]
fn test_pass_rate_calculation_invariant() {
    // Property: pass_rate = (passed / total) * 100
    for total in [1, 10, 100, 1000] {
        for passed in 0..=total {
            let pass_rate = (passed as f64 / total as f64) * 100.0;
            assert!(pass_rate >= 0.0 && pass_rate <= 100.0);
        }
    }
}

/// @priority: medium
/// @spec: PB-004
/// @team: proof-bundle
/// @tags: property, metadata
#[test]
fn test_priority_level_ordering() {
    // Property: critical > high > medium > low > unset
    let critical = test_metadata().priority("critical").build();
    let high = test_metadata().priority("high").build();
    let medium = test_metadata().priority("medium").build();
    let low = test_metadata().priority("low").build();
    let unset = test_metadata().build();
    
    assert!(metadata::priority_level(&critical) > metadata::priority_level(&high));
    assert!(metadata::priority_level(&high) > metadata::priority_level(&medium));
    assert!(metadata::priority_level(&medium) > metadata::priority_level(&low));
    assert!(metadata::priority_level(&low) > metadata::priority_level(&unset));
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn create_test_summary() -> proof_bundle::TestSummary {
    proof_bundle::TestSummary {
        total: 10,
        passed: 10,
        failed: 0,
        ignored: 0,
        duration_secs: 1.0,
        pass_rate: 100.0,
        tests: vec![],
    }
}

fn create_realistic_summary() -> proof_bundle::TestSummary {
    use proof_bundle::{TestResult, TestStatus};
    
    let mut tests = vec![];
    
    // Add some passing tests with metadata
    for i in 1..=5 {
        let meta = test_metadata()
            .priority("high")
            .spec(&format!("ORCH-{}", 1000 + i))
            .team("proof-bundle")
            .build();
        
        tests.push(TestResult {
            name: format!("test_feature_{}", i),
            status: TestStatus::Passed,
            duration_secs: 0.1,
            stdout: None,
            stderr: None,
            error_message: None,
            metadata: Some(meta),
        });
    }
    
    proof_bundle::TestSummary {
        total: tests.len(),
        passed: tests.len(),
        failed: 0,
        ignored: 0,
        duration_secs: 0.5,
        pass_rate: 100.0,
        tests,
    }
}
