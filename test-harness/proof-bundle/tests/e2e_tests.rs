//! End-to-End Tests with Golden File Validation
//!
//! These tests prove that proof-bundle V2 actually works end-to-end.
//! No mocks. No shortcuts. Real tests only.

use proof_bundle::{api, formatters, parsers, ProofBundle, LegacyTestType, TestStatus};

#[test]
fn test_parse_real_cargo_output_golden() {
    // Load real cargo test output (golden file)
    let output = include_str!("golden/cargo_test_output.txt");
    
    // Parse it
    let summary = parsers::parse_stable_output(output)
        .expect("Must parse real cargo output");
    
    // Validate against known good values
    assert_eq!(summary.total, 38, "proof-bundle has 38 tests");
    assert_eq!(summary.passed, 38, "All tests should pass");
    assert_eq!(summary.failed, 0, "No tests should fail");
    assert_eq!(summary.ignored, 0, "No tests ignored");
    assert_eq!(summary.pass_rate, 100.0, "100% pass rate");
    
    // Verify test names were extracted
    assert!(!summary.tests.is_empty(), "Must extract individual test results");
    
    // Verify specific tests exist
    let test_names: Vec<_> = summary.tests.iter().map(|t| t.name.as_str()).collect();
    assert!(test_names.contains(&"api::tests::test_mode_template"));
    assert!(test_names.contains(&"formatters::tests::test_generate_executive_summary"));
    assert!(test_names.contains(&"metadata::tests::test_metadata_builder"));
    assert!(test_names.contains(&"parsers::json::tests::test_parse_json_output_success"));
    assert!(test_names.contains(&"templates::tests::test_unit_test_fast"));
    
    println!("✅ Successfully parsed {} tests from real cargo output", summary.total);
}

#[test]
fn test_parser_extracts_all_test_names() {
    let output = include_str!("golden/cargo_test_output.txt");
    let summary = parsers::parse_stable_output(output).unwrap();
    
    // All 38 tests must be extracted
    assert_eq!(summary.tests.len(), 38, "Must extract all 38 individual test results");
    
    // All must have status
    for test in &summary.tests {
        assert!(!test.name.is_empty(), "Test name must not be empty");
        assert_eq!(test.status, TestStatus::Passed, "All tests in golden file passed");
    }
}

#[test]
fn test_formatters_with_real_data() {
    let output = include_str!("golden/cargo_test_output.txt");
    let summary = parsers::parse_stable_output(output).unwrap();
    
    // Generate all 4 reports
    let executive = formatters::generate_executive_summary(&summary);
    let developer = formatters::generate_test_report(&summary);
    let failure = formatters::generate_failure_report(&summary);
    let metadata_report = formatters::generate_metadata_report(&summary);
    
    // Executive summary must contain key sections
    assert!(executive.contains("Test Results Summary"), "Missing summary section");
    assert!(executive.contains("100.0% PASS RATE"), "Missing pass rate");
    assert!(executive.contains("Risk Assessment"), "Missing risk assessment");
    assert!(executive.contains("Recommendation"), "Missing recommendation");
    assert!(executive.contains("LOW RISK"), "Should show low risk for 100% pass");
    assert!(executive.contains("APPROVED"), "Should approve deployment");
    
    // Developer report must contain details
    assert!(developer.contains("Test Report"), "Missing title");
    assert!(developer.contains("Summary"), "Missing summary");
    assert!(developer.contains("38 tests"), "Missing test count");
    assert!(developer.contains("100.0%"), "Missing pass rate");
    
    // Failure report should show no failures
    assert!(failure.contains("Failure Report"), "Missing title");
    assert!(failure.contains("NO FAILURES"), "Should show no failures");
    
    // Metadata report should handle no metadata gracefully
    assert!(metadata_report.contains("Test Metadata Report"), "Missing title");
    
    println!("✅ All formatters generated valid reports from real data");
}

#[test]
fn test_reports_are_non_empty_and_useful() {
    let output = include_str!("golden/cargo_test_output.txt");
    let summary = parsers::parse_stable_output(output).unwrap();
    
    let executive = formatters::generate_executive_summary(&summary);
    let developer = formatters::generate_test_report(&summary);
    
    // Reports must be substantial (not just headers)
    assert!(executive.len() > 300, "Executive summary too short: {} bytes", executive.len());
    assert!(developer.len() > 300, "Developer report too short: {} bytes", developer.len());
    
    // Must contain actual test data (test names are in the report)
    assert!(developer.contains("test"), "Missing test keyword");
    assert!(developer.contains("38"), "Missing test count");
    
    println!("✅ Reports are substantial and contain real data");
}

#[test]
#[ignore] // Run manually: cargo test test_generate_real_proof_bundle -- --ignored --nocapture
fn test_generate_real_proof_bundle() {
    println!("\n🎯 Running REAL end-to-end proof bundle generation...\n");
    
    // THE REAL TEST: Generate proof bundle for proof-bundle itself
    let summary = api::generate_for_crate("proof-bundle", api::Mode::UnitFast)
        .expect("Must successfully generate proof bundle");
    
    // Validate results
    assert!(summary.total > 0, "Must capture real tests, got {}", summary.total);
    assert_eq!(summary.total, 38, "Should capture all 38 tests");
    assert!(summary.pass_rate >= 90.0, "Must have high pass rate: {:.1}%", summary.pass_rate);
    
    println!("📊 Summary:");
    println!("   Total: {}", summary.total);
    println!("   Passed: {}", summary.passed);
    println!("   Failed: {}", summary.failed);
    println!("   Pass rate: {:.1}%\n", summary.pass_rate);
    
    // Verify files were created
    let pb = ProofBundle::for_type(LegacyTestType::Unit)
        .expect("Must create proof bundle");
    
    let root = pb.root();
    println!("📁 Proof bundle location: {}\n", root.display());
    
    // Check all 7 files exist
    assert!(root.join("test_results.ndjson").exists(), "Missing test_results.ndjson");
    assert!(root.join("summary.json").exists(), "Missing summary.json");
    assert!(root.join("executive_summary.md").exists(), "Missing executive_summary.md");
    assert!(root.join("test_report.md").exists(), "Missing test_report.md");
    assert!(root.join("failure_report.md").exists(), "Missing failure_report.md");
    assert!(root.join("metadata_report.md").exists(), "Missing metadata_report.md");
    assert!(root.join("test_config.json").exists(), "Missing test_config.json");
    
    println!("✅ All 7 files generated successfully\n");
    
    // Read and validate executive summary
    let executive = std::fs::read_to_string(root.join("executive_summary.md"))
        .expect("Must read executive summary");
    
    assert!(executive.contains("Test Results Summary"), "Executive summary missing key content");
    assert!(executive.contains("PASS RATE"), "Executive summary missing pass rate");
    
    println!("✅ REAL END-TO-END TEST PASSED!");
    println!("   Check {} for generated reports\n", root.display());
}

#[test]
fn test_parser_handles_edge_cases() {
    // Test with minimal output
    let minimal = "running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out";
    let summary = parsers::parse_stable_output(minimal).unwrap();
    assert_eq!(summary.total, 0);
    assert_eq!(summary.passed, 0);
    
    // Test with single test
    let single = "running 1 tests\ntest my_test ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out";
    let summary = parsers::parse_stable_output(single).unwrap();
    assert_eq!(summary.total, 1);
    assert_eq!(summary.passed, 1);
    assert_eq!(summary.tests[0].name, "my_test");
}

#[test]
fn test_golden_file_exists_and_is_valid() {
    // Verify golden file is present and parseable
    let output = include_str!("golden/cargo_test_output.txt");
    
    assert!(!output.is_empty(), "Golden file must not be empty");
    assert!(output.contains("running 38 tests"), "Golden file must contain test header");
    assert!(output.contains("test result: ok"), "Golden file must contain result summary");
    
    // Must be parseable
    let summary = parsers::parse_stable_output(output);
    assert!(summary.is_ok(), "Golden file must be parseable: {:?}", summary.err());
}
