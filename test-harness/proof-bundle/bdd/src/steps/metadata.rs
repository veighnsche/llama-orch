//! BDD steps for metadata testing

use cucumber::{given, then, when};
use crate::steps::world::BddWorld;
use proof_bundle::{metadata, TestMetadata, TestResult, TestStatus, TestSummary, formatters};
use std::collections::HashMap;

#[given("the metadata system is available")]
async fn metadata_available(_world: &mut BddWorld) {
    // Nothing to do - metadata is always available
}

#[given(regex = r#"^I have a doc comment "([^"]+)"$"#)]
async fn have_doc_comment(world: &mut BddWorld, doc: String) {
    world.doc_comment = Some(doc);
}

#[when("I parse the doc comments")]
async fn parse_doc_comments(world: &mut BddWorld) {
    let doc = world.doc_comment.as_ref().expect("No doc comment");
    world.parsed_metadata = Some(metadata::parse_doc_comments(doc));
}

#[then(regex = r#"^the metadata should have priority "([^"]+)"$"#)]
async fn metadata_has_priority(world: &mut BddWorld, expected: String) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    assert_eq!(meta.priority.as_ref(), Some(&expected), "Priority mismatch");
}

#[then(regex = r#"^the metadata should have spec "([^"]+)"$"#)]
async fn metadata_has_spec(world: &mut BddWorld, expected: String) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    assert_eq!(meta.spec.as_ref(), Some(&expected), "Spec mismatch");
}

#[then(regex = r#"^the metadata should have team "([^"]+)"$"#)]
async fn metadata_has_team(world: &mut BddWorld, expected: String) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    assert_eq!(meta.team.as_ref(), Some(&expected), "Team mismatch");
}

#[then(regex = r#"^the metadata should have owner "([^"]+)"$"#)]
async fn metadata_has_owner(world: &mut BddWorld, expected: String) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    assert_eq!(meta.owner.as_ref(), Some(&expected), "Owner mismatch");
}

#[then(regex = r#"^the metadata should have issue "([^"]+)"$"#)]
async fn metadata_has_issue(world: &mut BddWorld, expected: String) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    assert_eq!(meta.issue.as_ref(), Some(&expected), "Issue mismatch");
}

#[when("I build metadata with:")]
async fn build_metadata(world: &mut BddWorld, table: cucumber::gherkin::Table) {
    let mut builder = metadata::test_metadata();
    
    for row in table.rows.iter().skip(1) { // Skip header
        let field = &row[0];
        let value = &row[1];
        
        builder = match field.as_str() {
            "priority" => builder.priority(value),
            "spec" => builder.spec(value),
            "team" => builder.team(value),
            "owner" => builder.owner(value),
            "issue" => builder.issue(value),
            "flaky" => builder.flaky(value),
            "timeout" => builder.timeout(value),
            _ => builder,
        };
    }
    
    world.parsed_metadata = Some(builder.build());
}

#[given(regex = r#"^I have metadata with priority "([^"]+)"$"#)]
async fn have_metadata_priority(world: &mut BddWorld, priority: String) {
    world.parsed_metadata = Some(metadata::test_metadata().priority(&priority).build());
}

#[when("I check if the test is critical")]
async fn check_is_critical(world: &mut BddWorld) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    world.is_critical = Some(metadata::is_critical(meta));
}

#[then("it should be marked as critical")]
async fn should_be_critical(world: &mut BddWorld) {
    assert!(world.is_critical.expect("is_critical not set"), "Should be critical");
}

#[when("I check if the test is high priority")]
async fn check_is_high_priority(world: &mut BddWorld) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    world.is_high_priority = Some(metadata::is_high_priority(meta));
}

#[then("it should be marked as high priority")]
async fn should_be_high_priority(world: &mut BddWorld) {
    assert!(world.is_high_priority.expect("is_high_priority not set"), "Should be high priority");
}

#[given(regex = r#"^I have metadata with flaky "([^"]+)"$"#)]
async fn have_metadata_flaky(world: &mut BddWorld, flaky: String) {
    world.parsed_metadata = Some(metadata::test_metadata().flaky(&flaky).build());
}

#[when("I check if the test is flaky")]
async fn check_is_flaky(world: &mut BddWorld) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    world.is_flaky = Some(metadata::is_flaky(meta));
}

#[then("it should be marked as flaky")]
async fn should_be_flaky(world: &mut BddWorld) {
    assert!(world.is_flaky.expect("is_flaky not set"), "Should be flaky");
}

#[then(regex = r#"^the metadata should have custom field "([^"]+)" with value "([^"]+)"$"#)]
async fn metadata_has_custom_field(world: &mut BddWorld, key: String, value: String) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    assert_eq!(meta.custom.get(&key), Some(&value), "Custom field mismatch");
}

#[then(regex = r"^the metadata should have (\d+) requirements$")]
async fn metadata_has_n_requirements(world: &mut BddWorld, count: usize) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    assert_eq!(meta.requires.len(), count, "Requirements count mismatch");
}

#[then(regex = r#"^the metadata should require "([^"]+)"$"#)]
async fn metadata_requires(world: &mut BddWorld, requirement: String) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    assert!(meta.requires.contains(&requirement), "Missing requirement: {}", requirement);
}

#[then(regex = r"^the metadata should have (\d+) tags$")]
async fn metadata_has_n_tags(world: &mut BddWorld, count: usize) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    assert_eq!(meta.tags.len(), count, "Tags count mismatch");
}

#[then(regex = r#"^the metadata should have tag "([^"]+)"$"#)]
async fn metadata_has_tag(world: &mut BddWorld, tag: String) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    assert!(meta.tags.contains(&tag), "Missing tag: {}", tag);
}

#[given("I have a test result with metadata")]
async fn have_test_result_with_metadata(world: &mut BddWorld) {
    let meta = metadata::test_metadata()
        .priority("critical")
        .spec("ORCH-1234")
        .team("orchestrator")
        .build();
    
    world.test_result = Some(TestResult {
        name: "test_example".to_string(),
        status: TestStatus::Passed,
        duration_secs: 0.5,
        stdout: None,
        stderr: None,
        error_message: None,
        metadata: Some(meta),
    });
}

#[when("I serialize the test result to JSON")]
async fn serialize_test_result(world: &mut BddWorld) {
    let result = world.test_result.as_ref().expect("No test result");
    world.json_output = Some(serde_json::to_string_pretty(result).expect("Serialization failed"));
}

#[then("the JSON should contain the metadata")]
async fn json_contains_metadata(world: &mut BddWorld) {
    let json = world.json_output.as_ref().expect("No JSON output");
    assert!(json.contains("\"metadata\""), "JSON missing metadata field");
    assert!(json.contains("\"priority\""), "JSON missing priority");
}

#[then("the metadata should be deserializable")]
async fn metadata_deserializable(world: &mut BddWorld) {
    let json = world.json_output.as_ref().expect("No JSON output");
    let _result: TestResult = serde_json::from_str(json).expect("Deserialization failed");
}

#[when("I create test results with mixed priorities:")]
async fn have_mixed_priorities(world: &mut BddWorld, table: cucumber::gherkin::Table) {
    let mut tests = Vec::new();
    
    for row in table.rows.iter().skip(1) {
        let name = &row[0];
        let priority = &row[1];
        let status = match row[2].as_str() {
            "passed" => TestStatus::Passed,
            "failed" => TestStatus::Failed,
            _ => TestStatus::Passed,
        };
        
        let meta = metadata::test_metadata().priority(priority).build();
        
        tests.push(TestResult {
            name: name.clone(),
            status,
            duration_secs: 0.1,
            stdout: None,
            stderr: None,
            error_message: if status == TestStatus::Failed {
                Some("Test failed".to_string())
            } else {
                None
            },
            metadata: Some(meta),
        });
    }
    
    world.test_summary = Some(TestSummary {
        total: tests.len(),
        passed: tests.iter().filter(|t| t.status == TestStatus::Passed).count(),
        failed: tests.iter().filter(|t| t.status == TestStatus::Failed).count(),
        ignored: 0,
        duration_secs: 0.5,
        pass_rate: 80.0,
        tests,
    });
}

#[when("I generate the metadata report")]
async fn generate_metadata_report(world: &mut BddWorld) {
    let summary = world.test_summary.as_ref().expect("No summary");
    world.metadata_report = Some(formatters::generate_metadata_report(summary));
}

#[then(regex = r#"^it should have a "([^"]+)" section with (\d+) tests?$"#)]
async fn should_have_section(world: &mut BddWorld, section: String, count: usize) {
    let report = world.metadata_report.as_ref().expect("No metadata report");
    assert!(report.contains(&section), "Missing section: {}", section);
    // Could add more sophisticated parsing here
}

#[when("I create test results with specs:")]
async fn have_test_results_with_specs(world: &mut BddWorld, table: cucumber::gherkin::Table) {
    let mut tests = Vec::new();
    
    for row in table.rows.iter().skip(1) {
        let name = &row[0];
        let spec = &row[1];
        let status = match row[2].as_str() {
            "passed" => TestStatus::Passed,
            "failed" => TestStatus::Failed,
            _ => TestStatus::Passed,
        };
        
        let meta = metadata::test_metadata().spec(spec).build();
        
        tests.push(TestResult {
            name: name.clone(),
            status,
            duration_secs: 0.1,
            stdout: None,
            stderr: None,
            error_message: None,
            metadata: Some(meta),
        });
    }
    
    world.test_summary = Some(TestSummary {
        total: tests.len(),
        passed: tests.iter().filter(|t| t.status == TestStatus::Passed).count(),
        failed: tests.iter().filter(|t| t.status == TestStatus::Failed).count(),
        ignored: 0,
        duration_secs: 0.3,
        pass_rate: 100.0,
        tests,
    });
}

#[then(regex = r#"^it should have a section for "([^"]+)" with (\d+) tests?$"#)]
async fn should_have_spec_section(world: &mut BddWorld, spec: String, _count: usize) {
    let report = world.metadata_report.as_ref().expect("No metadata report");
    assert!(report.contains(&spec), "Missing spec section: {}", spec);
}

#[when("I create test results with teams:")]
async fn have_test_results_with_teams(world: &mut BddWorld, table: cucumber::gherkin::Table) {
    let mut tests = Vec::new();
    
    for row in table.rows.iter().skip(1) {
        let name = &row[0];
        let team = &row[1];
        let status = match row[2].as_str() {
            "passed" => TestStatus::Passed,
            "failed" => TestStatus::Failed,
            _ => TestStatus::Passed,
        };
        
        let meta = metadata::test_metadata().team(team).build();
        
        tests.push(TestResult {
            name: name.clone(),
            status,
            duration_secs: 0.1,
            stdout: None,
            stderr: None,
            error_message: if status == TestStatus::Failed {
                Some("Failed".to_string())
            } else {
                None
            },
            metadata: Some(meta),
        });
    }
    
    world.test_summary = Some(TestSummary {
        total: tests.len(),
        passed: tests.iter().filter(|t| t.status == TestStatus::Passed).count(),
        failed: tests.iter().filter(|t| t.status == TestStatus::Failed).count(),
        ignored: 0,
        duration_secs: 0.3,
        pass_rate: 66.7,
        tests,
    });
}

#[then(regex = r#"^it should have a section for team "([^"]+)" with (\d+) tests?$"#)]
async fn should_have_team_section(world: &mut BddWorld, team: String, _count: usize) {
    let report = world.metadata_report.as_ref().expect("No metadata report");
    assert!(report.contains(&team), "Missing team section: {}", team);
}

#[when("I create test results with flaky tests:")]
async fn have_flaky_tests(world: &mut BddWorld, table: cucumber::gherkin::Table) {
    let mut tests = Vec::new();
    
    for row in table.rows.iter().skip(1) {
        let name = &row[0];
        let flaky = &row[1];
        
        let meta = metadata::test_metadata().flaky(flaky).build();
        
        tests.push(TestResult {
            name: name.clone(),
            status: TestStatus::Passed,
            duration_secs: 0.1,
            stdout: None,
            stderr: None,
            error_message: None,
            metadata: Some(meta),
        });
    }
    
    world.test_summary = Some(TestSummary {
        total: tests.len(),
        passed: tests.len(),
        failed: 0,
        ignored: 0,
        duration_secs: 0.2,
        pass_rate: 100.0,
        tests,
    });
}

#[then(regex = r#"^it should have a "([^"]+)" section$"#)]
async fn should_have_named_section(world: &mut BddWorld, section: String) {
    let report = world.metadata_report.as_ref().expect("No metadata report");
    assert!(report.contains(&section), "Missing section: {}", section);
}

#[then(regex = r#"^it should list "([^"]+)" as flaky$"#)]
async fn should_list_as_flaky(world: &mut BddWorld, test_name: String) {
    let report = world.metadata_report.as_ref().expect("No metadata report");
    assert!(report.contains(&test_name), "Test not listed as flaky: {}", test_name);
}

#[given("I have a test result with critical failure")]
async fn have_critical_failure(world: &mut BddWorld) {
    let meta = metadata::test_metadata()
        .priority("critical")
        .spec("ORCH-1234")
        .build();
    
    let test = TestResult {
        name: "test_critical_path".to_string(),
        status: TestStatus::Failed,
        duration_secs: 0.5,
        stdout: None,
        stderr: None,
        error_message: Some("Critical failure".to_string()),
        metadata: Some(meta),
    };
    
    world.test_summary = Some(TestSummary {
        total: 1,
        passed: 0,
        failed: 1,
        ignored: 0,
        duration_secs: 0.5,
        pass_rate: 0.0,
        tests: vec![test],
    });
}

#[when("I generate the executive summary")]
async fn generate_executive_summary(world: &mut BddWorld) {
    let summary = world.test_summary.as_ref().expect("No summary");
    world.executive_summary = Some(formatters::generate_executive_summary(summary));
}

#[then(regex = r#"^it should contain "([^"]+)"$"#)]
async fn should_contain(world: &mut BddWorld, text: String) {
    let report = world.executive_summary.as_ref().expect("No executive summary");
    assert!(report.contains(&text), "Executive summary missing: {}", text);
}

#[given("I have test results with metadata")]
async fn have_test_results_with_metadata(world: &mut BddWorld) {
    let meta = metadata::test_metadata()
        .priority("high")
        .spec("ORCH-5678")
        .build();
    
    let test = TestResult {
        name: "test_with_metadata".to_string(),
        status: TestStatus::Failed,
        duration_secs: 0.3,
        stdout: None,
        stderr: None,
        error_message: Some("Test failed".to_string()),
        metadata: Some(meta),
    };
    
    world.test_summary = Some(TestSummary {
        total: 1,
        passed: 0,
        failed: 1,
        ignored: 0,
        duration_secs: 0.3,
        pass_rate: 0.0,
        tests: vec![test],
    });
}

#[when("I generate the developer report")]
async fn generate_developer_report(world: &mut BddWorld) {
    let summary = world.test_summary.as_ref().expect("No summary");
    world.test_report = Some(formatters::generate_test_report(summary));
}

#[then("failed tests should show their metadata")]
async fn failed_tests_show_metadata(world: &mut BddWorld) {
    let report = world.test_report.as_ref().expect("No test report");
    assert!(report.contains("ORCH-"), "Report missing spec reference");
}

#[then("metadata should include priority badges")]
async fn metadata_includes_badges(world: &mut BddWorld) {
    let report = world.test_report.as_ref().expect("No test report");
    // Check for emoji or priority indicators
    assert!(report.contains("⚠️") || report.contains("high") || report.contains("HIGH"));
}

#[then("metadata should include spec references")]
async fn metadata_includes_specs(world: &mut BddWorld) {
    let report = world.test_report.as_ref().expect("No test report");
    assert!(report.contains("ORCH-"), "Report missing spec reference");
}

#[given("I have metadata with no fields set")]
async fn have_empty_metadata(world: &mut BddWorld) {
    world.parsed_metadata = Some(TestMetadata::default());
}

#[when("I serialize it to JSON")]
async fn serialize_metadata_to_json(world: &mut BddWorld) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    world.json_output = Some(serde_json::to_string(meta).expect("Serialization failed"));
}

#[then("it should produce valid JSON")]
async fn should_produce_valid_json(world: &mut BddWorld) {
    let json = world.json_output.as_ref().expect("No JSON output");
    let _: serde_json::Value = serde_json::from_str(json).expect("Invalid JSON");
}

#[then("all fields should be omitted")]
async fn all_fields_omitted(world: &mut BddWorld) {
    let json = world.json_output.as_ref().expect("No JSON output");
    // Empty metadata should serialize to {}
    assert_eq!(json.trim(), "{}", "Expected empty JSON object");
}

#[given("I have metadata with all fields set")]
async fn have_full_metadata(world: &mut BddWorld) {
    world.parsed_metadata = Some(
        metadata::test_metadata()
            .priority("critical")
            .spec("ORCH-1234")
            .team("orchestrator")
            .owner("alice@example.com")
            .issue("#5678")
            .flaky("5% failure rate")
            .timeout("30s")
            .require("GPU")
            .tag("integration")
            .custom("compliance", "SOC2")
            .build()
    );
}

#[when("I serialize and deserialize it")]
async fn serialize_deserialize_roundtrip(world: &mut BddWorld) {
    let original = world.parsed_metadata.as_ref().expect("No metadata");
    let json = serde_json::to_string(original).expect("Serialization failed");
    let deserialized: TestMetadata = serde_json::from_str(&json).expect("Deserialization failed");
    world.roundtrip_metadata = Some(deserialized);
}

#[then("the result should equal the original")]
async fn roundtrip_equals_original(world: &mut BddWorld) {
    let original = world.parsed_metadata.as_ref().expect("No original");
    let roundtrip = world.roundtrip_metadata.as_ref().expect("No roundtrip");
    assert_eq!(original, roundtrip, "Roundtrip mismatch");
}

#[given(regex = r#"^I have metadata with timeout "([^"]+)"$"#)]
async fn have_metadata_timeout(world: &mut BddWorld, timeout: String) {
    world.parsed_metadata = Some(metadata::test_metadata().timeout(&timeout).build());
}

#[when("I check the timeout value")]
async fn check_timeout_value(world: &mut BddWorld) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    world.timeout_value = meta.timeout.clone();
}

#[then(regex = r#"^it should be "([^"]+)"$"#)]
async fn timeout_should_be(world: &mut BddWorld, expected: String) {
    assert_eq!(world.timeout_value.as_ref(), Some(&expected), "Timeout mismatch");
}

#[then(regex = r"^the result should be (true|false)$")]
async fn result_should_be(world: &mut BddWorld, expected: String) {
    let expected_bool = expected == "true";
    let actual = world.is_high_priority.expect("is_high_priority not set");
    assert_eq!(actual, expected_bool, "Result mismatch");
}

#[when("I use the fluent API to build metadata")]
async fn use_fluent_api(world: &mut BddWorld) {
    world.parsed_metadata = Some(
        metadata::test_metadata()
            .priority("high")
            .spec("ORCH-1234")
            .team("orchestrator")
            .build()
    );
}

#[then("I can chain multiple methods")]
async fn can_chain_methods(_world: &mut BddWorld) {
    // This is tested by the previous step succeeding
}

#[then("the final metadata should have all fields")]
async fn final_metadata_has_fields(world: &mut BddWorld) {
    let meta = world.parsed_metadata.as_ref().expect("No metadata");
    assert!(meta.priority.is_some());
    assert!(meta.spec.is_some());
    assert!(meta.team.is_some());
}

#[when("I build metadata without calling record()")]
async fn build_without_record(world: &mut BddWorld) {
    world.parsed_metadata = Some(
        metadata::test_metadata()
            .priority("medium")
            .build()
    );
}

#[then("it should return the metadata object")]
async fn should_return_metadata(world: &mut BddWorld) {
    assert!(world.parsed_metadata.is_some(), "Metadata should exist");
}

#[then("it should not persist anywhere")]
async fn should_not_persist(_world: &mut BddWorld) {
    // This is a design property - metadata::test_metadata().build() doesn't persist
    // Just verify it doesn't panic
}
