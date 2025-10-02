Feature: V2 One-Liner API
  As a developer
  I want to generate proof bundles with one function call
  So that I don't have to write boilerplate

  Background:
    Given the proof-bundle V2 API is available

  Scenario: Generate proof bundle with UnitFast mode
    When I call generate_for_crate with package "proof-bundle" and mode "UnitFast"
    Then the API should succeed
    And the summary should have at least 1 test
    And the summary should have a pass rate above 90%
    And all 7 files should be generated

  Scenario: Parse real cargo test output
    Given I have real cargo test output with 38 tests
    When I parse it with the stable parser
    Then I should get 38 test results
    And all tests should have names
    And the pass rate should be 100%

  Scenario: Generate all 4 reports
    Given I have a test summary with real data
    When I generate all reports
    Then the executive summary should contain "Test Results Summary"
    And the test report should contain "Test Report"
    And the failure report should contain "Failure Report"
    And the metadata report should contain "Test Metadata Report"

  Scenario: Detect critical test failures
    Given I have a test summary with a critical failure
    When I generate the executive summary
    Then it should contain "CRITICAL ALERT"
    And it should contain "NOT APPROVED"
    And it should contain "CRITICAL RISK"

  Scenario: Handle metadata in reports
    Given I have tests with metadata annotations
    When I generate the metadata report
    Then it should group tests by priority
    And it should group tests by spec
    And it should show flaky tests separately

  Scenario: Validate golden files
    Given I have the golden cargo output file
    When I parse it
    Then it should match the expected test count
    And it should extract all test names correctly
