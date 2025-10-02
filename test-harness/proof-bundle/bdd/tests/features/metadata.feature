Feature: Test Metadata Annotations (PB-004)
  As a developer
  I want to annotate tests with metadata
  So that proof bundles show business context and priorities

  Background:
    Given the metadata system is available

  Scenario: Parse doc comment annotations
    Given I have a doc comment "@priority: critical @spec: ORCH-3250 @team: orchestrator @owner: alice@example.com"
    When I parse the doc comments
    Then the metadata should have priority "critical"
    And the metadata should have spec "ORCH-3250"
    And the metadata should have team "orchestrator"
    And the metadata should have owner "alice@example.com"

  Scenario: Build metadata programmatically
    When I build metadata with:
      | field    | value              |
      | priority | high               |
      | spec     | ORCH-1234          |
      | team     | pool-manager       |
      | issue    | #5678              |
    Then the metadata should have priority "high"
    And the metadata should have spec "ORCH-1234"
    And the metadata should have team "pool-manager"
    And the metadata should have issue "#5678"

  Scenario: Detect critical tests
    Given I have metadata with priority "critical"
    When I check if the test is critical
    Then it should be marked as critical

  Scenario: Detect high priority tests
    Given I have metadata with priority "high"
    When I check if the test is high priority
    Then it should be marked as high priority

  Scenario: Detect flaky tests
    Given I have metadata with flaky "5% failure rate"
    When I check if the test is flaky
    Then it should be marked as flaky

  Scenario: Parse custom fields
    Given I have a doc comment "@priority: medium @custom:compliance: SOC2 @custom:environment: staging"
    When I parse the doc comments
    Then the metadata should have custom field "compliance" with value "SOC2"
    And the metadata should have custom field "environment" with value "staging"

  Scenario: Parse requires list
    Given I have a doc comment "@requires: GPU @requires: CUDA @requires: 16GB VRAM"
    When I parse the doc comments
    Then the metadata should have 3 requirements
    And the metadata should require "GPU"
    And the metadata should require "CUDA"
    And the metadata should require "16GB VRAM"

  Scenario: Parse tags list
    Given I have a doc comment "@tags: integration @tags: slow @tags: gpu-required"
    When I parse the doc comments
    Then the metadata should have 3 tags
    And the metadata should have tag "integration"
    And the metadata should have tag "slow"
    And the metadata should have tag "gpu-required"

  Scenario: Metadata in test results
    Given I have a test result with metadata
    When I serialize the test result to JSON
    Then the JSON should contain the metadata
    And the metadata should be deserializable

  Scenario: Metadata report groups by priority
    When I create test results with mixed priorities:
      | name              | priority | status |
      | test_critical_1   | critical | passed |
      | test_critical_2   | critical | failed |
      | test_high_1       | high     | passed |
      | test_medium_1     | medium   | passed |
      | test_low_1        | low      | passed |
    And I generate the metadata report
    Then it should have a "Critical" section with 2 tests
    And it should have a "High" section with 1 test
    And it should have a "Medium" section with 1 test
    And it should have a "Low" section with 1 test

  Scenario: Metadata report groups by spec
    When I create test results with specs:
      | name        | spec       | status |
      | test_1      | ORCH-1234  | passed |
      | test_2      | ORCH-1234  | passed |
      | test_3      | ORCH-5678  | passed |
    And I generate the metadata report
    Then it should have a section for "ORCH-1234" with 2 tests
    And it should have a section for "ORCH-5678" with 1 test

  Scenario: Metadata report groups by team
    When I create test results with teams:
      | name        | team          | status |
      | test_1      | orchestrator  | passed |
      | test_2      | orchestrator  | failed |
      | test_3      | pool-manager  | passed |
    And I generate the metadata report
    Then it should have a section for team "orchestrator" with 2 tests
    And it should have a section for team "pool-manager" with 1 test

  Scenario: Metadata report shows flaky tests
    When I create test results with flaky tests:
      | name        | flaky                | status |
      | test_1      | 5% failure rate      | passed |
      | test_2      | intermittent timeout | passed |
    And I generate the metadata report
    Then it should have a "Known Flaky Tests" section
    And it should list "test_1" as flaky
    And it should list "test_2" as flaky

  Scenario: Critical failure alert in executive summary
    Given I have a test result with critical failure
    When I generate the executive summary
    Then it should contain "🚨 CRITICAL ALERT"
    And it should contain "CRITICAL RISK"
    And it should contain "NOT APPROVED"

  Scenario: Metadata in developer report
    Given I have test results with metadata
    When I generate the developer report
    Then failed tests should show their metadata
    And metadata should include priority badges
    And metadata should include spec references

  Scenario: Empty metadata is valid
    Given I have metadata with no fields set
    When I serialize it to JSON
    Then it should produce valid JSON
    And all fields should be omitted

  Scenario: Metadata serialization round-trip
    Given I have metadata with all fields set
    When I serialize and deserialize it
    Then the result should equal the original

  Scenario: Timeout metadata
    Given I have metadata with timeout "30s"
    When I check the timeout value
    Then it should be "30s"

  Scenario: Multiple owners not supported
    Given I have a doc comment "@owner: alice@example.com"
    When I parse the doc comments
    Then the metadata should have owner "alice@example.com"

  Scenario: Case-insensitive priority detection
    Given I have metadata with priority "CRITICAL"
    When I check if the test is critical
    Then it should be marked as critical

  Scenario Outline: Priority levels
    Given I have metadata with priority "<priority>"
    When I check if the test is high priority
    Then the result should be <is_high>

    Examples:
      | priority | is_high |
      | critical | true    |
      | high     | true    |
      | medium   | false   |
      | low      | false   |

  Scenario: Metadata builder fluent API
    When I use the fluent API to build metadata
    Then I can chain multiple methods
    And the final metadata should have all fields

  Scenario: Metadata without recording
    When I build metadata without calling record()
    Then it should return the metadata object
    And it should not persist anywhere
