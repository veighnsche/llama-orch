Feature: GGUF Format Validation
  As a security-conscious system
  I want to validate GGUF format before loading
  So that malformed models are rejected

  Scenario: Load valid GGUF file
    Given a valid GGUF model file
    When I load and validate the model
    Then the model loads successfully
    And the loaded bytes match the file contents

  Scenario: Reject invalid magic number
    Given a model file with invalid magic number
    When I load and validate the model
    Then the load fails with invalid format

  Scenario: Validate valid GGUF bytes in memory
    Given valid GGUF bytes in memory
    When I validate the bytes in memory
    Then the validation succeeds

  Scenario: Reject invalid GGUF bytes in memory
    Given invalid GGUF bytes in memory
    When I validate the bytes in memory
    Then the validation fails with invalid format
