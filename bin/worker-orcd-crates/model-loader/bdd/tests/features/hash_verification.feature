Feature: Hash Verification
  As a security-conscious system
  I want to verify model integrity via SHA-256 hashes
  So that poisoned models are rejected

  Background:
    Given a valid GGUF model file

  Scenario: Load model with correct hash
    Given a GGUF model file with hash "computed"
    When I load the model with hash verification
    Then the model loads successfully
    And the loaded bytes match the file contents

  Scenario: Load model with wrong hash
    When I load the model with wrong hash
    Then the load fails with hash mismatch

  Scenario: Load model without hash verification
    When I load and validate the model
    Then the model loads successfully

  Scenario: Reject invalid hash format (too short)
    Given valid GGUF bytes in memory
    When I validate with hash "abc123"
    Then the validation fails with invalid format

  Scenario: Reject invalid hash format (non-hex)
    Given valid GGUF bytes in memory
    When I validate with hash "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
    Then the validation fails with invalid format

  Scenario: Accept valid hash format
    Given valid GGUF bytes in memory
    When I validate with computed hash
    Then the validation succeeds
