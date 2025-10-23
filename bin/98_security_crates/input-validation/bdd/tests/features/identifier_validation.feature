Feature: Identifier Validation
  As a security-conscious system
  I want to validate identifiers (shard_id, task_id, pool_id)
  So that I prevent injection attacks and path traversal

  Background:
    Given a max length of 256

  Scenario: Accept valid identifier with alphanumeric and dash
    Given an identifier "shard-abc123"
    When I validate the identifier
    Then the validation should succeed

  Scenario: Accept valid identifier with underscore
    Given an identifier "task_gpu0"
    When I validate the identifier
    Then the validation should succeed

  Scenario: Reject empty identifier
    Given an identifier ""
    When I validate the identifier
    Then the validation should fail
    And the error should be "Empty"

  Scenario: Reject identifier that is too long
    Given an identifier with 257 characters
    And a max length of 256
    When I validate the identifier
    Then the validation should fail
    And the error should be "TooLong"

  Scenario: Reject identifier with null byte
    Given an identifier "shard\0null"
    When I validate the identifier
    Then the validation should fail
    And the error should be "NullByte"

  Scenario: Reject identifier with path traversal (../)
    Given an identifier "shard-../etc/passwd"
    When I validate the identifier
    Then the validation should fail
    And the validation should reject path traversal

  Scenario: Reject identifier with path traversal (./)
    Given an identifier "shard-./config"
    When I validate the identifier
    Then the validation should fail
    And the validation should reject path traversal

  Scenario: Reject identifier with invalid characters
    Given an identifier "shard@123"
    When I validate the identifier
    Then the validation should fail
    And the error should be "InvalidCharacters"

  Scenario: Reject identifier with spaces
    Given an identifier "shard 123"
    When I validate the identifier
    Then the validation should fail
    And the error should be "InvalidCharacters"

  Scenario: Accept identifier at exact length limit
    Given an identifier with 256 characters
    And a max length of 256
    When I validate the identifier
    Then the validation should succeed
