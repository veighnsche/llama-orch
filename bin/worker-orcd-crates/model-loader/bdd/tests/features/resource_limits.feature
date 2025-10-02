Feature: Resource Limits
  As a security-conscious system
  I want to enforce resource limits on model files
  So that resource exhaustion attacks are prevented

  Scenario: Reject file exceeding max size
    Given a model file that is too large
    When I load the model with max size 1000 bytes
    Then the load fails with file too large

  Scenario: Reject excessive tensor count
    Given a GGUF file with 100000 tensors
    When I validate the bytes in memory
    Then the validation fails with tensor count exceeded

  Scenario: Accept valid tensor count
    Given a GGUF file with 100 tensors
    When I validate the bytes in memory
    Then the validation succeeds

  Scenario: Reject oversized string
    Given a GGUF file with oversized string
    When I validate the bytes in memory
    Then the validation fails with string too long

  Scenario: Reject excessive metadata pairs
    Given a GGUF file with 10000 metadata pairs
    When I validate the bytes in memory
    Then the validation fails with invalid format
