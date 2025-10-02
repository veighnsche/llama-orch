Feature: Resource Limits
  As a security-conscious system
  I want to enforce resource limits on model files
  So that resource exhaustion attacks are prevented

  Scenario: Reject file exceeding max size
    Given a model file that is too large
    When I load the model with max size 1000 bytes
    Then the load fails with file too large

  # TODO(M0): Add tensor count limit tests
  # Scenario: Reject excessive tensor count
  #   Given a GGUF file with 100000 tensors
  #   When I load and validate the model
  #   Then the load fails with tensor count exceeded

  # TODO(M0): Add string length limit tests
  # Scenario: Reject oversized strings
  #   Given a GGUF file with 1MB string
  #   When I load and validate the model
  #   Then the load fails with string too long
