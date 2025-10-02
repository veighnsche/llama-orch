Feature: Path Security
  As a security-conscious system
  I want to prevent path traversal attacks
  So that arbitrary file access is prevented

  # TODO(M0): Enable once input-validation is integrated
  @skip
  Scenario: Reject path traversal sequence
    Given a model file with path traversal sequence
    When I load and validate the model
    Then the load fails with path validation error

  # TODO(M0): Add symlink tests
  # @skip
  # Scenario: Reject symlink escape
  #   Given a symlink pointing outside allowed directory
  #   When I load and validate the model
  #   Then the load fails with path validation error

  # TODO(M0): Add null byte tests
  # @skip
  # Scenario: Reject null byte in path
  #   Given a model path with null byte
  #   When I load and validate the model
  #   Then the load fails with path validation error
