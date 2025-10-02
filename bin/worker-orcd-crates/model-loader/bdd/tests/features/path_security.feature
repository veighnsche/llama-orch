Feature: Path Security
  As a security-conscious system
  I want to prevent path traversal attacks
  So that arbitrary file access is prevented

  Scenario: Reject path traversal sequence
    Given a model file with path traversal sequence
    When I attempt to load the model
    Then the load fails with path validation error

  Scenario: Reject symlink escape
    Given a symlink pointing outside allowed directory
    When I attempt to load the model
    Then the load fails with path validation error

  Scenario: Reject null byte in path
    Given a model path with null byte
    When I attempt to load the model
    Then the load fails with path validation error

  Scenario: Accept valid path within allowed directory
    Given a valid model file in allowed directory
    When I load and validate the model
    Then the model loads successfully
