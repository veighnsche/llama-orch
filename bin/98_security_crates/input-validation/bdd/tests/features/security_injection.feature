Feature: Security Injection Prevention
  As a security-conscious system
  I want to prevent injection attacks
  So that I protect against SQL, command, and log injection

  Scenario: Reject SQL injection in model reference
    Given a model reference "'; DROP TABLE models; --"
    When I validate the model reference
    Then the validation should fail
    And the validation should reject SQL injection

  Scenario: Reject command injection with semicolon
    Given a model reference "model; rm -rf /"
    When I validate the model reference
    Then the validation should fail
    And the validation should reject command injection

  Scenario: Reject command injection with pipe
    Given a model reference "model | cat /etc/passwd"
    When I validate the model reference
    Then the validation should fail
    And the validation should reject command injection

  Scenario: Reject command injection with ampersand
    Given a model reference "model && ls"
    When I validate the model reference
    Then the validation should fail
    And the validation should reject command injection

  Scenario: Reject log injection with newline
    Given a model reference "model\n[ERROR] Fake log entry"
    When I validate the model reference
    Then the validation should fail
    And the validation should reject log injection

  Scenario: Reject log injection with carriage return
    Given a model reference "model\r\nFake log"
    When I validate the model reference
    Then the validation should fail
    And the validation should reject log injection

  Scenario: Reject ANSI escape injection
    Given a string "text\x1b[31mRED"
    When I sanitize the string
    Then the validation should fail
    And the error should be "AnsiEscape"

  Scenario: Reject path traversal in model reference
    Given a model reference "file:../../../../etc/passwd"
    When I validate the model reference
    Then the validation should fail
    And the validation should reject path traversal

  Scenario: Accept safe model reference
    Given a model reference "meta-llama/Llama-3.1-8B"
    When I validate the model reference
    Then the validation should succeed

  Scenario: Accept safe model reference with colon
    Given a model reference "hf:org/repo"
    When I validate the model reference
    Then the validation should succeed
