Feature: Token Event Validation
  As a security-conscious audit system
  I want to validate token lifecycle events
  So that I prevent injection attacks in token management logs

  Scenario: Accept valid TokenCreated event
    Given a user ID "admin@example.com"
    And a token fingerprint "abc123def456"
    When I create a TokenCreated event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject TokenCreated with ANSI escape in token fingerprint
    Given a user ID "admin@example.com"
    And a token fingerprint "abc\x1b[31m123"
    When I create a TokenCreated event
    And I validate the event
    Then the validation should reject ANSI escape sequences

  Scenario: Accept valid TokenRevoked event
    Given a user ID "admin@example.com"
    And a token fingerprint "abc123def456"
    And a reason "expired"
    When I create a TokenRevoked event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject TokenRevoked with null byte in reason
    Given a user ID "admin@example.com"
    And a token fingerprint "abc123def456"
    And a reason "revoked\0malicious"
    When I create a TokenRevoked event
    And I validate the event
    Then the validation should reject null bytes

  Scenario: Reject TokenRevoked with control characters in reason
    Given a user ID "admin@example.com"
    And a token fingerprint "abc123def456"
    And a reason "revoked\r\n[ERROR] Fake"
    When I create a TokenRevoked event
    And I validate the event
    Then the validation should reject control characters
