Feature: Field Validation and Length Limits
  As a security-conscious audit system
  I want to enforce field validation rules
  So that I prevent DoS attacks and maintain data integrity

  Scenario: Accept field at maximum length
    Given a user ID with 1024 characters
    And a path "/v2/tasks"
    When I create an AuthSuccess event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject field exceeding maximum length
    Given a user ID with 2000 characters
    And a path "/v2/tasks"
    When I create an AuthSuccess event
    And I validate the event
    Then the validation should reject oversized fields

  Scenario: Accept valid session ID
    Given a user ID "user@example.com"
    And a session ID "session-abc123"
    And a path "/v2/tasks"
    When I create an AuthSuccess event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject session ID with ANSI escapes
    Given a user ID "user@example.com"
    And a session ID "session\x1b[31m-abc123"
    And a path "/v2/tasks"
    When I create an AuthSuccess event
    And I validate the event
    Then the validation should reject ANSI escape sequences

  Scenario: Accept newlines in structured fields
    Given a user ID "user@example.com"
    And a path "/v2/tasks"
    And a details string "line1\nline2"
    When I create a PolicyViolation event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject Unicode directional overrides
    Given a user ID "user\u{202E}evil\u{202D}@example.com"
    And a path "/v2/tasks"
    When I create an AuthSuccess event
    And I validate the event
    Then the validation should reject Unicode directional overrides
