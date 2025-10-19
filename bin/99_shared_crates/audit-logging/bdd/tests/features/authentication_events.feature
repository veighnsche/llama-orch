Feature: Authentication Event Validation
  As a security-conscious audit system
  I want to validate authentication events
  So that I prevent log injection and ensure data integrity

  Scenario: Accept valid AuthSuccess event
    Given a user ID "admin@example.com"
    And a path "/v2/tasks"
    When I create an AuthSuccess event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject AuthSuccess with ANSI escape in user ID
    Given a user ID "\x1b[31mFAKE ERROR\x1b[0m"
    And a path "/v2/tasks"
    When I create an AuthSuccess event
    And I validate the event
    Then the validation should reject ANSI escape sequences

  Scenario: Reject AuthSuccess with control characters in path
    Given a user ID "admin@example.com"
    And a path "/v2/tasks\r\n[CRITICAL] Fake log"
    When I create an AuthSuccess event
    And I validate the event
    Then the validation should reject control characters

  Scenario: Reject AuthFailure with null byte in user ID
    Given a user ID "admin\0malicious"
    And an IP address "192.168.1.1"
    And a path "/v2/auth"
    And a reason "invalid_credentials"
    When I create an AuthFailure event
    And I validate the event
    Then the validation should reject null bytes

  Scenario: Accept valid AuthFailure event
    Given a user ID "attacker@evil.com"
    And an IP address "10.0.0.1"
    And a path "/v2/auth"
    And a reason "invalid_token"
    When I create an AuthFailure event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject AuthFailure with log injection in reason
    Given a user ID "user@example.com"
    And an IP address "192.168.1.1"
    And a path "/v2/auth"
    And a reason "failed\n[ERROR] System compromised"
    When I create an AuthFailure event
    And I validate the event
    Then the validation should reject log injection
