Feature: Node Event Validation
  As a security-conscious audit system
  I want to validate node lifecycle events
  So that I prevent injection attacks in node management logs

  Scenario: Accept valid NodeRegistered event
    Given a user ID "admin@example.com"
    And a node ID "node-gpu-0"
    When I create a NodeRegistered event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject NodeRegistered with ANSI escape in node ID
    Given a user ID "admin@example.com"
    And a node ID "node-gpu-0\x1b[31mFAKE\x1b[0m"
    When I create a NodeRegistered event
    And I validate the event
    Then the validation should reject ANSI escape sequences

  Scenario: Accept valid NodeDeregistered event
    Given a user ID "admin@example.com"
    And a node ID "node-gpu-1"
    And a reason "maintenance"
    When I create a NodeDeregistered event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject NodeDeregistered with null byte in node ID
    Given a user ID "admin@example.com"
    And a node ID "node-gpu-1\0malicious"
    And a reason "maintenance"
    When I create a NodeDeregistered event
    And I validate the event
    Then the validation should reject null bytes

  Scenario: Reject NodeDeregistered with control characters in reason
    Given a user ID "admin@example.com"
    And a node ID "node-gpu-1"
    And a reason "removed\r\n[CRITICAL] Fake"
    When I create a NodeDeregistered event
    And I validate the event
    Then the validation should reject control characters
