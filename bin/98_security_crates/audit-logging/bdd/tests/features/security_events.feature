Feature: Security Incident Event Validation
  As a security-conscious audit system
  I want to validate security incident events
  So that I prevent injection attacks even in security event logs

  Scenario: Accept valid PathTraversalAttempt event
    Given a user ID "attacker@evil.com"
    And a path "../../../etc/passwd"
    And an endpoint "/v2/files"
    When I create a PathTraversalAttempt event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject PathTraversalAttempt with ANSI escape in endpoint
    Given a user ID "attacker@evil.com"
    And a path "../etc/passwd"
    And an endpoint "/v2/files\x1b[31mFAKE\x1b[0m"
    When I create a PathTraversalAttempt event
    And I validate the event
    Then the validation should reject ANSI escape sequences

  Scenario: Accept valid PolicyViolation event
    Given a worker ID "worker-gpu-0"
    And a details string "Attempted to access restricted VRAM region"
    When I create a PolicyViolation event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject PolicyViolation with control characters in details
    Given a worker ID "worker-gpu-0"
    And a details string "violation\r\n[ERROR] System compromised"
    When I create a PolicyViolation event
    And I validate the event
    Then the validation should reject control characters

  Scenario: Reject PolicyViolation with null byte in worker ID
    Given a worker ID "worker-gpu-0\0malicious"
    And a details string "Attempted unauthorized access"
    When I create a PolicyViolation event
    And I validate the event
    Then the validation should reject null bytes
