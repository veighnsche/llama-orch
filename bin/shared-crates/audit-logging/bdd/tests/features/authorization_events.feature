Feature: Authorization Event Validation
  As a security-conscious audit system
  I want to validate authorization events
  So that I prevent injection attacks in authorization logs

  Scenario: Accept valid AuthorizationGranted event
    Given a user ID "admin@example.com"
    And a resource ID "pool-123"
    And a resource type "pool"
    And an action "create"
    When I create an AuthorizationGranted event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject AuthorizationGranted with ANSI escape in action
    Given a user ID "admin@example.com"
    And a resource ID "pool-123"
    And a resource type "pool"
    And an action "create\x1b[31mFAKE\x1b[0m"
    When I create an AuthorizationGranted event
    And I validate the event
    Then the validation should reject ANSI escape sequences

  Scenario: Accept valid AuthorizationDenied event
    Given a user ID "user@example.com"
    And a resource ID "pool-123"
    And a resource type "pool"
    And an action "delete"
    And a reason "insufficient_permissions"
    When I create an AuthorizationDenied event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject AuthorizationDenied with null byte in reason
    Given a user ID "user@example.com"
    And a resource ID "pool-123"
    And a resource type "pool"
    And an action "delete"
    And a reason "denied\0malicious"
    When I create an AuthorizationDenied event
    And I validate the event
    Then the validation should reject null bytes

  Scenario: Accept valid PermissionChanged event
    Given a user ID "admin@example.com"
    And a subject "user-456"
    When I create a PermissionChanged event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject PermissionChanged with control characters in subject
    Given a user ID "admin@example.com"
    And a subject "user-456\r\nFAKE"
    When I create a PermissionChanged event
    And I validate the event
    Then the validation should reject control characters
