Feature: Compliance Event Validation (GDPR)
  As a security-conscious audit system
  I want to validate GDPR compliance events
  So that I maintain regulatory compliance and prevent injection attacks

  Scenario: Accept valid GdprDataAccessRequest event
    Given a customer ID "customer-123"
    And a requester "user@example.com"
    When I create a GdprDataAccessRequest event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject GdprDataAccessRequest with ANSI escape in requester
    Given a customer ID "customer-123"
    And a requester "user\x1b[31m@example.com"
    When I create a GdprDataAccessRequest event
    And I validate the event
    Then the validation should reject ANSI escape sequences

  Scenario: Accept valid GdprDataExport event
    Given a customer ID "customer-456"
    And an export format "json"
    When I create a GdprDataExport event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject GdprDataExport with null byte in export format
    Given a customer ID "customer-456"
    And an export format "json\0malicious"
    When I create a GdprDataExport event
    And I validate the event
    Then the validation should reject null bytes

  Scenario: Accept valid GdprRightToErasure event
    Given a customer ID "customer-789"
    When I create a GdprRightToErasure event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject GdprRightToErasure with control characters in customer ID
    Given a customer ID "customer-789\r\nFAKE"
    When I create a GdprRightToErasure event
    And I validate the event
    Then the validation should reject control characters
