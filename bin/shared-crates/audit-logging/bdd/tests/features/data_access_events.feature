Feature: Data Access Event Validation (GDPR)
  As a security-conscious audit system
  I want to validate data access events
  So that I maintain GDPR compliance and prevent injection attacks

  Scenario: Accept valid InferenceExecuted event
    Given a customer ID "customer-123"
    And a job ID "job-abc789"
    And a model reference "meta-llama/Llama-3.1-8B"
    When I create an InferenceExecuted event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject InferenceExecuted with ANSI escape in customer ID
    Given a customer ID "customer-123\x1b[0m"
    And a job ID "job-abc789"
    And a model reference "meta-llama/Llama-3.1-8B"
    When I create an InferenceExecuted event
    And I validate the event
    Then the validation should reject ANSI escape sequences

  Scenario: Accept valid ModelAccessed event
    Given a customer ID "customer-456"
    And a model reference "meta-llama/Llama-3.1-70B"
    And an access type "inference"
    When I create a ModelAccessed event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject ModelAccessed with null byte in access type
    Given a customer ID "customer-456"
    And a model reference "meta-llama/Llama-3.1-70B"
    And an access type "inference\0malicious"
    When I create a ModelAccessed event
    And I validate the event
    Then the validation should reject null bytes

  Scenario: Accept valid DataDeleted event
    Given a customer ID "customer-789"
    And a reason "user_requested"
    When I create a DataDeleted event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject DataDeleted with control characters in reason
    Given a customer ID "customer-789"
    And a reason "deleted\r\n[CRITICAL] Fake"
    When I create a DataDeleted event
    And I validate the event
    Then the validation should reject control characters
