Feature: Resource Operation Event Validation
  As a security-conscious audit system
  I want to validate resource operation events
  So that I prevent injection attacks and ensure data integrity

  Scenario: Accept valid PoolCreated event
    Given a user ID "admin@example.com"
    And a pool ID "pool-abc123"
    And a model reference "meta-llama/Llama-3.1-8B"
    And a node ID "node-gpu-0"
    When I create a PoolCreated event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject PoolCreated with path traversal in pool ID
    Given a user ID "admin@example.com"
    And a pool ID "pool-../../../etc/passwd"
    And a model reference "meta-llama/Llama-3.1-8B"
    And a node ID "node-gpu-0"
    When I create a PoolCreated event
    And I validate the event
    Then the validation should fail

  Scenario: Reject PoolCreated with ANSI escape in model reference
    Given a user ID "admin@example.com"
    And a pool ID "pool-123"
    And a model reference "\x1b[31mmalicious-model\x1b[0m"
    And a node ID "node-gpu-0"
    When I create a PoolCreated event
    And I validate the event
    Then the validation should reject ANSI escape sequences

  Scenario: Accept valid PoolDeleted event
    Given a user ID "admin@example.com"
    And a pool ID "pool-abc123"
    And a model reference "meta-llama/Llama-3.1-8B"
    And a node ID "node-gpu-0"
    And a reason "user_requested"
    When I create a PoolDeleted event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject PoolDeleted with control characters in reason
    Given a user ID "admin@example.com"
    And a pool ID "pool-123"
    And a model reference "meta-llama/Llama-3.1-8B"
    And a node ID "node-gpu-0"
    And a reason "deleted\r\n[CRITICAL] Unauthorized access"
    When I create a PoolDeleted event
    And I validate the event
    Then the validation should reject control characters

  Scenario: Accept valid TaskSubmitted event
    Given a user ID "user@example.com"
    And a task ID "task-xyz789"
    And a model reference "meta-llama/Llama-3.1-8B"
    When I create a TaskSubmitted event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject TaskSubmitted with null byte in task ID
    Given a user ID "user@example.com"
    And a task ID "task-123\0malicious"
    And a model reference "meta-llama/Llama-3.1-8B"
    When I create a TaskSubmitted event
    And I validate the event
    Then the validation should reject null bytes
