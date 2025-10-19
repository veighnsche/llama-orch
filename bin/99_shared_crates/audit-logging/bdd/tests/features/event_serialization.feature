Feature: Event Serialization
  As an audit system
  I want to serialize events to JSON
  So that I can store them in audit logs

  Scenario: Serialize valid AuthSuccess event
    Given a user ID "admin@example.com"
    And a path "/v2/tasks"
    When I create an AuthSuccess event
    And I validate the event
    And I serialize the event to JSON
    Then the event should be serializable

  Scenario: Serialize valid PoolCreated event
    Given a user ID "admin@example.com"
    And a pool ID "pool-abc123"
    And a model reference "meta-llama/Llama-3.1-8B"
    And a node ID "node-gpu-0"
    When I create a PoolCreated event
    And I validate the event
    And I serialize the event to JSON
    Then the event should be serializable

  Scenario: Serialize valid VramSealed event
    Given a shard ID "shard-abc123"
    And a worker ID "worker-gpu-0"
    When I create a VramSealed event
    And I validate the event
    And I serialize the event to JSON
    Then the event should be serializable

  Scenario: Sanitized data should be serializable
    Given a user ID "admin@example.com"
    And a path "/v2/tasks"
    When I create an AuthSuccess event
    And I validate the event
    Then the event should contain sanitized data
    When I serialize the event to JSON
    Then the event should be serializable
