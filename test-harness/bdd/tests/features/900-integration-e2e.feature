# Integration End-to-End Tests
# Created by: TEAM-083
# Priority: P0 - Critical for production readiness
#
# These tests verify complete workflows across multiple components

Feature: End-to-End Integration Tests
  As a system integrator
  I want to test complete workflows
  So that I verify all components work together

  @integration @e2e
  Scenario: Complete inference workflow
    Given queen-rbee is running
    And rbee-hive is running on workstation
    And worker-001 is registered with model "tinyllama-q4"
    When client sends inference request via queen-rbee
    Then queen-rbee routes to worker-001
    And worker-001 processes the request
    And tokens are streamed back to client
    And worker returns to idle state
    And metrics are recorded

  @integration @e2e
  Scenario: Worker failover
    Given queen-rbee is running
    And worker-001 is processing request "req-001"
    And worker-002 is available with same model
    When worker-001 crashes unexpectedly
    Then queen-rbee detects crash within 5 seconds
    And request "req-001" can be retried on worker-002
    And user receives result without data loss

  @integration @e2e
  Scenario: Model download and registration
    Given rbee-hive is running
    And model "tinyllama-q4" is not in catalog
    When rbee-hive downloads model from HuggingFace
    Then download completes successfully
    And model is registered in catalog
    And model is available for worker startup

  @integration @e2e
  Scenario: Concurrent worker registration
    Given queen-rbee is running
    When 3 rbee-hive instances register workers simultaneously
    Then all 3 workers are registered
    And each worker has unique ID
    And registry state is consistent

  @integration @e2e
  Scenario: SSE streaming with backpressure
    Given worker-001 is processing inference
    When tokens are generated faster than network can send
    Then SSE stream applies backpressure
    And no tokens are lost
    And client receives all tokens in order
