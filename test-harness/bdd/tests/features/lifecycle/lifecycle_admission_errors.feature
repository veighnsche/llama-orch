Feature: Lifecycle states with admission errors
  # Traceability: ORCH-3069..3073, ORCH-3093

  Scenario: Deprecated state blocks new sessions with error context
    Given a Control Plane API endpoint
    When I set model state Deprecated with deadline_ms
    Then new sessions are blocked with MODEL_DEPRECATED

  Scenario: Deprecated state blocks new sessions at enqueue
    Given a Control Plane API endpoint
    And an OrchQueue API endpoint
    When I enqueue a completion task with valid payload
    Then new sessions are blocked with MODEL_DEPRECATED
