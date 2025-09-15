Feature: Lifecycle states
  # Traceability: ORCH-3069..3074
  Scenario: Deprecate blocks sessions
    Given a Control Plane API endpoint
    When I set model state Deprecated with deadline_ms
    Then new sessions are blocked with MODEL_DEPRECATED
    And model_state gauge is exported

  Scenario: Retire unloads pools
    Given a Control Plane API endpoint
    When I set model state Retired
    Then pools unload and archives retained
