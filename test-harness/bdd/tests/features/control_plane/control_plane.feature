Feature: Control Plane endpoints
  # Traceability: ORCH-2101..2104, OC-CTRL-2001..2004
  Scenario: Pool health shows status and metrics
    Given a Control Plane API endpoint
    And a pool id
    When I request pool health
    Then I receive 200 with liveness readiness draining and metrics

  Scenario: Pool drain starts
    Given a Control Plane API endpoint
    And a pool id
    When I request pool drain with deadline_ms
    Then draining begins

  Scenario: Pool reload is atomic success
    Given a Control Plane API endpoint
    And a pool id
    When I request pool reload with new model_ref
    Then reload succeeds and is atomic

  Scenario: Pool reload fails and rolls back
    Given a Control Plane API endpoint
    And a pool id
    When I request pool reload with new model_ref
    Then reload fails and rolls back atomically

  Scenario: List replica sets
    Given a Control Plane API endpoint
    When I request replicasets
    Then I receive a list of replica sets with load and SLO snapshots
