Feature: Handoff Autobind Watcher
  # Traceability: B-BG-001 through B-BG-012
  # Background service that watches for engine-provisioner handoff files

  Scenario: Autobind adapter from handoff file
    Given a handoff file exists with pool_id "gpu-0" and replica_id "r1"
    When the handoff watcher processes the file
    Then an adapter is bound to pool "gpu-0" replica "r1"
    And the pool is registered as ready
    And a narration breadcrumb is emitted

  Scenario: Skip already bound pool
    Given a pool "gpu-0" is already bound
    And a handoff file for "gpu-0" exists
    When the handoff watcher processes the file
    Then the pool is not re-bound

  Scenario: Watcher runs continuously
    Given the handoff watcher is running
    When I create a new handoff file
    And I wait for the poll interval
    Then the new handoff is processed
