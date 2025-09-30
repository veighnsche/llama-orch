Feature: Pool Reload Lifecycle
  # Traceability: B-RELOAD-001 through B-RELOAD-009
  # Spec: OC-POOL-3038 - Atomic reload with rollback on failure
  
  Background:
    Given a running pool-managerd daemon
    And a pool "test-pool" is registered and ready
    And the pool is running model "llama-7b-v1"
  
  Scenario: Reload drains pool first
    Given the pool has 2 active leases
    When I request reload with model "llama-7b-v2"
    Then drain is initiated
    And reload waits for drain to complete
  
  Scenario: Reload stages new model via model-provisioner
    Given drain has completed
    When reload proceeds
    Then model-provisioner is called with "llama-7b-v2"
    And the new model is staged
  
  Scenario: Reload stops old engine process
    Given the new model is staged
    When reload proceeds
    Then the old engine process is stopped
    And the old PID file is removed
  
  Scenario: Reload starts new engine with new model
    Given the old engine is stopped
    And the new model is staged
    When reload proceeds
    Then a new engine process is spawned
    And the new engine uses model "llama-7b-v2"
    And a new PID file is created
  
  Scenario: Reload waits for new engine health check
    Given the new engine is spawned
    When reload proceeds
    Then health check polls the new engine
    And reload waits for HTTP 200 response
  
  Scenario: Reload sets ready=true on success
    Given the new engine health check succeeds
    When reload completes
    Then the registry health is live=true ready=true
    And the pool is marked as ready
  
  Scenario: Reload rolls back on failure (atomic)
    Given the new engine health check fails
    When reload detects failure
    Then the new engine process is killed
    And the old model is restored
    And the old engine is restarted
    And reload returns error
  
  Scenario: Reload updates engine_version in registry
    Given the new engine health check succeeds
    When reload completes
    Then the registry engine_version is "llama-7b-v2"
    And the handoff file reflects new version
  
  Scenario: Reload preserves pool_id and device_mask
    Given the pool has device_mask "0,1"
    When reload completes successfully
    Then the pool_id remains "test-pool"
    And the device_mask remains "0,1"
  
  Scenario: Reload with same model version is idempotent
    Given the pool is running model "llama-7b-v1"
    When I request reload with model "llama-7b-v1"
    Then reload skips model staging
    And reload completes successfully
  
  Scenario: Reload failure preserves original state
    Given the pool is running model "llama-7b-v1"
    And the new model "llama-7b-v2" fails health check
    When reload attempts and fails
    Then the pool is still running model "llama-7b-v1"
    And the pool is ready
    And no state corruption occurred
  
  Scenario: Reload emits metrics for reload duration
    When reload completes successfully
    Then reload_duration_ms metric is emitted
    And reload_success_total counter increments
  
  Scenario: Reload failure emits failure metrics
    When reload fails
    Then reload_failure_total counter increments
    And the failure reason is labeled
  
  Scenario: Reload with drain timeout fails gracefully
    Given the pool has leases that never complete
    When I request reload with drain deadline 1000ms
    And drain times out
    Then reload is aborted
    And the original engine remains running
    And reload returns drain timeout error
