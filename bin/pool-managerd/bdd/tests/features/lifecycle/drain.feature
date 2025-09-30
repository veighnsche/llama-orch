Feature: Pool Drain Lifecycle
  # Traceability: B-DRAIN-001 through B-DRAIN-007
  # Spec: OC-POOL-3010, OC-POOL-3031 - Drain lifecycle with deadline
  
  Background:
    Given a running pool-managerd daemon
    And a pool "test-pool" is registered and ready
    And the pool has 3 active leases
  
  Scenario: Drain request sets draining flag
    When I request drain for pool "test-pool" with deadline 5000ms
    Then the pool draining flag is true
    And the registry shows draining=true
  
  Scenario: Draining pool refuses new lease allocations
    Given the pool is draining
    When I attempt to allocate a new lease
    Then the allocation is refused
    And the error indicates pool is draining
  
  Scenario: Draining pool allows existing leases to complete
    Given the pool is draining
    And the pool has 2 active leases
    When an existing lease completes
    Then the active_leases count decrements
    And the pool remains draining
  
  Scenario: Drain waits for active_leases to reach 0
    Given the pool is draining
    And the pool has 3 active leases
    When all leases complete naturally
    Then active_leases reaches 0
    And drain completes successfully
  
  Scenario: Drain force-stops after deadline expires
    Given the pool is draining with deadline 1000ms
    And the pool has 2 active leases that never complete
    When the deadline expires
    Then drain force-stops the engine
    And the PID file is removed
    And drain completes with force-stop status
  
  Scenario: Drain stops engine process after leases drain
    Given the pool is draining
    And the pool has 1 active lease
    When the last lease completes
    Then the engine process is stopped gracefully
    And the PID file is removed
  
  Scenario: Drain updates registry health to not ready
    Given the pool is draining
    When drain completes
    Then the registry health is live=false ready=false
    And the pool status shows not ready
  
  Scenario: Drain with no active leases completes immediately
    Given a pool "empty-pool" is registered and ready
    And the pool has 0 active leases
    When I request drain for pool "empty-pool" with deadline 5000ms
    Then drain completes immediately
    And the engine process is stopped
  
  Scenario: Drain emits metrics for drain duration
    Given the pool is draining
    When drain completes
    Then drain_duration_ms metric is emitted
    And the metric includes pool_id label
  
  Scenario: Drain with inflight requests logs warning
    Given the pool is draining
    And the pool has 5 active leases
    When drain starts
    Then a warning is logged about inflight requests
    And the log includes active_leases count
