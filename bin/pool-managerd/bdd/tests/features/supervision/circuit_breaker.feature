Feature: Circuit Breaker for Restart Storms
  # Traceability: B-SUPER-020 through B-SUPER-025
  # Spec: OC-POOL-3011 - Circuit breaker prevents infinite restart loops
  
  Background:
    Given a running pool-managerd daemon
    And a pool "test-pool" is registered
    And circuit breaker is configured with threshold=5 timeout=300s
  
  Scenario: Circuit opens after N consecutive failures
    Given the engine has crashed 4 times consecutively
    When the engine crashes the 5th time
    Then the circuit breaker opens
    And no further restart attempts are made
    And the pool is marked as permanently failed
  
  Scenario: Open circuit prevents restart attempts
    Given the circuit breaker is open
    When the engine crashes again
    Then no restart is scheduled
    And the crash is logged but ignored
    And the pool remains in failed state
  
  Scenario: Circuit transitions to half-open after timeout
    Given the circuit breaker has been open for 300 seconds
    When the timeout expires
    Then the circuit transitions to half-open
    And one test restart is allowed
  
  Scenario: Half-open allows single test restart
    Given the circuit breaker is half-open
    When supervisor attempts restart
    Then exactly one restart is attempted
    And the circuit remains half-open during test
  
  Scenario: Circuit closes after successful test restart
    Given the circuit breaker is half-open
    And a test restart is attempted
    When the engine starts successfully
    And the engine runs stably for 60 seconds
    Then the circuit breaker closes
    And normal restart policy resumes
  
  Scenario: Circuit reopens if test restart fails
    Given the circuit breaker is half-open
    And a test restart is attempted
    When the engine crashes immediately
    Then the circuit breaker reopens
    And the timeout is extended by 2x
  
  Scenario: Circuit breaker logs state transitions
    When the circuit breaker changes state
    Then the log includes old_state and new_state
    And the log includes failure_count
    And the log includes pool_id
  
  Scenario: Circuit breaker emits metrics
    When the circuit breaker opens
    Then circuit_breaker_open_total counter increments
    And circuit_breaker_state gauge is set to 1 (open)
  
  Scenario: Circuit breaker respects manual reset
    Given the circuit breaker is open
    When an operator manually resets the circuit
    Then the circuit transitions to closed
    And the failure count resets to 0
    And normal restart policy resumes
  
  Scenario: Circuit breaker failure threshold is configurable
    Given circuit breaker threshold is set to 3
    When the engine crashes 3 times
    Then the circuit breaker opens
  
  Scenario: Circuit breaker distinguishes error types
    Given the circuit breaker tracks CUDA errors separately
    When 5 CUDA errors occur
    Then the CUDA circuit opens
    But the general circuit remains closed
