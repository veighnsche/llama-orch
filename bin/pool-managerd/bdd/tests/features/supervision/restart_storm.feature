Feature: Restart Storm Prevention
  # Traceability: B-SUPER-030 through B-SUPER-033
  # Spec: OC-POOL-3011 - Maximum restart rate enforcement
  
  Background:
    Given a running pool-managerd daemon
    And a pool "test-pool" is registered
    And restart rate limit is 10 restarts per 60 seconds
  
  Scenario: Restart counter increments on each restart
    Given the engine has restarted 3 times
    When the engine crashes and restarts again
    Then the restart_count is 4
    And the restart_count is persisted
  
  Scenario: Restart counter resets after stable period
    Given the engine has restarted 5 times
    When the engine runs stably for 600 seconds
    Then the restart_count resets to 0
    And the restart window resets
  
  Scenario: Restart storms are logged with restart_count
    Given the engine is restarting frequently
    When the restart_count exceeds 5 in 60 seconds
    Then a warning is logged about restart storm
    And the log includes restart_count
    And the log includes time_window
  
  Scenario: Maximum restart rate is enforced
    Given the engine has restarted 10 times in 60 seconds
    When the engine crashes again
    Then the restart is delayed beyond rate limit
    And a rate limit warning is logged
  
  Scenario: Restart rate limit uses sliding window
    Given 8 restarts occurred in the last 60 seconds
    When 30 seconds pass
    And 2 more restarts occur
    Then the rate limit is not exceeded
    And restarts proceed normally
  
  Scenario: Restart storm triggers circuit breaker
    Given the engine restarts 10 times in 30 seconds
    When the restart storm is detected
    Then the circuit breaker opens
    And further restarts are prevented
  
  Scenario: Restart storm emits critical alert
    Given the engine is in restart storm
    When the storm threshold is exceeded
    Then a critical alert is emitted
    And the alert includes pool_id and restart_count
  
  Scenario: Restart storm metrics are tracked
    When a restart storm occurs
    Then restart_storm_total counter increments
    And restart_rate gauge shows current rate
    And metrics include pool_id label
  
  Scenario: Restart storm distinguishes crash types
    Given 5 OOM crashes in 60 seconds
    When the pattern is detected
    Then the storm is classified as "oom_storm"
    And specific remediation is suggested
  
  Scenario: Restart storm respects manual override
    Given the circuit breaker is open due to restart storm
    When an operator manually allows restart
    Then one restart is permitted
    And the storm counter is not reset
