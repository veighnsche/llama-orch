Feature: Exponential Backoff Policy
  # Traceability: B-SUPER-010 through B-SUPER-014
  # Spec: OC-POOL-3011 - Restart storms bounded by exponential backoff
  
  Background:
    Given a running pool-managerd daemon
    And a pool "test-pool" is registered
    And backoff policy is configured with initial=1000ms max=60000ms
  
  Scenario: First restart has initial_ms delay
    Given the engine crashes for the first time
    When supervisor schedules restart
    Then the backoff delay is 1000ms
    And restart is scheduled after 1000ms
  
  Scenario: Subsequent restarts double delay (exponential)
    Given the engine has crashed 3 times
    When supervisor schedules the 4th restart
    Then the backoff delay is 8000ms
    And the delay follows exponential pattern: 1000, 2000, 4000, 8000
  
  Scenario: Backoff delay is capped at max_ms
    Given the engine has crashed 10 times
    When supervisor schedules restart
    Then the backoff delay is 60000ms
    And the delay does not exceed max_ms
  
  Scenario: Backoff includes jitter to prevent thundering herd
    Given the engine crashes
    When supervisor calculates backoff delay
    Then jitter is added to the base delay
    And jitter is between -10% and +10% of base delay
  
  Scenario: Backoff resets after stable run period
    Given the engine has crashed 5 times
    And backoff delay is 32000ms
    When the engine runs stably for 300 seconds
    Then the backoff delay resets to 1000ms
    And the crash counter resets to 0
  
  Scenario: Backoff policy logs delay decisions
    Given the engine crashes
    When supervisor calculates backoff
    Then the log includes backoff_ms
    And the log includes crash_count
    And the log includes next_restart_at timestamp
  
  Scenario: Backoff respects minimum delay
    Given backoff policy has min_delay=500ms
    When the first crash occurs
    Then the backoff delay is at least 500ms
  
  Scenario: Backoff delay increases per crash type
    Given the engine crashes with CUDA error
    When supervisor calculates backoff
    Then CUDA errors use 2x multiplier
    And the backoff delay is doubled
  
  Scenario: Backoff emits metrics
    When supervisor applies backoff
    Then backoff_delay_ms histogram is updated
    And restart_scheduled_total counter increments
    And the metric includes crash_reason label
