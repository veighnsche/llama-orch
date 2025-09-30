Feature: Crash Detection and Recovery
  # Traceability: B-SUPER-001 through B-SUPER-004
  # Spec: OC-POOL-3010 - Driver/CUDA errors trigger restart with backoff
  
  Background:
    Given a running pool-managerd daemon
    And a pool "test-pool" is registered and ready
    And supervision is enabled
  
  Scenario: Supervisor detects when engine process exits
    Given the engine process is running
    When the engine process exits unexpectedly
    Then the supervisor detects the exit
    And the exit code is captured
    And a crash event is logged
  
  Scenario: Supervisor detects when health check fails
    Given the engine process is running
    When health check polling fails 3 consecutive times
    Then the supervisor detects health failure
    And the failure is logged with timestamps
  
  Scenario: Supervisor detects driver/CUDA errors
    Given the engine process is running
    When the engine logs contain "CUDA error"
    Then the supervisor detects driver error
    And the error type is classified as CUDA
  
  Scenario: Supervisor transitions pool to unready on crash
    Given the engine process crashes
    When the supervisor detects the crash
    Then the registry health is set to live=false ready=false
    And the pool status shows not ready
    And last_error is updated with crash reason
  
  Scenario: Supervisor captures exit signals
    Given the engine process is running
    When the engine receives SIGSEGV
    Then the supervisor captures the signal
    And the crash reason includes "SIGSEGV"
  
  Scenario: Supervisor distinguishes graceful vs crash exit
    Given the engine process is running
    When the engine exits with code 0
    Then the supervisor recognizes graceful shutdown
    And no restart is attempted
  
  Scenario: Supervisor logs crash context
    Given the engine process crashes
    When the supervisor detects the crash
    Then the log includes pool_id
    And the log includes engine_version
    And the log includes uptime_seconds
    And the log includes exit_code
  
  Scenario: Supervisor increments crash counter
    Given the engine has crashed 2 times
    When the engine crashes again
    Then the crash_count is 3
    And the crash_count is persisted in registry
  
  Scenario: Supervisor detects OOM kills
    Given the engine process is running
    When the engine is killed by OOM killer
    Then the supervisor detects OOM condition
    And the crash reason is "out_of_memory"
    And a critical alert is logged
