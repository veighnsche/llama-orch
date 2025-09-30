Feature: Process Lifecycle Management
  # Traceability: B-PROC-001 through B-PROC-022
  # Spec: Engine process spawn, stop, and monitoring
  
  Background:
    Given a PreparedEngine with:
      | field        | value                      |
      | binary_path  | /usr/local/bin/llama-server |
      | flags        | --model,/models/test.gguf,--port,8080 |
      | host         | 127.0.0.1                  |
      | port         | 8080                       |
      | pool_id      | test-pool                  |
  
  Scenario: Spawn process with correct binary
    When I spawn the engine process
    Then the process is started
    And the command uses binary "/usr/local/bin/llama-server"
  
  Scenario: Spawn process with all flags
    When I spawn the engine process
    Then the process receives flag "--model"
    And the process receives flag "/models/test.gguf"
    And the process receives flag "--port"
    And the process receives flag "8080"
  
  Scenario: Stdout redirected to log file
    When I spawn the engine process
    Then stdout is redirected to ".runtime/engine-test-pool.log"
  
  Scenario: Stderr redirected to log file
    When I spawn the engine process
    Then stderr is redirected to ".runtime/engine-test-pool.log"
  
  Scenario: Log file created in .runtime directory
    When I spawn the engine process
    Then the log file exists at ".runtime/engine-test-pool.log"
  
  Scenario: Log file opened in append mode
    Given a log file already exists
    When I spawn the engine process
    Then the new logs are appended to existing content
  
  Scenario: PID captured after spawn
    When I spawn the engine process
    Then the process ID is captured
    And the PID is greater than 0
  
  Scenario: PID written to file
    When I spawn the engine process
    Then a PID file is created at ".runtime/test-pool.pid"
    And the PID file contains the process ID
  
  Scenario: PID file format is plain text
    When I spawn the engine process
    Then the PID file contains only the numeric process ID
  
  Scenario: Stop pool reads PID file
    Given an engine is running for pool "test-pool"
    And the PID file exists
    When I call stop_pool for "test-pool"
    Then the PID is read from ".runtime/test-pool.pid"
  
  Scenario: Stop pool sends SIGTERM first
    Given an engine is running for pool "test-pool"
    When I call stop_pool for "test-pool"
    Then SIGTERM is sent to the process
    And the system waits for graceful shutdown
  
  Scenario: Stop pool waits 5 seconds for graceful shutdown
    Given an engine is running for pool "test-pool"
    When I call stop_pool for "test-pool"
    Then the system waits up to 5 seconds
    And checks if process is still alive
  
  Scenario: Stop pool sends SIGKILL after grace period
    Given an engine is running for pool "test-pool"
    And the process does not respond to SIGTERM
    When I call stop_pool for "test-pool"
    And 5 seconds have elapsed
    Then SIGKILL is sent to the process
  
  Scenario: Stop pool removes PID file
    Given an engine is running for pool "test-pool"
    When I call stop_pool for "test-pool"
    And the process has stopped
    Then the PID file is removed
  
  Scenario: Stop pool fails if PID file missing
    Given no engine is running for pool "test-pool"
    And no PID file exists
    When I call stop_pool for "test-pool"
    Then the call returns an error
    And the error mentions missing PID file
