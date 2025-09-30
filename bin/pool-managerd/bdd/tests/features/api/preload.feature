Feature: Preload API Endpoint
  # Traceability: B-API-010 through B-API-016
  # Spec: POST /pools/{id}/preload HTTP endpoint behaviors
  
  Background:
    Given a running pool-managerd daemon
  
  Scenario: Preload accepts PreparedEngine JSON
    Given a valid PreparedEngine JSON payload
    When I POST to /pools/test-pool/preload
    Then I receive 200 OK
    And the response includes pool_id field
    And the response includes pid field
    And the response includes handoff_path field
  
  Scenario: Preload returns pool_id on success
    Given a PreparedEngine for pool "my-pool"
    And the engine will respond with HTTP 200 on /health
    When I POST to /pools/my-pool/preload
    Then I receive 200 OK
    And the pool_id field equals "my-pool"
  
  Scenario: Preload returns pid on success
    Given a PreparedEngine for pool "test-pool"
    And the engine will respond with HTTP 200 on /health
    When I POST to /pools/test-pool/preload
    Then I receive 200 OK
    And the pid field is a positive integer
  
  Scenario: Preload returns handoff_path on success
    Given a PreparedEngine for pool "test-pool"
    And the engine will respond with HTTP 200 on /health
    When I POST to /pools/test-pool/preload
    Then I receive 200 OK
    And the handoff_path field contains ".runtime/engines/engine.json"
  
  Scenario: Preload fails with 500 on spawn error
    Given a PreparedEngine with invalid binary path
    When I POST to /pools/test-pool/preload
    Then I receive 500 Internal Server Error
    And the error message contains "preload failed"
  
  Scenario: Preload fails with 500 on health check timeout
    Given a PreparedEngine for pool "test-pool"
    And the engine will never respond to health checks
    When I POST to /pools/test-pool/preload
    Then I receive 500 Internal Server Error
    And the error message contains "health check failed"
  
  Scenario: Preload fails with 500 on registry lock failure
    Given a PreparedEngine for pool "test-pool"
    And the registry lock is poisoned
    When I POST to /pools/test-pool/preload
    Then I receive 500 Internal Server Error
    And the error message contains "registry lock failed"
  
  Scenario: Spawned process writes to log file
    Given a PreparedEngine for pool "test-pool"
    And the engine will respond with HTTP 200 on /health
    When I POST to /pools/test-pool/preload
    Then I receive 200 OK
    And a log file exists at ".runtime/engine-test-pool.log"
  
  Scenario: PID file is created in .runtime
    Given a PreparedEngine for pool "test-pool"
    And the engine will respond with HTTP 200 on /health
    When I POST to /pools/test-pool/preload
    Then I receive 200 OK
    And a PID file exists at ".runtime/test-pool.pid"
  
  Scenario: Handoff JSON is written to .runtime/engines
    Given a PreparedEngine for pool "test-pool"
    And the engine will respond with HTTP 200 on /health
    When I POST to /pools/test-pool/preload
    Then I receive 200 OK
    And a handoff file exists at ".runtime/engines/engine.json"
  
  Scenario: Multiple preload requests for different pools
    Given a PreparedEngine for pool "pool1"
    And a PreparedEngine for pool "pool2"
    And both engines will respond with HTTP 200 on /health
    When I POST to /pools/pool1/preload
    And I POST to /pools/pool2/preload
    Then both requests return 200 OK
    And pool1 has its own PID file
    And pool2 has its own PID file
  
  Scenario: Preload with all PreparedEngine fields
    Given a PreparedEngine with:
      | binary_path    | /usr/local/bin/llama-server |
      | flags          | --model,/models/test.gguf   |
      | host           | 127.0.0.1                   |
      | port           | 8080                        |
      | model_path     | /models/test.gguf           |
      | engine_version | llamacpp-v1.2.3             |
      | pool_id        | test-pool                   |
      | replica_id     | r0                          |
    And the engine will respond with HTTP 200 on /health
    When I POST to /pools/test-pool/preload
    Then I receive 200 OK
    And all fields are processed correctly
