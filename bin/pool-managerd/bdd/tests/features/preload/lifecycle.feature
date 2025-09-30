Feature: Preload and Readiness Lifecycle
  # Traceability: B-PRELOAD-001 through B-PRELOAD-044
  # Spec: OC-POOL-3001, OC-POOL-3002, OC-POOL-3003
  # Complete preload flow: spawn, health check, handoff, registry update
  
  Background:
    Given a running pool-managerd daemon
    And an empty registry
  
  # Engine Spawn (B-PRELOAD-001 through B-PRELOAD-005)
  Scenario: Engine spawns with correct binary path
    Given a PreparedEngine with binary "/usr/local/bin/llama-server"
    When I execute preload
    Then the engine process is spawned
    And the binary path is "/usr/local/bin/llama-server"
  
  Scenario: Engine receives all flags from PreparedEngine
    Given a PreparedEngine with flags:
      | --model          |
      | /models/test.gguf |
      | --port           |
      | 8080             |
    When I execute preload
    Then all flags are passed to the engine process
  
  Scenario: Engine stdout redirected to log file
    Given a PreparedEngine for pool "test-pool"
    When I execute preload
    Then stdout is redirected to ".runtime/engine-test-pool.log"
  
  Scenario: PID file written before health check
    Given a PreparedEngine for pool "test-pool"
    When I execute preload
    Then the PID file ".runtime/test-pool.pid" is created
    And the PID file is written before health check starts
  
  # Health Check Wait (B-PRELOAD-010 through B-PRELOAD-014)
  Scenario: Health check polls engine HTTP endpoint
    Given a PreparedEngine with host "127.0.0.1" and port 8080
    When I execute preload
    Then health check polls "http://127.0.0.1:8080/health"
  
  Scenario: Health check retries with 500ms intervals
    Given a PreparedEngine with slow startup
    When I execute preload
    Then health check retries every 500 milliseconds
  
  Scenario: Health check succeeds on HTTP 200
    Given a PreparedEngine for pool "test-pool"
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then health check succeeds
    And preload completes successfully
  
  Scenario: Health check times out after 120 seconds
    Given a PreparedEngine for pool "test-pool"
    And the engine never responds to health checks
    When I execute preload
    Then health check times out after 120 seconds
    And preload returns an error
  
  Scenario: Health check accepts HTTP/1.1 200
    Given a PreparedEngine for pool "test-pool"
    And the engine responds with "HTTP/1.1 200 OK"
    When I execute preload
    Then health check succeeds
  
  Scenario: Health check accepts HTTP/1.0 200
    Given a PreparedEngine for pool "test-pool"
    And the engine responds with "HTTP/1.0 200 OK"
    When I execute preload
    Then health check succeeds
  
  # Readiness Gating (B-PRELOAD-020 through B-PRELOAD-024)
  Scenario: Pool not ready until health check succeeds (OC-POOL-3001)
    Given a PreparedEngine for pool "test-pool"
    And the engine takes 5 seconds to become healthy
    When I execute preload
    Then the pool is not ready during startup
    And the pool becomes ready only after health check succeeds
  
  Scenario: Pool becomes ready after successful health check (OC-POOL-3001)
    Given a PreparedEngine for pool "test-pool"
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then the pool health is live=true ready=true
  
  Scenario: Handoff file written only after health check success
    Given a PreparedEngine for pool "test-pool"
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then the handoff file is created
    And the handoff file is written after health check succeeds
  
  Scenario: Registry updated to ready after health check (OC-POOL-3003)
    Given a PreparedEngine for pool "test-pool"
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then the registry shows pool "test-pool" as ready
    And the registry health is live=true ready=true
  
  Scenario: Registry includes engine_version from PreparedEngine
    Given a PreparedEngine with engine_version "llamacpp-v1.2.3"
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then the registry engine_version is "llamacpp-v1.2.3"
  
  # Preload Failure Handling (B-PRELOAD-030 through B-PRELOAD-034)
  Scenario: Process killed if health check fails (OC-POOL-3002)
    Given a PreparedEngine for pool "test-pool"
    And the engine never responds to health checks
    When I execute preload
    And health check times out
    Then the spawned process is killed
  
  Scenario: PID file removed if health check fails (OC-POOL-3002)
    Given a PreparedEngine for pool "test-pool"
    And the engine never responds to health checks
    When I execute preload
    And health check times out
    Then the PID file is removed
  
  Scenario: Registry records last_error when health check fails (OC-POOL-3002, OC-POOL-3003)
    Given a PreparedEngine for pool "test-pool"
    And the engine never responds to health checks
    When I execute preload
    And health check times out
    Then the registry last_error contains "health check failed"
  
  Scenario: Preload returns error when health check times out (OC-POOL-3002)
    Given a PreparedEngine for pool "test-pool"
    And the engine never responds to health checks
    When I execute preload
    Then preload returns an error
    And the error message contains "health check failed"
  
  Scenario: Pool remains unready when preload fails (OC-POOL-3002, OC-POOL-3003)
    Given a PreparedEngine for pool "test-pool"
    And the engine never responds to health checks
    When I execute preload
    Then the pool is not ready
    And the registry shows ready=false
  
  # Handoff File Generation (B-PRELOAD-040 through B-PRELOAD-044)
  Scenario: Handoff JSON includes engine metadata
    Given a PreparedEngine with:
      | engine_version | llamacpp-v1.2.3 |
      | host          | 127.0.0.1       |
      | port          | 8080            |
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then the handoff JSON includes field "engine" with value "llamacpp"
    And the handoff JSON includes field "engine_version" with value "llamacpp-v1.2.3"
    And the handoff JSON includes field "url" with value "http://127.0.0.1:8080"
  
  Scenario: Handoff JSON includes pool and replica IDs
    Given a PreparedEngine with:
      | pool_id    | test-pool |
      | replica_id | r0        |
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then the handoff JSON includes field "pool_id" with value "test-pool"
    And the handoff JSON includes field "replica_id" with value "r0"
  
  Scenario: Handoff JSON includes model path
    Given a PreparedEngine with model_path "/models/llama-7b.gguf"
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then the handoff JSON includes field "model.path" with value "/models/llama-7b.gguf"
  
  Scenario: Handoff JSON includes flags array
    Given a PreparedEngine with flags:
      | --model          |
      | /models/test.gguf |
      | --port           |
      | 8080             |
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then the handoff JSON includes field "flags" as array
    And the flags array contains all PreparedEngine flags
  
  Scenario: Handoff file is pretty-printed JSON
    Given a PreparedEngine for pool "test-pool"
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then the handoff file is valid JSON
    And the handoff file is formatted with indentation
  
  # Complete Success Flow
  Scenario: Complete successful preload flow (OC-POOL-3001)
    Given a PreparedEngine for pool "test-pool"
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then the engine process is spawned
    And the PID file is created
    And health check succeeds
    And the handoff file is written
    And the registry is updated to ready
    And preload returns success with pool_id, pid, and handoff_path
  
  # Complete Failure Flow
  Scenario: Complete failure flow with cleanup (OC-POOL-3002)
    Given a PreparedEngine for pool "test-pool"
    And the engine never responds to health checks
    When I execute preload
    Then the engine process is spawned
    And the PID file is created
    And health check times out
    And the spawned process is killed
    And the PID file is removed
    And the registry records the error
    And preload returns an error
  
  # Edge Cases
  Scenario: Preload with minimal PreparedEngine
    Given a PreparedEngine with only required fields
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then preload succeeds
  
  Scenario: Preload creates .runtime directory if missing
    Given no .runtime directory exists
    And a PreparedEngine for pool "test-pool"
    When I execute preload
    Then the .runtime directory is created
    And the PID file is written successfully
  
  Scenario: Preload creates .runtime/engines directory for handoff
    Given no .runtime/engines directory exists
    And a PreparedEngine for pool "test-pool"
    And the engine responds with HTTP 200 on /health
    When I execute preload
    Then the .runtime/engines directory is created
    And the handoff file is written successfully
