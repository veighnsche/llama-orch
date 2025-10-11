# Traceability: TEST-001 (split by TEAM-077)
# Architecture: TEAM-037 (queen-rbee orchestration)
# Components: rbee-hive (pool manager daemon)
# Refactored by: TEAM-077 (reorganized to correct BDD architecture)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: rbee-hive Daemon Lifecycle
  As a system managing the rbee-hive daemon
  I want to control startup, shutdown, and health monitoring
  So that rbee-hive runs reliably and cleans up properly

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"

  @lifecycle @critical
  Scenario: Rbee-hive remains running as persistent HTTP daemon
    Given rbee-hive is started as HTTP daemon on port 8080
    And rbee-hive spawned a worker
    When the worker sends ready callback
    Then rbee-hive does NOT exit
    And rbee-hive continues monitoring worker health every 30s
    And rbee-hive enforces idle timeout of 5 minutes
    And rbee-hive remains available for new worker requests
    And rbee-hive HTTP API remains accessible

  @lifecycle
  Scenario: Rbee-hive monitors worker health
    Given rbee-hive is running as persistent daemon
    And a worker is registered
    When 30 seconds elapse
    Then rbee-hive sends health check to worker
    And if worker responds, rbee-hive updates last_activity
    And if worker does not respond, rbee-hive marks worker as unhealthy
    And if worker is unhealthy for 3 consecutive checks, rbee-hive removes it from registry
    And rbee-hive continues running (does NOT exit)

  @lifecycle @critical
  Scenario: Rbee-hive enforces idle timeout (worker dies, pool lives)
    Given rbee-hive is running as persistent daemon
    And a worker completed inference and is idle
    When 5 minutes elapse without new requests
    Then rbee-hive sends shutdown command to worker
    And rbee-hive removes worker from in-memory registry
    And worker releases resources and exits
    And rbee-hive continues running as daemon (does NOT exit)

  @lifecycle @critical
  Scenario: Cascading shutdown when rbee-hive receives SIGTERM
    Given rbee-hive is running as persistent daemon
    And 3 workers are registered and running
    When user sends SIGTERM to rbee-hive (Ctrl+C)
    Then rbee-hive sends "POST /v1/admin/shutdown" to all 3 workers
    And rbee-hive waits for workers to acknowledge (max 5s per worker)
    And all workers unload models and exit
    And rbee-hive clears in-memory registry
    And rbee-hive exits cleanly
    And model catalog (SQLite) persists on disk

  @lifecycle @error-handling
  Scenario: EH-014a - Worker ignores shutdown signal
    Given rbee-hive is running with 1 worker
    When rbee-hive sends shutdown command to worker
    And worker does not respond within 30s
    Then rbee-hive force-kills worker process
    And rbee-hive logs force-kill event
    And rbee-hive displays:
      """
      [rbee-hive] 🛑 Shutting down worker-abc123...
      [rbee-hive] ⏳ Waiting for graceful shutdown (30s timeout)...
      [rbee-hive] ⚠️  Worker did not respond, force-killing
      [rbee-hive] ✅ Worker terminated
      """

  @lifecycle @error-handling
  Scenario: EH-014b - Graceful shutdown with active request
    Given worker is processing inference request
    When rbee-hive sends shutdown command
    Then worker sets state to "draining"
    And worker rejects new inference requests with 503
    And worker waits for active request to complete (max 30s)
    And worker unloads model after request completes
    And worker exits with code 0

  @edge-case
  Scenario: EC10 - Idle timeout and worker auto-shutdown
    Given the worker has been idle for 30 minutes
    When rbee-hive checks idle timeout
    Then rbee-hive sends shutdown signal to worker
    And worker gracefully shuts down
    And rbee-hive removes worker from registry
    And rbee-hive logs:
      """
      Worker worker-abc123 idle for 30m, shutting down
      """
