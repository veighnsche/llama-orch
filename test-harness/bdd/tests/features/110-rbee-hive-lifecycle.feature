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

  # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # TEAM-098: Worker PID Tracking & Force-Kill Tests (15 scenarios)
  # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  @p0 @lifecycle @pid-tracking
  Scenario: LIFE-001 - Store worker PID on spawn
    Given rbee-hive is running as persistent daemon
    When rbee-hive spawns a worker process
    Then worker PID is stored in registry
    And PID is greater than 0
    And PID corresponds to running process

  @p0 @lifecycle @pid-tracking
  Scenario: LIFE-002 - Track PID across worker lifecycle
    Given rbee-hive spawned a worker with stored PID
    When worker transitions from Loading to Idle
    Then PID remains unchanged in registry
    And PID still corresponds to same process

  @p0 @lifecycle @force-kill
  Scenario: LIFE-003 - Force-kill worker after graceful timeout
    Given rbee-hive is running with 1 worker
    When rbee-hive sends shutdown command to worker
    And worker does not respond within 10s
    Then rbee-hive force-kills worker using stored PID
    And worker process terminates
    And rbee-hive logs force-kill event with PID

  @p0 @lifecycle @force-kill
  Scenario: LIFE-004 - Force-kill hung worker (SIGTERM → SIGKILL)
    Given rbee-hive is running with 1 worker
    And worker is hung and not responding
    When rbee-hive attempts graceful shutdown
    And worker ignores SIGTERM for 10s
    Then rbee-hive sends SIGKILL to worker PID
    And worker process is terminated forcefully
    And rbee-hive removes worker from registry

  @p0 @lifecycle @health-check
  Scenario: LIFE-005 - Process liveness check (not just HTTP)
    Given rbee-hive is running with 1 worker
    And worker PID is stored in registry
    When rbee-hive performs health check
    Then rbee-hive verifies process exists via PID
    And rbee-hive checks HTTP endpoint
    And if process dead but HTTP alive, mark as zombie
    And if process alive but HTTP dead, attempt restart

  @p0 @lifecycle @timeout
  Scenario: LIFE-006 - Ready timeout - kill if stuck in Loading > 30s
    Given rbee-hive spawned a worker
    And worker is in Loading state
    When 30 seconds elapse without ready callback
    Then rbee-hive force-kills worker using PID
    And rbee-hive removes worker from registry
    And rbee-hive logs timeout event

  @p0 @lifecycle @shutdown
  Scenario: LIFE-007 - Parallel worker shutdown (all workers concurrently)
    Given rbee-hive is running with 5 workers
    When rbee-hive receives SIGTERM
    Then rbee-hive sends shutdown to all 5 workers concurrently
    And rbee-hive waits for all workers in parallel
    And shutdown completes faster than sequential (< 15s for 5 workers)

  @p0 @lifecycle @shutdown
  Scenario: LIFE-008 - Shutdown timeout enforcement (30s total)
    Given rbee-hive is running with 3 workers
    When rbee-hive receives SIGTERM
    And 2 workers respond within 5s
    And 1 worker does not respond
    Then rbee-hive waits maximum 30s total
    And rbee-hive force-kills unresponsive worker at 30s
    And rbee-hive exits after all workers terminated

  @p0 @lifecycle @shutdown
  Scenario: LIFE-009 - Shutdown progress metrics logged
    Given rbee-hive is running with 4 workers
    When rbee-hive receives SIGTERM
    Then rbee-hive logs "Shutting down 4 workers..."
    And rbee-hive logs progress "1/4 workers stopped"
    And rbee-hive logs progress "2/4 workers stopped"
    And rbee-hive logs progress "3/4 workers stopped"
    And rbee-hive logs progress "4/4 workers stopped"
    And rbee-hive logs "All workers stopped, exiting"

  @p0 @lifecycle @cleanup
  Scenario: LIFE-010 - PID cleanup on worker removal
    Given rbee-hive is running with 1 worker
    And worker PID is stored in registry
    When rbee-hive removes worker from registry
    Then worker PID is cleared from memory
    And no references to PID remain in registry

  @p0 @lifecycle @crash-detection
  Scenario: LIFE-011 - Detect worker crash via PID (process not found)
    Given rbee-hive is running with 1 worker
    And worker PID is stored in registry
    When worker process crashes unexpectedly
    And rbee-hive performs health check
    Then rbee-hive detects PID no longer exists
    And rbee-hive marks worker as crashed
    And rbee-hive removes worker from registry
    And rbee-hive logs crash event with PID

  @p0 @lifecycle @cleanup
  Scenario: LIFE-012 - Zombie process cleanup
    Given rbee-hive spawned a worker
    And worker process exited but not reaped
    When rbee-hive detects zombie process via PID
    Then rbee-hive reaps zombie process
    And rbee-hive removes worker from registry
    And rbee-hive logs zombie cleanup event

  @p0 @lifecycle @force-kill
  Scenario: LIFE-013 - Multiple workers force-killed in parallel
    Given rbee-hive is running with 3 workers
    And all 3 workers are hung
    When rbee-hive receives SIGTERM
    And all workers ignore shutdown command
    Then rbee-hive force-kills all 3 workers concurrently
    And all 3 processes terminate
    And rbee-hive exits cleanly

  @p0 @lifecycle @audit
  Scenario: LIFE-014 - Force-kill audit logging
    Given rbee-hive is running with 1 worker
    When rbee-hive force-kills worker
    Then rbee-hive logs force-kill event
    And force-kill log includes worker_id
    And force-kill log includes PID
    And force-kill log includes reason
    And force-kill log includes signal type
    And force-kill log includes timestamp

  @p0 @lifecycle @graceful
  Scenario: LIFE-015 - Graceful shutdown preferred over force-kill
    Given rbee-hive is running with 1 worker
    When rbee-hive sends shutdown command
    And worker responds within 5s
    Then rbee-hive does NOT force-kill worker
    And worker exits gracefully
    And rbee-hive logs graceful shutdown success
