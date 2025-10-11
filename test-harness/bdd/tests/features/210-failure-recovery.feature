# Traceability: Gap Analysis P0 - Failure Recovery
# Created by: TEAM-079
# Modified by: TEAM-080 (architectural accuracy - removed scenarios requiring non-existent features)
# Priority: P0 - Critical for production readiness
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# ⚠️ These scenarios test detection and cleanup after failures
# ⚠️ Automatic request retry/failover requires state machine (v2.0 feature)

Feature: Failure Detection and Recovery
  As a system handling failures
  I want to detect crashes and clean up state
  So that system remains consistent

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"

  @failover @p0
  Scenario: Gap-F1 - Worker crash detection and registry cleanup
    Given worker-001 is registered in queen-rbee with last_heartbeat=T0
    And worker-001 is processing inference request (client connected directly)
    When worker-001 crashes unexpectedly
    Then heartbeat timeout occurs after 120s (4 missed heartbeats @ 30s interval)
    And queen-rbee marks worker-001 as stale
    And queen-rbee removes worker-001 from registry
    And subsequent routing requests do NOT include worker-001
    # Note: User must manually retry request - no automatic failover in v1.0
    # Automatic retry requires request state machine (planned for v2.0)

  @failover @p0
  Scenario: Gap-F2 - Model catalog database corruption
    Given the SQLite catalog database is corrupted
    When rbee-hive attempts to query catalog
    Then rbee-hive detects corruption via integrity check
    And rbee-hive creates backup at "~/.rbee/models.db.corrupt.backup"
    And rbee-hive initializes fresh catalog
    And rbee-keeper displays recovery instructions
    And system continues operating with empty catalog

  # DELETED by TEAM-080: Scenario Gap-F3 - Registry split-brain
  # Reason: v1.0 supports only SINGLE queen-rbee instance (no HA)
  # Split-brain requires multi-master setup with Raft/Paxos consensus
  # If HA is implemented in v2.0, add this scenario back with:
  # @future @v2.0 @requires-ha @requires-consensus

  @failover @p0
  Scenario: Gap-F4 - Partial download resume after crash
    Given model download interrupted at 60% (3MB/5MB)
    And partial file exists at "/tmp/tinyllama-q4.gguf.partial"
    When rbee-hive restarts download
    Then rbee-hive sends "Range: bytes=3145728-" header
    And download resumes from 60%
    And progress shows "Resuming from 60%..."
    And download completes successfully

  @failover @p1
  Scenario: Gap-F5 - Worker heartbeat timeout with active request
    Given worker-001 is processing request
    When heartbeat times out (>120s)
    Then queen-rbee marks worker as "stale-but-busy"
    And new requests are NOT routed to worker-001
    And existing request is allowed to complete
    And worker is removed after request completes or 5min timeout

  @failover @p1
  Scenario: Gap-F6 - rbee-hive restart and worker re-registration
    Given 3 workers are running and registered in queen-rbee
    And rbee-hive's local registry is in-memory (will be lost on restart)
    When rbee-hive restarts
    Then rbee-hive's local registry is empty (ephemeral)
    And workers continue running independently (separate processes)
    And workers send heartbeats to queen-rbee (maintain global registration)
    And in-flight requests continue processing at worker level
    # Note: rbee-hive must rebuild local state from queen-rbee or worker callbacks

  @failover @p1
  Scenario: Gap-F7 - Graceful shutdown with pending requests
    Given worker has 2 requests in progress
    When SIGTERM is received
    Then worker stops accepting new requests
    And worker completes existing requests (max 30s)
    And worker sends final status to queen-rbee
    And worker exits with code 0

  @failover @p2
  Scenario: Gap-F8 - Catalog backup and restore
    Given catalog contains 50 model entries
    When "rbee-keeper catalog backup" is executed
    Then backup is created at "~/.rbee/backups/models-<timestamp>.db"
    And backup integrity is verified
    And restore command is displayed
