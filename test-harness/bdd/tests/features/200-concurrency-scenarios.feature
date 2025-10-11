# Traceability: Gap Analysis P0 - Concurrency Testing
# Created by: TEAM-079
# Modified by: TEAM-080 (architectural clarity - specifies which registry layer)
# Priority: P0 - Critical for production readiness
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# ⚠️ These scenarios test race conditions in queen-rbee GLOBAL registry
# ⚠️ For worker-level slot concurrency, see 130-inference-execution.feature
# ⚠️ For rbee-hive local registry, see 060-rbee-hive-worker-registry.feature

Feature: queen-rbee Global Registry Concurrency
  As queen-rbee managing a global worker registry
  I want to handle concurrent updates from multiple rbee-hive instances
  So that registry state remains consistent

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"
    And queen-rbee has an empty global registry

  @concurrency @p0 @queen-rbee-registry
  Scenario: Gap-C1 - Concurrent worker registration to queen-rbee from multiple rbee-hive instances
    Given 3 rbee-hive instances are running on different nodes
    When all 3 instances POST to "http://localhost:8080/v1/workers/register" for "worker-001" simultaneously
    Then queen-rbee's global registry accepts only one registration
    And the other 2 receive HTTP 409 "WORKER_ALREADY_REGISTERED"
    And no database locks occur (in-memory registry uses Arc<RwLock>)
    And worker-001 appears exactly once in queen-rbee's global registry

  @concurrency @p0 @queen-rbee-registry
  Scenario: Gap-C2 - Race condition on worker state update in queen-rbee registry
    Given worker-001 is registered in queen-rbee with state "idle"
    And worker-001 has 1 slot available
    When rbee-hive-A sends PATCH to "http://localhost:8080/v1/workers/worker-001" with state "busy" at T+0ms
    And rbee-hive-B sends PATCH to "http://localhost:8080/v1/workers/worker-001" with state "busy" at T+1ms
    Then queen-rbee processes updates sequentially (RwLock guarantees)
    And both updates succeed (last-write-wins for state cache)
    And no state corruption occurs
    And worker state in queen-rbee is "busy"

  # DELETED by TEAM-080: Scenario Gap-C3
  # Reason: Each rbee-hive has separate SQLite catalog (~/.rbee/models.db)
  # No shared database = no concurrent INSERT conflicts possible
  # If shared catalog is implemented (PostgreSQL), add this scenario back

  # MOVED by TEAM-080: Scenario Gap-C4
  # Reason: Slot allocation happens AT THE WORKER, not in queen-rbee registry
  # queen-rbee only caches slot availability (eventually consistent)
  # See 130-inference-execution.feature for worker-level slot tests
  # See 060-rbee-hive-worker-registry.feature for slot tracking tests

  # MOVED by TEAM-080: Scenario Gap-C5
  # Reason: Download coordination happens at rbee-hive level, not queen-rbee
  # Each rbee-hive has separate catalog and downloads independently
  # For download coordination within single rbee-hive, see 030-model-provisioner.feature
  # Note: Current architecture allows duplicate downloads across nodes (by design)

  @concurrency @p1 @queen-rbee-registry
  Scenario: Gap-C6 - queen-rbee registry cleanup during active registration
    Given queen-rbee's stale worker cleanup task is running (removes workers with heartbeat >120s)
    When new worker registration arrives via POST to "/v1/workers/register"
    Then registration acquires write lock and completes successfully
    And cleanup task waits for write lock (no interference)
    And no deadlocks occur (RwLock prevents deadlock)
    And new worker appears in registry after cleanup completes

  @concurrency @p1 @queen-rbee-registry
  Scenario: Gap-C7 - Heartbeat update during state transition in queen-rbee
    Given worker-001 state in queen-rbee is transitioning from "idle" to "busy" (PATCH request in progress)
    When heartbeat POST to "/v1/workers/worker-001/heartbeat" arrives mid-transition
    Then heartbeat waits for write lock (RwLock sequential writes)
    And state transition completes first
    Then heartbeat update processes and updates last_heartbeat_unix
    And state remains consistent ("busy" with fresh heartbeat)
    And no partial updates occur (atomic write operations)
