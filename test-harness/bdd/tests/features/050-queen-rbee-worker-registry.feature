# Traceability: TEST-001 Phase 4 (global worker registry)
# Architecture: TEAM-037 (queen-rbee orchestration, in-memory registry)
# Components: queen-rbee (global orchestrator)
# Created by: TEAM-078
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# ⚠️ DO NOT use mock servers - wire up actual queen-rbee WorkerRegistry

Feature: queen-rbee Worker Registry
  As a global orchestrator
  I want to maintain a registry of all workers across rbee-hive instances
  So that I can route inference requests to available workers

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"

  Scenario: Register worker from rbee-hive
    Given queen-rbee has no workers registered
    When rbee-hive reports worker "worker-001" with capabilities ["cuda:0"]
    Then queen-rbee registers the worker
    And the request body is:
      """
      {
        "worker_id": "worker-001",
        "rbee_hive_url": "http://workstation.home.arpa:8081",
        "capabilities": ["cuda:0"],
        "models_loaded": ["tinyllama-q4"]
      }
      """
    And queen-rbee returns 201 Created
    And the worker is added to in-memory registry

  Scenario: Query all workers
    Given queen-rbee has workers:
      | worker_id  | rbee_hive_url                          | capabilities  | models_loaded      |
      | worker-001 | http://workstation.home.arpa:8081      | ["cuda:0"]    | ["tinyllama-q4"]   |
      | worker-002 | http://workstation.home.arpa:8081      | ["cuda:1"]    | ["llama2-7b-q4"]   |
    When rbee-keeper queries all workers
    Then queen-rbee returns 200 OK
    And the response contains 2 workers
    And each worker has worker_id, rbee_hive_url, capabilities, models_loaded

  Scenario: Filter by capability
    Given queen-rbee has workers:
      | worker_id  | rbee_hive_url                          | capabilities  | models_loaded      |
      | worker-001 | http://workstation.home.arpa:8081      | ["cuda:0"]    | ["tinyllama-q4"]   |
      | worker-002 | http://workstation.home.arpa:8081      | ["cpu"]       | ["tinyllama-q4"]   |
    When rbee-keeper queries workers with capability "cuda:0"
    Then queen-rbee returns 200 OK
    And the response contains 1 worker
    And the worker has worker_id "worker-001"

  Scenario: Update worker state
    Given queen-rbee has worker "worker-001" registered
    When rbee-hive updates worker state to "busy"
    Then queen-rbee receives PATCH request
    And the request body is:
      """
      {
        "state": "busy"
      }
      """
    And queen-rbee updates the worker state in registry
    And queen-rbee returns 200 OK

  Scenario: Remove worker
    Given queen-rbee has worker "worker-001" registered
    When rbee-hive removes worker "worker-001"
    Then queen-rbee receives DELETE request
    And queen-rbee removes the worker from registry
    And queen-rbee returns 204 No Content

  Scenario: Stale worker cleanup
    Given queen-rbee has workers:
      | worker_id  | last_heartbeat_unix | state  |
      | worker-001 | 1728508603          | ready  |
      | worker-002 | 1728508303          | ready  |
    And current time is 1728508903 (300 seconds later)
    When queen-rbee runs stale worker cleanup
    Then queen-rbee marks worker-002 as stale (no heartbeat for >120s)
    And queen-rbee removes worker-002 from registry
    And queen-rbee keeps worker-001 (heartbeat within 120s)
