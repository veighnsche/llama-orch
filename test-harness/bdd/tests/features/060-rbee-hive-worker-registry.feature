# Traceability: TEST-001 (split by TEAM-077)
# Architecture: TEAM-037 (queen-rbee orchestration)
# Components: queen-rbee, rbee-hive
# Refactored by: TEAM-077 (reorganized to correct BDD architecture)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: rbee-hive Worker Registry
  As a system managing rbee-hive instances
  I want to perform health checks and manage rbee-hive state
  So that I can ensure rbee-hive is ready for worker spawning

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"

  Scenario: Pool preflight health check succeeds
    Given node "workstation" is reachable
    When rbee-keeper sends GET to "http://workstation.home.arpa:9200/v1/health"
    Then the response status is 200
    And the response body contains:
      """
      {
        "status": "alive",
        "version": "0.1.0",
        "api_version": "v1"
      }
      """
    And rbee-keeper proceeds to model provisioning

  Scenario: Worker registry returns empty list
    Given no workers are registered
    When queen-rbee queries "http://workstation.home.arpa:9200/v1/workers/list"
    Then the response is:
      """
      {
        "workers": []
      }
      """
    And rbee-keeper proceeds to pool preflight

  Scenario: Worker registry returns matching idle worker
    Given a worker is registered with model_ref "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" and state "idle"
    When rbee-keeper queries the worker registry
    Then rbee-keeper skips to Phase 8 (inference execution)

  Scenario: Worker registry returns matching busy worker
    Given a worker is registered with model_ref "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" and state "busy"
    When rbee-keeper queries the worker registry
    Then rbee-keeper proceeds to Phase 8 but expects 503 response

  @error-handling
  Scenario: EH-002a - rbee-hive HTTP connection timeout
    Given rbee-hive is unreachable
    When queen-rbee attempts to connect with 10s timeout
    Then the connection times out
    And queen-rbee retries 3 times with exponential backoff
    And rbee-keeper displays:
      """
      [queen-rbee] 🔌 Connecting to rbee-hive at workstation.home.arpa:9200
      [queen-rbee] ⏳ Attempt 1/3 failed: Connection timeout
      [queen-rbee] ⏳ Attempt 2/3 failed: Connection timeout (delay 200ms)
      [queen-rbee] ⏳ Attempt 3/3 failed: Connection timeout (delay 400ms)
      [queen-rbee] ❌ Cannot connect to rbee-hive after 3 attempts
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-002b - rbee-hive returns malformed JSON
    Given rbee-hive is running but buggy
    When queen-rbee queries worker registry
    And rbee-hive returns invalid JSON: "{ workers: [ incomplete"
    Then queen-rbee detects JSON parse error
    And rbee-keeper displays:
      """
      [queen-rbee] ❌ Error: Invalid response from rbee-hive
        Expected valid JSON, got parse error at position 12
        
      Suggestion: rbee-hive may be corrupted, try restarting:
        ssh workstation pkill rbee-hive
      """
    And the exit code is 1

  Scenario: Pool preflight connection timeout with retry
    Given node "workstation" is unreachable
    When rbee-keeper attempts to connect with timeout 10s
    Then rbee-keeper retries 3 times with exponential backoff
    And attempt 1 has delay 0ms
    And attempt 2 has delay 200ms
    And attempt 3 has delay 400ms
    And rbee-keeper aborts with error "CONNECTION_TIMEOUT"
    And the error suggests checking if rbee-hive is running
    And the exit code is 1

  Scenario: Pool preflight detects version mismatch
    Given rbee-keeper version is "0.1.0"
    And rbee-hive version is "0.2.0"
    When rbee-keeper performs health check
    Then rbee-keeper aborts with error "VERSION_MISMATCH"
    And the error message includes both versions
    And the error suggests upgrading rbee-keeper
    And the exit code is 1

  @edge-case
  Scenario: EC8 - Version mismatch
    Given rbee-keeper version is "0.1.0"
    And rbee-hive version is "0.2.0"
    When rbee-keeper performs version check
    Then rbee-keeper displays:
      """
      Error: Version mismatch
        rbee-keeper: v0.1.0
        rbee-hive: v0.2.0
        
      Please upgrade rbee-keeper to v0.2.0:
        cargo install rbee-keeper --version 0.2.0
      """
    And the exit code is 1
