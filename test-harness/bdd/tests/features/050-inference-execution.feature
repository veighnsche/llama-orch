# Traceability: TEST-001 (split by TEAM-077)
# Architecture: TEAM-037 (queen-rbee orchestration)
# Components: rbee-keeper, llm-worker-rbee (worker)
# Refactored by: TEAM-077 (split from test-001.feature into focused feature files)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: Inference Execution
  As a system executing inference requests
  I want to send requests to workers and stream tokens
  So that users can get LLM responses

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"

  Scenario: Inference request with SSE streaming
    Given the worker is ready and idle
    When rbee-keeper sends inference request:
      """
      POST http://workstation.home.arpa:8081/v1/inference
      Authorization: Bearer <worker_api_key>
      Content-Type: application/json

      {
        "prompt": "write a short story",
        "max_tokens": 20,
        "temperature": 0.7,
        "stream": true
      }
      """
    Then the worker responds with SSE stream:
      """
      HTTP/1.1 200 OK
      Content-Type: text/event-stream

      data: {"token": "Once", "index": 0}
      data: {"token": " upon", "index": 1}
      data: {"token": " a", "index": 2}
      data: {"token": " time", "index": 3}
      data: {"done": true, "total_tokens": 20, "duration_ms": 1234}
      data: [DONE]
      """
    And rbee-keeper streams tokens to stdout in real-time
    And the worker transitions from "idle" to "busy" to "idle"
    And the exit code is 0

  @error-handling
  Scenario: EH-018a - Worker busy with all slots occupied
    Given the worker is in state "busy"
    When rbee-keeper sends inference request
    Then the worker responds with:
      """
      HTTP/1.1 503 Service Unavailable
      Content-Type: application/json

      {
        "error": {
          "code": "ALL_SLOTS_BUSY",
          "message": "Worker is busy, try again later",
          "slots_total": 1,
          "slots_busy": 1
        }
      }
      """
    And rbee-keeper retries with exponential backoff
    And retry 1 has delay 1 second
    And retry 2 has delay 2 seconds
    And retry 3 has delay 4 seconds
    And if still busy after 3 retries, rbee-keeper aborts
    And rbee-keeper displays:
      """
      [worker] ⏳ Worker busy, retrying in 1s...
      [worker] ⏳ Worker busy, retrying in 2s...
      [worker] ⏳ Worker busy, retrying in 4s...
      [rbee-keeper] ❌ Error: Worker still busy after 3 retries
      
      Suggestions:
        - Wait for current request to complete
        - Use a different node
        - Spawn additional worker for this model
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-013a - Worker crashes during inference
    Given inference is streaming tokens
    When worker process crashes unexpectedly
    Then rbee-keeper detects SSE stream closed
    And rbee-keeper saves partial results
    And rbee-keeper displays:
      """
      Once upon a time, in a small village, there lived a curious cat
      
      [rbee-keeper] ❌ Error: SSE stream closed unexpectedly
        Worker may have crashed
        
      Partial result saved to: /tmp/rbee-partial-abc123.txt
      Tokens generated: 12 / 20
      """
    And rbee-hive removes worker from registry
    And the exit code is 1

  @error-handling
  Scenario: EH-013b - Worker hangs during inference
    Given inference has started
    When worker stops responding
    And no tokens generated for 60 seconds
    Then rbee-keeper detects stall timeout
    And rbee-keeper cancels request
    And rbee-keeper displays:
      """
      [rbee-keeper] ❌ Error: Worker timeout - no response for 60s
        Request may be stuck
        
      Canceling request...
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-003a - Worker HTTP connection lost mid-inference
    Given inference is streaming tokens
    When network connection drops
    Then rbee-keeper detects connection loss within 5s
    And rbee-keeper displays partial results
    And rbee-keeper displays:
      """
      Once upon a time, in a small
      
      [rbee-keeper] ❌ Error: Worker connection lost
        Network may be down
        
      Partial result: 8 tokens generated
      """
    And the exit code is 1

  @edge-case
  Scenario: EC1 - Connection timeout with retry and backoff
    Given node "workstation" is unreachable
    When rbee-keeper attempts connection
    Then rbee-keeper displays:
      """
      Attempt 1: Connecting to workstation.home.arpa:8080... (timeout 10s)
      Attempt 2: Connecting to workstation.home.arpa:8080... (timeout 10s, delay 200ms)
      Attempt 3: Connecting to workstation.home.arpa:8080... (timeout 10s, delay 400ms)
      Error: Cannot connect to workstation.home.arpa:8080 after 3 attempts
      Suggestion: Check if rbee-hive is running on workstation
      """
    And the exit code is 1

  @edge-case
  Scenario: EC4 - Worker crash during inference
    Given the worker is streaming tokens
    When the worker process dies unexpectedly
    Then rbee-keeper detects SSE stream closed
    And rbee-keeper displays:
      """
      Once upon a time, in a small village, there lived a curious cat
      Error: SSE stream closed unexpectedly
      
      Partial result saved to: /tmp/rbee-partial-abc123.txt
      Tokens generated: 12 / 20
      """
    And rbee-hive removes worker from registry
    And rbee-hive logs crash event
    And the exit code is 1

  @edge-case
  Scenario: EC6 - Queue full with retry
    Given the worker has 1 slot total
    And 1 slot is busy
    When rbee-keeper sends inference request
    Then the worker returns 503 "ALL_SLOTS_BUSY"
    And rbee-keeper retries with backoff
    And rbee-keeper displays:
      """
      Worker is busy, retrying in 1 second...
      Worker is busy, retrying in 2 seconds...
      Worker is busy, retrying in 4 seconds...
      Error: Worker still busy after 3 retries
      Suggestion: Wait for current request to complete or use a different node
      """
    And the exit code is 1

  @error-handling @cancellation
  Scenario: Gap-G12a - Client cancellation with Ctrl+C
    Given inference is in progress
    When the user presses Ctrl+C
    Then rbee-keeper sends:
      """
      DELETE http://workstation.home.arpa:8081/v1/inference/<request_id>
      """
    And rbee-keeper waits for acknowledgment with timeout 5s
    And worker stops token generation immediately
    And worker releases slot and returns to idle
    And rbee-keeper displays:
      """
      Once upon a time, in a small
      
      ^C
      [rbee-keeper] 🛑 Canceling request...
      [worker] ✅ Request canceled, slot released
      """
    And the exit code is 130

  @error-handling @cancellation
  Scenario: Gap-G12b - Client disconnects during inference
    Given inference is streaming tokens
    When client closes connection unexpectedly
    Then worker detects SSE stream closure within 1s
    And worker stops token generation immediately
    And worker releases slot
    And worker logs cancellation event
    And worker returns to idle state

  @error-handling @cancellation
  Scenario: Gap-G12c - Explicit cancellation endpoint
    Given inference is in progress with request_id "req-123"
    When rbee-keeper sends:
      """
      DELETE http://workstation.home.arpa:8081/v1/inference/req-123
      """
    Then worker responds with:
      """
      HTTP/1.1 204 No Content
      """
    And worker stops inference
    And worker releases slot
    And subsequent DELETE requests are idempotent (also return 204)
