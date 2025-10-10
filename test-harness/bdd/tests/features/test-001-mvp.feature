# Traceability: TEST-001-MVP
# Architecture: TEAM-030 (in-memory worker registry, SQLite model catalog)
# Components: rbee-keeper, rbee-hive, llm-worker-rbee
# Scope: MVP scenarios only - critical path and essential edge cases

Feature: Cross-Node Inference MVP
  As a user on the control node
  I want to run inference on a remote compute node with minimal viable functionality
  So that I can validate the core distributed inference architecture

  Background:
    Given the following topology:
      | node | hostname           | components                         | capabilities |
      | blep | blep.home.arpa     | rbee-keeper, queen-rbee, rbee-hive | cpu          |
      | mac  | mac.home.arpa      | rbee-hive, llm-worker-rbee         | metal:0      |
    And I am on node "blep"
    And the model catalog is SQLite at "~/.rbee/models.db"
    And the worker registry is in-memory ephemeral

  # ============================================================================
  # MVP-001: HAPPY PATH - Cold Start
  # ============================================================================

  @mvp @critical @happy-path
  Scenario: MVP-001 - Cold start inference on remote node
    Given no workers are registered
    And node "mac" is reachable at "http://mac.home.arpa:8080"
    And node "mac" has 8000 MB of available RAM
    And node "mac" has Metal backend available
    And the model "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" is not in the catalog
    When I run:
      """
      rbee-keeper infer \
        --node mac \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --prompt "write a short story" \
        --max-tokens 20 \
        --temperature 0.7
      """
    Then rbee-keeper queries worker registry and finds no workers
    And rbee-keeper performs health check at "http://mac.home.arpa:8080/v1/health"
    And health check returns version "0.1.0" and status "alive"
    And rbee-hive checks model catalog and finds model missing
    And rbee-hive downloads model from Hugging Face with progress stream
    And rbee-keeper displays download progress bar
    And model is registered in SQLite catalog at "/models/tinyllama-q4.gguf"
    And rbee-hive performs RAM check: 8000 MB available >= 6000 MB required
    And rbee-hive performs backend check: Metal available
    And rbee-hive spawns worker on port 8081
    And worker sends ready callback to pool manager
    And worker is registered in in-memory registry with state "loading"
    And rbee-keeper polls worker readiness at "http://mac.home.arpa:8081/v1/ready"
    And worker streams loading progress showing layers loaded
    And worker completes loading and returns state "ready"
    And rbee-keeper sends inference request to worker
    And worker streams 20 tokens via SSE
    And rbee-keeper displays tokens to stdout
    And inference completes successfully
    And worker transitions to state "idle"
    And the exit code is 0

  # ============================================================================
  # MVP-002: HAPPY PATH - Warm Start
  # ============================================================================

  @mvp @critical @happy-path
  Scenario: MVP-002 - Warm start with existing idle worker
    Given a worker is registered with:
      | field      | value                                           |
      | id         | worker-abc123                                   |
      | url        | http://mac.home.arpa:8081                       |
      | model_ref  | hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF       |
      | state      | idle                                            |
      | backend    | metal                                           |
    And the worker is healthy
    When I run:
      """
      rbee-keeper infer \
        --node mac \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --prompt "write a poem" \
        --max-tokens 20
      """
    Then rbee-keeper queries worker registry
    And registry returns worker "worker-abc123" with state "idle"
    And rbee-keeper skips pool preflight, model provisioning, and worker startup
    And rbee-keeper sends inference request directly to "http://mac.home.arpa:8081/v1/inference"
    And worker streams 20 tokens via SSE
    And rbee-keeper displays tokens to stdout
    And inference completes successfully
    And total latency is under 5 seconds
    And the exit code is 0

  # ============================================================================
  # MVP-003: Model Provisioning
  # ============================================================================

  @mvp @critical
  Scenario: MVP-003 - Model found in catalog (skip download)
    Given the model catalog contains:
      | provider | reference                                 | local_path                  |
      | hf       | TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF    | /models/tinyllama-q4.gguf   |
    And no workers are registered
    When rbee-keeper initiates inference for the model
    Then rbee-hive queries SQLite catalog
    And catalog returns local_path "/models/tinyllama-q4.gguf"
    And rbee-hive skips download
    And rbee-hive proceeds directly to worker preflight

  @mvp @critical
  Scenario: MVP-004 - Model download with progress stream
    Given the model is not in the catalog
    And Hugging Face API is reachable
    When rbee-hive initiates model download
    Then rbee-hive creates SSE endpoint "/v1/models/download/progress"
    And rbee-keeper connects to the SSE stream
    And the stream emits:
      """
      data: {"stage": "downloading", "bytes_downloaded": 1048576, "bytes_total": 5242880, "speed_mbps": 45.2}
      data: {"stage": "downloading", "bytes_downloaded": 3145728, "bytes_total": 5242880, "speed_mbps": 48.1}
      data: {"stage": "complete", "local_path": "/models/tinyllama-q4.gguf"}
      data: [DONE]
      """
    And rbee-keeper displays progress bar: "[████████████████████----] 80% (4.0 MB / 5.0 MB) @ 45.2 MB/s"
    And rbee-hive inserts model into SQLite catalog
    And download completes successfully

  # ============================================================================
  # MVP-005: Worker Lifecycle
  # ============================================================================

  @mvp @critical
  Scenario: MVP-005 - Worker startup and ready callback
    Given worker preflight checks passed
    When rbee-hive spawns worker process
    Then the command is:
      """
      llm-worker-rbee \
        --model /models/tinyllama-q4.gguf \
        --backend metal \
        --device 0 \
        --port 8081 \
        --api-key <worker_api_key>
      """
    And worker HTTP server binds to port 8081
    And worker sends POST to "http://mac.home.arpa:8080/v1/workers/ready" with:
      """
      {
        "worker_id": "worker-abc123",
        "url": "http://mac.home.arpa:8081",
        "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "backend": "metal",
        "device": 0
      }
      """
    And rbee-hive registers worker in in-memory registry
    And model loading begins asynchronously
    And rbee-hive returns worker details with state "loading"

  @mvp @critical
  Scenario: MVP-006 - Worker loading progress stream
    Given worker is loading model to VRAM
    And rbee-keeper polls "/v1/ready" and receives state "loading"
    When rbee-keeper connects to "/v1/loading/progress"
    Then the SSE stream emits:
      """
      data: {"stage": "loading_to_vram", "layers_loaded": 8, "layers_total": 32, "vram_mb": 1024}
      data: {"stage": "loading_to_vram", "layers_loaded": 16, "layers_total": 32, "vram_mb": 2048}
      data: {"stage": "loading_to_vram", "layers_loaded": 24, "layers_total": 32, "vram_mb": 3072}
      data: {"stage": "loading_to_vram", "layers_loaded": 32, "layers_total": 32, "vram_mb": 4096}
      data: {"stage": "ready"}
      data: [DONE]
      """
    And rbee-keeper displays: "Loading model to VRAM... [████████████████████████] 100% (32/32 layers)"
    And worker transitions to state "ready"

  @mvp @critical
  Scenario: MVP-007 - Inference execution with SSE token stream
    Given worker is ready and idle
    When rbee-keeper sends POST to "/v1/inference" with:
      """
      {
        "prompt": "write a short story",
        "max_tokens": 20,
        "temperature": 0.7,
        "stream": true
      }
      """
    Then worker responds with SSE stream:
      """
      data: {"token": "Once", "index": 0}
      data: {"token": " upon", "index": 1}
      data: {"token": " a", "index": 2}
      data: {"token": " time", "index": 3}
      data: {"token": ",", "index": 4}
      data: {"token": " in", "index": 5}
      data: {"token": " a", "index": 6}
      data: {"token": " small", "index": 7}
      data: {"token": " village", "index": 8}
      data: {"done": true, "total_tokens": 20, "duration_ms": 1234}
      data: [DONE]
      """
    And rbee-keeper streams tokens to stdout in real-time
    And worker transitions: idle → busy → idle
    And the exit code is 0

  # ============================================================================
  # MVP-008: Pool Manager Lifecycle
  # ============================================================================

  @mvp @critical @lifecycle
  Scenario: MVP-008 - Pool manager remains running as persistent HTTP daemon
    Given rbee-hive is started as HTTP daemon on port 8080
    And rbee-hive spawned a worker
    When worker sends ready callback
    Then rbee-hive does NOT exit
    And rbee-hive continues monitoring worker health every 30 seconds
    And rbee-hive enforces idle timeout of 5 minutes
    And rbee-hive remains available for new worker requests
    And rbee-hive HTTP API remains accessible at "http://mac.home.arpa:8080"

  @mvp @critical @lifecycle
  Scenario: MVP-009 - Worker idle timeout and auto-shutdown (worker dies, pool lives)
    Given rbee-hive is running as persistent daemon
    And worker completed inference at T+0:00
    And worker is idle
    When 5 minutes elapse without new requests
    Then pool manager sends "POST /v1/admin/shutdown" at T+5:00
    And worker unloads model from VRAM at T+5:01
    And worker exits cleanly at T+5:02
    And pool manager removes worker from in-memory registry at T+5:02
    And VRAM is available for other applications
    And rbee-hive continues running as daemon (does NOT exit)

  @mvp @critical @lifecycle
  Scenario: MVP-010 - rbee-keeper exits after inference (CLI dies, daemons live)
    Given rbee-hive is running as persistent daemon on "mac"
    And worker is running and idle
    When rbee-keeper completes inference request
    Then rbee-keeper exits with code 0
    And rbee-hive continues running as daemon
    And worker continues running as daemon
    And worker remains in rbee-hive's in-memory registry

  @mvp @critical @lifecycle
  Scenario: MVP-011 - Cascading shutdown when rbee-hive receives SIGTERM
    Given rbee-hive is running as persistent daemon
    And 3 workers are registered and running
    When user sends SIGTERM to rbee-hive (Ctrl+C)
    Then rbee-hive sends "POST /v1/admin/shutdown" to all 3 workers
    And rbee-hive waits for workers to acknowledge (max 5s per worker)
    And all workers unload models and exit
    And rbee-hive clears in-memory registry
    And rbee-hive exits cleanly
    And model catalog (SQLite) persists on disk

  @mvp @critical @lifecycle
  Scenario: MVP-012 - rbee-hive spawned by rbee-keeper (ephemeral mode)
    Given rbee-keeper is configured to spawn rbee-hive
    When rbee-keeper runs inference command
    Then rbee-keeper spawns rbee-hive as child process
    And rbee-hive starts HTTP daemon
    And rbee-hive spawns worker
    And inference completes
    And rbee-keeper sends SIGTERM to rbee-hive
    And rbee-hive cascades shutdown to worker
    And worker exits
    And rbee-hive exits
    And rbee-keeper exits with code 0

  @mvp @critical @lifecycle
  Scenario: MVP-013 - rbee-hive pre-started (persistent mode)
    Given rbee-hive is already running as daemon on "mac"
    And rbee-hive was started manually by operator
    When rbee-keeper runs inference command
    Then rbee-keeper connects to existing rbee-hive HTTP API
    And rbee-keeper does NOT spawn rbee-hive
    And inference completes
    And rbee-keeper exits
    And rbee-hive continues running (was not spawned by rbee-keeper)
    And worker continues running (idle timeout not reached)

  # ============================================================================
  # MVP EDGE CASES (Essential Only)
  # ============================================================================

  @mvp @edge-case @critical
  Scenario: MVP-EC1 - Connection timeout with retry
    Given node "mac" is unreachable
    When rbee-keeper attempts connection with timeout 10s
    Then rbee-keeper retries 3 times with exponential backoff:
      | attempt | delay |
      | 1       | 0ms   |
      | 2       | 200ms |
      | 3       | 400ms |
    And rbee-keeper displays:
      """
      Attempt 1: Connecting to mac.home.arpa:8080... (timeout 10s)
      Attempt 2: Connecting to mac.home.arpa:8080... (timeout 10s, delay 200ms)
      Attempt 3: Connecting to mac.home.arpa:8080... (timeout 10s, delay 400ms)
      Error: Cannot connect to mac.home.arpa:8080 after 3 attempts
      Suggestion: Check if rbee-hive is running on mac
      """
    And the exit code is 1

  @mvp @edge-case @critical
  Scenario: MVP-EC2 - Insufficient RAM
    Given the model requires 6000 MB (5000 MB * 1.2)
    And node "mac" has only 4000 MB available RAM
    When rbee-hive performs RAM preflight check
    Then the check fails
    And rbee-keeper displays:
      """
      Error: Insufficient RAM on mac
        Required: 6000 MB
        Available: 4000 MB
        
      Suggestion: Try a smaller quantized model (Q4 instead of Q8)
      """
    And the exit code is 1

  @mvp @edge-case @critical
  Scenario: MVP-EC3 - Worker crash during inference
    Given worker is streaming tokens
    And 12 tokens have been generated
    When worker process dies unexpectedly
    Then rbee-keeper detects SSE stream closed
    And rbee-keeper displays:
      """
      Once upon a time, in a small village, there lived a curious cat
      Error: SSE stream closed unexpectedly
      
      Partial result saved to: /tmp/rbee-partial-abc123.txt
      Tokens generated: 12 / 20
      """
    And pool manager removes worker from registry
    And pool manager logs crash event
    And the exit code is 1

  @mvp @edge-case @critical
  Scenario: MVP-EC4 - Client cancellation with Ctrl+C
    Given inference is in progress
    And 8 tokens have been generated
    When user presses Ctrl+C
    Then rbee-keeper catches SIGINT signal
    And rbee-keeper sends "DELETE http://mac.home.arpa:8081/v1/inference/<request_id>"
    And rbee-keeper waits for acknowledgment with timeout 5s
    And rbee-keeper displays "\nCanceled."
    And worker stops token generation immediately
    And worker releases slot and returns to idle
    And the exit code is 130

  @mvp @edge-case @critical
  Scenario: MVP-EC5 - Worker busy (all slots occupied)
    Given worker has 1 slot total
    And 1 slot is busy with another request
    When rbee-keeper sends inference request
    Then worker responds with:
      """
      HTTP/1.1 503 Service Unavailable
      {
        "error": {
          "code": "ALL_SLOTS_BUSY",
          "message": "Worker is busy, try again later",
          "slots_total": 1,
          "slots_busy": 1
        }
      }
      """
    And rbee-keeper retries with exponential backoff:
      | retry | delay     |
      | 1     | 1 second  |
      | 2     | 2 seconds |
      | 3     | 4 seconds |
    And if still busy after 3 retries, rbee-keeper displays:
      """
      Worker is busy, retrying in 1 second...
      Worker is busy, retrying in 2 seconds...
      Worker is busy, retrying in 4 seconds...
      Error: Worker still busy after 3 retries
      Suggestion: Wait for current request to complete or use a different node
      """
    And the exit code is 1

  @mvp @edge-case
  Scenario: MVP-EC6 - Version mismatch
    Given rbee-keeper version is "0.1.0"
    And rbee-hive version is "0.2.0"
    When rbee-keeper performs health check
    Then rbee-keeper detects version mismatch
    And rbee-keeper displays:
      """
      Error: Version mismatch
        rbee-keeper: v0.1.0
        rbee-hive: v0.2.0
        
      Please upgrade rbee-keeper to v0.2.0:
        cargo install rbee-keeper --version 0.2.0
      """
    And the exit code is 1

  @mvp @edge-case
  Scenario: MVP-EC7 - Invalid API key
    Given rbee-keeper uses API key "wrong_key"
    When rbee-keeper sends request with "Authorization: Bearer wrong_key"
    Then rbee-hive responds with:
      """
      HTTP/1.1 401 Unauthorized
      {
        "error": {
          "code": "INVALID_API_KEY",
          "message": "Invalid or missing API key"
        }
      }
      """
    And rbee-keeper displays:
      """
      Error: Authentication failed
        Invalid API key for mac.home.arpa
        
      Check your configuration:
        ~/.rbee/config.yaml
      """
    And the exit code is 1

  @mvp @edge-case
  Scenario: MVP-EC8 - Model loading timeout
    Given worker is loading model
    And 5 minutes have elapsed
    And worker is stuck at 28/32 layers
    When rbee-keeper timeout expires
    Then rbee-keeper displays:
      """
      Error: Model loading timeout after 5 minutes
      Worker state: loading (stuck at 28/32 layers)
      
      Suggestion: Check worker logs on mac for errors
      """
    And the exit code is 1

  @mvp @edge-case
  Scenario: MVP-EC9 - Model download failure with retry
    Given model download fails at 40% with "Connection reset by peer"
    When rbee-hive retries download
    Then rbee-hive attempts up to 6 retries with exponential backoff
    And rbee-hive displays:
      """
      Downloading model... [████████------------] 40% (2.0 MB / 5.0 MB)
      Error: Connection reset by peer
      Retrying download (attempt 2/6)... delay 100ms
      Downloading model... [████████████--------] 60% (3.0 MB / 5.0 MB)
      """
    And if all 6 attempts fail, rbee-hive returns error "DOWNLOAD_FAILED"
    And rbee-keeper displays the error
    And the exit code is 1

  @mvp @edge-case
  Scenario: MVP-EC10 - Backend unavailable
    Given requested backend is "cuda"
    And node "mac" only has Metal backend
    When rbee-hive performs backend preflight check
    Then the check fails
    And rbee-hive returns error "BACKEND_UNAVAILABLE"
    And rbee-keeper displays:
      """
      Error: Backend unavailable on mac
        Requested: cuda
        Available: metal
        
      Suggestion: Use --backend metal or choose a different node
      """
    And the exit code is 1

  # ============================================================================
  # MVP ERROR FORMAT VALIDATION
  # ============================================================================

  @mvp
  Scenario: MVP - Error response format is consistent
    Given an error occurs with code "VRAM_EXHAUSTED"
    When the error is returned to rbee-keeper
    Then the response format is:
      """
      {
        "error": {
          "code": "VRAM_EXHAUSTED",
          "message": "Insufficient VRAM: need 6000 MB, have 4000 MB",
          "details": {
            "required": 6000,
            "available": 4000
          }
        }
      }
      """
    And the error code is one of:
      | code                  |
      | CONNECTION_TIMEOUT    |
      | VERSION_MISMATCH      |
      | MODEL_NOT_FOUND       |
      | DOWNLOAD_FAILED       |
      | VRAM_EXHAUSTED        |
      | INVALID_API_KEY       |
      | ALL_SLOTS_BUSY        |
      | WORKER_CRASHED        |
      | LOADING_TIMEOUT       |
      | REQUEST_CANCELED      |
      | BACKEND_UNAVAILABLE   |
    And the message is human-readable
    And the details provide actionable context

  # ============================================================================
  # MVP SUCCESS CRITERIA
  # ============================================================================

  @mvp @acceptance
  Scenario: MVP - Success criteria validation
    Given the MVP implementation is complete
    Then the following criteria are met:
      | criterion                                          | status |
      | Model downloads with progress bar                  | ✓      |
      | Worker starts and loads model                      | ✓      |
      | Inference streams tokens in real-time              | ✓      |
      | Worker auto-shuts down after 5 minutes idle        | ✓      |
      | Total latency < 30 seconds (cold start)            | ✓      |
      | Connection failures retry with backoff             | ✓      |
      | Version mismatches detected and reported           | ✓      |
      | RAM exhaustion prevents worker startup             | ✓      |
      | Worker crashes return partial results              | ✓      |
      | Ctrl+C cancels gracefully                          | ✓      |
      | Busy workers return 503 with retry suggestion      | ✓      |
      | Invalid auth returns 401 with helpful message      | ✓      |

  # ============================================================================
  # LIFECYCLE RULES - CRITICAL UNDERSTANDING
  # ============================================================================
  #
  # RULE 1: rbee-hive is a PERSISTENT HTTP DAEMON
  #   - Starts: `rbee-hive daemon` or spawned by rbee-keeper
  #   - Runs: Continuously as HTTP server on port 8080
  #   - Dies: ONLY when receiving SIGTERM (Ctrl+C) or explicit shutdown
  #   - Does NOT die: After spawning workers, after inference completes
  #
  # RULE 2: llm-worker-rbee is a PERSISTENT HTTP DAEMON
  #   - Starts: Spawned by rbee-hive
  #   - Runs: Continuously as HTTP server on port 8001+
  #   - Dies: When idle timeout (5 min) OR rbee-hive sends shutdown OR SIGTERM
  #   - Does NOT die: After inference completes (stays idle)
  #
  # RULE 3: rbee-keeper is a CLI (EPHEMERAL)
  #   - Starts: User runs command
  #   - Runs: Only during command execution
  #   - Dies: After command completes (exit code 0 or 1)
  #   - Does NOT die: Never stays running
  #
  # RULE 4: Ephemeral Mode (rbee-keeper spawns rbee-hive)
  #   - rbee-keeper spawns rbee-hive as child process
  #   - rbee-hive spawns worker
  #   - Inference completes
  #   - rbee-keeper sends SIGTERM to rbee-hive
  #   - rbee-hive cascades shutdown to worker
  #   - All processes exit
  #
  # RULE 5: Persistent Mode (rbee-hive pre-started)
  #   - Operator starts: `rbee-hive daemon &`
  #   - rbee-hive runs continuously
  #   - rbee-keeper connects to existing rbee-hive
  #   - Inference completes
  #   - rbee-keeper exits
  #   - rbee-hive continues running
  #   - Worker continues running (until idle timeout)
  #
  # RULE 6: Cascading Shutdown
  #   - SIGTERM → rbee-hive
  #   - rbee-hive → POST /v1/admin/shutdown → all workers
  #   - Workers unload models and exit
  #   - rbee-hive clears registry and exits
  #   - Model catalog (SQLite) persists on disk
  #
  # RULE 7: Worker Idle Timeout
  #   - Worker completes inference → idle
  #   - 5 minutes elapse without new requests
  #   - rbee-hive sends shutdown to worker
  #   - Worker exits, VRAM freed
  #   - rbee-hive continues running
  #
  # RULE 8: Process Ownership
  #   - IF rbee-keeper spawned rbee-hive → rbee-keeper owns lifecycle
  #   - IF operator started rbee-hive → operator owns lifecycle
  #   - rbee-hive always owns worker lifecycle
  #   - Workers never own their own lifecycle (managed by rbee-hive)

---
# Created by: TEAM-037 (Testing Team)
# Scope: MVP scenarios only - critical path and essential edge cases
# Full test suite: test-001.feature
# 
# LIFECYCLE SUMMARY:
# - rbee-hive: PERSISTENT HTTP DAEMON (dies only on SIGTERM)
# - llm-worker-rbee: PERSISTENT HTTP DAEMON (dies on idle timeout or shutdown)
# - rbee-keeper: EPHEMERAL CLI (dies after command completes)
# - Ephemeral mode: rbee-keeper spawns rbee-hive, controls lifecycle
# - Persistent mode: rbee-hive pre-started, rbee-keeper just connects
#
# Verified by Testing Team 🔍
