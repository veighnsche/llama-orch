# Traceability: TEST-001
# Architecture: TEAM-030 (in-memory worker registry, SQLite model catalog)
# Components: rbee-keeper, rbee-hive, llm-worker-rbee, queen-rbee

Feature: Cross-Node Inference Request Flow
  As a user on the control node
  I want to run inference on a remote compute node
  So that I can utilize distributed GPU resources across my homelab

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee, rbee-hive              | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
      | mac         | mac.home.arpa         | rbee-hive, llm-worker-rbee                      | metal:0                |
    And I am on node "blep"
    And the model catalog is SQLite at "~/.rbee/models.db"
    And the worker registry is in-memory ephemeral

  # ============================================================================
  # HAPPY PATH: Full inference flow from cold start
  # ============================================================================

  Scenario: Happy path - cold start inference on remote node
    Given no workers are registered for model "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    And node "mac" is reachable at "http://mac.home.arpa:8080"
    And node "mac" has 8000 MB of available RAM
    And node "mac" has Metal backend available
    When I run:
      """
      rbee-keeper infer \
        --node mac \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --prompt "write a short story" \
        --max-tokens 20 \
        --temperature 0.7
      """
    Then rbee-keeper queries the worker registry at "http://mac.home.arpa:8080/v1/workers/list"
    And the worker registry returns an empty list
    And rbee-keeper performs pool preflight check at "http://mac.home.arpa:8080/v1/health"
    And the health check returns version "0.1.0" and status "alive"
    And rbee-hive checks the model catalog for "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    And the model is not found in the catalog
    And rbee-hive downloads the model from Hugging Face
    And a download progress SSE stream is available at "/v1/models/download/progress"
    And rbee-keeper displays a progress bar showing download percentage and speed
    And the model download completes successfully
    And rbee-hive registers the model in SQLite catalog with local_path "/models/tinyllama-q4.gguf"
    And rbee-hive performs worker preflight checks
    And RAM check passes with 8000 MB available
    And Metal backend check passes
    And rbee-hive spawns worker process "llm-worker-rbee" on port 8081
    And the worker HTTP server starts on port 8081
    And the worker sends ready callback to "http://mac.home.arpa:8080/v1/workers/ready"
    And rbee-hive registers the worker in the in-memory registry
    And rbee-hive returns worker details to rbee-keeper
    And rbee-keeper polls worker readiness at "http://mac.home.arpa:8081/v1/ready"
    And the worker returns state "loading" with progress_url
    And rbee-keeper streams loading progress showing layers loaded
    And the worker completes loading and returns state "ready"
    And rbee-keeper sends inference request to "http://mac.home.arpa:8081/v1/inference"
    And the worker streams tokens via SSE
    And rbee-keeper displays tokens to stdout in real-time
    And the inference completes with 20 tokens generated
    And the worker transitions to state "idle"
    And the exit code is 0

  Scenario: Warm start - reuse existing idle worker
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
    Then rbee-keeper queries the worker registry
    And the registry returns worker "worker-abc123" with state "idle"
    And rbee-keeper skips pool preflight and model provisioning
    And rbee-keeper sends inference request directly to "http://mac.home.arpa:8081/v1/inference"
    And the worker streams tokens via SSE
    And the inference completes successfully
    And the total latency is under 5 seconds
    And the exit code is 0

  # ============================================================================
  # PHASE 1: Worker Registry Check
  # ============================================================================

  Scenario: Worker registry returns empty list
    Given no workers are registered
    When rbee-keeper queries "http://mac.home.arpa:8080/v1/workers/list"
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

  # ============================================================================
  # PHASE 2: Pool Preflight
  # ============================================================================

  Scenario: Pool preflight health check succeeds
    Given node "mac" is reachable
    When rbee-keeper sends GET to "http://mac.home.arpa:8080/v1/health"
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

  Scenario: Pool preflight detects version mismatch
    Given rbee-keeper version is "0.1.0"
    And rbee-hive version is "0.2.0"
    When rbee-keeper performs health check
    Then rbee-keeper aborts with error "VERSION_MISMATCH"
    And the error message includes both versions
    And the error suggests upgrading rbee-keeper
    And the exit code is 1

  Scenario: Pool preflight connection timeout with retry
    Given node "mac" is unreachable
    When rbee-keeper attempts to connect with timeout 10s
    Then rbee-keeper retries 3 times with exponential backoff
    And attempt 1 has delay 0ms
    And attempt 2 has delay 200ms
    And attempt 3 has delay 400ms
    And rbee-keeper aborts with error "CONNECTION_TIMEOUT"
    And the error suggests checking if rbee-hive is running
    And the exit code is 1

  # ============================================================================
  # PHASE 3: Model Provisioning
  # ============================================================================

  Scenario: Model found in SQLite catalog
    Given the model catalog contains:
      | provider | reference                                 | local_path                  |
      | hf       | TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF    | /models/tinyllama-q4.gguf   |
    When rbee-hive checks the model catalog
    Then the query returns local_path "/models/tinyllama-q4.gguf"
    And rbee-hive skips model download
    And rbee-hive proceeds to worker preflight

  Scenario: Model not found - download with progress
    Given the model is not in the catalog
    When rbee-hive initiates download from Hugging Face
    Then rbee-hive creates SSE endpoint "/v1/models/download/progress"
    And rbee-keeper connects to the SSE stream
    And the stream emits progress events:
      """
      data: {"stage": "downloading", "bytes_downloaded": 1048576, "bytes_total": 5242880, "speed_mbps": 45.2}
      data: {"stage": "downloading", "bytes_downloaded": 2097152, "bytes_total": 5242880, "speed_mbps": 48.1}
      data: {"stage": "complete", "local_path": "/models/tinyllama-q4.gguf"}
      data: [DONE]
      """
    And rbee-keeper displays progress bar with percentage and speed
    And rbee-hive inserts model into SQLite catalog
    And rbee-hive proceeds to worker preflight

  Scenario: Model download fails with retry
    Given the model is not in the catalog
    When rbee-hive attempts download
    And the download fails with "Connection reset by peer" at 40% progress
    Then rbee-hive retries download with delay 100ms
    And rbee-hive resumes from last checkpoint
    And rbee-hive retries up to 6 times
    And if all retries fail, rbee-hive returns error "DOWNLOAD_FAILED"
    And rbee-keeper displays the error to the user
    And the exit code is 1

  Scenario: Model catalog registration after download
    Given the model downloaded successfully to "/models/tinyllama-q4.gguf"
    And the model size is 5242880 bytes
    When rbee-hive registers the model in the catalog
    Then the SQLite INSERT statement is:
      """
      INSERT INTO models (id, provider, reference, local_path, size_bytes, downloaded_at_unix)
      VALUES ('tinyllama-q4', 'hf', 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', 
              '/models/tinyllama-q4.gguf', 5242880, 1728508603)
      """
    And the catalog query now returns the model

  # ============================================================================
  # PHASE 4: Worker Preflight
  # ============================================================================

  Scenario: Worker preflight RAM check passes
    Given the model size is 5000 MB
    And the node has 8000 MB of available RAM
    When rbee-hive performs RAM check
    Then rbee-hive calculates required RAM as model_size * 1.2 = 6000 MB
    And the check passes because 8000 MB >= 6000 MB
    And rbee-hive proceeds to backend check

  Scenario: Worker preflight RAM check fails
    Given the model size is 5000 MB
    And the node has 4000 MB of available RAM
    When rbee-hive performs RAM check
    Then rbee-hive calculates required RAM as 6000 MB
    And the check fails because 4000 MB < 6000 MB
    And rbee-hive returns error "VRAM_EXHAUSTED"
    And the error includes required and available amounts
    And rbee-keeper suggests using a smaller quantized model
    And the exit code is 1

  Scenario: Worker preflight backend check passes
    Given the requested backend is "metal"
    And node "mac" has Metal backend available
    When rbee-hive performs backend check
    Then the check passes
    And rbee-hive proceeds to worker startup

  Scenario: Worker preflight backend check fails
    Given the requested backend is "cuda"
    And node "mac" does not have CUDA available
    When rbee-hive performs backend check
    Then the check fails
    And rbee-hive returns error "BACKEND_UNAVAILABLE"
    And the error message includes the requested backend
    And the exit code is 1

  # ============================================================================
  # PHASE 5: Worker Startup
  # ============================================================================

  Scenario: Worker startup sequence
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
    And the worker HTTP server binds to port 8081
    And the worker sends ready callback to pool manager
    And the ready callback includes worker_id, url, model_ref, backend, device
    And model loading begins asynchronously
    And rbee-hive returns worker details to rbee-keeper with state "loading"

  Scenario: Worker ready callback
    Given the worker HTTP server started successfully
    When the worker sends ready callback
    Then the request is:
      """
      POST http://mac.home.arpa:8080/v1/workers/ready
      {
        "worker_id": "worker-abc123",
        "url": "http://mac.home.arpa:8081",
        "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "backend": "metal",
        "device": 0
      }
      """
    And rbee-hive acknowledges the callback
    And rbee-hive updates the in-memory registry

  # ============================================================================
  # PHASE 6: Worker Registration
  # ============================================================================

  Scenario: Worker registration in in-memory registry
    Given the worker sent ready callback
    When rbee-hive registers the worker
    Then the in-memory HashMap is updated with:
      | field            | value                                           |
      | id               | worker-abc123                                   |
      | url              | http://mac.home.arpa:8081                       |
      | model_ref        | hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF       |
      | backend          | metal                                           |
      | device           | 0                                               |
      | state            | loading                                         |
      | slots_total      | 1                                               |
      | slots_available  | 0                                               |
    And the registration is ephemeral (lost on rbee-hive restart)

  # ============================================================================
  # PHASE 7: Worker Health Check
  # ============================================================================

  Scenario: Worker health check while loading
    Given the worker is in state "loading"
    When rbee-keeper polls "http://mac.home.arpa:8081/v1/ready"
    Then the response is:
      """
      {
        "ready": false,
        "state": "loading",
        "progress_url": "http://mac.home.arpa:8081/v1/loading/progress"
      }
      """
    And rbee-keeper connects to the progress SSE stream
    And the stream emits layer loading progress

  Scenario: Worker loading progress stream
    Given the worker is loading model to VRAM
    When rbee-keeper connects to "/v1/loading/progress"
    Then the SSE stream emits:
      """
      data: {"stage": "loading_to_vram", "layers_loaded": 12, "layers_total": 32, "vram_mb": 2048}
      data: {"stage": "loading_to_vram", "layers_loaded": 24, "layers_total": 32, "vram_mb": 4096}
      data: {"stage": "ready"}
      data: [DONE]
      """
    And rbee-keeper displays progress bar with layers loaded

  Scenario: Worker health check when ready
    Given the worker completed model loading
    When rbee-keeper polls "/v1/ready"
    Then the response is:
      """
      {
        "ready": true,
        "state": "idle",
        "model_loaded": true
      }
      """
    And rbee-keeper proceeds to inference execution

  Scenario: Worker loading timeout
    Given the worker is loading for 5 minutes
    And the worker is stuck at 28/32 layers
    When rbee-keeper timeout expires
    Then rbee-keeper aborts with error "LOADING_TIMEOUT"
    And the error includes current loading state
    And the error suggests checking worker logs
    And the exit code is 1

  # ============================================================================
  # PHASE 8: Inference Execution
  # ============================================================================

  Scenario: Inference request with SSE streaming
    Given the worker is ready and idle
    When rbee-keeper sends inference request:
      """
      POST http://mac.home.arpa:8081/v1/inference
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

  Scenario: Inference request when worker is busy
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
    And the error suggests waiting or using a different node
    And the exit code is 1

  # ============================================================================
  # EDGE CASES
  # ============================================================================

  Scenario: EC1 - Connection timeout with retry and backoff
    Given node "mac" is unreachable
    When rbee-keeper attempts connection
    Then rbee-keeper displays:
      """
      Attempt 1: Connecting to mac.home.arpa:8080... (timeout 10s)
      Attempt 2: Connecting to mac.home.arpa:8080... (timeout 10s, delay 200ms)
      Attempt 3: Connecting to mac.home.arpa:8080... (timeout 10s, delay 400ms)
      Error: Cannot connect to mac.home.arpa:8080 after 3 attempts
      Suggestion: Check if rbee-hive is running on mac
      """
    And the exit code is 1

  Scenario: EC2 - Model download failure with retry
    Given model download fails at 40% with "Connection reset by peer"
    When rbee-hive retries download
    Then rbee-hive displays:
      """
      Downloading model... [████████------------] 40% (2.0 MB / 5.0 MB)
      Error: Connection reset by peer
      Retrying download (attempt 2/6)... delay 100ms
      Downloading model... [████████████--------] 60% (3.0 MB / 5.0 MB)
      """
    And if all 6 attempts fail, error "DOWNLOAD_FAILED" is returned
    And the exit code is 1

  Scenario: EC3 - Insufficient VRAM
    Given the model requires 6000 MB
    And only 4000 MB is available
    When rbee-hive performs VRAM check
    Then rbee-keeper displays:
      """
      Error: Insufficient VRAM on mac
        Required: 6000 MB
        Available: 4000 MB
        
      Suggestion: Try a smaller quantized model (Q4 instead of Q8)
      """
    And the exit code is 1

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
    And pool manager removes worker from registry
    And pool manager logs crash event
    And the exit code is 1

  Scenario: EC5 - Client cancellation with Ctrl+C
    Given inference is in progress
    When the user presses Ctrl+C
    Then rbee-keeper sends:
      """
      DELETE http://mac.home.arpa:8081/v1/inference/<request_id>
      """
    And rbee-keeper waits for acknowledgment with timeout 5s
    And rbee-keeper displays "\nCanceled."
    And the worker stops token generation
    And the worker releases slot and returns to idle
    And the exit code is 130

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

  Scenario: EC7 - Model loading timeout
    Given the worker is loading for over 5 minutes
    When rbee-keeper timeout expires
    Then rbee-keeper displays:
      """
      Error: Model loading timeout after 5 minutes
      Worker state: loading (stuck at 28/32 layers)
      
      Suggestion: Check worker logs on mac for errors
      """
    And the exit code is 1

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

  Scenario: EC9 - Invalid API key
    Given rbee-keeper uses API key "wrong_key"
    When rbee-keeper sends request with "Authorization: Bearer wrong_key"
    Then rbee-hive returns:
      """
      HTTP/1.1 401 Unauthorized
      Content-Type: application/json

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

  Scenario: EC10 - Idle timeout and worker auto-shutdown
    Given inference completed at T+0:00
    And the worker is idle
    When 5 minutes elapse
    Then pool manager sends "POST /v1/admin/shutdown" at T+5:00
    And the worker unloads model from VRAM at T+5:01
    And the worker exits cleanly at T+5:02
    And pool manager removes worker from registry at T+5:02
    And VRAM is available for other applications
    And the next inference request triggers cold start

  # ============================================================================
  # POOL MANAGER LIFECYCLE
  # ============================================================================

  @lifecycle @critical
  Scenario: Pool manager remains running as persistent HTTP daemon
    Given rbee-hive is started as HTTP daemon on port 8080
    And rbee-hive spawned a worker
    When the worker sends ready callback
    Then rbee-hive does NOT exit
    And rbee-hive continues monitoring worker health every 30s
    And rbee-hive enforces idle timeout of 5 minutes
    And rbee-hive remains available for new worker requests
    And rbee-hive HTTP API remains accessible

  @lifecycle
  Scenario: Pool manager monitors worker health
    Given rbee-hive is running as persistent daemon
    And a worker is registered
    When 30 seconds elapse
    Then pool manager sends health check to worker
    And if worker responds, pool manager updates last_activity
    And if worker does not respond, pool manager marks worker as unhealthy
    And if worker is unhealthy for 3 consecutive checks, pool manager removes it from registry
    And rbee-hive continues running (does NOT exit)

  @lifecycle @critical
  Scenario: Pool manager enforces idle timeout (worker dies, pool lives)
    Given rbee-hive is running as persistent daemon
    And a worker completed inference and is idle
    When 5 minutes elapse without new requests
    Then pool manager sends shutdown command to worker
    And pool manager removes worker from in-memory registry
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

  @lifecycle @critical
  Scenario: rbee-keeper exits after inference (CLI dies, daemons live)
    Given rbee-hive is running as persistent daemon
    And worker is running and idle
    When rbee-keeper completes inference request
    Then rbee-keeper exits with code 0
    And rbee-hive continues running as daemon
    And worker continues running as daemon
    And worker remains in rbee-hive's in-memory registry

  @lifecycle @critical
  Scenario: Ephemeral mode - rbee-keeper spawns rbee-hive
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

  @lifecycle @critical
  Scenario: Persistent mode - rbee-hive pre-started
    Given rbee-hive is already running as daemon
    And rbee-hive was started manually by operator
    When rbee-keeper runs inference command
    Then rbee-keeper connects to existing rbee-hive HTTP API
    And rbee-keeper does NOT spawn rbee-hive
    And inference completes
    And rbee-keeper exits
    And rbee-hive continues running (was not spawned by rbee-keeper)
    And worker continues running (idle timeout not reached)

  # ============================================================================
  # ERROR RESPONSE FORMAT
  # ============================================================================

  Scenario: Error response structure validation
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
    And the error code is one of the defined error codes
    And the message is human-readable
    And the details provide actionable context

  # ============================================================================
  # CLI COMMANDS
  # ============================================================================

  Scenario: CLI command - basic inference
    When I run:
      """
      rbee-keeper infer \
        --node mac \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --prompt "write a short story" \
        --max-tokens 20 \
        --temperature 0.7
      """
    Then the command executes the full inference flow
    And tokens are streamed to stdout
    And the exit code is 0

  Scenario: CLI command - list workers
    Given workers are registered on multiple nodes
    When I run "rbee-keeper workers list"
    Then the output shows all registered workers with their state
    And the exit code is 0

  Scenario: CLI command - check worker health
    When I run "rbee-keeper workers health --node mac"
    Then the output shows health status of workers on mac
    And the exit code is 0

  Scenario: CLI command - manually shutdown worker
    Given a worker with id "worker-abc123" is running
    When I run "rbee-keeper workers shutdown --id worker-abc123"
    Then the worker receives shutdown command
    And the worker unloads model and exits
    And the exit code is 0

  Scenario: CLI command - view logs
    When I run "rbee-keeper logs --node mac --follow"
    Then logs from mac are streamed to stdout
    And the stream continues until Ctrl+C
    And the exit code is 0 or 130

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

# Created by: TEAM-037 (Testing Team)
# 
# LIFECYCLE SUMMARY:
# - rbee-hive: PERSISTENT HTTP DAEMON (dies only on SIGTERM)
# - llm-worker-rbee: PERSISTENT HTTP DAEMON (dies on idle timeout or shutdown)
# - rbee-keeper: EPHEMERAL CLI (dies after command completes)
# - Ephemeral mode: rbee-keeper spawns rbee-hive, controls lifecycle
# - Persistent mode: rbee-hive pre-started, rbee-keeper just connects
#
# Verified by Testing Team 🔍
