# Traceability: TEST-001 (split by TEAM-077)
# Architecture: TEAM-037 (queen-rbee orchestration, in-memory worker registry)
# Components: rbee-hive (pool manager), llm-worker-rbee (worker)
# Refactored by: TEAM-077 (split from test-001.feature into focused feature files)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: worker-rbee Daemon Lifecycle
  As a system managing worker-rbee daemons
  I want to spawn, register, and monitor worker-rbee processes
  So that they can serve inference requests

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"

  Scenario: Worker startup sequence
    Given worker preflight checks passed
    When rbee-hive spawns worker process
    Then the command is:
      """
      llm-worker-rbee \
        --model /models/tinyllama-q4.gguf \
        --backend cuda \
        --device 1 \
        --port 8081 \
        --api-key <worker_api_key>
      """
    And the worker HTTP server binds to port 8081
    And the worker sends ready callback to rbee-hive
    And the ready callback includes worker_id, url, model_ref, backend, device
    And model loading begins asynchronously
    And rbee-hive returns worker details to rbee-keeper with state "loading"

  Scenario: Worker ready callback
    Given the worker HTTP server started successfully
    When the worker sends ready callback
    Then the request is:
      """
      POST http://workstation.home.arpa:9200/v1/workers/ready
      {
        "worker_id": "worker-abc123",
        "url": "http://workstation.home.arpa:8081",
        "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "backend": "cuda",
        "device": 1
      }
      """
    And rbee-hive acknowledges the callback
    And rbee-hive updates the in-memory registry

  @error-handling
  Scenario: EH-012a - Worker binary not found
    Given worker preflight checks passed
    When rbee-hive attempts to spawn worker
    And worker binary does not exist at expected path
    Then spawn fails immediately
    And rbee-keeper displays:
      """
      [rbee-hive] ❌ Error: Worker binary not found
        Expected path: /home/vince/rbee/target/release/llm-worker-rbee
        
      Install worker binary:
        rbee-keeper setup install --node workstation
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-012b - Worker port already in use
    Given port 8001 is already occupied by another process
    When rbee-hive spawns worker on port 8001
    Then worker fails to bind port
    And rbee-hive detects bind failure
    And rbee-hive tries next available port 8002
    And rbee-keeper displays:
      """
      [rbee-hive] ⚠️  Port 8001 in use, trying 8002...
      [worker] 🚀 HTTP server ready on port 8002
      """
    And worker successfully starts on port 8002

  @error-handling
  Scenario: EH-012c - Worker crashes during startup
    Given rbee-hive spawns worker process
    When worker crashes during initialization
    And worker process exits with code 1
    Then rbee-hive detects startup failure within 30s
    And rbee-keeper displays:
      """
      [worker] ❌ Initialization failed: CUDA device 1 not found
      [rbee-hive] ❌ Worker startup failed
      
      Check worker logs for details.
      """
    And the exit code is 1

  Scenario: Worker registration in in-memory registry
    Given the worker sent ready callback
    When rbee-hive registers the worker
    Then the in-memory HashMap is updated with:
      | field            | value                                           |
      | id               | worker-abc123                                   |
      | url              | http://workstation.home.arpa:8081               |
      | model_ref        | hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF       |
      | backend          | cuda                                            |
      | device           | 1                                               |
      | state            | loading                                         |
      | slots_total      | 1                                               |
      | slots_available  | 0                                               |
    And the registration is ephemeral (lost on rbee-hive restart)

  Scenario: Worker health check while loading
    Given the worker is in state "loading"
    When rbee-keeper polls "http://workstation.home.arpa:8081/v1/ready"
    Then the response is:
      """
      {
        "ready": false,
        "state": "loading",
        "progress_url": "http://workstation.home.arpa:8081/v1/loading/progress"
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

  @error-handling
  Scenario: EH-016a - Worker loading timeout
    Given the worker is loading for 5 minutes
    And the worker is stuck at 28/32 layers
    When rbee-keeper timeout expires
    Then rbee-keeper aborts with error "LOADING_TIMEOUT"
    And rbee-keeper displays:
      """
      [rbee-keeper] ❌ Error: Model loading timeout after 5 minutes
        Worker state: loading (stuck at 28/32 layers)
        Elapsed: 5m 0s
        
      Suggestion: Check worker logs on workstation:
        ssh workstation tail -f ~/.rbee/logs/worker-abc123.log
      """
    And the exit code is 1

  @edge-case
  Scenario: EC7 - Model loading timeout
    Given the worker is loading for over 5 minutes
    When rbee-keeper timeout expires
    Then rbee-keeper displays:
      """
      Error: Model loading timeout after 5 minutes
      Worker state: loading (stuck at 28/32 layers)
      
      Suggestion: Check worker logs on workstation for errors
      """
    And the exit code is 1
