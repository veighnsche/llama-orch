# Traceability: TEST-001 (split by TEAM-077)
# Architecture: TEAM-037 (queen-rbee orchestration, in-memory worker registry, SQLite model catalog)
# Components: rbee-keeper, queen-rbee, rbee-hive, llm-worker-rbee
# Updated by: TEAM-036 (GGUF support, installation system)
# Updated by: TEAM-041 (rbee-hive Registry, SSH setup)
# Refactored by: TEAM-077 (reorganized to correct BDD architecture)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: End-to-End Integration Flows
  As a system integrating all components
  I want to execute complete workflows from start to finish
  So that I can verify the entire system works together correctly

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"
    And the model catalog is SQLite at "~/.rbee/models.db"
    And the worker registry is in-memory ephemeral per node
    And the rbee-hive registry is SQLite at "~/.rbee/beehives.db"

  @critical
  Scenario: Happy path - cold start inference on remote node
    Given no workers are registered for model "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    And node "workstation" is registered in rbee-hive registry with SSH details
    And node "workstation" is reachable at "http://workstation.home.arpa:8080"
    And node "workstation" has 8000 MB of available RAM
    And node "workstation" has CUDA backend available
    When I run:
      """
      rbee-keeper infer \
        --node workstation \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --prompt "write a short story" \
        --max-tokens 20 \
        --temperature 0.7 \
        --backend cuda \
        --device 1
      """
    Then rbee-keeper sends request to queen-rbee at "http://localhost:8080/v2/tasks"
    And queen-rbee queries rbee-hive registry for node "workstation"
    And the registry returns SSH details for node "workstation"
    And queen-rbee establishes SSH connection using registry details
    And queen-rbee starts rbee-hive via SSH at "workstation.home.arpa"
    And queen-rbee updates registry with last_connected_unix
    And queen-rbee queries rbee-hive worker registry at "http://workstation.home.arpa:9200/v1/workers/list"
    And the worker registry returns an empty list
    And queen-rbee performs pool preflight check at "http://workstation.home.arpa:9200/v1/health"
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
    And CUDA backend check passes
    And rbee-hive spawns worker process "llm-worker-rbee" on port 8001 with cuda device 1
    And the worker HTTP server starts on port 8001
    And the worker sends ready callback to "http://workstation.home.arpa:9200/v1/workers/ready"
    And rbee-hive registers the worker in the in-memory registry
    And rbee-hive returns worker details to queen-rbee
    And queen-rbee returns worker URL to rbee-keeper
    And rbee-keeper polls worker readiness at "http://workstation.home.arpa:8001/v1/ready"
    And the worker returns state "loading" with progress_url
    And rbee-keeper streams loading progress showing layers loaded
    And the worker completes loading and returns state "ready"
    And rbee-keeper sends inference request to "http://workstation.home.arpa:8001/v1/inference"
    And the worker streams tokens via SSE
    And rbee-keeper displays tokens to stdout in real-time
    And the inference completes with 20 tokens generated
    And the worker transitions to state "idle"
    And the exit code is 0

  Scenario: Warm start - reuse existing idle worker
    Given node "workstation" is registered in rbee-hive registry with SSH details
    And a worker is registered with:
      | field      | value                                           |
      | id         | worker-abc123                                   |
      | url        | http://workstation.home.arpa:8001               |
      | model_ref  | hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF       |
      | state      | idle                                            |
      | backend    | cuda                                            |
      | device     | 1                                               |
    And the worker is healthy
    When I run:
      """
      rbee-keeper infer \
        --node workstation \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --prompt "write a poem" \
        --max-tokens 20 \
        --backend cuda \
        --device 1
      """
    Then rbee-keeper queries the worker registry
    And the registry returns worker "worker-abc123" with state "idle"
    And queen-rbee skips pool preflight and model provisioning
    And rbee-keeper sends inference request directly to "http://workstation.home.arpa:8001/v1/inference"
    And the worker streams tokens via SSE
    And the inference completes successfully
    And the total latency is under 5 seconds
    And the exit code is 0
