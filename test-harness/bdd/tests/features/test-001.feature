# Traceability: TEST-001
# Architecture: TEAM-037 (queen-rbee orchestration, in-memory worker registry, SQLite model catalog)
# Components: rbee-keeper (config + testing tool), queen-rbee (orchestrator), rbee-hive (pool manager), llm-worker-rbee (worker)
# Updated by: TEAM-038 (aligned with queen-rbee orchestration and GGUF support)
# Updated by: TEAM-041 (added rbee-hive Registry module, SSH setup flow, rbee-keeper configuration mode)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# ⚠️ DO NOT use mock servers - wire up actual rbee-hive and llm-worker-rbee libraries
# ⚠️ See TEAM_063_REAL_HANDOFF.md for implementation requirements

Feature: Cross-Node Inference Request Flow
  As a user on the control node
  I want to run inference on a remote compute node
  So that I can utilize distributed GPU resources across my homelab

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    # NOTE: This test suite uses workstation node with cuda backend on device 1
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"
    And the model catalog is SQLite at "~/.rbee/models.db"
    And the worker registry is in-memory ephemeral per node
    And the rbee-hive registry is SQLite at "~/.rbee/beehives.db"

  # ============================================================================
  # PREREQUISITES: rbee-hive Registry Setup (TEAM-041)
  # ============================================================================
  # CRITICAL: Before any inference can happen, remote rbee-hive nodes must be
  # configured through rbee-keeper. The queen-rbee maintains a persistent
  # rbee-hive Registry with SSH connection details.
  # ============================================================================

  @setup @critical
  Scenario: Add remote rbee-hive node to registry
    Given queen-rbee is running
    And the rbee-hive registry is empty
    When I run:
      """
      rbee-keeper setup add-node \
        --name workstation \
        --ssh-host workstation.home.arpa \
        --ssh-user vince \
        --ssh-key ~/.ssh/id_ed25519 \
        --git-repo https://github.com/user/llama-orch.git \
        --git-branch main \
        --install-path ~/rbee
      """
    Then rbee-keeper sends request to queen-rbee at "http://localhost:8080/v2/registry/beehives/add"
    And queen-rbee validates SSH connection with:
      """
      ssh -i /home/vince/.ssh/id_ed25519 vince@workstation.home.arpa "echo 'connection test'"
      """
    And the SSH connection succeeds
    And queen-rbee saves node to rbee-hive registry:
      | field              | value                                      |
      | node_name          | workstation                                |
      | ssh_host           | workstation.home.arpa                      |
      | ssh_port           | 22                                         |
      | ssh_user           | vince                                      |
      | ssh_key_path       | /home/vince/.ssh/id_ed25519                |
      | git_repo_url       | https://github.com/user/llama-orch.git     |
      | git_branch         | main                                       |
      | install_path       | /home/vince/rbee                           |
      | last_connected_unix| 1728508603                                 |
      | status             | reachable                                  |
    And rbee-keeper displays:
      """
      [queen-rbee] 🔌 Testing SSH connection to workstation.home.arpa
      [queen-rbee] ✅ SSH connection successful! Node 'workstation' saved to registry
      """
    And the exit code is 0

  @setup @error-handling
  Scenario: EH-001a - SSH connection timeout
    When I run:
      """
      rbee-keeper setup add-node \
        --name unreachable \
        --ssh-host unreachable.home.arpa \
        --ssh-user vince \
        --ssh-key ~/.ssh/id_ed25519 \
        --git-repo https://github.com/user/llama-orch.git \
        --install-path ~/rbee
      """
    Then queen-rbee attempts SSH connection with 10s timeout
    And the SSH connection fails with timeout
    And queen-rbee retries 3 times with exponential backoff
    And queen-rbee does NOT save node to registry
    And rbee-keeper displays:
      """
      [queen-rbee] 🔌 Testing SSH connection to unreachable.home.arpa
      [queen-rbee] ⏳ Attempt 1/3 failed: Connection timeout
      [queen-rbee] ⏳ Attempt 2/3 failed: Connection timeout (delay 200ms)
      [queen-rbee] ⏳ Attempt 3/3 failed: Connection timeout (delay 400ms)
      [queen-rbee] ❌ SSH connection failed after 3 attempts
      """
    And the exit code is 1

  @setup @error-handling
  Scenario: EH-001b - SSH authentication failure
    Given SSH key at "~/.ssh/id_ed25519" has wrong permissions
    When I run:
      """
      rbee-keeper setup add-node \
        --name workstation \
        --ssh-host workstation.home.arpa \
        --ssh-user vince \
        --ssh-key ~/.ssh/id_ed25519 \
        --git-repo https://github.com/user/llama-orch.git \
        --install-path ~/rbee
      """
    Then queen-rbee attempts SSH connection
    And the SSH connection fails with "Permission denied (publickey)"
    And queen-rbee does NOT save node to registry
    And rbee-keeper displays:
      """
      [queen-rbee] 🔌 Testing SSH connection to workstation.home.arpa
      [queen-rbee] ❌ SSH authentication failed: Permission denied
      
      Suggestion: Check SSH key permissions:
        chmod 600 ~/.ssh/id_ed25519
      """
    And the exit code is 1

  @setup @error-handling
  Scenario: EH-001c - SSH command execution failure
    Given SSH connection succeeds
    But rbee-hive binary does not exist on remote node
    When queen-rbee attempts to start rbee-hive via SSH
    Then the SSH command fails with "command not found"
    And rbee-keeper displays:
      """
      [queen-rbee] ❌ Failed to start rbee-hive: command not found
      
      Suggestion: Install rbee-hive on workstation:
        rbee-keeper setup install --node workstation
      """
    And the exit code is 1

  @setup
  Scenario: Install rbee-hive on remote node
    Given node "workstation" is registered in rbee-hive registry
    When I run:
      """
      rbee-keeper setup install --node workstation
      """
    Then queen-rbee loads SSH details from registry
    And queen-rbee executes installation via SSH:
      """
      ssh -i /home/vince/.ssh/id_ed25519 vince@workstation.home.arpa << 'EOF'
        cd /home/vince/rbee
        git clone https://github.com/user/llama-orch.git .
        git checkout main
        cargo build --release --bin rbee-hive
        cargo build --release --bin llm-worker-rbee
      EOF
      """
    And rbee-keeper displays:
      """
      [queen-rbee] 📦 Cloning repository on workstation
      [queen-rbee] 🔨 Building rbee-hive and llm-worker-rbee
      [queen-rbee] ✅ Installation complete on workstation!
      """
    And the exit code is 0

  @setup
  Scenario: List registered rbee-hive nodes
    Given multiple nodes are registered in rbee-hive registry
    When I run "rbee-keeper setup list-nodes"
    Then rbee-keeper displays:
      """
      Registered rbee-hive Nodes:
      
      workstation (workstation.home.arpa)
        Status: reachable
        Last connected: 2024-10-09 14:22:15
        Install path: /home/vince/rbee
      
      blep (blep.home.arpa)
        Status: reachable
        Last connected: 2024-10-09 14:22:15
        Install path: /home/vince/rbee
      """
    And the exit code is 0

  @setup
  Scenario: Remove node from rbee-hive registry
    Given node "workstation" is registered in rbee-hive registry
    When I run "rbee-keeper setup remove-node --name workstation"
    Then queen-rbee removes node from registry
    And rbee-keeper displays:
      """
      [queen-rbee] ✅ Node 'workstation' removed from registry
      """
    And the exit code is 0

  @setup @error-handling
  Scenario: EH-011a - Invalid SSH key path
    When I run:
      """
      rbee-keeper setup add-node \
        --name workstation \
        --ssh-host workstation.home.arpa \
        --ssh-user vince \
        --ssh-key /nonexistent/key \
        --git-repo https://github.com/user/llama-orch.git \
        --install-path ~/rbee
      """
    Then rbee-keeper validates SSH key path before sending to queen-rbee
    And validation fails with "SSH key not found: /nonexistent/key"
    And rbee-keeper displays:
      """
      [rbee-keeper] ❌ Error: SSH key not found
        Path: /nonexistent/key
        
      Check the key path and try again.
      """
    And the exit code is 1

  @setup @error-handling
  Scenario: EH-011b - Duplicate node name
    Given node "workstation" already exists in registry
    When I run:
      """
      rbee-keeper setup add-node \
        --name workstation \
        --ssh-host workstation2.home.arpa \
        --ssh-user vince \
        --ssh-key ~/.ssh/id_ed25519 \
        --git-repo https://github.com/user/llama-orch.git \
        --install-path ~/rbee
      """
    Then queen-rbee detects duplicate node name
    And rbee-keeper displays:
      """
      [queen-rbee] ❌ Error: Node 'workstation' already exists in registry
      
      To update this node, run:
        rbee-keeper setup update-node --name workstation ...
      
      To remove and re-add:
        rbee-keeper setup remove-node --name workstation
      """
    And the exit code is 1

  @setup @critical
  Scenario: Inference fails when node not in registry
    Given the rbee-hive registry does not contain node "workstation"
    When I run:
      """
      rbee-keeper infer \
        --node workstation \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --prompt "test"
      """
    Then queen-rbee queries rbee-hive registry for node "workstation"
    And the query returns no results
    And rbee-keeper displays:
      """
      [queen-rbee] ❌ ERROR: Node 'workstation' not found in rbee-hive registry
      
      To add this node, run:
        rbee-keeper setup add-node --name workstation --ssh-host workstation.home.arpa ...
      """
    And the exit code is 1

  # ============================================================================
  # HAPPY PATH: Full inference flow from cold start
  # ============================================================================

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

  # ============================================================================
  # PHASE 1: Worker Registry Check
  # ============================================================================

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

  # ============================================================================
  # PHASE 2: Pool Preflight
  # ============================================================================

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

  @error-handling
  Scenario: EH-002a - rbee-hive HTTP connection timeout
    Given queen-rbee started rbee-hive via SSH
    But rbee-hive process crashed immediately
    When queen-rbee queries worker registry at "http://workstation.home.arpa:9200/v1/workers/list"
    Then the HTTP request times out after 10s
    And queen-rbee retries 3 times with exponential backoff
    And all retries fail
    And rbee-keeper displays:
      """
      [queen-rbee] ❌ Error: Cannot connect to rbee-hive on workstation
        Attempted 3 times, all failed with timeout
        
      Suggestion: Check rbee-hive logs on workstation:
        ssh workstation journalctl -u rbee-hive -n 50
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

  Scenario: Pool preflight detects version mismatch
    Given rbee-keeper version is "0.1.0"
    And rbee-hive version is "0.2.0"
    When rbee-keeper performs health check
    Then rbee-keeper aborts with error "VERSION_MISMATCH"
    And the error message includes both versions
    And the error suggests upgrading rbee-keeper
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

  @error-handling
  Scenario: EH-007a - Model not found on Hugging Face
    Given model "hf:NonExistent/FakeModel" does not exist
    When rbee-hive attempts to download model
    Then Hugging Face returns 404 Not Found
    And rbee-hive detects 404 status code
    And rbee-keeper displays:
      """
      [rbee-hive] ❌ Error: Model not found
        Model: hf:NonExistent/FakeModel
        
      Suggestion: Check model reference on https://huggingface.co/
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-007b - Model repository is private
    Given model "hf:PrivateOrg/PrivateModel" requires authentication
    When rbee-hive attempts to download without credentials
    Then Hugging Face returns 403 Forbidden
    And rbee-keeper displays:
      """
      [rbee-hive] ❌ Error: Access denied to model
        Model: hf:PrivateOrg/PrivateModel
        
      This model requires authentication. Provide HF token:
        export HF_TOKEN=your_token_here
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-008a - Model download timeout
    Given the model is not in the catalog
    When rbee-hive attempts download
    And network becomes very slow
    And no progress for 60 seconds
    Then rbee-hive detects stall timeout
    And rbee-hive retries download with exponential backoff
    And rbee-keeper displays:
      """
      [rbee-hive] ⏳ Download stalled, retrying (attempt 2/6)...
      """
    And if all 6 attempts fail, error "DOWNLOAD_TIMEOUT" is returned
    And the exit code is 1

  @error-handling
  Scenario: EH-008b - Model download fails with retry
    Given the model is not in the catalog
    When rbee-hive attempts download
    And the download fails with "Connection reset by peer" at 40% progress
    Then rbee-hive retries download with delay 100ms
    And rbee-hive resumes from last checkpoint
    And rbee-hive retries up to 6 times with exponential backoff
    And rbee-keeper displays progress:
      """
      [rbee-hive] Downloading... [████████------------] 40% (2.0 MB / 5.0 MB)
      [rbee-hive] ❌ Connection reset by peer
      [rbee-hive] ⏳ Retrying download (attempt 2/6)... delay 100ms
      [rbee-hive] Downloading... [████████████--------] 60% (3.0 MB / 5.0 MB)
      """
    And if all retries fail, rbee-hive returns error "DOWNLOAD_FAILED"
    And the exit code is 1

  @error-handling
  Scenario: EH-008c - Downloaded model checksum mismatch
    Given model download completes
    When rbee-hive verifies checksum
    And checksum does not match expected value
    Then rbee-hive deletes corrupted file
    And rbee-hive retries download
    And rbee-keeper displays:
      """
      [rbee-hive] ✅ Download complete
      [rbee-hive] 🔍 Verifying checksum...
      [rbee-hive] ❌ Checksum mismatch: file corrupted
      [rbee-hive] 🗑️  Deleting corrupted file
      [rbee-hive] ⏳ Retrying download (attempt 2/6)...
      """
    And the exit code is 1 if all retries fail

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
  # PHASE 3.5: GGUF Model Support (TEAM-036)
  # ============================================================================

  @gguf @team-036
  Scenario: GGUF model detection by file extension
    Given a model file at "/models/tinyllama-q4.gguf"
    When llm-worker-rbee loads the model
    Then the model factory detects ".gguf" extension
    And the factory creates a QuantizedLlama model variant
    And the model is loaded using candle's quantized_llama module
    And GGUF metadata is extracted from the file header

  @gguf @team-036
  Scenario: GGUF metadata extraction
    Given a GGUF file at "/models/tinyllama-q4.gguf"
    When llm-worker-rbee reads the GGUF header
    Then the following metadata is extracted:
      | field          | value  |
      | vocab_size     | 32000  |
      | eos_token_id   | 2      |
      | quantization   | Q4_K_M |
    And the vocab_size is used for model initialization
    And the eos_token_id is used for generation stopping

  @gguf @team-036
  Scenario: GGUF quantization formats supported
    Given the following GGUF models are available:
      | model                          | quantization | size_mb |
      | tinyllama-q4_k_m.gguf          | Q4_K_M       | 669     |
      | tinyllama-q5_k_m.gguf          | Q5_K_M       | 817     |
      | tinyllama-q8_0.gguf            | Q8_0         | 1260    |
    When llm-worker-rbee loads each model
    Then all quantization formats are supported
    And inference completes successfully for each model
    And VRAM usage is proportional to quantization level

  @gguf @team-036
  Scenario: GGUF model size calculation
    Given a GGUF file at "/models/tinyllama-q4.gguf"
    When rbee-hive calculates model size
    Then the file size is read from disk
    And the size is used for RAM preflight checks
    And the size is stored in the model catalog

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

  @error-handling
  Scenario: EH-004a - Worker preflight RAM check fails
    Given the model size is 5000 MB
    And the node has 4000 MB of available RAM
    When rbee-hive performs RAM check
    Then rbee-hive calculates required RAM as 6000 MB
    And the check fails because 4000 MB < 6000 MB
    And rbee-hive returns error "INSUFFICIENT_RAM"
    And rbee-keeper displays:
      """
      [rbee-hive] ❌ Error: Insufficient RAM on workstation
        Required: 6000 MB (model size * 1.2)
        Available: 4000 MB
        
      Suggestions:
        - Close other applications to free RAM
        - Use a smaller quantized model (Q4 instead of Q8)
        - Try CPU backend with smaller context size
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-004b - RAM exhausted during model loading
    Given model loading has started
    And worker is loading model to RAM
    When system RAM is exhausted by another process
    Then worker detects OOM condition
    And worker exits with error
    And rbee-hive detects worker crash
    And rbee-keeper displays:
      """
      [worker] ❌ Out of memory during model loading
      [rbee-hive] ❌ Worker crashed: OOM killed
      
      Suggestion: Free up RAM and try again
      """
    And the exit code is 1

  Scenario: Worker preflight backend check passes
    Given the requested backend is "cuda"
    And node "workstation" has CUDA backend available
    When rbee-hive performs backend check
    Then the check passes
    And rbee-hive proceeds to worker startup

  @error-handling
  Scenario: EH-005a - VRAM exhausted on CUDA device
    Given CUDA device 1 has 2000 MB VRAM
    And model requires 4000 MB VRAM
    When rbee-hive performs VRAM check
    Then the check fails
    And rbee-keeper displays:
      """
      [rbee-hive] ❌ Error: Insufficient VRAM on CUDA device 1
        Required: 4000 MB
        Available: 2000 MB
        
      Suggestions:
        - Use smaller quantized model (Q4_K_M instead of Q8_0)
        - Try CPU backend: --backend cpu
        - Free VRAM by closing other GPU applications
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-009a - Backend not available
    Given the requested backend is "metal"
    And node "workstation" does not have Metal available
    When rbee-hive performs backend check
    Then the check fails
    And rbee-hive returns error "BACKEND_UNAVAILABLE"
    And rbee-keeper displays:
      """
      [rbee-hive] ❌ Error: Backend not available
        Requested: metal
        Available: ["cpu", "cuda"]
        
      Metal is only available on macOS.
      Try: --backend cuda --device 0
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-009b - CUDA not installed
    Given the requested backend is "cuda"
    And node "workstation" has no CUDA installed
    When rbee-hive performs backend check
    Then the check fails
    And rbee-keeper displays:
      """
      [rbee-hive] ❌ Error: CUDA backend not available
        CUDA drivers not found on workstation
        
      Available backends: ["cpu"]
      
      To install CUDA:
        https://developer.nvidia.com/cuda-downloads
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-006a - Insufficient disk space for model download
    Given node "workstation" has 1000 MB free disk space
    And model "TinyLlama" requires 5000 MB
    When rbee-hive checks disk space before download
    Then the check fails
    And rbee-keeper displays:
      """
      [rbee-hive] ❌ Error: Insufficient disk space
        Required: 5000 MB
        Available: 1000 MB
        Free up: 4000 MB
        
      Suggestion: Remove unused models:
        rbee-keeper models rm <model_name>
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-006b - Disk fills up during download
    Given model download has started
    When disk space is exhausted mid-download
    Then download fails with "No space left on device"
    And rbee-hive cleans up partial download
    And rbee-keeper displays:
      """
      [rbee-hive] Downloading... [████████------------] 40% (2.0 MB / 5.0 MB)
      [rbee-hive] ❌ Error: Disk full during download
      [rbee-hive] 🗑️  Cleaning up partial download
      
      Free up disk space and try again.
      """
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

  # ============================================================================
  # PHASE 6: Worker Registration
  # ============================================================================

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

  # ============================================================================
  # PHASE 7: Worker Health Check
  # ============================================================================

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

  # ============================================================================
  # PHASE 8: Inference Execution
  # ============================================================================

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

  # ============================================================================
  # EDGE CASES
  # ============================================================================

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
      Error: Insufficient VRAM on workstation
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
    And rbee-hive removes worker from registry
    And rbee-hive logs crash event
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
      
      Suggestion: Check worker logs on workstation for errors
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

  @error-handling @validation
  Scenario: EH-015a - Invalid model reference format
    When I run:
      """
      rbee-keeper infer \
        --node workstation \
        --model invalid-format \
        --prompt "test"
      """
    Then rbee-keeper validates model reference format
    And validation fails
    And rbee-keeper displays:
      """
      [rbee-keeper] ❌ Error: Invalid model reference format
        Got: invalid-format
        
      Expected formats:
        - hf:org/repo (Hugging Face)
        - file:///path/to/model.gguf (Local file)
        
      Example: hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
      """
    And the exit code is 1

  @error-handling @validation
  Scenario: EH-015b - Invalid backend name
    When I run:
      """
      rbee-keeper infer \
        --node workstation \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --backend quantum \
        --prompt "test"
      """
    Then rbee-keeper validates backend name
    And validation fails
    And rbee-keeper displays:
      """
      [rbee-keeper] ❌ Error: Invalid backend
        Got: quantum
        
      Valid backends: ["cpu", "cuda", "metal"]
      
      Example: --backend cuda --device 0
      """
    And the exit code is 1

  @error-handling @validation
  Scenario: EH-015c - Device number out of range
    Given node "workstation" has 2 CUDA devices (0, 1)
    When I run:
      """
      rbee-keeper infer \
        --node workstation \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --backend cuda \
        --device 5 \
        --prompt "test"
      """
    Then rbee-hive validates device number
    And validation fails
    And rbee-keeper displays:
      """
      [rbee-hive] ❌ Error: Device 5 not available
        Requested: cuda device 5
        Available: [0, 1]
        
      Try: --device 0 or --device 1
      """
    And the exit code is 1

  @error-handling @authentication
  Scenario: EH-017a - Missing API key
    Given rbee-hive requires API key
    When queen-rbee sends request without Authorization header
    Then rbee-hive returns:
      """
      HTTP/1.1 401 Unauthorized
      Content-Type: application/json

      {
        "error": {
          "code": "MISSING_API_KEY",
          "message": "Missing API key"
        }
      }
      """
    And rbee-keeper displays:
      """
      [rbee-keeper] ❌ Error: Authentication required
        Missing API key for workstation.home.arpa
        
      Configure API key in:
        ~/.rbee/config.yaml
      """
    And the exit code is 1

  @error-handling @authentication
  Scenario: EH-017b - Invalid API key
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
      [rbee-keeper] ❌ Error: Authentication failed
        Invalid API key for workstation.home.arpa
        
      Check your configuration:
        ~/.rbee/config.yaml
      """
    And the exit code is 1

  Scenario: EC10 - Idle timeout and worker auto-shutdown
    Given inference completed at T+0:00
    And the worker is idle
    When 5 minutes elapse
    Then rbee-hive sends "POST /v1/admin/shutdown" at T+5:00
    And the worker unloads model from VRAM at T+5:01
    And the worker exits cleanly at T+5:02
    And rbee-hive removes worker from registry at T+5:02
    And VRAM is available for other applications
    And the next inference request triggers cold start

  # ============================================================================
  # RBEE-HIVE LIFECYCLE
  # ============================================================================

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
  # CLI COMMANDS (TEAM-036 Installation System)
  # ============================================================================

  @install @team-036
  Scenario: CLI command - install to user paths
    When I run "rbee-keeper install"
    Then binaries are installed to "~/.local/bin/"
    And config directory is created at "~/.config/rbee/"
    And data directory is created at "~/.local/share/rbee/models/"
    And default config file is generated at "~/.config/rbee/config.toml"
    And the following binaries are copied:
      | binary           | source                          | destination                  |
      | rbee-keeper      | target/release/rbee-keeper      | ~/.local/bin/rbee-keeper     |
      | rbee-hive        | target/release/rbee-hive        | ~/.local/bin/rbee-hive       |
      | llm-worker-rbee  | target/release/llm-worker-rbee  | ~/.local/bin/llm-worker-rbee |
    And installation instructions are displayed
    And the exit code is 0

  @install @team-036
  Scenario: CLI command - install to system paths
    When I run "rbee-keeper install --system"
    Then binaries are installed to "/usr/local/bin/"
    And config directory is created at "/etc/rbee/"
    And data directory is created at "/var/lib/rbee/models/"
    And default config file is generated at "/etc/rbee/config.toml"
    And sudo permissions are required
    And the exit code is 0

  @install @team-036
  Scenario: Config file loading with XDG priority
    Given the following config files exist:
      | path                           | priority |
      | /tmp/custom.toml               | 1 (RBEE_CONFIG env var) |
      | ~/.config/rbee/config.toml     | 2 (user config) |
      | /etc/rbee/config.toml          | 3 (system config) |
    When RBEE_CONFIG="/tmp/custom.toml" is set
    Then rbee-keeper loads config from "/tmp/custom.toml"
    When RBEE_CONFIG is not set
    And "~/.config/rbee/config.toml" exists
    Then rbee-keeper loads config from "~/.config/rbee/config.toml"
    When neither RBEE_CONFIG nor user config exist
    Then rbee-keeper loads config from "/etc/rbee/config.toml"

  @install @team-036
  Scenario: Remote binary path configuration
    Given config file contains:
      """
      [remote]
      binary_path = "/opt/rbee/bin/rbee-hive"
      git_repo_dir = "/opt/rbee/repo"
      """
    When rbee-keeper executes remote command on "workstation"
    Then the command uses "/opt/rbee/bin/rbee-hive" instead of "rbee-hive"
    And git commands use "/opt/rbee/repo" instead of "~/llama-orch"

  # ============================================================================
  # CLI COMMANDS (General)
  # ============================================================================

  Scenario: CLI command - basic inference
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
    Then the command executes the full inference flow
    And tokens are streamed to stdout
    And the exit code is 0

  Scenario: CLI command - list workers
    Given workers are registered on multiple nodes
    When I run "rbee-keeper workers list"
    Then the output shows all registered workers with their state
    And the exit code is 0

  Scenario: CLI command - check worker health
    When I run "rbee-keeper workers health --node workstation"
    Then the output shows health status of workers on workstation
    And the exit code is 0

  Scenario: CLI command - manually shutdown worker
    Given a worker with id "worker-abc123" is running
    When I run "rbee-keeper workers shutdown --id worker-abc123"
    Then the worker receives shutdown command
    And the worker unloads model and exits
    And the exit code is 0

  Scenario: CLI command - view logs
    When I run "rbee-keeper logs --node workstation --follow"
    Then logs from workstation are streamed to stdout
    And the stream continues until Ctrl+C
    And the exit code is 0 or 130

  # ============================================================================
  # LIFECYCLE RULES - CRITICAL UNDERSTANDING
  # ============================================================================
  #
  # RULE 1: rbee-keeper is a CONFIGURATION + TESTING TOOL (EPHEMERAL CLI)
  #   - Purpose: Configure rbee-hive nodes + run inference for testing
  #   - Commands: `setup add-node`, `setup install`, `setup list-nodes`, `infer`, etc.
  #   - Starts: User runs command
  #   - Runs: Only during command execution
  #   - Dies: After command completes (exit code 0 or 1)
  #   - Does NOT die: Never stays running
  #   - Production: Use llama-orch SDK → queen-rbee directly
  #   - CRITICAL: NOT just for testing - also manages rbee-hive registry via queen-rbee
  #
  # RULE 2: queen-rbee is a PERSISTENT HTTP DAEMON (ORCHESTRATOR)
  #   - Starts: `queen-rbee daemon` or spawned by rbee-keeper
  #   - Runs: Continuously as HTTP server on port 8080
  #   - Dies: ONLY when receiving SIGTERM or explicit shutdown
  #   - Does NOT die: After inference completes
  #   - Controls: ALL rbee-hive instances via SSH
  #   - Maintains: rbee-hive Registry (SQLite at ~/.rbee/beehives.db)
  #   - Registry stores: SSH connection details for all remote nodes
  #
  # RULE 3: rbee-hive is a PERSISTENT HTTP DAEMON (RBEE-HIVE)
  #   - Starts: Spawned by queen-rbee via SSH
  #   - Runs: Continuously as HTTP server on port 9200
  #   - Dies: When queen-rbee sends SIGTERM via SSH
  #   - Does NOT die: After spawning workers, after inference completes
  #   - Controls: Workers on its node
  #
  # RULE 4: llm-worker-rbee is a PERSISTENT HTTP DAEMON (WORKER)
  #   - Starts: Spawned by rbee-hive
  #   - Runs: Continuously as HTTP server on port 8001+
  #   - Dies: When idle timeout (5 min) OR rbee-hive sends shutdown OR SIGTERM
  #   - Does NOT die: After inference completes (stays idle)
  #
  # RULE 5: Ephemeral Mode (rbee-keeper spawns queen-rbee)
  #   - rbee-keeper spawns queen-rbee as child process
  #   - queen-rbee spawns rbee-hive via SSH
  #   - rbee-hive spawns worker
  #   - Inference completes
  #   - rbee-keeper sends SIGTERM to queen-rbee
  #   - queen-rbee cascades shutdown to all rbee-hive instances via SSH
  #   - rbee-hive cascades shutdown to workers
  #   - All processes exit
  #
  # RULE 6: Persistent Mode (queen-rbee pre-started)
  #   - Operator starts: `queen-rbee daemon &`
  #   - queen-rbee runs continuously
  #   - rbee-keeper connects to existing queen-rbee
  #   - Inference completes
  #   - rbee-keeper exits
  #   - queen-rbee continues running
  #   - rbee-hive continues running
  #   - Worker continues running (until idle timeout)
  #
  # RULE 7: Cascading Shutdown (CRITICAL)
  #   - SIGTERM → queen-rbee
  #   - queen-rbee → SSH SIGTERM → all rbee-hive instances
  #   - rbee-hive → POST /v1/admin/shutdown → all workers
  #   - Workers unload models and exit
  #   - rbee-hive clears registry and exits
  #   - queen-rbee exits
  #   - Model catalog (SQLite) persists on disk
  #
  # RULE 8: Worker Idle Timeout
  #   - Worker completes inference → idle
  #   - 5 minutes elapse without new requests
  #   - rbee-hive sends shutdown to worker
  #   - Worker exits, VRAM freed
  #   - rbee-hive continues running
  #   - queen-rbee continues running
  #
  # RULE 9: Process Ownership
  #   - IF rbee-keeper spawned queen-rbee → rbee-keeper owns lifecycle
  #   - IF operator started queen-rbee → operator owns lifecycle
  #   - queen-rbee always owns rbee-hive lifecycle (via SSH)
  #   - rbee-hive always owns worker lifecycle
  #   - Workers never own their own lifecycle (managed by rbee-hive)

# Created by: TEAM-037 (Testing Team)
# Updated by: TEAM-038 (aligned with queen-rbee orchestration)
# Updated by: TEAM-041 (added rbee-hive Registry, SSH setup, configuration mode)
# 
# LIFECYCLE SUMMARY:
# - rbee-keeper: EPHEMERAL CLI (config + testing tool, dies after command completes)
# - queen-rbee: PERSISTENT HTTP DAEMON (orchestrator, dies only on SIGTERM)
#   - Maintains rbee-hive Registry (SQLite at ~/.rbee/beehives.db)
#   - Registry stores SSH connection details for all remote nodes
# - rbee-hive: PERSISTENT HTTP DAEMON (pool manager, dies when queen-rbee sends SIGTERM)
# - llm-worker-rbee: PERSISTENT HTTP DAEMON (worker, dies on idle timeout or shutdown)
# - Ephemeral mode: rbee-keeper spawns queen-rbee, controls lifecycle
# - Persistent mode: queen-rbee pre-started, rbee-keeper just connects
# - Cascading shutdown: queen-rbee → rbee-hive → workers
#
# CONFIGURATION FLOW (TEAM-041):
# - rbee-keeper setup add-node: Add remote node with SSH validation
# - rbee-keeper setup install: Remote installation via SSH (git clone + cargo build)
# - rbee-keeper setup list-nodes: List all registered rbee-hive nodes
# - queen-rbee uses registry to establish SSH connections for inference
#
# GGUF Support: TEAM-036 added quantized model support (Q4_K_M, Q5_K_M, etc.)
# Installation: TEAM-036 added XDG-compliant installation system
#
# Verified by Testing Team 🔍
# Updated by Implementation Team 🛠️
