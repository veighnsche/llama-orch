# Traceability: TEST-001 (split by TEAM-077, TEAM-078)
# Architecture: TEAM-037 (queen-rbee orchestration, HuggingFace downloads)
# Components: rbee-hive (pool manager), ModelProvisioner
# Created by: TEAM-078 (split from 020-model-provisioning.feature)
# Updated by: TEAM-036 (GGUF support)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# ⚠️ DO NOT use mock servers - wire up actual rbee-hive and ModelProvisioner libraries

Feature: Model Provisioner
  As a system provisioning models
  I want to download models from Hugging Face with progress tracking
  So that workers can load them for inference

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"
    And the model catalog is SQLite at "~/.rbee/models.db"

  Scenario: Model download with progress tracking
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

  # ============================================================================
  # GGUF Model Support (TEAM-036)
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

  @edge-case
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
