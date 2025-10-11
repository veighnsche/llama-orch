# Traceability: TEST-001 (split by TEAM-077)
# Architecture: TEAM-037 (queen-rbee orchestration)
# Components: rbee-hive (pool manager), worker preflight checks
# Refactored by: TEAM-077 (split from test-001.feature into focused feature files)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: Worker Preflight Checks
  As a system preparing to start a worker
  I want to validate resources before startup
  So that I can fail fast if resources are insufficient

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"

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

  @edge-case
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
