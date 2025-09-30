Feature: GPU-Only Preflight Enforcement
  # Traceability: B-PREFLIGHT-001 through B-PREFLIGHT-013
  # Spec: OC-POOL-3012 - CPU inference spillover is disallowed
  
  Scenario: CUDA available when nvcc on PATH
    Given nvcc is available on PATH
    When I check cuda_available
    Then the result is true
  
  Scenario: CUDA available when nvidia-smi on PATH
    Given nvidia-smi is available on PATH
    When I check cuda_available
    Then the result is true
  
  Scenario: CUDA not available when neither tool on PATH
    Given nvcc is not on PATH
    And nvidia-smi is not on PATH
    When I check cuda_available
    Then the result is false
  
  Scenario: assert_gpu_only succeeds when CUDA available
    Given nvcc is available on PATH
    When I call assert_gpu_only
    Then the call succeeds
  
  Scenario: assert_gpu_only fails when CUDA not available
    Given nvcc is not on PATH
    And nvidia-smi is not on PATH
    When I call assert_gpu_only
    Then the call fails with error
    And the error message contains "GPU-only enforcement"
    And the error message contains "nvcc/nvidia-smi not found"
  
  Scenario: Preflight checks PATH environment
    Given PATH is set to "/usr/local/cuda/bin"
    And nvcc exists at "/usr/local/cuda/bin/nvcc"
    When I check cuda_available
    Then the result is true
  
  Scenario: Preflight fails fast on missing CUDA
    Given no CUDA toolkit is installed
    When I call assert_gpu_only
    Then the call fails immediately
    And no CPU fallback is attempted
