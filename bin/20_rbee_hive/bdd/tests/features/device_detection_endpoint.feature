# Created by: TEAM-159
# Tests rbee-hive device detection endpoint (happy flow lines 43-44)

Feature: Hive Device Detection Endpoint
  As rbee-hive
  I want to detect and report device capabilities
  So that queen-rbee can schedule jobs appropriately

  # ============================================================================
  # Happy Flow: Lines 43-44
  # ============================================================================

  Scenario: Device detection returns CPU and GPU information
    Given rbee-hive is running
    When queen requests device detection via GET /v1/devices
    Then the response should be 200 OK
    And the response should include CPU information:
      | cores | ram_gb |
      | 8     | 32     |
    And the response should include GPU list
    And the response should include model count
    And the response should include worker count

  Scenario: Device detection with CUDA GPUs
    Given rbee-hive is running on a system with CUDA GPUs
    When queen requests device detection via GET /v1/devices
    Then the response should include GPUs:
      | id   | name       | vram_gb |
      | gpu0 | RTX 3060   | 12      |
      | gpu1 | RTX 3090   | 24      |
    And all GPUs should have valid names
    And all GPUs should have positive VRAM

  Scenario: Device detection with CPU only (no GPUs)
    Given rbee-hive is running on a CPU-only system
    When queen requests device detection via GET /v1/devices
    Then the response should include CPU information
    And the GPU list should be empty
    And the response should be valid JSON

  Scenario: Device detection with Metal backend (macOS)
    Given rbee-hive is running on macOS with Metal support
    When queen requests device detection via GET /v1/devices
    Then the response should include Metal GPUs
    And the GPU backend should be detected as Metal

  Scenario: Device detection includes model catalog count
    Given rbee-hive has 3 models in the catalog
    When queen requests device detection via GET /v1/devices
    Then the response should show 3 models
    And the model count should be accurate

  Scenario: Device detection includes worker count
    Given rbee-hive has 2 active workers
    When queen requests device detection via GET /v1/devices
    Then the response should show 2 workers
    And the worker count should be accurate

  Scenario: Device detection with no models or workers
    Given rbee-hive has an empty model catalog
    And rbee-hive has no active workers
    When queen requests device detection via GET /v1/devices
    Then the response should show 0 models
    And the response should show 0 workers

  # ============================================================================
  # Real Hardware Detection
  # ============================================================================

  Scenario: Device detection calls real device-detection crate
    Given rbee-hive is running
    When queen requests device detection via GET /v1/devices
    Then rbee-hive should call detect_backends()
    And rbee-hive should call detect_gpus()
    And rbee-hive should call get_system_ram_gb()
    And the response should contain real hardware information

  Scenario: Device detection uses nvidia-smi for GPU detection
    Given rbee-hive is running on a system with nvidia-smi
    When queen requests device detection via GET /v1/devices
    Then nvidia-smi should be executed
    And GPU information should be parsed from nvidia-smi output
    And GPU names should match nvidia-smi output

  Scenario: Device detection handles nvidia-smi not found
    Given rbee-hive is running on a system without nvidia-smi
    When queen requests device detection via GET /v1/devices
    Then the GPU list should be empty
    And the response should still be valid
    And CPU information should still be present

  # ============================================================================
  # Response Format
  # ============================================================================

  Scenario: Device detection response has correct JSON structure
    Given rbee-hive is running
    When queen requests device detection via GET /v1/devices
    Then the response should be valid JSON
    And the response should have field "cpu"
    And the response should have field "gpus"
    And the response should have field "models"
    And the response should have field "workers"
    And cpu should have field "cores"
    And cpu should have field "ram_gb"
    And each GPU should have field "id"
    And each GPU should have field "name"
    And each GPU should have field "vram_gb"

  Scenario: GPU IDs follow correct format
    Given rbee-hive is running with 3 GPUs
    When queen requests device detection via GET /v1/devices
    Then GPU IDs should be "gpu0", "gpu1", "gpu2"
    And GPU IDs should be sequential
    And GPU IDs should start from 0

  # ============================================================================
  # Edge Cases
  # ============================================================================

  Scenario: Device detection with maximum GPU count
    Given rbee-hive is running with 8 GPUs
    When queen requests device detection via GET /v1/devices
    Then all 8 GPUs should be reported
    And the response should be valid

  Scenario: Device detection with unusual RAM sizes
    Given rbee-hive is running with 512 GB RAM
    When queen requests device detection via GET /v1/devices
    Then the CPU RAM should be reported as 512 GB
    And the response should be valid

  Scenario: Device detection with high core count
    Given rbee-hive is running with 128 CPU cores
    When queen requests device detection via GET /v1/devices
    Then the CPU cores should be reported as 128
    And the response should be valid

  Scenario: Device detection is idempotent
    Given rbee-hive is running
    When queen requests device detection via GET /v1/devices
    And queen requests device detection via GET /v1/devices again
    Then both responses should be identical
    And both responses should be valid

  # ============================================================================
  # Performance
  # ============================================================================

  Scenario: Device detection completes quickly
    Given rbee-hive is running
    When queen requests device detection via GET /v1/devices
    Then the response should be received within 2 seconds
    And the response should be valid

  Scenario: Device detection can handle concurrent requests
    Given rbee-hive is running
    When 5 queens request device detection simultaneously
    Then all 5 requests should succeed
    And all responses should be identical
    And all responses should be valid
