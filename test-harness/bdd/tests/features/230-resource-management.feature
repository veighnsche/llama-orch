# Traceability: Gap Analysis P1 - Advanced Resource Management
# Created by: TEAM-079
# Priority: P1 - Important for production optimization
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# ⚠️ These scenarios test dynamic resource allocation and monitoring

Feature: Advanced Resource Management
  As a system managing GPU and CPU resources
  I want to optimize resource allocation dynamically
  So that hardware is used efficiently

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "workstation"
    And rbee-hive is running at "http://localhost:9200"

  @resources @p1
  Scenario: Gap-R1 - Multi-GPU automatic selection
    Given device 0 has 2GB VRAM free
    And device 1 has 6GB VRAM free
    And model requires 4GB VRAM
    When user requests "--backend cuda" (no device specified)
    Then rbee-hive selects device 1 (most free VRAM)
    And rbee-hive logs "Selected CUDA device 1 (6GB free)"
    And worker starts on device 1

  @resources @p1
  Scenario: Gap-R2 - Dynamic RAM monitoring during preflight
    Given preflight starts with 8GB RAM available
    When another process allocates 4GB during check
    Then preflight re-checks RAM availability
    And detects only 4GB now available
    And fails if model requires 6GB
    And displays "RAM decreased during preflight check"

  @resources @p1
  Scenario: Gap-R3 - GPU temperature monitoring
    Given GPU temperature is 85°C
    And safe threshold is 80°C
    When rbee-hive checks GPU health
    Then preflight fails with "GPU_TOO_HOT"
    And displays "GPU temperature: 85°C (max: 80°C)"
    And suggests "Wait for GPU to cool down"

  @resources @p1
  Scenario: Gap-R4 - CPU core pinning for CPU backend
    Given system has 16 CPU cores
    And cores 0-7 are busy
    When rbee-hive starts CPU worker
    Then worker is pinned to cores 8-15
    And worker logs "Pinned to CPU cores 8-15"
    And other cores remain available

  @resources @p1
  Scenario: Gap-R5 - VRAM fragmentation detection
    Given GPU has 8GB total VRAM
    And 6GB is allocated in fragments
    And largest contiguous block is 1GB
    When model requires 2GB contiguous VRAM
    Then preflight fails with "VRAM_FRAGMENTED"
    And suggests "Restart workers to defragment VRAM"

  @resources @p2
  Scenario: Gap-R6 - Bandwidth throttling for downloads
    Given system bandwidth limit is 10 MB/s
    When model download starts
    Then download speed is throttled to 10 MB/s
    And progress shows "Downloading at 10 MB/s (throttled)"
    And other network traffic is not starved

  @resources @p2
  Scenario: Gap-R7 - Disk I/O monitoring
    Given disk I/O is at 90% capacity
    When model loading starts
    Then rbee-hive detects high I/O
    And warns "High disk I/O may slow loading"
    And continues with degraded performance
