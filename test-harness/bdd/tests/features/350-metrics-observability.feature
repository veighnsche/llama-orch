# Traceability: RC-P2-METRICS (Release Candidate P2 Metrics & Observability)
# Created by: TEAM-100 (THE CENTENNIAL TEAM! 💯🎉)
# Components: pool-managerd, queen-rbee, llm-worker-rbee, narration-core
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# 🎀 SPECIAL: TEAM-100 integrates narration-core for human-readable debugging!

Feature: Metrics & Observability - Prometheus Metrics with Narration
  As a system operator
  I want comprehensive Prometheus metrics with human-readable narration
  So that I can monitor system health and debug issues effectively

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And narration capture is enabled

  @p2 @metrics @observability
  Scenario: MET-001 - Expose /metrics endpoint with narration
    Given pool-managerd is running at "http://0.0.0.0:9090"
    When I enable the metrics endpoint
    Then narration event is emitted with actor "pool-managerd"
    And narration event is emitted with action "metrics_enable"
    And narration human field contains "Metrics endpoint enabled"
    When I request "/metrics"
    Then the response status is 200
    And narration event is emitted with action "metrics_serve"
    And narration human field contains "Serving metrics"
    And narration correlation_id is present

  @p2 @metrics @prometheus
  Scenario: MET-002 - Prometheus format compliance
    Given pool-managerd is running with metrics enabled
    When I request "/metrics"
    Then the response is in Prometheus text format
    And the response contains "# HELP" comments
    And the response contains "# TYPE" declarations
    And metric names follow prometheus naming conventions
    And narration event confirms "Metrics formatted in Prometheus text format"

  @p2 @metrics @worker-state
  Scenario: MET-003 - Worker count by state with narration
    Given pool-managerd is running with 3 workers
    And worker "worker-1" is in state "live"
    And worker "worker-2" is in state "live"
    And worker "worker-3" is in state "down"
    When I request "/metrics"
    Then metric "pool_mgr_workers_total{status=\"live\"}" equals 2
    And metric "pool_mgr_workers_total{status=\"down\"}" equals 1
    And narration event contains "Worker state metrics collected"
    And narration correlation_id propagates to worker queries

  @p2 @metrics @latency
  Scenario: MET-004 - Request latency histogram
    Given pool-managerd is running
    When I make 10 requests with varying latencies
    Then metric "pool_mgr_request_duration_seconds" is a histogram
    And histogram has buckets [0.001, 0.01, 0.1, 1.0, 10.0]
    And histogram sum matches total request time
    And narration events include duration_ms for each request
    And narration correlation_ids link requests to metrics

  @p2 @metrics @errors
  Scenario: MET-005 - Error rate counter with narration
    Given pool-managerd is running
    When 5 requests succeed
    And 3 requests fail with error "VRAM exhausted"
    Then metric "pool_mgr_errors_total{error_kind=\"vram_exhausted\"}" equals 3
    And metric "pool_mgr_requests_total" equals 8
    And narration events include error_kind field
    And narration human field describes each error clearly
    And secrets are redacted in narration events

  @p2 @metrics @vram
  Scenario: MET-006 - VRAM usage gauge
    Given pool-managerd is running with GPU workers
    And GPU 0 has 8192 MB total VRAM
    And GPU 0 has 2048 MB allocated VRAM
    When I request "/metrics"
    Then metric "pool_mgr_gpu_vram_total_bytes{gpu=\"0\"}" equals 8589934592
    And metric "pool_mgr_gpu_vram_allocated_bytes{gpu=\"0\"}" equals 2147483648
    And narration event confirms "VRAM metrics collected for GPU 0"
    And narration includes device field "GPU0"

  @p2 @metrics @download
  Scenario: MET-007 - Model download progress with narration
    Given model provisioner is downloading "hf:meta-llama/Llama-3.2-1B"
    And download is 60% complete (600 MB of 1000 MB)
    When I request "/metrics"
    Then metric "model_download_progress_percent{model=\"Llama-3.2-1B\"}" equals 60
    And metric "model_download_bytes_total{model=\"Llama-3.2-1B\"}" equals 1048576000
    And metric "model_download_bytes_downloaded{model=\"Llama-3.2-1B\"}" equals 629145600
    And narration event contains "Model download progress: 60%"
    And narration includes model_ref field

  @p2 @metrics @health
  Scenario: MET-008 - Health check success rate
    Given pool-managerd performs health checks every 5 seconds
    When 8 health checks succeed
    And 2 health checks fail
    Then metric "pool_mgr_health_checks_total{result=\"success\"}" equals 8
    And metric "pool_mgr_health_checks_total{result=\"failure\"}" equals 2
    And narration events describe each health check result
    And narration correlation_id groups health check cycles

  @p2 @metrics @crash-rate
  Scenario: MET-009 - Crash rate by model with narration
    Given workers are running with different models
    When worker with "llama-7b" crashes
    And worker with "llama-7b" crashes
    And worker with "mistral-7b" crashes
    Then metric "worker_crashes_total{model=\"llama-7b\"}" equals 2
    And metric "worker_crashes_total{model=\"mistral-7b\"}" equals 1
    And narration events include error_kind "worker_crash"
    And narration human field explains crash reason
    And narration includes model_ref and worker_id

  @p2 @metrics @throughput
  Scenario: MET-010 - Request throughput counter
    Given pool-managerd is running
    When 100 requests are processed in 10 seconds
    Then metric "pool_mgr_requests_total" equals 100
    And metric rate is approximately 10 requests/second
    And narration events track request lifecycle
    And narration correlation_ids enable request tracing

  @p2 @metrics @narration
  Scenario: MET-011 - Narration events emitted for metric updates
    Given pool-managerd is running with narration enabled
    When metrics are collected
    Then narration event is emitted with action "metrics_collect"
    And narration human field contains "Collected metrics"
    And narration includes metric count
    And narration correlation_id is present
    And narration emitted_by field contains "pool-managerd@"
    And narration emitted_at_ms is within last 1000 ms

  @p2 @metrics @correlation
  Scenario: MET-012 - Correlation IDs in metric labels
    Given pool-managerd is running
    And request has correlation_id "req-test-100"
    When I make a request with correlation_id "req-test-100"
    Then metric labels include correlation_id "req-test-100"
    And narration events use same correlation_id
    And I can trace request across services using correlation_id

  @p2 @metrics @secrets
  Scenario: MET-013 - Secret redaction in metric labels
    Given pool-managerd is running
    When I make request with Bearer token "secret-token-123"
    Then metric labels do not contain "secret-token-123"
    And metric labels contain "[REDACTED]" for sensitive fields
    And narration events redact Bearer tokens
    And narration events redact API keys
    And narration human field does not leak secrets

  @p2 @metrics @cute-mode
  Scenario: MET-014 - Cute mode narration for metrics (optional)
    Given pool-managerd is running with cute mode enabled
    When metrics endpoint is accessed
    Then narration cute field is present
    And narration cute field contains emoji
    And narration cute field describes metrics whimsically
    And narration cute field is under 150 characters
    And example cute message: "Pool-managerd proudly shows off its metrics! 📊✨"

  @p2 @metrics @story-mode
  Scenario: MET-015 - Story mode for multi-service metric flows
    Given queen-rbee requests metrics from pool-managerd
    When pool-managerd responds with metrics
    Then narration story field is present
    And narration story field contains dialogue
    And narration story field shows request-response conversation
    And example story: "\"How are your workers?\" asked queen-rbee. \"Great!\" replied pool-managerd, \"All 5 workers healthy!\""
    And narration story field is under 200 characters
