Feature: Worker-Orcd Narration Integration
  As the narration-core team
  We verify that worker-orcd correctly integrates our logging patterns
  Including correlation IDs, performance metrics, and editorial standards

  Background:
    Given narration capture is enabled

  Scenario: Worker emits inference start narration
    Given worker-orcd receives inference request with correlation_id "req-123"
    When worker starts inference for job "job-456"
    Then a narration event is emitted with actor "worker-orcd"
    And the narration event has action "inference_start"
    And the narration event has correlation_id "req-123"
    And the narration event has target "job-456"
    And the human field includes "Starting inference"

  Scenario: Worker propagates correlation IDs throughout request lifecycle
    Given orchestratord sends request with correlation_id "req-abc"
    When worker-orcd processes the request
    And worker emits start narration
    And worker emits progress narration
    And worker emits completion narration
    Then all narration events include correlation_id "req-abc"

  Scenario: Worker emits performance metrics on completion
    Given worker completes inference in 2500 ms
    And worker generated 150 tokens
    When worker emits completion narration
    Then the narration event has duration_ms 2500
    And the narration event has tokens_out 150
    And the human field includes "150 tokens"
    And the human field includes "2500 ms"

  Scenario: Worker includes model context in narration
    Given worker is running model "llama-7b"
    When worker emits inference start narration
    Then the narration event has model_ref "llama-7b"
    And the human field includes "llama-7b"

  Scenario: Worker emits heartbeat narration
    Given worker is alive
    When worker sends heartbeat to pool-managerd
    Then a narration event is emitted with actor "worker-orcd"
    And the narration event has action "heartbeat_send"
    And the narration event has target "pool-managerd"
    And the human field includes "heartbeat"

  Scenario: Worker emits ready callback narration
    Given worker has started successfully
    And worker is running engine "llamacpp-v1" version "b1234"
    When worker sends ready callback to pool-managerd
    Then a narration event is emitted with actor "worker-orcd"
    And the narration event has action "ready_callback"
    And the narration event has engine "llamacpp-v1"
    And the narration event has engine_version "b1234"
    And the human field includes "ready"

  Scenario: Worker emits error narration with context
    Given worker encounters CUDA out of memory error
    When worker emits error narration
    Then a narration event is emitted with actor "worker-orcd"
    And the narration event has error_kind "cuda_oom"
    And the human field includes "CUDA out of memory"
    And the human field includes specific memory amounts

  Scenario: Worker narration follows editorial guidelines (under 100 chars)
    When worker emits any narration
    Then the human field is under 100 characters
    And the human field uses present tense
    And the human field uses active voice

  Scenario: Worker narration includes correlation ID when available
    Given orchestratord provides correlation_id "req-xyz"
    When worker emits any narration
    Then the narration event has correlation_id "req-xyz"

  Scenario: Worker narration handles missing correlation ID gracefully
    Given no correlation_id is provided
    When worker emits narration
    Then the narration event is still valid
    And the correlation_id field is absent


  Scenario: Worker emits cancellation narration
    Given worker receives cancellation request for job "job-789"
    When worker processes cancellation
    Then a narration event is emitted with actor "worker-orcd"
    And the narration event has action "cancel"
    And the narration event has target "job-789"
    And the human field includes "cancel"

  Scenario: Worker narration includes worker_id for identification
    Given worker has worker_id "worker-gpu0-r1"
    When worker emits any narration
    Then the narration event has worker_id "worker-gpu0-r1"

  Scenario: Worker narration is specific and actionable
    Given worker fails to allocate VRAM
    When worker emits error narration
    Then the human field includes the requested amount
    And the human field includes the available amount
    And the human field includes the GPU identifier
    And the human field does not say "error occurred"

  Scenario: Worker uses narrate_auto for provenance
    When worker emits narration using narrate_auto
    Then the narration event has emitted_by field
    And the narration event has emitted_at_ms field
    And emitted_by includes service name and version

  Scenario: Worker propagates correlation ID in outgoing HTTP requests
    Given worker has correlation_id "req-def"
    When worker sends HTTP request to pool-managerd
    Then the request includes header "X-Correlation-Id" with value "req-def"

  Scenario: Editorial review - inference start narration quality
    When worker emits inference start narration
    Then the human field is clear and specific
    And the human field includes job_id
    And the human field includes model_ref
    And the human field is under 100 characters
    And the human field uses present tense "Starting"

  Scenario: Editorial review - inference complete narration quality
    When worker emits inference complete narration
    Then the human field includes token count
    And the human field includes duration in milliseconds
    And the human field is under 100 characters
    And the human field is actionable for debugging

  Scenario: Editorial review - error narration quality
    When worker emits error narration
    Then the human field explains what failed
    And the human field explains why it failed
    And the human field includes specific values
    And the human field does not use error codes without explanation
    And the human field is under 100 characters

  Scenario: Worker narration supports distributed tracing
    Given OpenTelemetry is enabled
    And current trace_id is "trace-123"
    When worker emits narration
    Then the narration event has trace_id "trace-123"
    And the narration event has span_id
    And the span_id is valid

  Scenario: Worker emits token generation metrics
    Given worker generates tokens at 60 tokens/second
    When worker emits completion narration
    Then the narration event has tokens_out
    And the narration event has decode_time_ms
    And performance can be calculated from metrics

  Scenario: Worker narration timeline is coherent
    Given worker processes job "job-999"
    When worker emits start narration
    And worker emits completion narration
    Then start narration has correlation_id
    And completion narration has same correlation_id
    And both narrations have same job_id "job-999"
    And timeline is traceable via correlation_id
