# BDD Feature Mapping: Behaviors → Features → Scenarios → Steps

**Purpose**: Map all 200+ behaviors to BDD features, scenarios, and step definitions  
**Date**: 2025-09-30

---

## 📋 Structure

```
Feature (Gherkin .feature file)
  └─ Scenario (test case)
      └─ Steps (Given/When/Then)
          └─ Step Function (Rust implementation)
              └─ Behaviors (B-XXX-NNN codes)
```

---

## Feature 1: Middleware and Request Processing

**File**: `tests/features/middleware/correlation_id.feature`

### Scenario 1.1: Request with correlation ID
```gherkin
Given a client request with X-Correlation-Id header "test-123"
When I make a request to any endpoint
Then the response includes X-Correlation-Id header "test-123"
```
**Steps**:
- `given_request_with_correlation_id(world, id)` → sets `world.extra_headers`
- `when_make_request(world, endpoint)` → calls `world.http_call()`
- `then_response_has_correlation_id(world, expected)` → checks `world.last_headers`

**Behaviors**: B-MW-001, B-MW-003, B-MW-004

---

### Scenario 1.2: Request without correlation ID
```gherkin
Given a client request without X-Correlation-Id header
When I make a request to any endpoint
Then the response includes a generated X-Correlation-Id header
And the correlation ID is a valid UUIDv4
```
**Steps**:
- `given_request_without_correlation_id(world)` → clears headers
- `when_make_request(world, endpoint)` → calls `world.http_call()`
- `then_response_has_generated_correlation_id(world)` → checks header exists
- `then_correlation_id_is_uuid(world)` → validates UUID format

**Behaviors**: B-MW-002, B-MW-003, B-MW-004

---

### Scenario 1.3: API key validation - missing key
```gherkin
Given no API key is provided
When I make a request to /v2/tasks
Then I receive 401 Unauthorized
And the response includes X-Correlation-Id header
```
**Steps**:
- `given_no_api_key(world)` → sets `world.api_key = None`
- `when_make_request(world, "/v2/tasks")` → POST request
- `then_status_code(world, 401)` → checks status
- `then_response_has_correlation_id(world)` → checks header

**Behaviors**: B-MW-011, B-MW-014

---

### Scenario 1.4: API key validation - invalid key
```gherkin
Given an invalid API key is provided
When I make a request to /v2/tasks
Then I receive 403 Forbidden
And the response includes X-Correlation-Id header
```
**Steps**:
- `given_invalid_api_key(world)` → sets `world.api_key = Some("bad")`
- `when_make_request(world, "/v2/tasks")` → POST request
- `then_status_code(world, 403)` → checks status
- `then_response_has_correlation_id(world)` → checks header

**Behaviors**: B-MW-012, B-MW-014

---

### Scenario 1.5: Metrics endpoint bypasses auth
```gherkin
Given no API key is provided
When I make a request to /metrics
Then I receive 200 OK
And the response contains Prometheus metrics
```
**Steps**:
- `given_no_api_key(world)` → sets `world.api_key = None`
- `when_make_request(world, "/metrics")` → GET request
- `then_status_code(world, 200)` → checks status
- `then_response_contains_prometheus_metrics(world)` → validates format

**Behaviors**: B-MW-010

---

## Feature 2: Control Plane - Capabilities

**File**: `tests/features/control_plane/capabilities.feature`

### Scenario 2.1: Get capabilities
```gherkin
Given a Control Plane API endpoint
When I request capabilities
Then I receive 200 OK
And the response includes api_version
And the response includes engines array
```
**Steps**:
- `given_control_plane_endpoint(world)` → no-op
- `when_request_capabilities(world)` → GET `/v2/meta/capabilities`
- `then_status_code(world, 200)` → checks status
- `then_response_includes_field(world, "api_version")` → checks JSON
- `then_response_includes_field(world, "engines")` → checks JSON

**Behaviors**: B-CP-001, B-CP-002, B-CP-003, B-CP-004

---

### Scenario 2.2: Capabilities are cached
```gherkin
Given a Control Plane API endpoint
When I request capabilities twice
Then both responses are identical
And the second response is served from cache
```
**Steps**:
- `given_control_plane_endpoint(world)` → no-op
- `when_request_capabilities_twice(world)` → two GET requests
- `then_responses_identical(world)` → compares bodies
- `then_second_from_cache(world)` → checks timing/logs

**Behaviors**: B-CP-001, B-CP-002

---

## Feature 3: Control Plane - Pool Management

**File**: `tests/features/control_plane/pool_health.feature`

### Scenario 3.1: Get pool health - existing pool
```gherkin
Given a pool "default" exists
When I request pool health for "default"
Then I receive 200 OK
And the response includes live status
And the response includes ready status
And the response includes draining status
And the response includes metrics
```
**Steps**:
- `given_pool_exists(world, pool_id)` → registers pool in state
- `when_request_pool_health(world, pool_id)` → GET `/v2/pools/{id}/health`
- `then_status_code(world, 200)` → checks status
- `then_response_includes_field(world, "live")` → checks JSON
- `then_response_includes_field(world, "ready")` → checks JSON
- `then_response_includes_field(world, "draining")` → checks JSON
- `then_response_includes_field(world, "metrics")` → checks JSON

**Behaviors**: B-CP-010, B-CP-011, B-CP-013, B-CP-014, B-CP-015

---

### Scenario 3.2: Get pool health - non-existent pool
```gherkin
Given a pool "unknown" does not exist
When I request pool health for "unknown"
Then I receive 200 OK
And the response shows live=true and ready=true (defaults)
```
**Steps**:
- `given_pool_does_not_exist(world, pool_id)` → ensures not registered
- `when_request_pool_health(world, pool_id)` → GET request
- `then_status_code(world, 200)` → checks status
- `then_response_has_defaults(world)` → checks live=true, ready=true

**Behaviors**: B-CP-010, B-CP-012, B-CP-015

---

### Scenario 3.3: Drain pool
```gherkin
Given a pool "default" exists
When I request pool drain with deadline_ms 5000
Then I receive 202 Accepted
And the pool is marked as draining
```
**Steps**:
- `given_pool_exists(world, pool_id)` → registers pool
- `when_request_pool_drain(world, pool_id, deadline_ms)` → POST `/v2/pools/{id}/drain`
- `then_status_code(world, 202)` → checks status
- `then_pool_is_draining(world, pool_id)` → checks state.draining_pools

**Behaviors**: B-CP-020, B-CP-021, B-CP-022

---

### Scenario 3.4: Reload pool - success
```gherkin
Given a pool "default" exists
When I request pool reload with new_model_ref "model-v2"
Then I receive 200 OK
And the model_state metric is updated
```
**Steps**:
- `given_pool_exists(world, pool_id)` → registers pool
- `when_request_pool_reload(world, pool_id, model_ref)` → POST `/v2/pools/{id}/reload`
- `then_status_code(world, 200)` → checks status
- `then_metric_updated(world, "model_state")` → checks metrics

**Behaviors**: B-CP-030, B-CP-032, B-CP-033

---

### Scenario 3.5: Reload pool - failure (test sentinel)
```gherkin
Given a pool "default" exists
When I request pool reload with new_model_ref "bad"
Then I receive 409 Conflict
```
**Steps**:
- `given_pool_exists(world, pool_id)` → registers pool
- `when_request_pool_reload(world, pool_id, "bad")` → POST request
- `then_status_code(world, 409)` → checks status

**Behaviors**: B-CP-030, B-CP-031

---

### Scenario 3.6: Purge pool
```gherkin
Given a pool "default" exists
When I request pool purge
Then I receive 202 Accepted
```
**Steps**:
- `given_pool_exists(world, pool_id)` → registers pool
- `when_request_pool_purge(world, pool_id)` → POST `/v2/pools/{id}/purge`
- `then_status_code(world, 202)` → checks status

**Behaviors**: B-CP-040, B-CP-041

---

## Feature 4: Control Plane - Worker Registration

**File**: `tests/features/control_plane/worker_registration.feature`

### Scenario 4.1: Register worker - missing token
```gherkin
Given no Bearer token is provided
When I register a worker
Then I receive 401 Unauthorized
And the response includes code 40101
And the response includes message "MISSING_TOKEN"
```
**Steps**:
- `given_no_bearer_token(world)` → clears Authorization header
- `when_register_worker(world, body)` → POST `/v2/workers/register`
- `then_status_code(world, 401)` → checks status
- `then_response_code(world, 40101)` → checks JSON code
- `then_response_message(world, "MISSING_TOKEN")` → checks JSON message

**Behaviors**: B-CP-050, B-CP-051

---

### Scenario 4.2: Register worker - invalid token
```gherkin
Given a Bearer token "bad-token" is provided
And AUTH_TOKEN environment variable is "good-token"
When I register a worker
Then I receive 401 Unauthorized
And the response includes code 40102
And the response includes message "BAD_TOKEN"
```
**Steps**:
- `given_bearer_token(world, token)` → sets Authorization header
- `given_auth_token_env(world, expected)` → sets env var
- `when_register_worker(world, body)` → POST request
- `then_status_code(world, 401)` → checks status
- `then_response_code(world, 40102)` → checks JSON code
- `then_response_message(world, "BAD_TOKEN")` → checks JSON message

**Behaviors**: B-CP-050, B-CP-052, B-CP-053

---

### Scenario 4.3: Register worker - valid token
```gherkin
Given a Bearer token "valid-token" is provided
And AUTH_TOKEN environment variable is "valid-token"
When I register a worker with pool_id "gpu-0" and replica_id "r1"
Then I receive 200 OK
And the response includes identity fingerprint
And the response includes pool_id "gpu-0"
And the response includes replica_id "r1"
And an identity breadcrumb is logged
```
**Steps**:
- `given_bearer_token(world, token)` → sets Authorization header
- `given_auth_token_env(world, expected)` → sets env var
- `when_register_worker_with_params(world, pool_id, replica_id)` → POST with body
- `then_status_code(world, 200)` → checks status
- `then_response_includes_field(world, "identity")` → checks JSON
- `then_response_field_equals(world, "pool_id", "gpu-0")` → checks JSON
- `then_response_field_equals(world, "replica_id", "r1")` → checks JSON
- `then_identity_breadcrumb_logged(world)` → checks state.logs

**Behaviors**: B-CP-050, B-CP-053, B-CP-054, B-CP-055, B-CP-056, B-CP-058

---

## Feature 5: Data Plane - Task Admission

**File**: `tests/features/data_plane/task_admission.feature`

### Scenario 5.1: Enqueue valid task
```gherkin
Given an OrchQueue API endpoint
When I enqueue a completion task with valid payload
Then I receive 202 Accepted
And the response includes task_id
And the response includes queue_position
And the response includes predicted_start_ms
And the response includes streams object
And the response includes preparation object
And budget headers are present
And correlation ID is present
```
**Steps**:
- `given_orchqueue_endpoint(world)` → no-op
- `when_enqueue_valid_task(world)` → POST `/v2/tasks` with valid body
- `then_status_code(world, 202)` → checks status
- `then_response_includes_field(world, "task_id")` → checks JSON
- `then_response_includes_field(world, "queue_position")` → checks JSON
- `then_response_includes_field(world, "predicted_start_ms")` → checks JSON
- `then_response_includes_field(world, "streams")` → checks JSON
- `then_response_includes_field(world, "preparation")` → checks JSON
- `then_budget_headers_present(world)` → checks headers
- `then_correlation_id_present(world)` → checks headers

**Behaviors**: B-DP-010, B-DP-011, B-DP-012, B-DP-015, B-DP-020, B-DP-021, B-DP-022, B-DP-023, B-DP-024, B-DP-025, B-DP-026, B-DP-027, B-DP-028

---

### Scenario 5.2: Invalid ctx parameter
```gherkin
Given an OrchQueue API endpoint
When I enqueue a task with ctx=-1
Then I receive 400 Bad Request
And the error code is INVALID_PARAMS
And the error message mentions "ctx must be >= 0"
```
**Steps**:
- `given_orchqueue_endpoint(world)` → no-op
- `when_enqueue_task_with_invalid_ctx(world)` → POST with ctx=-1
- `then_status_code(world, 400)` → checks status
- `then_error_code(world, "INVALID_PARAMS")` → checks JSON
- `then_error_message_contains(world, "ctx must be >= 0")` → checks JSON

**Behaviors**: B-DP-001, B-ERR-001

---

### Scenario 5.3: Invalid deadline
```gherkin
Given an OrchQueue API endpoint
When I enqueue a task with deadline_ms=0
Then I receive 400 Bad Request
And the error code is DEADLINE_UNMET
```
**Steps**:
- `given_orchqueue_endpoint(world)` → no-op
- `when_enqueue_task_with_deadline(world, 0)` → POST with deadline_ms=0
- `then_status_code(world, 400)` → checks status
- `then_error_code(world, "DEADLINE_UNMET")` → checks JSON

**Behaviors**: B-DP-002, B-ERR-002

---

### Scenario 5.4: Queue full - reject policy
```gherkin
Given an OrchQueue API endpoint with reject policy
And the queue is at capacity
When I enqueue a task
Then I receive 429 Too Many Requests
And the error code is ADMISSION_REJECT
And the Retry-After header is present
And the X-Backoff-Ms header is present
And the error envelope includes retry_after_ms
And the error envelope includes policy_label
```
**Steps**:
- `given_orchqueue_with_policy(world, "reject")` → sets env var
- `given_queue_at_capacity(world)` → fills queue
- `when_enqueue_task(world)` → POST request
- `then_status_code(world, 429)` → checks status
- `then_error_code(world, "ADMISSION_REJECT")` → checks JSON
- `then_header_present(world, "Retry-After")` → checks header
- `then_header_present(world, "X-Backoff-Ms")` → checks header
- `then_error_includes_field(world, "retry_after_ms")` → checks JSON
- `then_error_includes_field(world, "policy_label")` → checks JSON

**Behaviors**: B-DP-013, B-DP-031, B-DP-032, B-DP-033, B-DP-034, B-DP-035, B-ERR-005, B-ERR-015, B-ERR-016, B-ERR-017

---

### Scenario 5.5: Queue full - drop-lru policy
```gherkin
Given an OrchQueue API endpoint with drop-lru policy
And the queue is at capacity
When I enqueue a task with expected_tokens=2000000
Then I receive 429 Too Many Requests
And the error code is QUEUE_FULL_DROP_LRU
And the oldest task is dropped
```
**Steps**:
- `given_orchqueue_with_policy(world, "drop-lru")` → sets env var
- `given_queue_at_capacity(world)` → fills queue
- `when_enqueue_task_with_expected_tokens(world, 2000000)` → POST request
- `then_status_code(world, 429)` → checks status
- `then_error_code(world, "QUEUE_FULL_DROP_LRU")` → checks JSON
- `then_oldest_task_dropped(world)` → checks queue state

**Behaviors**: B-DP-003, B-DP-014, B-DP-035, B-ERR-006

---

## Feature 6: Data Plane - Task Streaming

**File**: `tests/features/data_plane/task_streaming.feature`

### Scenario 6.1: Stream task - basic flow
```gherkin
Given a task "t-1" has been enqueued
When I stream task events for "t-1"
Then I receive 200 OK
And Content-Type is text/event-stream
And correlation ID is present in headers
And budget headers are present
And I receive SSE event "started"
And I receive SSE event "token"
And I receive SSE event "end"
```
**Steps**:
- `given_task_enqueued(world, task_id)` → enqueues task
- `when_stream_task_events(world, task_id)` → GET `/v2/tasks/{id}/events`
- `then_status_code(world, 200)` → checks status
- `then_content_type(world, "text/event-stream")` → checks header
- `then_correlation_id_present(world)` → checks header
- `then_budget_headers_present(world)` → checks headers
- `then_sse_event_received(world, "started")` → parses SSE body
- `then_sse_event_received(world, "token")` → parses SSE body
- `then_sse_event_received(world, "end")` → parses SSE body

**Behaviors**: B-DP-100, B-DP-101, B-DP-102, B-DP-103, B-DP-104, B-DP-120, B-DP-121, B-DP-123, B-DP-124

---

### Scenario 6.2: Stream with health-gated dispatch
```gherkin
Given a pool "default" is ready with slots_free > 0
And an adapter is bound to pool "default"
And a task "t-1" has been enqueued
When I stream task events for "t-1"
Then the task is dispatched to the adapter
And I receive real token events from the adapter
```
**Steps**:
- `given_pool_ready_with_slots(world, pool_id, slots)` → sets pool health
- `given_adapter_bound(world, pool_id)` → binds adapter
- `given_task_enqueued(world, task_id)` → enqueues task
- `when_stream_task_events(world, task_id)` → GET request
- `then_task_dispatched_to_adapter(world)` → checks adapter calls
- `then_real_tokens_received(world)` → checks SSE body

**Behaviors**: B-STR-001, B-STR-002, B-STR-003, B-STR-004, B-STR-005, B-STR-011, B-STR-012, B-STR-013

---

### Scenario 6.3: Stream with pool not ready - fallback
```gherkin
Given a pool "default" is not ready
And a task "t-1" has been enqueued
When I stream task events for "t-1"
Then the deterministic fallback is used
And I receive event "started"
And I receive one event "token" with text "Hello"
And I receive event "end"
```
**Steps**:
- `given_pool_not_ready(world, pool_id)` → sets pool health
- `given_task_enqueued(world, task_id)` → enqueues task
- `when_stream_task_events(world, task_id)` → GET request
- `then_deterministic_fallback_used(world)` → checks logs
- `then_sse_event_received(world, "started")` → parses SSE
- `then_sse_token_with_text(world, "Hello")` → parses SSE
- `then_sse_event_received(world, "end")` → parses SSE

**Behaviors**: B-DP-112, B-DP-150, B-DP-151, B-DP-153

---

### Scenario 6.4: SSE started event includes queue metadata
```gherkin
Given a task "t-1" has been enqueued at position 3
When I stream task events for "t-1"
Then the "started" event includes queue_position=3
And the "started" event includes predicted_start_ms
```
**Steps**:
- `given_task_enqueued_at_position(world, task_id, position)` → enqueues task
- `when_stream_task_events(world, task_id)` → GET request
- `then_started_event_includes(world, "queue_position", 3)` → parses SSE
- `then_started_event_includes_field(world, "predicted_start_ms")` → parses SSE

**Behaviors**: B-DP-120, B-DP-153

---

### Scenario 6.5: SSE transcript is persisted
```gherkin
Given a task "t-1" has been enqueued
When I stream task events for "t-1"
Then an SSE transcript artifact is created
And the artifact includes all events
And the artifact ID is a SHA-256 hash
```
**Steps**:
- `given_task_enqueued(world, task_id)` → enqueues task
- `when_stream_task_events(world, task_id)` → GET request
- `then_transcript_artifact_created(world)` → checks state.artifacts
- `then_artifact_includes_all_events(world)` → checks artifact content
- `then_artifact_id_is_sha256(world)` → validates ID format

**Behaviors**: B-DP-140, B-DP-141, B-STR-030, B-STR-031, B-STR-032, B-STR-033

---

## Feature 7: Data Plane - Task Cancellation

**File**: `tests/features/data_plane/task_cancellation.feature`

### Scenario 7.1: Cancel queued task
```gherkin
Given a task "t-1" has been enqueued
When I cancel task "t-1"
Then I receive 204 No Content
And the task is marked as cancelled
And a cancellation log is emitted
And tasks_canceled_total metric is incremented
```
**Steps**:
- `given_task_enqueued(world, task_id)` → enqueues task
- `when_cancel_task(world, task_id)` → POST `/v2/tasks/{id}/cancel`
- `then_status_code(world, 204)` → checks status
- `then_task_marked_cancelled(world, task_id)` → checks state.cancellations
- `then_cancellation_logged(world, task_id)` → checks state.logs
- `then_metric_incremented(world, "tasks_canceled_total")` → checks metrics

**Behaviors**: B-DP-200, B-DP-201, B-DP-202, B-DP-203, B-DP-205

---

### Scenario 7.2: Cancel during stream - race-free
```gherkin
Given a task "t-1" is streaming
When I cancel task "t-1" mid-stream
Then no further token events are emitted
And the stream terminates cleanly
```
**Steps**:
- `given_task_streaming(world, task_id)` → starts stream
- `when_cancel_task_mid_stream(world, task_id)` → POST cancel
- `then_no_further_tokens(world)` → checks SSE body
- `then_stream_terminates(world)` → checks SSE body

**Behaviors**: B-DP-130, B-DP-131, B-DP-132, B-DP-133, B-DP-206

---

## Feature 8: Sessions

**File**: `tests/features/data_plane/sessions.feature`

### Scenario 8.1: Get session - creates if missing
```gherkin
Given a session "sess-1" does not exist
When I query session "sess-1"
Then I receive 200 OK
And the session is created with default values
And ttl_ms_remaining is 600000
And turns is 0
```
**Steps**:
- `given_session_does_not_exist(world, session_id)` → ensures not in state
- `when_query_session(world, session_id)` → GET `/v2/sessions/{id}`
- `then_status_code(world, 200)` → checks status
- `then_session_created(world, session_id)` → checks state.sessions
- `then_response_field_equals(world, "ttl_ms_remaining", 600000)` → checks JSON
- `then_response_field_equals(world, "turns", 0)` → checks JSON

**Behaviors**: B-SS-001, B-SS-002, B-SS-003, B-SS-004, B-SS-005

---

### Scenario 8.2: Delete session
```gherkin
Given a session "sess-1" exists
When I delete session "sess-1"
Then I receive 204 No Content
And the session is removed
```
**Steps**:
- `given_session_exists(world, session_id)` → creates session in state
- `when_delete_session(world, session_id)` → DELETE `/v2/sessions/{id}`
- `then_status_code(world, 204)` → checks status
- `then_session_removed(world, session_id)` → checks state.sessions

**Behaviors**: B-SS-010, B-SS-011

---

### Scenario 8.3: Delete non-existent session - idempotent
```gherkin
Given a session "sess-1" does not exist
When I delete session "sess-1"
Then I receive 204 No Content
```
**Steps**:
- `given_session_does_not_exist(world, session_id)` → ensures not in state
- `when_delete_session(world, session_id)` → DELETE request
- `then_status_code(world, 204)` → checks status

**Behaviors**: B-SS-010, B-SS-012

---

## Feature 9: Catalog Management

**File**: `tests/features/catalog/catalog_crud.feature`

### Scenario 9.1: Create model
```gherkin
Given a catalog endpoint
When I create a model with id "llama-3-8b" and digest "sha256:abc123"
Then I receive 201 Created
And the response includes id "llama-3-8b"
And the response includes digest "sha256:abc123"
And the model is persisted to FsCatalog
```
**Steps**:
- `given_catalog_endpoint(world)` → no-op
- `when_create_model(world, id, digest)` → POST `/v2/catalog/models`
- `then_status_code(world, 201)` → checks status
- `then_response_field_equals(world, "id", "llama-3-8b")` → checks JSON
- `then_response_field_equals(world, "digest", "sha256:abc123")` → checks JSON
- `then_model_persisted(world, id)` → checks catalog

**Behaviors**: B-CAT-001, B-CAT-003, B-CAT-004, B-CAT-005, B-CAT-006

---

### Scenario 9.2: Create model - missing id
```gherkin
Given a catalog endpoint
When I create a model without an id
Then I receive 400 Bad Request
And the error message is "id required"
```
**Steps**:
- `given_catalog_endpoint(world)` → no-op
- `when_create_model_without_id(world)` → POST with empty id
- `then_status_code(world, 400)` → checks status
- `then_error_message(world, "id required")` → checks JSON

**Behaviors**: B-CAT-001, B-CAT-002

---

### Scenario 9.3: Get model - exists
```gherkin
Given a model "llama-3-8b" exists in catalog
When I get model "llama-3-8b"
Then I receive 200 OK
And the response includes id and digest
```
**Steps**:
- `given_model_exists(world, id)` → creates model in catalog
- `when_get_model(world, id)` → GET `/v2/catalog/models/{id}`
- `then_status_code(world, 200)` → checks status
- `then_response_includes_field(world, "id")` → checks JSON
- `then_response_includes_field(world, "digest")` → checks JSON

**Behaviors**: B-CAT-010, B-CAT-011

---

### Scenario 9.4: Get model - not found
```gherkin
Given a model "unknown" does not exist
When I get model "unknown"
Then I receive 404 Not Found
```
**Steps**:
- `given_model_does_not_exist(world, id)` → ensures not in catalog
- `when_get_model(world, id)` → GET request
- `then_status_code(world, 404)` → checks status

**Behaviors**: B-CAT-010, B-CAT-012

---

### Scenario 9.5: Verify model
```gherkin
Given a model "llama-3-8b" exists in catalog
When I verify model "llama-3-8b"
Then I receive 202 Accepted
And last_verified_ms is updated
```
**Steps**:
- `given_model_exists(world, id)` → creates model
- `when_verify_model(world, id)` → POST `/v2/catalog/models/{id}/verify`
- `then_status_code(world, 202)` → checks status
- `then_last_verified_updated(world, id)` → checks catalog

**Behaviors**: B-CAT-020, B-CAT-021, B-CAT-022

---

### Scenario 9.6: Set model state
```gherkin
Given a model "llama-3-8b" exists with state Active
When I set model state to Retired
Then I receive 202 Accepted
And the model state is Retired
```
**Steps**:
- `given_model_with_state(world, id, "Active")` → creates model
- `when_set_model_state(world, id, "Retired")` → POST `/v2/catalog/models/{id}/state`
- `then_status_code(world, 202)` → checks status
- `then_model_state_is(world, id, "Retired")` → checks catalog

**Behaviors**: B-CAT-030, B-CAT-031, B-CAT-032, B-CAT-034, B-CAT-035

---

### Scenario 9.7: Delete model
```gherkin
Given a model "llama-3-8b" exists in catalog
When I delete model "llama-3-8b"
Then I receive 204 No Content
And the model is removed from catalog
```
**Steps**:
- `given_model_exists(world, id)` → creates model
- `when_delete_model(world, id)` → DELETE `/v2/catalog/models/{id}`
- `then_status_code(world, 204)` → checks status
- `then_model_removed(world, id)` → checks catalog

**Behaviors**: B-CAT-040, B-CAT-041

---

## Feature 10: Artifacts

**File**: `tests/features/artifacts/artifacts.feature`

### Scenario 10.1: Create artifact
```gherkin
Given an artifacts endpoint
When I create an artifact with document {"key": "value"}
Then I receive 201 Created
And the response includes id (SHA-256 hash)
And the artifact is persisted
```
**Steps**:
- `given_artifacts_endpoint(world)` → no-op
- `when_create_artifact(world, doc)` → POST `/v2/artifacts`
- `then_status_code(world, 201)` → checks status
- `then_response_includes_field(world, "id")` → checks JSON
- `then_artifact_id_is_sha256(world)` → validates ID
- `then_artifact_persisted(world)` → checks store

**Behaviors**: B-ART-001, B-ART-002, B-ART-003, B-ART-004, B-ART-022

---

### Scenario 10.2: Get artifact - exists
```gherkin
Given an artifact with id "abc123" exists
When I get artifact "abc123"
Then I receive 200 OK
And the response is the artifact document
```
**Steps**:
- `given_artifact_exists(world, id, doc)` → creates artifact
- `when_get_artifact(world, id)` → GET `/v2/artifacts/{id}`
- `then_status_code(world, 200)` → checks status
- `then_response_equals_document(world, doc)` → checks JSON

**Behaviors**: B-ART-010, B-ART-011, B-ART-012

---

### Scenario 10.3: Get artifact - not found
```gherkin
Given an artifact with id "unknown" does not exist
When I get artifact "unknown"
Then I receive 404 Not Found
```
**Steps**:
- `given_artifact_does_not_exist(world, id)` → ensures not in store
- `when_get_artifact(world, id)` → GET request
- `then_status_code(world, 404)` → checks status

**Behaviors**: B-ART-010, B-ART-013

---

### Scenario 10.4: Artifact storage is idempotent
```gherkin
Given an artifacts endpoint
When I create the same artifact twice
Then both requests return the same id
```
**Steps**:
- `given_artifacts_endpoint(world)` → no-op
- `when_create_same_artifact_twice(world, doc)` → two POST requests
- `then_both_return_same_id(world)` → compares IDs

**Behaviors**: B-ART-022, B-ART-023

---

## Feature 11: Observability - Metrics

**File**: `tests/features/observability/metrics.feature`

### Scenario 11.1: Get Prometheus metrics
```gherkin
Given a metrics endpoint
When I request /metrics
Then I receive 200 OK
And Content-Type is text/plain
And the response includes TYPE headers
And the response includes pre-registered metrics
```
**Steps**:
- `given_metrics_endpoint(world)` → no-op
- `when_request_metrics(world)` → GET `/metrics`
- `then_status_code(world, 200)` → checks status
- `then_content_type(world, "text/plain")` → checks header
- `then_response_includes_type_headers(world)` → parses Prometheus text
- `then_response_includes_metrics(world, ["tasks_enqueued_total", ...])` → parses

**Behaviors**: B-OBS-001, B-OBS-002, B-OBS-003, B-OBS-007

---

### Scenario 11.2: Metrics include labels
```gherkin
Given tasks have been enqueued
When I request /metrics
Then tasks_enqueued_total includes labels (engine, pool_id, priority)
```
**Steps**:
- `given_tasks_enqueued(world, count)` → enqueues tasks
- `when_request_metrics(world)` → GET `/metrics`
- `then_metric_includes_labels(world, "tasks_enqueued_total", ["engine", "pool_id", "priority"])` → parses

**Behaviors**: B-OBS-006

---

## Feature 12: Background Services - Handoff Autobind

**File**: `tests/features/background/handoff_autobind.feature`

### Scenario 12.1: Autobind from handoff file
```gherkin
Given a handoff file exists at .runtime/engines/pool-gpu0.json
And the handoff contains url, pool_id, replica_id
When the handoff watcher processes the file
Then an adapter is bound to the pool
And the pool is registered as ready
And the pool is marked as bound
And a narration breadcrumb is emitted
```
**Steps**:
- `given_handoff_file_exists(world, filename, content)` → writes file
- `when_handoff_watcher_processes(world)` → triggers watcher (or waits)
- `then_adapter_bound(world, pool_id, replica_id)` → checks adapter_host
- `then_pool_registered_ready(world, pool_id)` → checks pool_manager
- `then_pool_marked_bound(world, pool_id)` → checks state.bound_pools
- `then_narration_emitted(world, "autobind")` → checks logs

**Behaviors**: B-BG-001, B-BG-003, B-BG-004, B-BG-006, B-BG-007, B-BG-008, B-BG-009, B-BG-010, B-BG-011

---

### Scenario 12.2: Skip already bound pool
```gherkin
Given a pool "gpu-0" is already bound
And a handoff file for "gpu-0" exists
When the handoff watcher processes the file
Then the pool is not re-bound
```
**Steps**:
- `given_pool_already_bound(world, pool_id)` → marks in state.bound_pools
- `given_handoff_file_exists(world, filename, content)` → writes file
- `when_handoff_watcher_processes(world)` → triggers watcher
- `then_pool_not_rebound(world, pool_id)` → checks adapter_host calls

**Behaviors**: B-BG-005

---

### Scenario 12.3: Watcher runs continuously
```gherkin
Given the handoff watcher is running
When I create a new handoff file
And I wait for the poll interval
Then the new handoff is processed
```
**Steps**:
- `given_handoff_watcher_running(world)` → starts watcher
- `when_create_handoff_file(world, filename, content)` → writes file
- `when_wait_for_poll_interval(world)` → sleeps
- `then_handoff_processed(world, filename)` → checks state

**Behaviors**: B-BG-001, B-BG-002, B-BG-012

---

## Summary Statistics

**Total Features**: 12  
**Total Scenarios**: 50+  
**Total Step Functions**: 100+  
**Total Behaviors Covered**: 200+

---

## Step Function Implementation Guide

### Common Step Patterns

#### Given Steps (Setup)
```rust
#[given(regex = r"^a pool \"(.+)\" exists$")]
async fn given_pool_exists(world: &mut World, pool_id: String) {
    let mut reg = world.state.pool_manager.lock().unwrap();
    reg.register_pool(&pool_id, /* ... */);
}
```

#### When Steps (Actions)
```rust
#[when(regex = r"^I request pool health for \"(.+)\"$")]
async fn when_request_pool_health(world: &mut World, pool_id: String) {
    let path = format!("/v2/pools/{}/health", pool_id);
    world.http_call(Method::GET, &path, None).await.unwrap();
}
```

#### Then Steps (Assertions)
```rust
#[then(regex = r"^I receive (\d+)")]
async fn then_status_code(world: &mut World, code: u16) {
    assert_eq!(world.last_status, Some(StatusCode::from_u16(code).unwrap()));
}
```

---

## Next Steps

1. **Create missing .feature files** for Features 8-12
2. **Implement missing step functions** in `src/steps/*.rs`
3. **Add behavior IDs as comments** in step functions
4. **Create traceability matrix**: Behavior → Step → Scenario → Feature
5. **Run BDD suite** and achieve 100% scenario pass rate
