# orchestratord Behaviors Catalog

**Purpose**: Complete catalog of ALL behaviors in orchestratord for BDD test coverage  
**Date**: 2025-09-30  
**Source**: Code flow analysis of all endpoints and services

---

## 📋 Table of Contents

1. [Middleware Behaviors](#middleware-behaviors)
2. [Control Plane Behaviors](#control-plane-behaviors)
3. [Data Plane Behaviors](#data-plane-behaviors)
4. [Catalog Behaviors](#catalog-behaviors)
5. [Session Behaviors](#session-behaviors)
6. [Artifact Behaviors](#artifact-behaviors)
7. [Streaming Behaviors](#streaming-behaviors)
8. [Observability Behaviors](#observability-behaviors)
9. [Background Service Behaviors](#background-service-behaviors)

---

## Middleware Behaviors

### Correlation ID Middleware
- **B-MW-001**: When request has `X-Correlation-Id` header → echo it in response
- **B-MW-002**: When request lacks `X-Correlation-Id` → generate UUIDv4 and add to response
- **B-MW-003**: Correlation ID attached to request extensions for handler access
- **B-MW-004**: All responses (including errors) include `X-Correlation-Id` header

### API Key Middleware
- **B-MW-010**: When `/metrics` endpoint → skip API key check
- **B-MW-011**: When no `X-API-Key` header → return 401 Unauthorized with correlation ID
- **B-MW-012**: When `X-API-Key` != "valid" → return 403 Forbidden with correlation ID
- **B-MW-013**: When `X-API-Key` == "valid" → allow request through
- **B-MW-014**: Early auth failures still include correlation ID in response

### Bearer Identity Middleware
- **B-MW-020**: Extracts Bearer token from Authorization header
- **B-MW-021**: Attaches identity to request extensions when present
- **B-MW-022**: Allows requests without Bearer token (optional auth)

---

## Control Plane Behaviors

### Capabilities Discovery (`GET /v2/meta/capabilities`)
- **B-CP-001**: First request → compute capabilities snapshot and cache
- **B-CP-002**: Subsequent requests → serve from cache
- **B-CP-003**: Response includes API version and engine metadata
- **B-CP-004**: Returns 200 OK with JSON body

### Pool Health (`GET /v2/pools/:id/health`)
- **B-CP-010**: Query pool-managerd registry for health status
- **B-CP-011**: When pool exists → return `{live, ready, draining, metrics, last_error}`
- **B-CP-012**: When pool doesn't exist → return default `{live: true, ready: true}`
- **B-CP-013**: Check draining state from `state.draining_pools`
- **B-CP-014**: Include queue_depth in metrics (currently hardcoded to 0)
- **B-CP-015**: Returns 200 OK with JSON body

### Pool Drain (`POST /v2/pools/:id/drain`)
- **B-CP-020**: Accept `DrainRequest` with `deadline_ms`
- **B-CP-021**: Mark pool as draining in `state.draining_pools`
- **B-CP-022**: Return 202 Accepted immediately (async operation)
- **B-CP-023**: Does NOT wait for in-flight tasks (stub implementation)
- **B-CP-024**: Does NOT prevent new admissions (stub implementation)

### Pool Reload (`POST /v2/pools/:id/reload`)
- **B-CP-030**: Accept `ReloadRequest` with `new_model_ref`
- **B-CP-031**: When `new_model_ref == "bad"` → return 409 Conflict (test sentinel)
- **B-CP-032**: When valid → update `model_state` gauge metric
- **B-CP-033**: Return 200 OK on success
- **B-CP-034**: Does NOT perform atomic swap (stub implementation)
- **B-CP-035**: Does NOT rollback on failure (stub implementation)

### Pool Purge (`POST /v2/pools/:id/purge`)
- **B-CP-040**: Accept untyped JSON body (pre-1.0 semantics)
- **B-CP-041**: Return 202 Accepted (stub, no actual purge)

### Worker Registration (`POST /v2/workers/register`)
- **B-CP-050**: Require `Authorization: Bearer <token>` header
- **B-CP-051**: When no Bearer token → return 401 with `{code: 40101, message: "MISSING_TOKEN"}`
- **B-CP-052**: When token doesn't match `AUTH_TOKEN` env → return 401 with `{code: 40102, message: "BAD_TOKEN"}`
- **B-CP-053**: Use timing-safe comparison for token validation
- **B-CP-054**: When valid → log identity breadcrumb `{identity: "token:<fp6>", event: "worker_register"}`
- **B-CP-055**: Extract `pool_id` from body (default: "default")
- **B-CP-056**: Extract `replica_id` from body (default: "r0")
- **B-CP-057**: When `mock-adapters` feature enabled → bind MockAdapter to pool
- **B-CP-058**: Return 200 OK with `{ok: true, identity, pool_id, replica_id}`

---

## Data Plane Behaviors

### Task Admission (`POST /v2/tasks`)

#### Validation Behaviors
- **B-DP-001**: When `ctx < 0` → return 400 with `INVALID_PARAMS` error
- **B-DP-002**: When `deadline_ms <= 0` → return 400 with `DEADLINE_UNMET` error
- **B-DP-003**: When `expected_tokens >= 2_000_000` → return 429 with `QUEUE_FULL_DROP_LRU` error
- **B-DP-004**: When `expected_tokens >= 1_000_000` → return 429 with `ADMISSION_REJECT` error
- **B-DP-005**: Test sentinels removed (was: `model_ref == "pool-unavailable"` → 503)
- **B-DP-006**: Test sentinels removed (was: `prompt == "cause-internal"` → 500)

#### Queue Admission Behaviors
- **B-DP-010**: Map `priority: "interactive"` → Interactive queue
- **B-DP-011**: Map `priority: "batch"` → Batch queue
- **B-DP-012**: Hash `task_id` to stable u32 for queue identity
- **B-DP-013**: When queue full + reject policy → return 429 with `ADMISSION_REJECT`
- **B-DP-014**: When queue full + drop-lru policy → drop oldest, enqueue new
- **B-DP-015**: On successful enqueue → return queue position (0-based)

#### Response Building Behaviors
- **B-DP-020**: Calculate `predicted_start_ms = queue_position * 100` (heuristic)
- **B-DP-021**: Build `streams` object with SSE URLs (`/v2/tasks/{id}/events`, `?verbose=true`)
- **B-DP-022**: Build `preparation` object (empty steps for now)
- **B-DP-023**: Store `AdmissionSnapshot` with queue info + original request
- **B-DP-024**: Return 202 Accepted with `AdmissionResponse`
- **B-DP-025**: Include budget headers: `X-Budget-Tokens-Remaining`, `X-Budget-Time-Remaining-Ms`, `X-Budget-Cost-Remaining`
- **B-DP-026**: Emit admission log with queue_position and predicted_start_ms
- **B-DP-027**: Emit narration breadcrumb for admission decision
- **B-DP-028**: Increment `tasks_enqueued_total` metric

#### Error Response Behaviors
- **B-DP-030**: All errors include correlation ID
- **B-DP-031**: 429 errors include `Retry-After` header (seconds)
- **B-DP-032**: 429 errors include `X-Backoff-Ms` header (milliseconds)
- **B-DP-033**: Error envelope includes `{code, message, engine, retriable, retry_after_ms, policy_label}`
- **B-DP-034**: Increment `tasks_rejected_total` metric on rejection
- **B-DP-035**: Increment `admission_backpressure_events_total` metric on queue full

### Task Streaming (`GET /v2/tasks/:id/events`)

#### Request Handling Behaviors
- **B-DP-100**: Extract correlation ID from request extensions
- **B-DP-101**: Add `X-Correlation-Id` to SSE response headers
- **B-DP-102**: Set `Content-Type: text/event-stream`
- **B-DP-103**: Include budget headers in SSE response
- **B-DP-104**: Return 200 OK with SSE stream

#### Dispatch Behaviors
- **B-DP-110**: Check if adapter bound for target pool
- **B-DP-111**: Query pool health: require `live=true AND ready=true AND slots_free > 0`
- **B-DP-112**: When pool not ready → fall back to deterministic SSE
- **B-DP-113**: Retrieve admission snapshot to build TaskRequest
- **B-DP-114**: Use `placement.pin_pool_id` if present, else "default"
- **B-DP-115**: When no admission snapshot → build synthetic request
- **B-DP-116**: Submit request to adapter via `adapter_host.submit()`
- **B-DP-117**: When adapter dispatch fails → fall back to deterministic SSE

#### SSE Event Generation Behaviors
- **B-DP-120**: Emit `event: started` with `{queue_position, predicted_start_ms}`
- **B-DP-121**: Emit `event: token` for each token with `{t: text, i: index}`
- **B-DP-122**: Emit `event: metrics` with `{queue_depth, ...}` (optional)
- **B-DP-123**: Emit `event: end` with `{tokens_out, decode_time_ms}`
- **B-DP-124**: Event ordering: `started → token* → [metrics*] → end`
- **B-DP-125**: When error occurs → emit `event: error` with `{code, message, engine}`
- **B-DP-126**: After error event → terminate stream (no further events)

#### Cancellation Behaviors
- **B-DP-130**: Check `state.cancellations` set for task_id
- **B-DP-131**: When cancelled → stop emitting tokens immediately
- **B-DP-132**: When cancelled → emit only one token event (deterministic path)
- **B-DP-133**: When cancelled → no metrics or end events after token

#### Persistence Behaviors
- **B-DP-140**: Persist SSE transcript as artifact
- **B-DP-141**: Transcript includes all events with type and data
- **B-DP-142**: Clear cancellation flag after stream completes
- **B-DP-143**: Clear admission snapshot after stream completes
- **B-DP-144**: Log tokens_out count with job_id, engine, pool_id
- **B-DP-145**: Increment `tasks_started_total` metric

#### Deterministic Fallback Behaviors
- **B-DP-150**: When no adapter bound → use deterministic SSE
- **B-DP-151**: Deterministic: emit started, one token ("Hello"), metrics, end
- **B-DP-152**: Deterministic: respect cancellation (stop after one token)
- **B-DP-153**: Deterministic: include queue_position and predicted_start_ms in started

### Task Cancellation (`POST /v2/tasks/:id/cancel`)
- **B-DP-200**: Add task_id to `state.cancellations` set
- **B-DP-201**: Log cancellation event with task_id
- **B-DP-202**: Emit narration breadcrumb for cancel
- **B-DP-203**: Increment `tasks_canceled_total` metric with labels (engine, pool, replica, reason)
- **B-DP-204**: Update `queue_depth` gauge to 0
- **B-DP-205**: Return 204 No Content
- **B-DP-206**: Cancellation is race-free (streaming checks set before emitting tokens)

---

## Session Behaviors

### Get Session (`GET /v2/sessions/:id`)
- **B-SS-001**: Get or create session via SessionService
- **B-SS-002**: Return session info: `{ttl_ms_remaining, turns, kv_bytes, kv_warmth, tokens_budget_remaining, time_budget_remaining_ms, cost_budget_remaining}`
- **B-SS-003**: Default TTL: 600,000ms (10 minutes)
- **B-SS-004**: Default turns: 0
- **B-SS-005**: Return 200 OK with JSON body

### Delete Session (`DELETE /v2/sessions/:id`)
- **B-SS-010**: Remove session from `state.sessions` map
- **B-SS-011**: Return 204 No Content
- **B-SS-012**: Idempotent (deleting non-existent session succeeds)

### Session Service Behaviors
- **B-SS-020**: `get_or_create()` → create if missing with default values
- **B-SS-021**: `tick()` → decrement TTL (not currently used)
- **B-SS-022**: `note_turn()` → increment turns counter (not currently used)
- **B-SS-023**: Sessions stored in-memory (not persisted)

---

## Catalog Behaviors

### Create Model (`POST /v2/catalog/models`)
- **B-CAT-001**: Extract `id` from request body
- **B-CAT-002**: When `id` empty → return 400 with `{error: "id required"}`
- **B-CAT-003**: Parse optional `digest` field (format: "algo:value")
- **B-CAT-004**: Create CatalogEntry with:
  - `id` from request
  - `local_path` = `~/.cache/models/{id}`
  - `lifecycle` = Active
  - `digest` if provided
  - `last_verified_ms` = None
- **B-CAT-005**: Persist entry to FsCatalog
- **B-CAT-006**: Return 201 Created with entry metadata
- **B-CAT-007**: Include optional fields: `source_url`, `manifests`, `signatures`, `sbom`, `trust_policy`

### Get Model (`GET /v2/catalog/models/:id`)
- **B-CAT-010**: Query FsCatalog for model by id
- **B-CAT-011**: When found → return 200 OK with `{id, digest}`
- **B-CAT-012**: When not found → return 404 with `{error: "not found"}`

### Verify Model (`POST /v2/catalog/models/:id/verify`)
- **B-CAT-020**: Query FsCatalog for model by id
- **B-CAT-021**: When found → update `last_verified_ms` to current timestamp
- **B-CAT-022**: When found → return 202 Accepted
- **B-CAT-023**: When not found → return 404
- **B-CAT-024**: Does NOT perform actual digest verification (stub)

### Set Model State (`POST /v2/catalog/models/:id/state`)
- **B-CAT-030**: Accept `{state: "Active"|"Retired", deadline_ms?: number}`
- **B-CAT-031**: Map "Active" → LifecycleState::Active
- **B-CAT-032**: Map "Retired" → LifecycleState::Retired
- **B-CAT-033**: Unknown state → default to Active
- **B-CAT-034**: Update catalog entry state
- **B-CAT-035**: Return 202 Accepted

### Delete Model (`DELETE /v2/catalog/models/:id`)
- **B-CAT-040**: Delete entry from FsCatalog
- **B-CAT-041**: When deleted → return 204 No Content
- **B-CAT-042**: When not found → return 404
- **B-CAT-043**: Attempts to delete local artifact file/directory

---

## Artifact Behaviors

### Create Artifact (`POST /v2/artifacts`)
- **B-ART-001**: Accept arbitrary JSON document
- **B-ART-002**: Persist via configured artifact store (CAS with SHA-256 ID)
- **B-ART-003**: Also store in `state.artifacts` in-memory map
- **B-ART-004**: Return 201 Created with `{id: "<sha256>"}`

### Get Artifact (`GET /v2/artifacts/:id`)
- **B-ART-010**: Query configured artifact store by ID
- **B-ART-011**: Fallback to `state.artifacts` in-memory map
- **B-ART-012**: When found → return 200 OK with document
- **B-ART-013**: When not found → return 404

### Artifact Store Behaviors
- **B-ART-020**: InMemStore: stores in HashMap (default)
- **B-ART-021**: FsStore: stores in `~/.cache/llama-orch/artifacts/{id}.json`
- **B-ART-022**: Content-addressed: ID = SHA-256 of JSON string
- **B-ART-023**: Idempotent: same content → same ID

---

## Streaming Behaviors

### Health-Gated Dispatch
- **B-STR-001**: Query pool-managerd for health status
- **B-STR-002**: Require `live=true AND ready=true` for dispatch
- **B-STR-003**: Require `slots_free > 0` for dispatch
- **B-STR-004**: When pool not ready → skip adapter dispatch
- **B-STR-005**: When pool ready → attempt adapter dispatch

### Adapter Integration
- **B-STR-010**: Query adapter-host for bound adapter
- **B-STR-011**: Build TaskRequest from admission snapshot
- **B-STR-012**: Submit to adapter via `adapter_host.submit(pool_id, request)`
- **B-STR-013**: Receive TokenStream from adapter
- **B-STR-014**: Convert adapter events to SSE format
- **B-STR-015**: When adapter fails → fall back to deterministic SSE

### SSE Encoding
- **B-STR-020**: Use BufWriter to minimize allocations
- **B-STR-021**: Format: `event: {type}\ndata: {json}\n\n`
- **B-STR-022**: Build entire SSE string in memory
- **B-STR-023**: Return as String (Axum converts to streaming response)

### Transcript Persistence
- **B-STR-030**: After stream completes → persist transcript as artifact
- **B-STR-031**: Transcript format: `{events: [{type, data}, ...]}`
- **B-STR-032**: Artifact ID is SHA-256 of transcript JSON
- **B-STR-033**: Transcript includes all events (started, token, metrics, end)

---

## Observability Behaviors

### Metrics Endpoint (`GET /metrics`)
- **B-OBS-001**: Return Prometheus text format
- **B-OBS-002**: Include TYPE headers for all metrics
- **B-OBS-003**: Pre-registered counters: `tasks_enqueued_total`, `tasks_started_total`, `tasks_canceled_total`, `tasks_rejected_total`, `tokens_in_total`, `tokens_out_total`, `admission_backpressure_events_total`, `catalog_verifications_total`
- **B-OBS-004**: Pre-registered gauges: `queue_depth`, `kv_cache_usage_ratio`, `gpu_utilization`, `vram_used_bytes`, `model_state`
- **B-OBS-005**: Pre-registered histograms: `latency_first_token_ms`, `latency_decode_ms`
- **B-OBS-006**: Metrics include labels (engine, pool_id, replica_id, priority, etc.)
- **B-OBS-007**: Return 200 OK with `Content-Type: text/plain; version=0.0.4`

### Metrics Collection
- **B-OBS-010**: `inc_counter()` → increment counter with labels
- **B-OBS-011**: `set_gauge()` → set gauge value with labels
- **B-OBS-012**: `observe_histogram()` → record histogram sample
- **B-OBS-013**: `observe_histogram_throttled()` → sample every N observations
- **B-OBS-014**: Thread-safe via Mutex-protected global state

### Logging & Narration
- **B-OBS-020**: Structured logs stored in `state.logs` Vec
- **B-OBS-021**: Log format: JSON strings with relevant fields
- **B-OBS-022**: Narration via `observability_narration_core::human()`
- **B-OBS-023**: Narration includes: component, event, subject, human-readable message
- **B-OBS-024**: Logs include: job_id, session_id, engine, pool_id, tokens_out, etc.

---

## Background Service Behaviors

### Handoff Autobind Watcher
- **B-BG-001**: Watch directory: `ORCHD_RUNTIME_DIR` (default: `.runtime/engines`)
- **B-BG-002**: Poll interval: `ORCHD_HANDOFF_WATCH_INTERVAL_MS` (default: 1000ms)
- **B-BG-003**: Scan for `*.json` files in watch directory
- **B-BG-004**: Parse handoff JSON: `{url, pool_id, replica_id, engine_version, device_mask, slots_total, slots_free}`
- **B-BG-005**: Skip if pool already bound (check `state.bound_pools`)
- **B-BG-006**: Create LlamaCppHttpAdapter with URL from handoff
- **B-BG-007**: Bind adapter to pool via `adapter_host.bind(pool_id, replica_id, adapter)`
- **B-BG-008**: Update pool-managerd registry via `register_ready_from_handoff()`
- **B-BG-009**: Mark pool as bound in `state.bound_pools`
- **B-BG-010**: Log info message with pool_id, replica_id, URL
- **B-BG-011**: Emit narration breadcrumb for autobind event
- **B-BG-012**: Continue watching indefinitely (background task)

### Placement Service
- **B-BG-020**: Placement cache with TTL (default: 60s)
- **B-BG-021**: `prefilter_route_default()` → always returns "default" pool (stub)
- **B-BG-022**: Cache key: `{model_ref, engine, workload}`
- **B-BG-023**: Cache hit → return cached decision
- **B-BG-024**: Cache miss → compute and store decision

---

## Error Handling Behaviors

### Error Status Code Mapping
- **B-ERR-001**: `InvalidParams` → 400 Bad Request
- **B-ERR-002**: `DeadlineUnmet` → 400 Bad Request
- **B-ERR-003**: `PoolUnavailable` → 503 Service Unavailable
- **B-ERR-004**: `Internal` → 500 Internal Server Error
- **B-ERR-005**: `AdmissionReject` → 429 Too Many Requests
- **B-ERR-006**: `QueueFullDropLru` → 429 Too Many Requests

### Error Envelope Construction
- **B-ERR-010**: All errors include `code` (ErrorKind enum)
- **B-ERR-011**: All errors include optional `message`
- **B-ERR-012**: All errors include optional `engine` (stub: "llamacpp")
- **B-ERR-013**: Retriable errors include `retriable: true`
- **B-ERR-014**: Retriable errors include `retry_after_ms` when applicable
- **B-ERR-015**: Admission errors include `policy_label`
- **B-ERR-016**: 429 errors set `Retry-After` header (seconds)
- **B-ERR-017**: 429 errors set `X-Backoff-Ms` header (milliseconds)

---

## Configuration Behaviors

### Environment Variables
- **B-CFG-001**: `ORCHD_ADMISSION_CAPACITY` → queue capacity (default: 16)
- **B-CFG-002**: `ORCHD_ADMISSION_POLICY` → "reject" or "drop-lru" (default: reject)
- **B-CFG-003**: `ORCHD_LLAMACPP_URL` → adapter URL (when `llamacpp-adapter` feature enabled)
- **B-CFG-004**: `ORCHD_LLAMACPP_POOL` → pool ID for adapter (default: "default")
- **B-CFG-005**: `ORCHD_LLAMACPP_REPLICA` → replica ID for adapter (default: "r0")
- **B-CFG-006**: `ORCHD_RUNTIME_DIR` → handoff watch directory (default: ".runtime/engines")
- **B-CFG-007**: `ORCHD_HANDOFF_WATCH_INTERVAL_MS` → poll interval (default: 1000)
- **B-CFG-008**: `AUTH_TOKEN` → expected Bearer token for worker registration
- **B-CFG-009**: `ORCHD_ADDR` → listen address (default: "127.0.0.1:8080")
- **B-CFG-010**: `ORCHD_PREFER_H2` → enable HTTP/2 preference (default: false)

### Feature Flags
- **B-CFG-020**: `llamacpp-adapter` → enable llamacpp adapter binding at startup
- **B-CFG-021**: `mock-adapters` → enable mock adapter binding in worker registration
- **B-CFG-022**: `metrics` → enable advanced observability/logging

---

## Summary Statistics

**Total Behaviors Cataloged**: 200+

**By Category**:
- Middleware: 8 behaviors
- Control Plane: 24 behaviors
- Data Plane: 56 behaviors
- Sessions: 9 behaviors
- Catalog: 17 behaviors
- Artifacts: 9 behaviors
- Streaming: 18 behaviors
- Observability: 15 behaviors
- Background Services: 16 behaviors
- Error Handling: 17 behaviors
- Configuration: 13 behaviors

**Coverage Gaps** (behaviors not tested in BDD):
- B-DP-005, B-DP-006 (removed sentinels)
- B-DP-114 (pin override routing)
- B-DP-122 (metrics SSE event with on_time_probability)
- B-BG-001 through B-BG-012 (handoff autobind watcher)
- B-CFG-010 (HTTP/2 preference)
- B-STR-015 (adapter fallback)

---

## Next Steps

1. **Add Missing BDD Scenarios** for uncovered behaviors
2. **Update Existing Scenarios** to match removed sentinels
3. **Add Integration Tests** for background services
4. **Document Behavior Dependencies** (which behaviors depend on others)
5. **Create Behavior Traceability Matrix** (behavior → test → requirement)
