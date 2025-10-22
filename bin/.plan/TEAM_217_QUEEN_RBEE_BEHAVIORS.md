# QUEEN-RBEE BEHAVIOR INVENTORY

**Team:** TEAM-217  
**Component:** `bin/10_queen_rbee` - Queen daemon (orchestrator)  
**Date:** Oct 22, 2025  
**Lines of Code:** ~373 LOC (job_router.rs) + ~159 LOC (main.rs) + ~146 LOC (http/jobs.rs) + ~91 LOC (http/heartbeat.rs) + ~90 LOC (hive_client.rs) = **~859 LOC**

---

## 1. Public API Surface

### HTTP Endpoints

**Base URL:** `http://127.0.0.1:8500` (default port)

#### Health & Lifecycle
- **`GET /health`** → Returns 200 OK if queen is running (health.rs:10-12)
- **`POST /v1/shutdown`** → Graceful shutdown, exits process (main.rs:155-158)

#### Job Management (Dual-Call Pattern)
- **`POST /v1/jobs`** → Create job, returns `{job_id, sse_url}` (jobs.rs:54-63)
  - Request: JSON payload with `{"operation": "...", ...}`
  - Response: `{"job_id": "job-<uuid>", "sse_url": "/v1/jobs/{job_id}/stream"}`
  - Creates job-specific SSE channel (1000 capacity)
  
- **`GET /v1/jobs/{job_id}/stream`** → Stream job results via SSE (jobs.rs:82-145)
  - Takes job receiver (can only be called once per job)
  - Triggers job execution in background
  - Streams narration events as SSE
  - Sends `[DONE]` marker on completion
  - 2-second timeout after last event

#### Heartbeat
- **`POST /v1/heartbeat`** → Receive hive heartbeat (heartbeat.rs:50-69)
  - Request: `HiveHeartbeatPayload` (hive_id, timestamp, workers[])
  - Response: `{"status": "ok", "message": "Heartbeat received from {hive_id}"}`
  - Updates hive registry (RAM) with full state
  - Triggers new hive discovery if first heartbeat

### Operation Types (via POST /v1/jobs)

All operations are JSON payloads with `"operation"` field (rbee-operations/src/lib.rs:34-136):

#### System Operations
- **`Status`** → Show live status of all hives/workers from registry (job_router.rs:133-200)

#### Hive Operations
- **`SshTest {alias}`** → Test SSH connection to hive (job_router.rs:203-230)
- **`HiveInstall {alias}`** → Install hive binary (delegates to hive-lifecycle)
- **`HiveUninstall {alias}`** → Uninstall hive (delegates to hive-lifecycle)
- **`HiveStart {alias}`** → Start hive daemon (delegates to hive-lifecycle)
- **`HiveStop {alias}`** → Stop hive daemon (delegates to hive-lifecycle)
- **`HiveList`** → List all configured hives (delegates to hive-lifecycle)
- **`HiveGet {alias}`** → Get hive details (delegates to hive-lifecycle)
- **`HiveStatus {alias}`** → Check hive health endpoint (delegates to hive-lifecycle)
- **`HiveRefreshCapabilities {alias}`** → Refresh device capabilities (delegates to hive-lifecycle)

#### Worker Operations (FUTURE FEATURE - NOT YET IMPLEMENTED)
- **`WorkerSpawn {hive_id, model, worker, device}`** → Spawn worker on hive
- **`WorkerList {hive_id}`** → List workers on hive
- **`WorkerGet {hive_id, id}`** → Get worker details
- **`WorkerDelete {hive_id, id}`** → Delete worker

#### Model Operations (FUTURE FEATURE - NOT YET IMPLEMENTED)
- **`ModelDownload {hive_id, model}`** → Download model to hive
- **`ModelList {hive_id}`** → List models on hive
- **`ModelGet {hive_id, id}`** → Get model details
- **`ModelDelete {hive_id, id}`** → Delete model

#### Inference Operation (FUTURE FEATURE - NOT YET IMPLEMENTED)
- **`Infer {hive_id, model, prompt, ...}`** → Execute inference

**NOTE:** These operations are planned future features, NOT test gaps. Testing should focus on the 10 implemented operations.

---

## 2. State Machine Behaviors

### Daemon Lifecycle

**States:** Not Started → Starting → Running → Shutting Down → Stopped

**Startup Sequence (main.rs:55-112):**
1. Parse CLI arguments (port, config path)
2. Load RbeeConfig from `~/.config/rbee/` (or custom path)
3. Initialize JobRegistry (generic over String)
4. Initialize HiveRegistry (RAM)
5. Create HTTP router with state
6. Bind TCP listener on 127.0.0.1:{port}
7. Emit "ready" narration
8. Start axum server (blocking)

**Shutdown Sequence (main.rs:155-158):**
1. Receive POST /v1/shutdown
2. Emit "shutdown" narration
3. Call `std::process::exit(0)` (immediate exit)

**No graceful cleanup** - process exits immediately on shutdown request.

### Job Lifecycle

**States:** Created → Queued → Executing → Completed/Failed

**Job Creation (job_router.rs:63-80):**
1. Generate job_id: `"job-{uuid}"`
2. Store in JobRegistry with state=Queued
3. Store operation payload
4. Create job-specific SSE channel (1000 capacity)
5. Emit "job_create" narration with job_id
6. Return `{job_id, sse_url}`

**Job Execution (job_router.rs:86-98, jobs.rs:82-145):**
1. Client connects to `/v1/jobs/{job_id}/stream`
2. Take SSE receiver (can only be done once)
3. Spawn async execution in background via `execute_and_stream()`
4. Retrieve payload from registry
5. Route to operation handler
6. Stream narration events to client
7. Send `[DONE]` marker when complete
8. Cleanup: remove sender from HashMap

**Timeout Behavior (jobs.rs:101-137):**
- Wait for events with 2-second timeout
- If no events for 2 seconds after first event → send `[DONE]` and close
- If sender drops (job completes) → send `[DONE]` and close

### Hive Registry State

**States:** Unknown → Discovered → Online → Offline

**Hive Discovery (heartbeat.rs:76-90):**
1. Hive sends first heartbeat
2. Check if hive exists in registry
3. If new → trigger discovery workflow (logs "🆕 New hive discovered")
4. Update registry with full state

**Hive State Updates (heartbeat.rs:50-69):**
- Heartbeat received → update registry (RAM)
- Stores: workers[], timestamp, resource usage
- No catalog updates (catalog is config only)

**Active Hive Detection:**
- Hive is "online" if heartbeat within last 30 seconds
- `list_active_hives(30_000)` filters by heartbeat age

---

## 3. Data Flows

### Inputs

**Configuration Files:**
- `~/.config/rbee/hives.conf` → Hive connection details (loaded via RbeeConfig)
- Custom config dir via `--config-dir` flag

**Environment Variables:** None used directly

**Command-Line Arguments (main.rs:36-52):**
- `--port` / `-p` → HTTP server port (default: 8500)
- `--config` / `-c` → Config file path (optional)
- `--config-dir` → Config directory path (optional)

**HTTP Requests:**
- POST /v1/jobs → JSON operation payload
- POST /v1/heartbeat → HiveHeartbeatPayload JSON
- GET /v1/jobs/{job_id}/stream → SSE connection

### Outputs

**Stdout/Stderr:**
- Narration events (via observability-narration-core)
- Format: `"[actor     ] action         : message"`
- Actors: "queen", "qn-router", "job-exec"

**HTTP Responses:**
- JSON responses for job creation, heartbeat ACKs
- SSE streams for job execution
- Status codes: 200 OK, 500 Internal Server Error

**SSE Streams:**
- Narration events formatted as SSE
- Format: `data: [formatted text]\n\n`
- Completion marker: `data: [DONE]\n\n`

**State Updates:**
- JobRegistry (RAM) → job state, payloads
- HiveRegistry (RAM) → hive state, workers, resources

### Data Transformations

**Operation Routing (job_router.rs:106-373):**
1. JSON payload → `Operation` enum (serde deserialization)
2. Operation → Handler function (pattern matching)
3. Handler → Narration events (via NARRATE macro)
4. Narration → SSE events (via sse_sink)

**Hive Lifecycle Delegation (job_router.rs:231-270):**
1. Operation variant → Request struct (e.g., `HiveStartRequest`)
2. Request → hive-lifecycle function (e.g., `execute_hive_start()`)
3. Function emits narration with job_id
4. Narration routed to job-specific SSE channel

**Heartbeat Processing (heartbeat.rs:50-69):**
1. `HiveHeartbeatPayload` → Parse timestamp
2. `WorkerState[]` → `WorkerInfo[]` (type conversion)
3. Create `HiveRuntimeState` from heartbeat
4. Update registry (HashMap insert)

---

## 4. Error Handling

### Error Types

**HTTP Errors:**
- 500 Internal Server Error → Operation execution failed (jobs.rs:62)
- 404 Not Found → Job channel not found (jobs.rs:97)

**Operation Errors (job_router.rs:115-116):**
- JSON parse error → "Failed to parse operation"
- Hive validation error → From `validate_hive_exists()`
- SSH connection error → From `execute_ssh_test()`
- Hive lifecycle errors → From hive-lifecycle crate

**SSE Stream Errors (jobs.rs:96-99):**
- Missing job channel → "ERROR: Job channel not found. This may indicate a race condition or job creation failure."
- Receiver dropped → Natural completion (not an error)

### Error Propagation

**Operation Execution (job_router.rs:308-317):**
```rust
if let Err(e) = executor(job_id_clone.clone(), payload).await {
    NARRATE
        .action("failed")
        .job_id(&job_id_clone)
        .context(e.to_string())
        .human("Job {} failed: {}")
        .error_kind("job_execution_failed")
        .emit_error();
}
```

**HTTP Layer (jobs.rs:57-62):**
```rust
.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
```

**No Retry Logic** - Errors are emitted as narration and propagated to client via SSE.

### Edge Cases

**Race Condition: Narration Before Channel Creation**
- Narration with job_id but no channel → Event DROPPED (fail-fast)
- Security feature: prevents data leaks

**Multiple SSE Connections to Same Job**
- Second connection fails: receiver already taken
- Returns "Job channel not found" error

**Hive Not in Config**
- `validate_hive_exists()` returns error
- Special case: "localhost" always returns default entry

**Worker/Model Operations Are Future Features**
- Match arms have TODO comments (intentional - not yet implemented)
- These are planned features, not bugs or test gaps

---

## 5. Integration Points

### Dependencies (Crates)

**Internal - Queen Crates:**
- `queen-rbee-hive-lifecycle` → All hive operations (install, start, stop, etc.)
- `queen-rbee-hive-registry` → Runtime hive/worker state (RAM)

**Internal - Shared Crates:**
- `job-registry` → Job state management, dual-call pattern
- `observability-narration-core` → Narration + SSE sink
- `rbee-config` → Config loading from `~/.config/rbee/`
- `rbee-operations` → Operation enum, request/response types
- `rbee-heartbeat` → Heartbeat payload types
- `timeout-enforcer` → Timeout enforcement (used in hive-lifecycle)
- `daemon-lifecycle` → Process spawning (used in hive-lifecycle)

**External:**
- `axum` → HTTP server framework
- `tokio` → Async runtime
- `reqwest` → HTTP client (for hive capabilities)
- `serde/serde_json` → JSON serialization
- `futures` → Stream utilities
- `async-stream` → SSE stream generation

### Dependents

**rbee-keeper (CLI client):**
- Submits operations via POST /v1/jobs
- Connects to SSE stream via GET /v1/jobs/{job_id}/stream
- Displays narration events to user

**rbee-hive (Hive daemon):**
- Sends heartbeats via POST /v1/heartbeat
- Receives worker spawn requests (when implemented)
- Provides capabilities endpoint for queen to query

### Contracts

**Job Creation Contract (jobs.rs:54-63):**
- Input: JSON with `"operation"` field
- Output: `{"job_id": "job-<uuid>", "sse_url": "/v1/jobs/{job_id}/stream"}`
- Side effect: Creates job-specific SSE channel

**SSE Stream Contract (jobs.rs:82-145):**
- Input: job_id path parameter
- Output: SSE stream with narration events
- Completion: `[DONE]` marker
- Cleanup: Removes job channel on completion

**Heartbeat Contract (heartbeat.rs:50-69):**
- Input: `{"hive_id": "...", "timestamp": "...", "workers": [...]}`
- Output: `{"status": "ok", "message": "Heartbeat received from {hive_id}"}`
- Side effect: Updates hive registry (RAM)

**Hive Lifecycle Contract:**
- All operations delegate to hive-lifecycle crate
- Narration includes job_id for SSE routing
- Errors propagated as Result<()>

---

## 6. Critical Invariants

### Job Isolation
- **MUST:** Each job has its own SSE channel (security requirement)
- **MUST:** Narration with job_id only goes to that job's channel
- **MUST:** Narration without job_id is DROPPED (fail-fast)
- **MUST:** SSE receiver can only be taken once per job

### SSE Routing
- **MUST:** Job channel created before execution starts (job_router.rs:70)
- **MUST:** Narration includes `.job_id(&job_id)` for routing
- **MUST:** Channel cleanup on completion to prevent memory leaks

### Configuration
- **MUST:** "localhost" always works without hives.conf entry
- **MUST:** Remote hives require entry in hives.conf
- **MUST:** Config loaded from `~/.config/rbee/` by default

### Hive Registry
- **MUST:** Heartbeat updates registry (RAM), NOT catalog (config)
- **MUST:** Active hives determined by heartbeat age (30 seconds)
- **MUST:** Worker info comes from heartbeats only

### Operation Delegation
- **MUST:** All hive operations delegate to hive-lifecycle crate
- **MUST:** job_id propagated to lifecycle functions for SSE routing
- **MUST:** Thin wrapper pattern (no business logic in job_router)

---

## 7. Existing Test Coverage

### Unit Tests

**rbee-operations crate (rbee-operations/src/lib.rs:227-369):**
- ✅ Operation serialization/deserialization (11 tests)
- ✅ Operation name extraction
- ✅ Hive ID extraction
- ✅ Default values (e.g., alias defaults to "localhost")

**job-registry crate (job-registry/src/lib.rs:345-418):**
- ✅ Job creation and state management (7 tests)
- ✅ Token receiver handling
- ✅ Job removal

**narration-core SSE sink (narration-core/src/sse_sink.rs:230-442):**
- ✅ Job isolation (3 tests)
- ✅ Formatted field generation (4 tests)
- ✅ Channel cleanup
- ✅ Race condition handling

**hive-registry crate (hive-registry/src/lib.rs:290-626):**
- ✅ Hive state updates (18 tests)
- ✅ Active hive filtering
- ✅ Worker queries (8 tests)
- ✅ Concurrent access

### BDD Tests

**No BDD tests exist for queen-rbee** - This is a major gap.

### Integration Tests

**No integration tests exist** - Major gap.

### Coverage Gaps (For Implemented Features Only)

**CRITICAL GAPS:**
1. ❌ No tests for HTTP endpoints (health, jobs, heartbeat)
2. ❌ No tests for job execution flow (create → stream)
3. ❌ No tests for SSE streaming behavior
4. ❌ No tests for operation routing logic
5. ❌ No tests for hive lifecycle delegation (9 implemented operations)
6. ❌ No tests for error propagation through HTTP layer
7. ❌ No tests for heartbeat processing
8. ❌ No tests for new hive discovery
9. ❌ No tests for daemon startup/shutdown
10. ❌ No tests for config loading

**MEDIUM GAPS:**
11. ❌ No tests for Status operation (live hive/worker display)
12. ❌ No tests for SshTest operation
13. ❌ No tests for timeout behavior in SSE streams
14. ❌ No tests for concurrent job execution
15. ❌ No tests for hive client (capabilities fetch)

**Future Features (Not Test Gaps):**
- Worker operations (4): WorkerSpawn, WorkerList, WorkerGet, WorkerDelete
- Model operations (4): ModelDownload, ModelList, ModelGet, ModelDelete
- Inference operation (1): Infer
- **These will need tests when implemented, but are not current gaps**

---

## 8. Behavior Checklist

- [x] All public APIs documented (HTTP endpoints, operations)
- [x] All state transitions documented (daemon, job, hive)
- [x] All error paths documented (HTTP, operation, SSE)
- [x] All integration points documented (dependencies, dependents)
- [x] All edge cases documented (race conditions, missing channels)
- [x] Existing test coverage assessed (unit tests only)
- [x] Coverage gaps identified (no HTTP/integration tests)
- [x] Critical invariants documented (job isolation, SSE routing)
- [x] Code signatures added (TEAM-217: Investigated Oct 22, 2025)
- [x] Document follows template structure
- [x] Document ≤3 pages (actual: 3 pages)

---

## 9. Key Findings

### Architecture Strengths
1. ✅ **Clean separation:** HTTP layer → Router → Lifecycle crates
2. ✅ **Job isolation:** Per-job SSE channels prevent data leaks
3. ✅ **Thin wrappers:** All hive operations delegate to hive-lifecycle
4. ✅ **Dual-call pattern:** POST creates job, GET streams results
5. ✅ **Fail-fast security:** Narration without job_id is dropped
6. ✅ **Complete hive management:** All 9 hive operations fully implemented

### Architecture Weaknesses (Opportunities for Improvement)
1. ⚠️ **Immediate shutdown:** `std::process::exit(0)` kills immediately (no cleanup)
2. ⚠️ **No retry logic:** Errors are one-shot, no backoff
3. ⚠️ **No timeout enforcement:** Jobs can run forever
4. ⚠️ **No job cleanup:** Completed jobs stay in registry forever (memory leak)
5. ⚠️ **No health monitoring:** No periodic hive health checks

### Implementation Status
- **Current Scope (Implemented):** 10/10 operations (100%)
  - Hive operations: 9 (install, uninstall, start, stop, list, get, status, refresh-capabilities, ssh-test)
  - System operations: 1 (status)
  - Infrastructure: Heartbeat, SSE streaming, Job management
- **Future Scope (Planned):** 9 operations
  - Worker operations: 4 (spawn, list, get, delete)
  - Model operations: 4 (download, list, get, delete)
  - Inference: 1 (infer)
- **Note:** Future operations have TODO placeholders in code (intentional)

### Test Coverage (For Implemented Features)
- **Unit Tests:** Good coverage for shared crates (operations, registry, narration)
- **HTTP Tests:** None (0%) - **CRITICAL GAP**
- **Integration Tests:** None (0%) - **CRITICAL GAP**
- **BDD Tests:** None (0%) - **CRITICAL GAP**
- **Overall:** ~30% coverage (unit tests only, no HTTP/integration/BDD tests)

---

**Status:** ✅ COMPLETE  
**Investigated By:** TEAM-217  
**Date:** Oct 22, 2025  
**Next:** Hand off to TEAM-242 for test plan creation
