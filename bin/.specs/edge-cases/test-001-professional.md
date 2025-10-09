# Test-001: Cross-Node Inference Request Flow

## Test Scenario Overview

**Topology:**
- **blep** (`blep.home.arpa`): Control plane node running `rbee-keeper`, `queen-rbee`, and `rbee-hive`. Can run workers on CPU.
- **workstation** (`workstation.home.arpa`): Compute node running `rbee-hive` and `llm-worker-rbee`. Can run workers on CUDA devices 0, 1, and CPU.
- **mac** (`mac.home.arpa`): Compute node running `rbee-hive` and `llm-worker-rbee`. Can run workers on Metal backend only.

**Test Objective:**  
From `blep`, initiate an inference request targeting the `mac` node using a user-friendly CLI command.

**Inference Parameters:**
- Model: `hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`
- Prompt: `"write a short story"`
- Max tokens: `20`
- Temperature: `0.7`
- Backend: `metal`
- Device: `0`

---

## Expected Process Flow

### Phase 1: Worker Registry Check (rbee-keeper)

**Step 1.1:** `rbee-keeper` queries local worker registry (SQLite).

**Decision Point:**
- **IF** a worker on `mac` with the requested model (`TinyLlama-1.1B-Chat-v1.0-GGUF`) is:
  - Registered as healthy
  - Not currently executing inference
  - Model loaded and ready
- **THEN** skip to Phase 4 (direct inference execution)
- **ELSE** proceed to Phase 2 (pool preflight)

---

### Phase 2: Pool Preflight (rbee-keeper → rbee-hive on mac)

**Step 2.1:** `rbee-keeper` initiates preflight check with `rbee-hive` on `mac`.

**Step 2.2:** Verify `rbee-hive` version and health:
- Check connectivity to `mac.home.arpa`
  - **IF** connection fails → **ABORT** with error message
- Check installed `rbee-hive` version against latest
  - **IF** version mismatch → trigger update process
  - **IF** update fails → **ABORT** with error message
- Perform additional preflight checks (TBD)

---

### Phase 3: Model Provisioning (rbee-hive on mac)

**Step 3.1:** `rbee-hive` queries local model catalog for `hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`.

**Decision Point:**
- **IF** model found in catalog → skip to Step 3.5
- **ELSE** proceed to Step 3.2

**Step 3.2:** `rbee-hive` delegates model download to model provisioner.

**Step 3.3:** Model provisioner downloads model from Hugging Face:
- Stream download progress (loading bar) to `blep` stdout via SSE or similar
- Handle download errors (network failure, disk space, etc.)

**Step 3.4:** Model provisioner notifies `rbee-hive` of successful download with local path.

**Step 3.5:** `rbee-hive` registers model in local catalog with filesystem path.

---

### Phase 4: Worker Preflight (rbee-hive on mac)

**Step 4.1:** `rbee-hive` performs resource availability checks:
- Check available RAM against model requirements
  - **IF** insufficient RAM → **ABORT** with error message
- Check Metal backend availability
- Perform additional preflight checks (TBD)

---

### Phase 5: Worker Startup (rbee-hive on mac)

**Step 5.1:** `rbee-hive` spawns `llm-worker-rbee` process with:
- Backend: Metal
- Device: 0
- Model path: (from catalog)

**Step 5.2:** `llm-worker-rbee` initializes:
- HTTP server starts and binds to port
- Worker notifies `rbee-hive` that HTTP server is ready
- Model loading begins (asynchronous, may take time)

**Step 5.3:** `rbee-hive` returns worker details to `rbee-keeper`:
- Worker URL (e.g., `http://mac.home.arpa:<port>`)
- Worker ID
- Status: `initializing`

**Step 5.4:** `rbee-hive` process lifecycle:
- **QUESTION:** Does `rbee-hive` terminate after worker handoff, or does it remain running to manage worker lifecycle?

---

### Phase 6: Worker Registration (rbee-keeper)

**Step 6.1:** `rbee-keeper` updates local worker registry:
- Add/update worker entry with URL, ID, model, backend, device
- Set `last_seen_alive` timestamp
- Set status: `initializing`

---

### Phase 7: Worker Health Check (rbee-keeper → llm-worker-rbee)

**Step 7.1:** `rbee-keeper` polls worker health endpoint.

**Response Handling:**
- **IF** status = `loading`:
  - Return SSE URL for model loading progress
  - Stream loading bar to stdout
  - Continue polling until status = `ready`
- **IF** status = `ready`:
  - Return `204 No Content`
  - Proceed to Phase 8
- **IF** status = `error`:
  - **ABORT** with error details

---

### Phase 8: Inference Execution (rbee-keeper → llm-worker-rbee)

**Step 8.1:** `rbee-keeper` sends inference request to worker:
- Endpoint: `POST /inference` (or similar)
- Payload: `{ "prompt": "write a short story", "max_tokens": 20, "temperature": 0.7 }`

**Step 8.2:** `llm-worker-rbee` executes inference:
- Stream generated tokens via SSE to `rbee-keeper`
- Worker remains alive after completion

**Step 8.3:** `rbee-keeper` streams tokens to stdout in real-time.

**Step 8.4:** Inference completes:
- Worker returns final status/metadata
- Worker transitions to `idle` state (ready for next request)

---

## Additional Gaps Identified from Reference Implementations

### G12: Request Cancellation and Context Management

**Missing:** No specification for handling request cancellation, timeouts, or client disconnections.

**Ollama's Approach:**
- Uses Go's `context.Context` throughout request lifecycle
- Detects client disconnection via `c.Request.Context().Err()`
- Returns HTTP 499 for canceled requests
- Gracefully handles context cancellation during model loading

**llama.cpp's Approach:**
- Tracks connection state via `req.is_connection_closed` callback
- Stops streaming immediately when client disconnects
- Supports explicit task cancellation via `SERVER_TASK_TYPE_CANCEL`

**Required for rbee:**
```yaml
# Worker API
POST /inference
{
  "prompt": "...",
  "timeout_seconds": 300  # Optional request timeout
}

# Client disconnection handling
- Worker detects SSE stream closure
- Stops token generation immediately
- Releases slot/resources
- Logs cancellation event

# Explicit cancellation
DELETE /inference/{request_id}
Response: 204 No Content
```

**Pool manager responsibilities:**
- Track active request IDs
- Forward cancellation to worker
- Update worker state (busy → idle)
- Reset idle timer

---

### G13: Request Queue Management and Backpressure

**Missing:** No specification for request queuing, queue limits, or backpressure handling.

**Ollama's Approach:**
- Configurable queue size via `OLLAMA_MAX_QUEUE` (default: 512)
- Returns `503 Service Unavailable` with `ErrMaxQueue` when queue full
- Requests wait in queue until runner available
- Queue is per-scheduler (global across all models)

**llama.cpp's Approach:**
- Queue is per-server instance
- Deferred task queue for requests that can't be scheduled immediately
- Metrics expose: `n_idle_slots`, `n_processing_slots`, `n_tasks_deferred`

**Required for rbee:**

**rbee-keeper (global queue):**
```yaml
ctl:
  max_pending_requests: 100
  queue_timeout_seconds: 300
```

**Behavior:**
```
Request arrives at rbee-keeper:
  IF queue.size < max_pending_requests:
    - Add to queue
    - Return 202 Accepted with request_id
  ELSE:
    - Return 503 Service Unavailable
    - Error: "Queue full, try again later"

Queue processing:
  - FIFO by default
  - Optional priority field for future extension
```

**Worker-level slots:**
```yaml
worker:
  max_concurrent_requests: 1  # Most models support 1, some support parallel
```

**Metrics to expose:**
```
rbees_ctl_queue_size
rbees_ctl_queue_capacity
rbees_worker_slots_total
rbees_worker_slots_busy
```

---

### G14: Progress Streaming for Long Operations

**Missing:** No specification for streaming progress during model download and loading.

**Ollama's Approach:**
- Downloads in parallel parts (16 parts, 100MB-1GB each)
- Streams progress via SSE: `{"status": "downloading", "completed": 1234, "total": 5678}`
- Exponential backoff retry (max 6 retries)
- Stall detection (part hasn't updated in timeout period)

**llama.cpp's Approach:**
- Simple progress percentage during model load
- No parallel downloads

**Required for rbee:**

**Model download progress (pool → ctl):**
```
SSE stream from pool manager:
{
  "stage": "downloading",
  "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "bytes_downloaded": 1234567890,
  "bytes_total": 2345678901,
  "download_speed_mbps": 45.2,
  "eta_seconds": 23
}
```

**Model loading progress (worker → ctl via pool):**
```
{
  "stage": "loading_to_vram",
  "model_ref": "...",
  "layers_loaded": 12,
  "layers_total": 32,
  "vram_allocated_mb": 2048,
  "eta_seconds": 5
}
```

**Implementation:**
- Pool manager exposes SSE endpoint: `GET /models/{model_id}/download/progress`
- Worker exposes SSE endpoint: `GET /loading/progress`
- rbee-keeper aggregates and streams to user's stdout

---

### G15: Health Check and Readiness Probes

**Missing:** No specification for health vs. readiness distinction.

**llama.cpp's Approach:**
- `/health` endpoint (public, no auth)
- Returns server state: `LOADING`, `READY`, `ERROR`
- Includes basic metrics

**Kubernetes-style distinction:**

**Liveness probe** (is process alive?):
```
GET /health
Response: 200 OK (always, unless process dead)
```

**Readiness probe** (can it serve traffic?):
```
GET /ready
Response:
  200 OK - ready to accept inference requests
  503 Service Unavailable - loading model, not ready
```

**Required for rbee:**

**Worker endpoints:**
```
GET /health
{
  "status": "alive",
  "uptime_seconds": 1234
}

GET /ready
{
  "ready": true,
  "model_loaded": true,
  "state": "idle"  // idle, busy, loading
}
```

**Pool manager endpoints:**
```
GET /health
{
  "status": "alive",
  "workers_managed": 3
}

GET /ready
{
  "ready": true,
  "workers_available": 2
}
```

---

### G16: Metrics and Observability

**Missing:** No specification for metrics endpoints and format.

**llama.cpp's Approach:**
```
GET /metrics
{
  "idle": 2,
  "processing": 1,
  "deferred": 0,
  "n_prompt_tokens_processed_total": 12345,
  "n_tokens_predicted_total": 67890,
  "slots": [...]
}
```

**Required for rbee (Prometheus format):**

**Worker metrics:**
```
# HELP rbees_worker_inference_requests_total Total inference requests
# TYPE rbees_worker_inference_requests_total counter
rbees_worker_inference_requests_total{model="tinyllama",backend="cuda",device="0"} 42

# HELP rbees_worker_tokens_generated_total Total tokens generated
# TYPE rbees_worker_tokens_generated_total counter
rbees_worker_tokens_generated_total{model="tinyllama"} 123456

# HELP rbees_worker_inference_duration_seconds Inference duration histogram
# TYPE rbees_worker_inference_duration_seconds histogram
rbees_worker_inference_duration_seconds_bucket{le="1.0"} 10
rbees_worker_inference_duration_seconds_bucket{le="5.0"} 35

# HELP rbees_worker_vram_allocated_bytes VRAM currently allocated
# TYPE rbees_worker_vram_allocated_bytes gauge
rbees_worker_vram_allocated_bytes{device="0"} 4294967296

# HELP rbees_worker_state Worker state (0=starting, 1=ready, 2=busy, 3=idle)
# TYPE rbees_worker_state gauge
rbees_worker_state 3
```

**Pool manager metrics:**
```
# HELP rbees_pool_workers_total Total workers by state
# TYPE rbees_pool_workers_total gauge
rbees_pool_workers_total{state="idle"} 2
rbees_pool_workers_total{state="busy"} 1

# HELP rbees_pool_vram_total_bytes Total VRAM per device
# TYPE rbees_pool_vram_total_bytes gauge
rbees_pool_vram_total_bytes{device="0"} 8589934592

# HELP rbees_pool_vram_allocated_bytes Allocated VRAM per device
# TYPE rbees_pool_vram_allocated_bytes gauge
rbees_pool_vram_allocated_bytes{device="0"} 4294967296
```

**rbee-keeper metrics:**
```
# HELP rbees_ctl_queue_size Current queue size
# TYPE rbees_ctl_queue_size gauge
rbees_ctl_queue_size 5

# HELP rbees_ctl_requests_total Total requests by status
# TYPE rbees_ctl_requests_total counter
rbees_ctl_requests_total{status="success"} 100
rbees_ctl_requests_total{status="error"} 5
rbees_ctl_requests_total{status="queue_full"} 2
```

---

### G17: Error Taxonomy and HTTP Status Codes

**Missing:** No specification for error types and appropriate HTTP status codes.

**Ollama's Approach:**
```
400 Bad Request - Invalid input (missing required fields, bad template)
404 Not Found - Model not found
499 Client Closed Request - Context canceled
503 Service Unavailable - Queue full (ErrMaxQueue)
500 Internal Server Error - Unexpected errors
```

**llama.cpp's Approach:**
```
ERROR_TYPE_INVALID_REQUEST
ERROR_TYPE_NOT_SUPPORTED
ERROR_TYPE_NOT_FOUND
ERROR_TYPE_SERVER
ERROR_TYPE_PERMISSION
ERROR_TYPE_UNAVAILABLE
```

**Required for rbee:**

**Error response format:**
```json
{
  "error": {
    "code": "QUEUE_FULL",
    "message": "Request queue is full, try again later",
    "details": {
      "queue_size": 100,
      "queue_capacity": 100
    }
  }
}
```

**Error codes:**
```
# Client errors (4xx)
INVALID_REQUEST (400) - Malformed request
MODEL_NOT_FOUND (404) - Model doesn't exist
WORKER_NOT_READY (503) - Worker still loading
QUEUE_FULL (503) - Too many pending requests
REQUEST_TIMEOUT (408) - Request exceeded timeout
REQUEST_CANCELED (499) - Client disconnected

# Server errors (5xx)
WORKER_FAILED (500) - Worker crashed during inference
VRAM_EXHAUSTED (507) - Insufficient VRAM
INTERNAL_ERROR (500) - Unexpected error
```

---

### G18: Retry Logic and Exponential Backoff

**Missing:** No specification for retry behavior on transient failures.

**Ollama's Approach (download.go):**
```go
const maxRetries = 6

func newBackoff(maxBackoff time.Duration) func(ctx context.Context) error {
    var n int
    return func(ctx context.Context) error {
        n++
        // n^2 backoff: smoother than 2^n
        d := min(time.Duration(n*n)*10*time.Millisecond, maxBackoff)
        // Randomize delay between 0.5-1.5x to avoid thundering herd
        jitter := time.Duration(rand.Float64() + 0.5)
        time.Sleep(d * jitter)
        return nil
    }
}
```

**Required for rbee:**

**rbee-keeper retry policy (for pool/worker communication):**
```yaml
retry:
  max_attempts: 3
  initial_delay_ms: 100
  max_delay_ms: 5000
  backoff_multiplier: 2.0
  jitter: true
  
retryable_errors:
  - WORKER_NOT_READY
  - CONNECTION_TIMEOUT
  - TEMPORARY_UNAVAILABLE
  
non_retryable_errors:
  - INVALID_REQUEST
  - MODEL_NOT_FOUND
  - QUEUE_FULL
```

**Implementation:**
```python
def retry_with_backoff(func, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return func()
        except RetryableError as e:
            if attempt == max_attempts - 1:
                raise
            delay = min(100 * (2 ** attempt), 5000) / 1000.0
            jitter = random.uniform(0.5, 1.5)
            time.sleep(delay * jitter)
```

---

### G19: Model Alias and Version Management

**Missing:** No specification for model aliasing or version pinning.

**Ollama's Approach:**
- Supports model tags: `llama2:7b`, `llama2:13b`, `llama2:latest`
- Digest-based versioning for reproducibility
- Model name parsing with validation

**Required for rbee:**

**Model reference format:**
```
# Hugging Face
hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M
   └─ provider:org/repo:variant

# Local file
file:///models/llama2-7b-q4.gguf

# With version pin (future)
hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF@sha256:abc123...
```

**Model catalog schema:**
```sql
CREATE TABLE models (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,  -- hf, file, http
  reference TEXT NOT NULL,  -- org/repo or file path
  variant TEXT,  -- Q4_K_M, Q8_0, etc.
  digest TEXT,  -- sha256 hash for verification
  size_bytes INTEGER,
  downloaded_at_unix INTEGER,
  last_used_unix INTEGER,
  local_path TEXT NOT NULL
);
```

---

### G20: Graceful Shutdown and Cleanup

**Missing:** No specification for graceful shutdown of components.

**Ollama's Approach:**
- Signal handling (SIGINT, SIGTERM)
- Waits for in-flight requests to complete
- Unloads all runners
- Closes database connections

**Required for rbee:**

**Worker shutdown sequence:**
```
1. Receive SIGTERM or POST /admin/shutdown
2. Set state to "draining"
3. Reject new inference requests (503)
4. Wait for active requests to complete (max 30s timeout)
5. Unload model from VRAM
6. Close HTTP server
7. Exit with code 0
```

**Pool manager shutdown sequence:**
```
1. Receive SIGTERM
2. Stop accepting new worker registration
3. Send shutdown signal to all managed workers
4. Wait for workers to terminate (max 60s)
5. Force-kill any remaining workers
6. Close database connections
7. Exit with code 0
```

**rbee-keeper shutdown:**
```
1. Receive SIGINT (Ctrl+C)
2. Cancel active request context
3. Send cancellation to worker
4. Wait for graceful response or timeout (5s)
5. Print cancellation message to stderr
6. Exit with code 130 (128 + SIGINT)
```

---

## Identified Contradictions

### C1: Pool Manager Lifecycle
**Line 35:** "pool manager dies, worker lives"  
**Line 36:** "ctl adds the worker details is last seen alive in the worker registry"

**Contradiction:** If the pool manager dies immediately after starting the worker, how does it manage worker lifecycle (health checks, shutdown, etc.)? Who is responsible for:
- Periodic worker health monitoring?
- Worker cleanup on failure?
- Worker shutdown on idle timeout?

**Resolution Needed:** Clarify whether `rbee-hive` is:
- A short-lived orchestrator that hands off to `rbee-keeper` for lifecycle management, OR
- A persistent daemon that continues managing local workers

---

### C2: Worker Health Check Responsibility
**Lines 37-40:** `rbee-keeper` performs worker preflight health checks  
**Lines 28-31:** `rbee-hive` performs worker preflight resource checks

**Overlap:** Both `rbee-keeper` and `rbee-hive` perform preflight checks. This creates ambiguity:
- Who is authoritative for resource availability?
- What happens if `rbee-hive` says "OK" but `rbee-keeper` health check fails?

**Resolution Needed:** Define clear separation of concerns:
- `rbee-hive`: Local resource checks (RAM, GPU, disk)
- `rbee-keeper`: Network reachability and worker API health

---

### C3: Model Loading State Handling
**Line 33:** "http server is loaded says the worker to the pool (but model is still loading to ram)"  
**Lines 38-40:** Health check returns loading status with SSE URL

**Ambiguity:** The flow suggests the worker notifies `rbee-hive` that HTTP is ready, but then `rbee-keeper` must poll to discover loading status. This creates a race condition:
- What if `rbee-keeper` polls before the worker transitions from `initializing` to `loading`?
- Should `rbee-hive` wait for model loading to complete before returning worker details?

**Resolution Needed:** Define explicit state machine for worker lifecycle:
- `starting` → `http_ready` → `loading_model` → `ready` → `busy` → `idle`

---

## Identified Gaps

### G1: Error Recovery and Retry Logic
**Missing:** No specification for:
- Retry behavior on transient failures (network, timeout)
- Exponential backoff strategy
- Maximum retry attempts
- User notification of retry attempts

---

### G2: Concurrent Request Handling
**Missing:** No specification for:
- What happens if multiple inference requests target the same worker?
- Request queuing mechanism
- Worker capacity/slot management
- Load balancing across multiple workers with the same model

**Ollama's Approach:**
- Configurable via `OLLAMA_NUM_PARALLEL` (default: 1)
- Single worker can handle N parallel requests if model supports it
- Embedding models forced to `parallel=1`
- Some models (e.g., mllama) don't support parallel due to encoder cache limitations

**llama.cpp's Approach:**
- `--parallel N` or `-np N` flag (default: 1)
- Creates N slots per model
- Each slot has independent KV cache
- Context size divided by parallel count: `n_ctx_slot = n_ctx / n_parallel`
- Batch processing: `max(n_batch, n_parallel)` tokens per step

**Required for rbee:**

**Worker configuration:**
```yaml
worker:
  max_parallel_requests: 1  # Start with 1, configurable per model
  slot_ctx_size: 2048  # Per-slot context (total_ctx / max_parallel)
```

**Slot-based architecture:**
```rust
struct WorkerSlot {
    id: usize,
    state: SlotState,  // idle, processing_prompt, generating
    request_id: Option<String>,
    kv_cache: KVCache,
    ctx_size: usize,
    tokens_processed: usize,
}

enum SlotState {
    Idle,
    ProcessingPrompt,
    Generating,
}
```

**Request routing:**
```
Request arrives at worker:
  1. Find idle slot
     IF no idle slots:
       - Return 503 Service Unavailable
       - Error: "All slots busy, try again later"
  2. Assign request to slot
  3. Process in parallel with other slots
  4. Release slot when done
```

**Metrics:**
```
rbees_worker_slots_total{worker_id="abc"} 4
rbees_worker_slots_idle{worker_id="abc"} 2
rbees_worker_slots_busy{worker_id="abc"} 2
```

**Model compatibility:**
```yaml
# Model catalog includes parallel support flag
models:
  - id: "tinyllama-q4"
    max_parallel: 4  # Supports up to 4 parallel requests
  - id: "mllama-vision"
    max_parallel: 1  # Vision models typically single-threaded
```

**Load balancing (multiple workers):**
```
rbee-keeper has multiple workers with same model:
  1. Query each worker's available slots
  2. Select worker with most idle slots
  3. If all workers full, queue or return 503
  4. Track per-worker load in registry
```

---

### G3: Worker Shutdown and Cleanup
**Missing:** No specification for:
- When/how workers are shut down (idle timeout, explicit command, resource pressure)
- Cleanup of model files on disk
- Handling of in-flight requests during shutdown

**Note:** This gap is fully covered by **G20: Graceful Shutdown and Cleanup** and **G11: VRAM Resource Contention** (idle timeout).

**Summary of resolution:**
- Workers shut down after idle timeout (default: 5 minutes)
- Graceful shutdown waits for in-flight requests (max 30s)
- Model files remain on disk (cached for future use)
- Pool manager triggers shutdown via `POST /admin/shutdown`

---

### G4: Authentication and Authorization
**Missing:** No specification for:
- How `rbee-keeper` authenticates to `rbee-hive` and `llm-worker-rbee`
- Authorization model (who can request inference on which nodes)
- API key management or mTLS

**Ollama's Approach:**
- SSH key-based authentication for remote model registry
- Uses `~/.ollama/id_ed25519` private key
- Signs requests with SSH signature: `<pubkey>:<signature>`
- No authentication for local server (trusts localhost)

**llama.cpp's Approach:**
- Optional API key authentication via `--api-key` flag
- Supports multiple API keys
- Bearer token in `Authorization` header
- Public endpoints exempt: `/health`, `/v1/health`, `/models`, `/api/tags`
- Returns 401 with `ERROR_TYPE_AUTHENTICATION` on failure
- CORS-aware: OPTIONS requests skip auth (browser preflight)

**Required for rbee (MVP: Simple API Key):**

**Configuration:**
```yaml
# Pool manager config
pool:
  api_key: "rbees_pool_secret_abc123"  # Optional, if not set = no auth
  
# Worker config  
worker:
  api_key: "rbees_worker_secret_xyz789"  # Optional
  
# CTL config
ctl:
  pools:
    - hostname: "mac.home.arpa"
      api_key: "rbees_pool_secret_abc123"
  workers:
    - hostname: "workstation.home.arpa"
      api_key: "rbees_worker_secret_xyz789"
```

**Authentication flow:**
```
rbee-keeper → rbee-hive:
  Request:
    POST /workers HTTP/1.1
    Authorization: Bearer rbees_pool_secret_abc123
    
  Response (success):
    200 OK
    
  Response (failure):
    401 Unauthorized
    {
      "error": {
        "code": "INVALID_API_KEY",
        "message": "Invalid or missing API key"
      }
    }
```

**Public endpoints (no auth required):**
```
GET /health
GET /ready
GET /metrics  # Debatable - may want to protect in production
```

**Protected endpoints:**
```
# Pool manager
POST /workers
DELETE /workers/{id}
POST /models/download
POST /admin/shutdown

# Worker
POST /inference
DELETE /inference/{id}
POST /admin/shutdown
```

**Security considerations:**
```yaml
# For homelab (MVP):
- API keys in config files (chmod 600)
- HTTPS optional (local network)
- No user-level auth (single user)

# For production (post-MVP):
- mTLS with client certificates
- JWT tokens with expiration
- Role-based access control (RBAC)
- Secrets management (Vault, etc.)
```

**Implementation:**
```rust
// Middleware for API key validation
fn validate_api_key(req: &Request, configured_key: &Option<String>) -> Result<(), AuthError> {
    let configured_key = match configured_key {
        Some(key) => key,
        None => return Ok(()), // No auth configured
    };
    
    let auth_header = req.headers()
        .get("Authorization")
        .ok_or(AuthError::MissingHeader)?
        .to_str()
        .map_err(|_| AuthError::InvalidHeader)?;
    
    if !auth_header.starts_with("Bearer ") {
        return Err(AuthError::InvalidScheme);
    }
    
    let provided_key = &auth_header[7..]; // Skip "Bearer "
    
    if provided_key != configured_key {
        return Err(AuthError::InvalidKey);
    }
    
    Ok(())
}
```

**Deferred to post-MVP:**
- SSH key-based auth (Ollama-style)
- Multi-user support
- Fine-grained permissions (per-model, per-node)
- Token rotation
- Audit logging of auth events

---

### G5: Observability and Logging
**Missing:** No specification for:
- Structured logging format
- Metrics emission (request latency, token throughput, error rates)
- Distributed tracing across components
- Audit trail for inference requests

**Note:** This gap is fully covered by **G16: Metrics and Observability**.

**Summary of resolution:**
- Structured JSON logging with log levels
- Prometheus metrics endpoints on all components
- Request ID propagation for distributed tracing
- Audit logs for all inference requests

---

### G6: Model Catalog Synchronization
**Missing:** No specification for:
- How model catalog is synchronized across nodes
- Handling of model version updates
- Model eviction policy (LRU, manual, etc.)

**Ollama's Approach:**
- Each node has independent model storage (`~/.ollama/models`)
- No automatic synchronization between nodes
- Models pulled on-demand per node
- Digest-based versioning ensures consistency
- No automatic eviction (manual `ollama rm`)

**llama.cpp's Approach:**
- Single model per server instance
- No catalog concept (stateless server)
- Model specified at startup via `--model` flag

**Required for rbee (MVP: Independent Catalogs):**

**Per-node model catalog:**
```sql
-- Each pool manager maintains local catalog
CREATE TABLE models (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  reference TEXT NOT NULL,
  variant TEXT,
  digest TEXT,  -- sha256 for verification
  size_bytes INTEGER,
  local_path TEXT NOT NULL,
  downloaded_at_unix INTEGER,
  last_used_unix INTEGER,
  use_count INTEGER DEFAULT 0
);
```

**No synchronization (MVP):**
- Each node downloads models independently
- `rbee-keeper` queries each pool for available models
- User explicitly targets node or lets ctl choose
- Models cached locally, no cross-node sharing

**Model eviction (manual for MVP):**
```bash
# User manually removes models
rbee-keeper models rm hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --node mac

# Pool manager deletes from disk and catalog
```

**Deferred to post-MVP:**
- Automatic LRU eviction when disk space low
- Peer-to-peer model sharing between nodes
- Centralized model registry
- Automatic version updates

---

### G7: Network Partition Handling
**Missing:** No specification for:
- Behavior when `mac` becomes unreachable mid-inference
- Worker registry staleness detection
- Automatic failover to alternative nodes

**Ollama's Approach:**
- No distributed system features
- Single-node design
- Client retries on connection failure

**llama.cpp's Approach:**
- No distributed system features
- Single-node design

**Required for rbee (MVP: Fail Fast):**

**Network failure detection:**
```yaml
ctl:
  connection_timeout_seconds: 10
  request_timeout_seconds: 300
```

**Behavior on network partition:**
```
rbee-keeper → rbee-hive (connection fails):
  1. Retry with exponential backoff (3 attempts)
  2. If all retries fail:
     - Mark node as unreachable in local cache
     - Return error to user
     - User can retry or target different node
     
Mid-inference network failure:
  1. SSE stream breaks
  2. rbee-keeper detects EOF or timeout
  3. Return error to user with partial results (if any)
  4. Worker continues processing (orphaned request)
  5. Worker eventually times out or completes
  6. Worker transitions to idle after timeout
```

**Worker registry staleness:**
```
Pool manager tracks worker health:
  - Last heartbeat timestamp
  - If no heartbeat for 60s → mark as stale
  - If no heartbeat for 300s → remove from registry
  - Next request to that worker will fail and trigger cleanup
```

**No automatic failover (MVP):**
- User must manually retry
- `rbee-keeper` can suggest alternative nodes
- No request migration between workers

**Deferred to post-MVP:**
- Automatic failover to backup nodes
- Request migration (save state, resume elsewhere)
- Distributed consensus (Raft, etcd)
- Split-brain detection and resolution

---

### G8: Resource Quotas and Limits
**Missing:** No specification for:
- Per-user or per-request resource limits
- Fair scheduling across multiple users
- Preemption policy for high-priority requests

**Ollama's Approach:**
- No per-user quotas (single-user design)
- FIFO queue (first-come-first-served)
- No priority levels
- Global queue limit (`OLLAMA_MAX_QUEUE`)

**llama.cpp's Approach:**
- No per-user quotas
- FIFO queue
- No priority levels

**Required for rbee (MVP: Single User, FIFO):**

**No quotas for MVP:**
```yaml
# Single user assumed
# All requests treated equally
# FIFO queue processing
```

**Deferred to post-MVP:**
```yaml
# Per-user quotas
quotas:
  users:
    - name: "alice"
      max_concurrent_requests: 2
      max_tokens_per_day: 1000000
      priority: 10
    - name: "bob"
      max_concurrent_requests: 1
      max_tokens_per_day: 100000
      priority: 5

# Fair scheduling
scheduling:
  policy: "weighted_fair_queue"  # or "priority", "round_robin"
  preemption: false  # Don't kill running requests
  
# Priority levels
priorities:
  - level: 10  # High
  - level: 5   # Normal
  - level: 1   # Low
```

**Implementation (post-MVP):**
- User identification via API key
- Token bucket rate limiting
- Weighted fair queuing algorithm
- Metrics per user

---

### G9: Streaming Error Handling
**Missing:** No specification for:
- Handling of SSE connection drops during token streaming
- Client reconnection and resume from last token
- Timeout for stalled streams

**Ollama's Approach:**
- SSE stream with newline-delimited JSON
- No resume capability (request is one-shot)
- Client must retry from beginning if connection drops
- Context cancellation stops generation immediately

**llama.cpp's Approach:**
- SSE stream with `data:` prefix
- Ends with `data: [DONE]\n\n`
- Detects client disconnect via callback
- No resume capability

**Required for rbee:**

**SSE format:**
```
# Token stream
data: {"token": "Hello", "index": 0}

data: {"token": " world", "index": 1}

data: {"token": "!", "index": 2}

# Completion
data: {"done": true, "total_tokens": 3, "duration_ms": 1234}

# Stream end marker
data: [DONE]

```

**Connection drop handling:**
```
Worker detects client disconnect:
  1. SSE write fails or connection closed
  2. Stop token generation immediately
  3. Log event: "Client disconnected mid-stream"
  4. Release slot
  5. Transition to idle
  6. No cleanup needed (stateless)
```

**Timeout for stalled streams:**
```yaml
worker:
  stream_timeout_seconds: 30  # Max time between tokens
  total_timeout_seconds: 300  # Max total inference time
```

**Timeout behavior:**
```
Worker monitors token generation:
  - Start timer on each token
  - If no token generated in 30s → timeout
  - If total time exceeds 300s → timeout
  - On timeout:
    - Send error event via SSE
    - Close stream
    - Release slot
```

**Error events in stream:**
```
data: {"error": {"code": "GENERATION_TIMEOUT", "message": "Token generation stalled"}}

data: [DONE]

```

**No resume capability (MVP):**
- Stateless generation (no saved state)
- Client must retry entire request
- Use request ID for idempotency (future)

**Deferred to post-MVP:**
```yaml
# Resumable streams
POST /inference
{
  "prompt": "...",
  "resume_token": "abc123"  # Resume from checkpoint
}

# Worker saves checkpoints
checkpoints:
  - request_id: "req-123"
    tokens_generated: 50
    kv_cache_snapshot: "..."
    expires_at: 1234567890
```

---

### G10: Version Compatibility Matrix
**Missing:** No specification for:
- Minimum compatible versions across `rbee-keeper`, `rbee-hive`, `llm-worker-rbee`
- Upgrade path and backward compatibility guarantees
- Protocol versioning (API contracts)

**Ollama's Approach:**
- Single version string: `version.Version = "0.0.0"`
- No explicit compatibility checks
- Breaking changes require full upgrade
- Client and server must match versions

**llama.cpp's Approach:**
- No version negotiation
- API is relatively stable
- Breaking changes communicated via release notes

**Required for rbee (MVP: Strict Version Match):**

**Version format (SemVer):**
```
rbee v0.1.0
  ├─ rbee-keeper v0.1.0
  ├─ rbee-hive v0.1.0
  └─ llm-worker-rbee v0.1.0
```

**Version exchange:**
```
# Pool manager /health endpoint
GET /health
{
  "status": "alive",
  "version": "0.1.0",
  "api_version": "v1"
}

# Worker /health endpoint
GET /health
{
  "status": "alive",
  "version": "0.1.0",
  "api_version": "v1"
}
```

**Compatibility check:**
```rust
fn check_compatibility(ctl_version: &str, remote_version: &str) -> Result<(), VersionError> {
    // MVP: Exact match required
    if ctl_version != remote_version {
        return Err(VersionError::Mismatch {
            expected: ctl_version.to_string(),
            actual: remote_version.to_string(),
        });
    }
    Ok(())
}
```

**Error on mismatch:**
```
rbee-keeper v0.1.0 → rbee-hive v0.2.0:
  Error: Version mismatch
    rbee-keeper: v0.1.0
    rbee-hive: v0.2.0
  
  Please upgrade rbee-keeper to v0.2.0 or downgrade rbee-hive to v0.1.0
```

**API versioning:**
```
# All endpoints prefixed with version
POST /v1/inference
GET /v1/health
GET /v1/metrics

# Future: v2 API can coexist
POST /v2/inference  # New features
POST /v1/inference  # Deprecated but supported
```

**Deferred to post-MVP:**
```yaml
# Compatibility matrix
compatibility:
  rbee-keeper:
    - version: "0.3.0"
      compatible_pool: ["0.3.0", "0.2.0"]  # Backward compatible
      compatible_worker: ["0.3.0", "0.2.0"]
  
  rbee-hive:
    - version: "0.3.0"
      compatible_worker: ["0.3.0", "0.2.0", "0.1.0"]
      
# Graceful degradation
features:
  - name: "parallel_requests"
    min_worker_version: "0.2.0"
    fallback: "single_request_mode"
```

**Upgrade path (post-MVP):**
1. Upgrade workers first (backward compatible)
2. Upgrade pool managers (can talk to old workers)
3. Upgrade ctl last (can talk to old pools)
4. Rolling upgrade without downtime

---

---

## G11: VRAM Resource Contention (CRITICAL)

**Missing:** No specification for handling VRAM contention when user launches GPU-intensive applications (games, video editing, etc.) on the same machine as workers.

### **Problem Statement**

When a worker loads a model on a GPU (CUDA/Metal), the VRAM is **locked and reserved** by the process. The OS does **not** automatically reclaim CUDA/Metal memory for other applications. This creates conflicts:

- User launches game → game fails to allocate VRAM → crashes or falls back to degraded performance
- Worker sits idle holding VRAM → blocks legitimate user activities on their own hardware

### **Industry Solutions Analysis**

#### **Ollama's Approach** (reference: `reference/ollama/server/sched.go`)

Ollama implements a **sophisticated scheduler with automatic unloading**:

1. **KeepAlive Duration** (default: 5 minutes)
   - Configurable via `OLLAMA_KEEP_ALIVE` environment variable
   - Can be set per-request via API
   - Supports: duration (e.g., "5m"), zero (immediate unload), negative (infinite)

2. **Reference Counting**
   - Each active inference increments `refCount`
   - When request completes, `refCount` decrements
   - When `refCount` reaches 0, starts expiration timer

3. **Expiration Timer**
   - Starts when worker becomes idle (`refCount == 0`)
   - Resets on new request
   - On expiration: triggers unload sequence

4. **VRAM Recovery Wait**
   - After unloading, **waits for VRAM to be reported as free by driver**
   - CUDA memory reporting can lag 0.5-1.5 seconds after process exit
   - Timeout: 5 seconds
   - Prevents race conditions where next model load fails due to stale VRAM reporting

5. **Eviction Policy**
   - When new model needs space: finds least-recently-used (LRU) runner
   - Sets `sessionDuration = 0` to force immediate expiration
   - Waits for unload before proceeding

#### **llama.cpp Server Approach** (reference: `reference/llama.cpp/tools/server/server.cpp`)

llama.cpp uses a **slot-based system**:

1. **Slot States**
   - `SLOT_STATE_IDLE`: No active inference
   - `SLOT_STATE_PROCESSING_PROMPT`: Loading prompt
   - `SLOT_STATE_DONE_PROMPT`: Generating tokens

2. **KV Cache Clearing**
   - When all slots are idle, optionally clears KV cache
   - Model remains loaded in VRAM
   - **Does not unload model automatically**

3. **Manual Lifecycle**
   - Server process must be explicitly stopped to release VRAM
   - No automatic idle timeout

### **Recommended Architecture for rbee**

**Decision: Pool Manager MUST be a persistent daemon** (resolves C1)

#### **1. Idle Timeout with Automatic Unload**

**Configuration (per-node in topology):**
```yaml
pool:
  worker_keep_alive_seconds: 300  # 5 minutes default
  worker_keep_alive_policy: "auto"  # auto | infinite | immediate
```

**Worker State Machine:**
```
starting → http_ready → loading_model → ready → busy → idle → (timeout) → unloading → terminated
                                          ↑                ↓
                                          └────(request)───┘
```

**Lifecycle:**
1. Worker completes inference → transitions to `idle`
2. Pool manager starts keep-alive timer (5 minutes)
3. If new request arrives → cancel timer, transition to `busy`
4. If timer expires → pool manager sends `POST /admin/shutdown` to worker
5. Worker gracefully shuts down, releases VRAM
6. Pool manager removes from registry
7. Pool manager waits for VRAM recovery (poll `nvidia-smi` or NVML)

#### **2. Pool Manager Responsibilities**

**Persistent daemon on each node:**
- Monitor local workers via periodic health checks (`GET /health`)
- Enforce keep-alive policy
- Track worker idle time
- Trigger graceful shutdown on timeout
- Wait for VRAM recovery before marking resources available
- Report worker state to `rbee-keeper` via heartbeat

**Registry Schema Update:**
```sql
CREATE TABLE workers (
  id TEXT PRIMARY KEY,
  node_hostname TEXT NOT NULL,
  model_ref TEXT NOT NULL,
  backend TEXT NOT NULL,  -- cuda, metal, cpu
  device_id INTEGER,
  url TEXT NOT NULL,
  state TEXT NOT NULL,  -- starting, ready, busy, idle, unloading
  idle_since_unix INTEGER,
  keep_alive_seconds INTEGER,
  vram_allocated_mb INTEGER,
  last_health_check_unix INTEGER,
  created_at_unix INTEGER NOT NULL
);
```

#### **3. Worker API Requirements**

**Health Endpoint:**
```
GET /health
Response:
{
  "status": "idle",  // loading, ready, busy, idle
  "model_loaded": true,
  "vram_mb": 4096,
  "idle_since_unix": 1728508603,
  "active_requests": 0
}
```

**Admin Shutdown Endpoint:**
```
POST /admin/shutdown
{
  "graceful": true,
  "timeout_seconds": 30
}
Response: 202 Accepted
```

**Shutdown Behavior:**
- Reject new inference requests (503 Service Unavailable)
- Wait for active requests to complete (up to timeout)
- Unload model from VRAM
- Terminate HTTP server
- Exit process cleanly

#### **4. VRAM Recovery Monitoring**

**Pool manager after worker shutdown:**
```python
def wait_for_vram_recovery(device_id: int, expected_free_mb: int, timeout: float = 5.0):
    """
    Poll GPU memory until free VRAM increases by expected amount.
    CUDA memory reporting can lag 0.5-1.5s after process termination.
    """
    start = time.time()
    baseline_free = get_gpu_free_memory(device_id)
    
    while time.time() - start < timeout:
        current_free = get_gpu_free_memory(device_id)
        recovered = current_free - baseline_free
        
        if recovered >= expected_free_mb * 0.9:  # 90% threshold
            return True
        
        time.sleep(0.1)
    
    # Timeout - proceed anyway but log warning
    return False
```

#### **5. User-Friendly Behavior**

**Scenario: User launches game while worker is idle**

1. Worker has been idle for 5 minutes
2. Pool manager sends shutdown signal
3. Worker unloads model, releases VRAM
4. Pool manager confirms VRAM recovery
5. User launches game → game allocates VRAM successfully
6. User later requests inference → pool manager starts new worker, reloads model

**Trade-off:**
- **Pro:** User's machine remains responsive for their own use
- **Con:** Model reload latency on next inference (typically 5-30 seconds depending on model size)

**Mitigation:**
- Allow per-model keep-alive override (e.g., frequently-used models can have longer timeout)
- Show loading progress via SSE during reload
- Cache model files on disk (no re-download needed)

---

## Recommendations

1. **Define Component Lifecycle:** ✅ **RESOLVED** - `rbee-hive` MUST be a persistent daemon with worker lifecycle management responsibilities.

2. **State Machine Documentation:** Create explicit state diagrams for worker lifecycle with transitions and error states.

3. **Error Taxonomy:** Define error codes and recovery strategies for each failure mode.

4. **API Contract:** Formalize REST/SSE API contracts with OpenAPI specs for all inter-component communication.

5. **Observability First:** Instrument all components with structured logs, metrics, and traces from the start.

6. **Security Model:** Define authentication/authorization before implementing network communication.

7. **Concurrency Model:** Specify request queuing, worker slots, and load balancing strategy.

8. **Failure Modes:** Document expected behavior for each identified gap (network partition, resource exhaustion, etc.).

9. **VRAM Management:** ✅ **RESOLVED** - Implement Ollama-style keep-alive with automatic unload and VRAM recovery monitoring.

---

## Summary of Findings

**Total gaps identified:** 20 (G1-G20)  
**Total contradictions identified:** 3 (C1-C3)  
**Resolved with industry standards:** 11 gaps + 1 contradiction

### **Resolved Items**
- **C1:** Pool manager lifecycle ✅ (persistent daemon)
- **G2:** Concurrent request handling ✅ (slot-based architecture)
- **G3:** Worker shutdown ✅ (covered by G20 + G11)
- **G4:** Authentication ✅ (Bearer token API key)
- **G5:** Observability ✅ (covered by G16)
- **G6:** Model catalog sync ✅ (independent per-node)
- **G7:** Network partitions ✅ (fail-fast with retry)
- **G8:** Resource quotas ✅ (single-user FIFO)
- **G9:** Streaming errors ✅ (SSE with timeouts)
- **G10:** Version compatibility ✅ (strict match)
- **G11:** VRAM contention ✅ (idle timeout + auto-unload)

### **Critical Gaps Requiring Immediate Attention**

**High Priority (blocking MVP):**
- **G11:** VRAM resource contention ✅ RESOLVED
- **G12:** Request cancellation and context management
- **G13:** Request queue management and backpressure
- **G15:** Health check and readiness probes
- **G17:** Error taxonomy and HTTP status codes
- **G20:** Graceful shutdown and cleanup

**Medium Priority (needed for production):**
- **G1:** Error recovery and retry logic (overlaps with G18)
- **G12:** Request cancellation and context management
- **G13:** Request queue management and backpressure
- **G14:** Progress streaming for long operations
- **G15:** Health check and readiness probes
- **G16:** Metrics and observability
- **G17:** Error taxonomy and HTTP status codes
- **G18:** Retry logic and exponential backoff
- **G20:** Graceful shutdown and cleanup

**Lower Priority (quality of life):**
- **G2:** Concurrent request handling ✅ RESOLVED (slot-based, start with 1 slot)
- **G3:** Worker shutdown and cleanup ✅ RESOLVED (covered by G20 + G11)
- **G4:** Authentication and authorization ✅ RESOLVED (Bearer token API key for MVP)
- **G5:** Observability and logging ✅ RESOLVED (covered by G16)
- **G6:** Model catalog synchronization ✅ RESOLVED (independent per-node catalogs for MVP)
- **G7:** Network partition handling ✅ RESOLVED (fail-fast with retry, no auto-failover for MVP)
- **G8:** Resource quotas and limits ✅ RESOLVED (single-user FIFO for MVP, defer quotas)
- **G9:** Streaming error handling ✅ RESOLVED (SSE with timeouts, no resume for MVP)
- **G10:** Version compatibility matrix ✅ RESOLVED (strict version match for MVP)
- **G19:** Model alias and version management (nice-to-have, defer to post-MVP)

---

## Next Steps

### **Phase 1: Core Architecture (Week 1)**
- [x] **C1 RESOLVED:** Pool manager is persistent daemon
- [ ] Resolve C2 (preflight responsibility split)
- [ ] Resolve C3 (model loading state machine)
- [ ] Define worker state machine with all transitions
- [ ] Create sequence diagrams for happy path and error scenarios

### **Phase 2: API Contracts (Week 1-2)**
- [ ] Define OpenAPI specs for all endpoints:
  - [ ] Worker: `/health`, `/ready`, `/metrics`, `/inference`, `/admin/shutdown`, `/loading/progress`
  - [ ] Pool: `/health`, `/ready`, `/metrics`, `/workers`, `/models/{id}/download/progress`
  - [ ] CTL: Internal queue management API
- [ ] Document error response format and codes (G17)
- [ ] Define SSE streaming format for progress and tokens

### **Phase 3: Resource Management (Week 2)**
- [x] **G11 RESOLVED:** VRAM contention handled via idle timeout + unload
- [ ] Implement keep-alive timer in pool manager
- [ ] Add NVML/nvidia-smi integration for VRAM monitoring
- [ ] Implement VRAM recovery wait logic
- [ ] Add graceful shutdown to all components (G20)

### **Phase 4: Request Lifecycle (Week 2-3)**
- [ ] Implement request cancellation (G12)
- [ ] Implement queue management with backpressure (G13)
- [ ] Add retry logic with exponential backoff (G18)
- [ ] Implement progress streaming (G14)
- [ ] Handle SSE connection drops and reconnection (G9)

### **Phase 5: Observability (Week 3)**
- [ ] Implement Prometheus metrics endpoints (G16)
- [ ] Add structured logging to all components
- [ ] Implement health and readiness probes (G15)
- [ ] Add distributed tracing headers (optional)

### **Phase 6: Testing & Validation (Week 3-4)**
- [ ] Write BDD scenarios covering:
  - [ ] Happy path (test-001 flow)
  - [ ] VRAM contention and auto-unload
  - [ ] Request cancellation
  - [ ] Queue full scenarios
  - [ ] Worker crash recovery
  - [ ] Network failures
  - [ ] Graceful shutdown
- [ ] Implement proof-of-concept with minimal viable flow
- [ ] Load testing with concurrent requests
- [ ] Chaos testing (kill workers, network partitions)

### **Deferred to Post-MVP**
- [ ] Authentication and authorization (G4)
- [ ] Model catalog synchronization across nodes (G6)
- [ ] Network partition handling (G7)
- [ ] Resource quotas and fair scheduling (G8)
- [ ] Version compatibility matrix (G10)
- [ ] Model aliasing and version pinning (G19)
- [ ] Concurrent request handling per worker (G2)
