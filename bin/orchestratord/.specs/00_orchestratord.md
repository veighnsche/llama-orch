# Orchestratord SPEC — The Brain (ORCH-1xxx)

**Status**: Draft  
**Applies to**: `bin/orchestratord/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

Orchestratord is **THE BRAIN** of llama-orch. It makes ALL intelligent decisions: admission, scheduling, worker selection (placement), eviction, and catalog management. Pool managers and workers are dumb executors.

**Why it exists:**
- Need centralized intelligence for multi-node GPU clusters
- Orchestratord can run without GPUs (queries pool managers for state)
- Clean separation: orchestratord decides → pool managers execute → workers run
- Stateless decision-making (all state from pool manager queries)

**What it does:**
- Make all intelligent decisions (admission, scheduling, placement, eviction)
- Provide client-facing HTTP API (task submission, SSE streaming)
- Query pool managers for GPU/worker state
- Command pool managers to start/stop workers
- Route requests directly to workers
- Manage model catalog
- Track metrics and observability

**What it does NOT do:**
- ❌ Execute inference (workers do this)
- ❌ Manage GPU hardware (pool managers do this via NVML)
- ❌ Spawn worker processes (pool managers do this)
- ❌ Allocate VRAM (workers do this via CUDA)
- ❌ Validate model compatibility (pool managers do this preflight)

**Decision hierarchy:**
```
Orchestratord (brain)
    ↓ commands
Pool Manager (levers)
    ↓ spawns
Worker (executor)
```

**Crate structure:**
- `orchestrator-core` — Queue, admission, core logic
- `scheduling` — Combined job ordering + worker selection
- `platform-api` — Client-facing HTTP API
- `agentic-api` — Agent-specific endpoints
- `streaming` — SSE relay from workers to clients
- `task-cancellation` — Cancellation propagation
- `backpressure` — Queue backpressure handling
- `job-timeout` — Timeout enforcement
- `pool-registry` — Track available pool managers

**Parent spec**: `.specs/00_llama-orch.md`

---

## 1. Core Responsibilities

### [ORCH-1001] Intelligence Boundary
Orchestratord MUST be the ONLY component that makes planning, scheduling, placement, routing, and eviction decisions. Pool managers and workers MUST NOT make these decisions.

### [ORCH-1002] Stateless Operation
Orchestratord MUST be able to run on machines without GPUs. All GPU state MUST be obtained by querying pool managers.

### [ORCH-1003] Control Plane
Orchestratord MUST provide the client-facing HTTP API for task submission, SSE streaming, catalog management, and observability.

### [ORCH-1004] Worker Status Values
Orchestratord MUST recognize the following worker status values (aligned with pool-managerd):

- `starting` — Worker process spawned, initializing CUDA context and loading model
- `ready` — Worker successfully loaded model and is ready to accept inference requests
- `busy` — Worker is currently processing an inference request
- `draining` — Worker is finishing current request but not accepting new ones (shutdown in progress)
- `failed` — Worker encountered a fatal error and is unavailable

**Status transitions**:
- `starting` → `ready` (successful model load and callback to pool-managerd)
- `starting` → `failed` (model load failure or timeout)
- `ready` → `busy` (inference request received)
- `busy` → `ready` (inference completed)
- `ready` → `draining` (shutdown command received)
- `draining` → `failed` (forced termination)
- Any status → `failed` (fatal error)

---

## 2. Queue & Admission

### [ORCH-1010] Queue Management
Orchestratord MUST maintain a bounded FIFO queue with two priorities: `interactive` and `batch`.

### [ORCH-1011] Admission Checks
Before enqueuing, orchestratord MUST validate:
- Context length within model limits
- Token budget within session limits
- Model exists in catalog

### [ORCH-1012] Backpressure
When queue is full, orchestratord MUST respond with HTTP 429, `Retry-After`, `X-Backoff-Ms`, and JSON body containing `policy_label`, `retriable`, `retry_after_ms`.

### [ORCH-1013] Queue Metrics
Orchestratord MUST expose Prometheus metrics: `queue_depth`, `tasks_enqueued_total`, `tasks_rejected_total`, `tasks_dropped_total`.

---

## 3. Scheduling & Worker Selection

### [ORCH-1020] Combined Decision
Orchestratord MUST make a combined decision: "Which job is next + where to run it". This includes job ordering (scheduling) and worker selection (placement) in a single decision point.

### [ORCH-1021] Priority Handling
Orchestratord MUST process `interactive` priority jobs before `batch` priority jobs within the same queue.

### [ORCH-1022] Fair Scheduling
Within the same priority, orchestratord SHOULD use FIFO ordering. Starvation prevention SHOULD be implemented for batch jobs.

### [ORCH-1030] State Query
Before making scheduling decisions, orchestratord MUST query pool managers for current state (GPU VRAM, running workers, worker URIs).

### [ORCH-1031] Worker Selection Algorithms
Orchestratord MUST select a worker using the configured algorithm:
- **least-loaded**: Select worker with fewest active jobs
- **most-vram-free**: Select worker on GPU with most free VRAM
- **round-robin**: Cycle through workers deterministically

### [ORCH-1032] Model Matching
Orchestratord MUST only select workers that have the requested model loaded.

### [ORCH-1033] No Worker Available
If no suitable worker is available, orchestratord MUST decide whether to:
- Command pool manager to start a new worker
- Reject the job with `POOL_UNAVAILABLE`
- Wait and retry scheduling

---

## 4. Worker Startup Decisions

### [ORCH-1040] Start New Worker
When orchestratord decides to start a new worker, it MUST:
1. Query pool managers for GPU VRAM capacity
2. Select a GPU with sufficient free VRAM
3. Send command to pool manager: `POST /v2/workers/start { model_ref, gpu_id }`
4. Wait for pool manager to report worker ready
5. Retry placement once worker is available

### [ORCH-1041] VRAM Capacity Check
Orchestratord MUST NOT command a worker start if no GPU has sufficient VRAM. It MUST reject the job instead.

### [ORCH-1042] Startup Timeout
If a worker does not become ready within a configured timeout (default 60s), orchestratord SHOULD retry once, then fail the job.

---

## 5. Eviction Decisions

### [ORCH-1050] Eviction Policy
Orchestratord MUST decide which workers to stop when VRAM is needed. Policy options:
- **LRU**: Evict least recently used model
- **LFU**: Evict least frequently used model
- **Manual**: Only evict on explicit admin command

### [ORCH-1051] Eviction Command
When orchestratord decides to evict, it MUST:
1. Send command to pool manager: `POST /v2/workers/stop { worker_id }`
2. Wait for pool manager to confirm worker stopped
3. Update internal state to reflect VRAM freed

### [ORCH-1052] Eviction Safety
Orchestratord MUST NOT evict a worker with active jobs. It MUST drain the worker first or wait for jobs to complete.

---

## 6. Routing (Direct Worker Connection)

### [ORCH-1060] Direct Connection
Orchestratord MUST connect directly to workers using the URI provided by pool managers. It MUST NOT proxy requests through pool managers.
SSE streams MUST be sourced directly from workers and relayed by orchestratord to clients; pool-managerd is bypassed entirely for inference streams.

### [ORCH-1061] Worker URI
Orchestratord MUST use the `uri` field from pool manager worker registry (e.g., `http://localhost:8001`).

### [ORCH-1062] Connection Pooling
Orchestratord SHOULD maintain HTTP connection pools per worker to reduce latency.

### [ORCH-1063] Request Format
Orchestratord MUST send inference requests to workers in the format expected by worker API (see `bin/worker-orcd/.specs/00_worker-orcd.md`).

---

## 7. Streaming (SSE to Clients)

### [ORCH-1070] SSE Framing
Orchestratord MUST stream results to clients via Server-Sent Events with event types: `started`, `token`, `metrics`, `end`, `error`.

### [ORCH-1071] SSE Relay
Orchestratord MUST relay SSE events from workers to clients, adding orchestrator-specific metadata (e.g., `queue_position`, `predicted_start_ms`).

### [ORCH-1072] Stream Cancellation
When a client cancels (disconnects), orchestratord MUST propagate cancellation to the worker: `POST /cancel`.
Additional requirements:
- Idempotency: Multiple cancellations for the same `job_id` MUST be safe and yield the same terminal outcome.
- Promptness: Orchestratord MUST issue cancel immediately on client request or stream disconnect.
- Deadline: Orchestratord SHOULD enforce a cancellation deadline (default 5s); after the deadline it treats the job as cancelled and closes the client SSE with an `error` event.

### [ORCH-1073] Stream Timeout
Orchestratord SHOULD apply a timeout to worker streams (default 5 minutes). After timeout, it MUST cancel the worker job and emit SSE `error` event.

---

## 8. Catalog Management

### [ORCH-1080] Model Registry
Orchestratord MUST maintain a catalog of available models with metadata: `model_id`, `name`, `size_bytes`, `context_length`, `lifecycle_state`.

### [ORCH-1081] Lifecycle States
Models MUST have lifecycle state: `Active` or `Retired`. Orchestratord MUST reject jobs for `Retired` models.

### [ORCH-1082] Verification
Orchestratord MUST provide API to verify model integrity (checksums, signatures). Verification failures SHOULD warn but not block unless policy requires.

### [ORCH-1083] Catalog API
Orchestratord MUST expose catalog endpoints:
- `POST /v2/catalog/models` — Register model
- `GET /v2/catalog/models/{id}` — Get metadata
- `POST /v2/catalog/models/{id}/verify` — Verify integrity
- `POST /v2/catalog/models/{id}/state` — Update lifecycle state

### [ORCH-1084] Model Ref Resolution
Orchestratord MUST resolve client-provided `model` (alias or scheme) to a canonical `model_ref` before scheduling:
- Accepts `hf:{org}/{repo}[@{rev}][::file={path}]`, `file:/abs/path.gguf`, or alias (e.g., `llama-7b`).
- SHOULD pin `@rev` to a commit SHA and select a concrete artifact file for determinism.
- MUST pass the resolved `model_ref` in `POST /v2/workers/start`.

---

## 9. Observability

### [ORCH-1090] Structured Logging
Orchestratord MUST emit structured logs (JSON) with fields: `job_id`, `session_id`, `model_ref`, `queue_position`, `worker_id`, `gpu_id`, `scheduling_latency_ms`, `execution_latency_ms`.

### [ORCH-1091] Prometheus Metrics
Orchestratord MUST expose metrics:
- `orchd_queue_depth{priority}`
- `orchd_tasks_enqueued_total{outcome}`
- `orchd_tasks_dispatched_total{worker_id, outcome}`
- `orchd_scheduling_latency_ms{algorithm}`
- `orchd_worker_connections_active{worker_id}`

### [ORCH-1092] Correlation ID
Orchestratord MUST generate or accept `X-Correlation-Id` header and include it in all logs and responses.

### [ORCH-1093] Human Narration
Orchestratord SHOULD emit human-readable narration at key points: admission, placement, worker start, eviction, stream start/end.

---

## 10. HTTP API

### [ORCH-1100] Client-Facing Endpoints
Orchestratord MUST expose:
- `POST /v2/tasks` — Submit job (returns 202 + job_id)
- `GET /v2/tasks/{id}/events` — SSE stream
- `POST /v2/tasks/{id}/cancel` — Cancel job
- `GET /v2/sessions/{id}` — Session status
- `DELETE /v2/sessions/{id}` — Delete session

### [ORCH-1101] Admin Endpoints
Orchestratord MUST expose:
- `GET /v2/meta/capabilities` — Capabilities
- `GET /v2/pools/{id}/health` — Pool health
- `POST /v2/pools/{id}/drain` — Drain pool
- `POST /v2/pools/{id}/reload` — Reload pool

### [ORCH-1102] Catalog Endpoints
See [ORCH-1083] for catalog API requirements.

---

## 11. Pool Manager Communication

### [ORCH-1110] State Query API
Orchestratord MUST query pool managers via:
- `GET /v2/state` — Get GPUs, workers, VRAM

Response format:
```json
{
  "pool_id": "pool-1",
  "gpus": [
    {
      "id": 0,
      "total_vram": 24000000000,
      "allocated_vram": 16000000000,
      "available_vram": 8000000000,
      "workers": ["worker-abc"]
    }
  ],
  "workers": [
    {
      "id": "worker-abc",
      "model_ref": "hf:author/repo@rev::file=models/llama-7b.Q4_K_M.gguf",
      "gpu": 0,
      "vram_used": 16000000000,
      "uri": "http://localhost:8001",
      "status": "ready",
      "started_at": "2025-10-03T00:00:00Z"
    }
  ]
}
```

### [ORCH-1111] Worker Lifecycle Commands
Orchestratord MUST command pool managers via:
- `POST /v2/workers/start { model_ref, gpu_id }` — Start worker
- `POST /v2/workers/stop { worker_id }` — Stop worker
- `POST /v2/workers/{id}/drain` — Drain worker

### [ORCH-1112] Pool Manager Discovery
Orchestratord MUST support configuration of pool manager endpoints (comma-separated list or service discovery).

---

## 12. Error Handling

### [ORCH-1120] Error Taxonomy
Orchestratord MUST use stable error codes:
- `ADMISSION_REJECT` — Job rejected at admission
- `QUEUE_FULL` — Queue capacity reached
- `INVALID_PARAMS` — Invalid request
- `POOL_UNAVAILABLE` — No workers/capacity available
- `WORKER_UNAVAILABLE` — Selected worker is down
- `WORKER_TIMEOUT` — Worker did not respond
- `INSUFFICIENT_VRAM` — GPU doesn't have enough free VRAM (from pool-managerd)
- `GPU_UNAVAILABLE` — GPU is in error state (from pool-managerd)
- `MODEL_NOT_FOUND` — Model not in cache and download failed (from pool-managerd)
- `WORKER_START_FAILED` — Worker process failed to start (from pool-managerd)
- `WORKER_START_TIMEOUT` — Worker didn't call back in time (from pool-managerd)
- `WORKER_NOT_FOUND` — Unknown worker_id (from pool-managerd)
- `INTERNAL` — Internal error

### [ORCH-1121] Retriable Errors
Orchestratord MUST mark errors as retriable: `QUEUE_FULL`, `POOL_UNAVAILABLE`, `WORKER_TIMEOUT`, `INSUFFICIENT_VRAM`, `GPU_UNAVAILABLE`, `WORKER_START_TIMEOUT`.

### [ORCH-1122] Error Propagation
Orchestratord MUST propagate worker errors to clients via SSE `error` event with original error code.

---

## 13. Security

### [ORCH-1130] Authentication
Orchestratord MAY support Bearer token authentication via `auth-min` crate. Home profile has no auth (localhost only).

### [ORCH-1131] Secrets Redaction
Orchestratord MUST NOT log secrets, API tokens, or sensitive data. It MUST use redaction helpers.

### [ORCH-1132] Rate Limiting
Orchestratord SHOULD support per-token rate limiting (requests/sec, tokens/sec).

---

## 14. Configuration

### [ORCH-1140] Required Config
Orchestratord MUST accept configuration:
- `bind_addr` — Bind address (default `0.0.0.0:8080`)
- `pool_managers` — Comma-separated list of pool manager URLs
- `queue_capacity` — Max queue size (default 100)
- `queue_policy` — `reject` or `drop-lru`

### [ORCH-1141] Optional Config
Orchestratord MAY accept:
- `placement_algorithm` — `least-loaded`, `most-vram-free`, `round-robin`
- `eviction_policy` — `lru`, `lfu`, `manual`
- `worker_timeout_sec` — Worker response timeout (default 300)
- `catalog_path` — Catalog storage directory

---

## 15. Traceability

**Code**: `bin/orchestratord/src/`  
**Tests**: `bin/orchestratord/tests/`, `bin/orchestratord/bdd/`  
**Parent**: `.specs/00_llama-orch.md`  
**Crates**: `orchestrator-core`, `scheduling`, `platform-api`, `agentic-api`, `streaming`, `task-cancellation`, `backpressure`, `job-timeout`, `pool-registry`

---

## 16. Refinement Opportunities

### 16.1 Advanced Scheduling
- Incorporate measured decode time into worker selection
- Predict `predicted_start_ms` based on queue depth and historical latency
- Multi-objective optimization (latency + VRAM + load balance)

### 16.2 Predictive Worker Startup
- Pre-warm workers based on traffic patterns
- Start workers speculatively before jobs arrive
- Adaptive worker pool sizing

### 16.3 Multi-Region Support
- Support pool managers in different datacenters
- Region-aware worker selection (minimize latency)
- Cross-region failover

---

**End of Specification**
