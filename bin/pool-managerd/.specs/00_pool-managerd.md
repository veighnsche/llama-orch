# Pool Managerd SPEC — State Reporter & Worker Factory (POOL-2xxx)

**Author**: Specs Team
**Date**: 2025-10-03
**Status**: Draft  
**Applies to**: `bin/pool-managerd/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

Pool managerd is the **control plane with all the levers**. It tracks local GPU state, validates model compatibility, manages worker processes, and reports everything to orchestratord. It is DUMB (makes NO decisions) but RICH (has all controls).

**Why it exists:**
- Orchestratord needs a local agent on each GPU node to execute commands
- Need preflight validation before spawning workers (capability matching, VRAM checks)
- Need system-wide GPU monitoring (NVML) separate from per-process worker CUDA contexts
- Need worker process lifecycle management (spawn, monitor, stop)

**What it does:**
- Track GPU state system-wide (via `gpu-inventory` using NVML FFI)
- Download and stage models (via `model-cache`, `model-provisioner`)
- Validate model compatibility before spawn (via `capability-matcher`)
- Spawn and monitor worker processes (via `worker-lifecycle`)
- Report state to orchestratord (via `control-api`)
- Update VRAM accounting when workers start/stop

**What it does NOT do:**
- ❌ Make placement decisions (orchestratord does this)
- ❌ Allocate VRAM (workers do this within their CUDA contexts)
- ❌ Route inference requests (orchestratord connects directly to workers)
- ❌ Schedule jobs (orchestratord does this)
- ❌ Decide eviction (orchestratord decides, pool manager executes)

**FFI Boundary:**
- Uses **NVML** (NVIDIA Management Library) for read-only GPU queries
- Does NOT use CUDA (workers use CUDA for VRAM allocation)

**Crate structure:**
- `gpu-inventory` — NVML FFI for system-wide GPU/VRAM tracking
- `capability-matcher` — Preflight model compatibility validation
- `model-cache` — Model storage and RAM staging
- `model-provisioner` — Model download orchestration
- `model-catalog` — Model metadata registry
- `worker-lifecycle` — Worker process spawning and monitoring
- `control-api` — HTTP API for orchestratord commands
- `error-recovery` — Pool-level error handling
- `pool-registration-client` — Register with orchestratord

**Parent spec**: `.specs/00_llama-orch.md`

---

## 1. Core Responsibilities

### [POOL-2001] State Reporting
Pool managerd MUST track and report the state of all local GPUs and workers to orchestratord on demand.

### [POOL-2002] Worker Lifecycle
Pool managerd MUST spawn, monitor, and stop worker processes when commanded by orchestratord.

### [POOL-2003] No Intelligence
Pool managerd MUST NOT make placement, routing, scheduling, or eviction decisions. It MUST only execute commands from orchestratord.

### [POOL-2004] GPU-Local Operation
Pool managerd MUST run on machines with NVIDIA GPUs. One pool managerd instance per physical host.

---

## 2. GPU Inventory

### [POOL-2010] GPU Discovery
At startup, pool managerd MUST discover all NVIDIA GPUs on the local machine using `nvidia-smi` or NVML.

### [POOL-2011] VRAM Tracking
Pool managerd MUST track per-GPU:
- `total_vram_bytes` — Total VRAM capacity
- `allocated_vram_bytes` — Sum of VRAM used by all workers on this GPU
- `available_vram_bytes` — Free VRAM (total - allocated)

### [POOL-2012] VRAM Updates
When a worker starts, pool managerd MUST update `allocated_vram_bytes` for that GPU. When a worker stops, pool managerd MUST decrement `allocated_vram_bytes`.

### [POOL-2013] GPU State Query
Pool managerd MUST provide API: `GET /v2/state` returning GPU inventory and worker registry.

**Response format**:
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

---

## 3. Worker Registry

### [POOL-2020] Worker Tracking
Pool managerd MUST maintain a registry of all running workers with fields:
- `worker_id` — Unique identifier
- `model_ref` — Model loaded (e.g., "llama-7b")
- `gpu_device` — GPU device ID (0, 1, ...)
- `vram_bytes` — VRAM used by this worker
- `uri` — Worker HTTP endpoint (e.g., "http://localhost:8001")
- `status` — Worker status: `starting`, `ready`, `busy`, `draining`, `failed`
- `pid` — Process ID
- `started_at` — Timestamp

### [POOL-2021] Worker Registration
When a worker becomes ready, it MUST call back to pool managerd: `POST /v2/internal/workers/ready { worker_id, model, vram_bytes, uri }`.

Pool managerd MUST:
1. Update worker registry status to `ready`
2. Update GPU inventory `allocated_vram_bytes`
3. Log registration event

### [POOL-2022] Worker Deregistration
When a worker stops (graceful or crash), pool managerd MUST:
1. Remove worker from registry
2. Update GPU inventory `allocated_vram_bytes`
3. Log deregistration event

### [POOL-2023] Worker Health Checks
Pool managerd SHOULD periodically check worker health: `GET /health`. If a worker is unreachable, pool managerd MUST mark it as `failed` and update state.

---

## 4. Model Cache (RAM Staging)

### [POOL-2030] Model Download
Pool managerd MUST download models from sources when needed:
- `hf:{org}/{repo}[@{rev}::file={path}]` — Hugging Face (canonical model_ref)
- `file:/abs/path/to/model.gguf` — Local filesystem
Other schemes (e.g., `https:`, `s3:`) are out of scope for now.

### [POOL-2031] RAM Pre-Staging
Pool managerd MAY pre-stage frequently used models in RAM (mmap, shared memory) to speed up worker startup.

### [POOL-2032] Cache Location
Pool managerd MUST use configured cache directory (default `~/.cache/llama-orch/models/`).

### [POOL-2033] Cache Management
Pool managerd SHOULD implement LRU eviction for RAM-staged models when RAM capacity is reached. Disk-cached models SHOULD be kept unless explicitly deleted.

### [POOL-2034] Model Location API
When starting a worker, pool managerd MUST provide the model location:
- `ModelLocation::Disk { path }` — Load from disk
- `ModelLocation::RamStaged { shared_mem_name, size_bytes }` — Load from shared memory

---

## 5. Worker Lifecycle

### [POOL-2040] Start Worker Command
When orchestratord sends `POST /v2/workers/start { model_ref, gpu_id }`, pool managerd MUST:

1. **Check VRAM**: Verify GPU has sufficient free VRAM (query gpu-inventory)
2. **Check model**: Ensure model is downloaded (download if needed)
3. **Stage model**: Pre-stage in RAM if not already staged
4. **Allocate port**: Select available port for worker HTTP server
5. **Spawn process**: Start `worker-orcd` process with args:
   ```bash
   worker-orcd \
     --worker-id worker-{uuid} \
     --model /path/to/model.gguf \
     --gpu-device {gpu_id} \
     --port {port} \
     --callback-url http://localhost:9200/v2/internal/workers/ready
   ```
6. **Wait for callback**: Worker will call `/v2/internal/workers/ready` when loaded
7. **Update registry**: Add worker to registry with status `starting`
8. **Return**: Respond to orchestratord with `worker_id`

### [POOL-2041] Start Worker Timeout
If worker does not call back within timeout (default 60s), pool managerd MUST:
1. Kill worker process
2. Mark worker as `failed` in registry
3. Log failure event
4. Notify orchestratord of failure (if possible)

### [POOL-2042] Stop Worker Command
When orchestratord sends `POST /v2/workers/stop { worker_id }`, pool managerd MUST:

1. **Mark draining**: Set worker status to `draining`
2. **Graceful shutdown**: Send `POST /shutdown` to worker (if API exists) or `SIGTERM`
3. **Wait**: Allow grace period (default 30s) for worker to finish active jobs
4. **Force kill**: If worker doesn't exit, send `SIGKILL`
5. **Update state**: Remove worker from registry, update GPU inventory
6. **Return**: Respond to orchestratord

### [POOL-2043] Worker Crash Handling
If a worker crashes (process exits unexpectedly), pool managerd MUST:
1. Detect crash via process monitoring
2. Remove worker from registry
3. Update GPU inventory (free VRAM)
4. Log crash event with exit code
5. Do NOT auto-restart (orchestratord decides)

---

## 6. Health Monitoring

### [POOL-2050] Worker Health Checks
Pool managerd SHOULD periodically check worker health:
- `GET /health` to each worker (default interval: 10s)
- If worker is unreachable for N consecutive checks (default 3), mark as `failed`

### [POOL-2051] GPU Health
Pool managerd SHOULD monitor GPU health:
- CUDA driver errors
- GPU temperature
- Power state
- Memory errors (ECC)

If GPU enters error state, pool managerd SHOULD:
1. Mark all workers on that GPU as `failed`
2. Stop all workers on that GPU
3. Mark GPU as `unavailable` in inventory
4. Log critical event

### [POOL-2052] Heartbeat to Orchestratord
Pool managerd MAY send periodic heartbeats to orchestratord (if multi-pool):
- `POST /v2/pools/{id}/heartbeat` with full state snapshot
- Default interval: 15s

---

## 7. Internal API

### [POOL-2060] Worker Callback Endpoint
Pool managerd MUST expose:
- `POST /v2/internal/workers/ready`

Request body:
```json
{
  "worker_id": "worker-abc",
  "model_ref": "llama-7b",
  "vram_bytes": 16000000000,
  "uri": "http://localhost:8001"
}
```

Pool managerd MUST validate:
- `worker_id` exists in registry with status `starting`
- `vram_bytes` fits within GPU capacity
- `uri` is reachable

Then update registry status to `ready`.

### [POOL-2061] State Query Endpoint
Pool managerd MUST expose:
- `GET /v2/state`

Returns GPU inventory + worker registry (see [POOL-2013]).

### [POOL-2062] Worker Lifecycle Endpoints
Pool managerd MUST expose:
- `POST /v2/workers/start { model, gpu_id }` — Start worker
- `POST /v2/workers/stop { worker_id }` — Stop worker
- `POST /v2/workers/{id}/drain` — Drain worker (mark draining, stop accepting new jobs)

---

## 8. Observability

### [POOL-2070] Structured Logging
Pool managerd MUST emit structured logs with fields:
- `pool_id` — Pool manager ID
  - `worker_id` — For worker events
  - `gpu_id` — For GPU events
  - `model_ref` — Model being loaded/unloaded
  - `vram_allocated` — VRAM changes
  - `event` — `worker_started`, `worker_stopped`, `worker_registered`, `worker_failed`, `gpu_discovered`, `model_downloaded`

### [POOL-2071] Prometheus Metrics
Pool managerd MUST expose metrics:
- `pool_mgr_gpus_total` — Number of GPUs
- `pool_mgr_gpu_vram_total_bytes{gpu_id}` — Total VRAM per GPU
- `pool_mgr_gpu_vram_allocated_bytes{gpu_id}` — Allocated VRAM per GPU
- `pool_mgr_workers_total{status}` — Worker count by status
- `pool_mgr_worker_starts_total{outcome}` — Worker start attempts
- `pool_mgr_worker_stops_total{reason}` — Worker stop events
- `pool_mgr_model_downloads_total{outcome}` — Model download attempts

### [POOL-2072] Human Narration
Pool managerd SHOULD emit human-readable narration for key events:
- Worker started/stopped
- GPU discovered/unavailable
- Model downloaded
- Worker registration

---

## 9. Configuration

### [POOL-2080] Required Config
Pool managerd MUST accept configuration:
- `bind_addr` — Bind address (default `0.0.0.0:9200`)
- `pool_id` — Unique pool identifier (default hostname)
- `model_cache_dir` — Model cache directory (default `~/.cache/llama-orch/models`)

### [POOL-2081] Optional Config
Pool managerd MAY accept:
- `orchestratord_url` — Orchestratord URL for registration (multi-node)
- `worker_start_timeout_sec` — Worker startup timeout (default 60)
- `worker_health_check_interval_sec` — Health check interval (default 10)
- `ram_staging_enabled` — Enable RAM pre-staging (default true)
- `ram_staging_max_bytes` — Max RAM for staging (default 64GB)

---

## 10. Error Handling

### [POOL-2090] Error Taxonomy
Pool managerd MUST use stable error codes:
- `INSUFFICIENT_VRAM` — GPU doesn't have enough free VRAM
- `GPU_UNAVAILABLE` — GPU is in error state
- `MODEL_NOT_FOUND` — Model not in cache and download failed
- `WORKER_START_FAILED` — Worker process failed to start
- `WORKER_START_TIMEOUT` — Worker didn't call back in time
- `WORKER_NOT_FOUND` — Unknown worker_id

### [POOL-2091] Error Responses
Pool managerd MUST return errors to orchestratord with:
- `error_code` — Stable error code
- `message` — Human-readable description
- `retriable` — Boolean (true if orchestratord can retry)
- `details` — Additional context (e.g., available VRAM)

---

## 11. Security

### [POOL-2100] Worker Callbacks
Pool managerd SHOULD validate worker callbacks using shared secret or process correlation (verify PID matches spawned process).

### [POOL-2101] Internal API
Pool managerd internal API (`/v2/internal/*`) SHOULD only be accessible from localhost or trusted network.

### [POOL-2102] Orchestratord API
Pool managerd external API (`/v2/workers/*`, `/v2/state`) MAY use Bearer token authentication if configured.

---

## 12. GPU Requirements

### [POOL-2110] NVIDIA Only
Pool managerd MUST require NVIDIA GPUs with CUDA drivers. It MUST fail fast if no NVIDIA GPUs are detected.

### [POOL-2111] VRAM-Only Policy
Pool managerd MUST NOT allow CPU inference fallback. Workers MUST fail to start if VRAM is insufficient.

### [POOL-2112] Driver Compatibility
Pool managerd SHOULD check CUDA driver version at startup and warn if incompatible with worker requirements.

---

## 13. Multi-Pool Support (Optional)

### [POOL-2120] Pool Registration
If `orchestratord_url` is configured, pool managerd MUST register with orchestratord:
- `POST /v2/pools/register { pool_id, gpus, bind_addr }`

### [POOL-2121] Heartbeat
Pool managerd MUST send periodic heartbeats to orchestratord with full state snapshot.

### [POOL-2122] Deregistration
On graceful shutdown, pool managerd SHOULD deregister:
- `DELETE /v2/pools/{pool_id}`

---

## 14. Traceability

**Code**: `bin/pool-managerd/src/`  
**Tests**: `bin/pool-managerd/tests/`, `bin/pool-managerd/bdd/`  
**Parent**: `.specs/00_llama-orch.md`  
**Crates**: `gpu-inventory`, `worker-registry` (pool-registry), `model-cache`, `lifecycle`, `health-monitor`, `api`

---

## 15. Refinement Opportunities

### 15.1 Advanced Model Cache
- Differential downloads (only fetch changed layers)
- Model deduplication (share common layers)
- Compression (decompress on-the-fly)

### 15.2 Predictive Staging
- Pre-download models based on orchestratord hints
- Pre-stage frequently used models in RAM
- Adaptive cache sizing based on usage patterns

### 15.3 Worker Process Isolation
- Containerization (Docker, Podman)
- Resource limits (cgroups)
- Network isolation (separate network namespaces)

---

**End of Specification**
