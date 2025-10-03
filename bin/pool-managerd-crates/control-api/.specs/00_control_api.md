# Control API SPEC — Pool Manager HTTP Endpoints (CAPI-9xxx)

**Status**: Draft  
**Applies to**: `bin/pool-managerd-crates/control-api/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `control-api` crate provides the HTTP API layer for pool-managerd. It exposes control plane endpoints for orchestratord to command worker lifecycle and query state.

**Why it exists:**
- Orchestratord needs HTTP API to command pool manager
- Pool manager exposes "all the levers" for orchestratord
- Clean separation: control API vs. internal callbacks

**What it does:**
- Expose endpoints for orchestratord commands (start/stop workers)
- Expose state query endpoints (GPU inventory, worker registry)
- Expose internal callback endpoints (worker registration)
- Handle request validation and error responses

**What it does NOT do:**
- ❌ Make placement decisions (orchestratord does this)
- ❌ Execute inference (workers do this)
- ❌ Download models (separate model-cache/provisioner layer)

---

## 1. Core Responsibilities

### [CAPI-9001] HTTP Server
The crate MUST provide HTTP server bound to configured port (default 9200).

### [CAPI-9002] Command Endpoints
The crate MUST expose endpoints for orchestratord commands.

### [CAPI-9003] State Query Endpoints
The crate MUST expose endpoints for state queries.

### [CAPI-9004] Internal Callback Endpoints
The crate MUST expose internal endpoints for worker callbacks.

---

## 2. Command Endpoints (External - Orchestratord)

### [CAPI-9010] Start Worker
`POST /v2/workers/start`

Request:
```json
{
  "model_ref": "hf:author/repo@rev::file=models/model.Q4_K_M.gguf",
  "gpu_id": 0
}
```

Notes:
- Only normalized `model_ref` values are accepted (schemes: `hf:`, `file:`). Alias resolution (e.g., `"llama-7b"`) happens in orchestrator per `.specs/00_llama-orch.md §6.0`.

Response:
```json
{
  "worker_id": "worker-abc",
  "status": "starting"
}
```

### [CAPI-9011] Stop Worker
`POST /v2/workers/stop`

Request:
```json
{
  "worker_id": "worker-abc"
}
```

Response:
```json
{
  "worker_id": "worker-abc",
  "status": "draining"
}
```

### [CAPI-9012] Drain Worker
`POST /v2/workers/{id}/drain`

Mark worker as draining (stop accepting new jobs, finish active jobs).

---

## 3. State Query Endpoints (External - Orchestratord)

### [CAPI-9020] Get State
`GET /v2/state`

Response:
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

### [CAPI-9021] Get Worker
`GET /v2/workers/{id}`

Get single worker details.

---

## 4. Internal Callback Endpoints (Worker → Pool Manager)

### [CAPI-9030] Worker Ready Callback
`POST /v2/internal/workers/ready`

Request (from worker):
```json
{
  "worker_id": "worker-abc",
  "model_ref": "llama-7b",
  "vram_bytes": 16000000000,
  "uri": "http://localhost:8001"
}
```

Pool manager MUST:
1. Validate worker_id exists in registry with status `starting`
2. Update worker status to `ready`
3. Update GPU inventory `allocated_vram`
4. Return 200 OK

---

## 5. Error Handling

### [CAPI-9040] HTTP Error Codes
- `200` — Success
- `400` — Invalid request
- `404` — Resource not found (worker, GPU)
- `409` — Conflict (e.g., insufficient VRAM)
- `500` — Internal error

### [CAPI-9041] Error Response Format
```json
{
  "error": {
    "code": "INSUFFICIENT_VRAM",
    "message": "GPU 0 has only 8GB free, need 16GB",
    "retriable": false
  }
}
```

---

## 6. Dependencies

### [CAPI-9050] Required Crates
```toml
[dependencies]
axum = { workspace = true }
tokio = { workspace = true }
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
tracing = { workspace = true }
tower-http = { workspace = true, features = ["cors"] }
```

---

## 7. Traceability

**Code**: `bin/pool-managerd-crates/control-api/src/`  
**Tests**: `bin/pool-managerd-crates/control-api/tests/`  
**Parent**: `bin/pool-managerd/.specs/00_pool-managerd.md`  
**Used by**: `pool-managerd`  
**Spec IDs**: CAPI-9001 to CAPI-9050

---

**End of Specification**
