# Agentic API SPEC — Standard Orchestrator HTTP API (AAPI-15xxx)

**Status**: Draft  
**Applies to**: `bin/orchestratord-crates/agentic-api/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `agentic-api` crate provides the standard HTTP API for orchestratord in home/single-node mode. This is the user-facing API for customers NOT going through the marketplace.

**Why it exists:**
- Provide standard orchestrator API for home users
- Direct customer access to orchestrator (no marketplace intermediary)
- Agent-friendly API design (streaming, cancellation, session management)

**What it does:**
- Expose HTTP endpoints for task submission, streaming, cancellation
- Provide session management (create, query, delete sessions)
- Expose catalog management (register models, query metadata)
- Provide observability endpoints (metrics, health)

**What it does NOT do:**
- ❌ Route to other orchestrators (platform-api does this)
- ❌ Handle billing (platform-api does this)
- ❌ Enforce multi-tenancy quotas (platform-api does this)

**Distinction from platform-api:**
- **agentic-api**: Direct customer → orchestrator (home/single-node)
- **platform-api**: Customer → marketplace → provider orchestrators (federation)

---

## 1. Core Responsibilities

### [AAPI-15001] Task Submission
The crate MUST expose `POST /v2/tasks` for inference requests.

### [AAPI-15002] SSE Streaming
The crate MUST stream results via Server-Sent Events.

### [AAPI-15003] Session Management
The crate MUST expose session endpoints (create, query, delete).

### [AAPI-15004] Catalog Management
The crate MUST expose catalog endpoints (register models, query metadata).

---

## 2. Task Endpoints

### [AAPI-15010] Submit Task
`POST /v2/tasks`

Request:
```json
{
  "session_id": "sess-abc",
  "model": "llama-3.1-8b",
  "prompt": "Hello world",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42,
  "priority": "interactive"
}
```

Response (202 Accepted):
```json
{
  "job_id": "job-xyz",
  "status": "queued",
  "queue_position": 2,
  "predicted_start_ms": 50,
  "events_url": "/v2/tasks/job-xyz/events"
}
```

### [AAPI-15011] Stream Events
`GET /v2/tasks/{job_id}/events`

SSE stream with events:
- `queued` — Job in queue
- `started` — Inference started
- `token` — Token generated
- `metrics` — Intermediate metrics
- `end` — Inference complete
- `error` — Error occurred

### [AAPI-15012] Cancel Task
`POST /v2/tasks/{job_id}/cancel`

Response:
```json
{
  "job_id": "job-xyz",
  "status": "cancelled"
}
```

---

## 3. Session Endpoints

### [AAPI-15020] Create Session
`POST /v2/sessions`

Request:
```json
{
  "session_id": "sess-abc",
  "token_budget": 10000,
  "context_mode": "stateless"
}
```

Response:
```json
{
  "session_id": "sess-abc",
  "token_budget": 10000,
  "tokens_used": 0,
  "tokens_remaining": 10000,
  "created_at": "2025-10-03T00:00:00Z"
}
```

### [AAPI-15021] Get Session
`GET /v2/sessions/{id}`

Response:
```json
{
  "session_id": "sess-abc",
  "token_budget": 10000,
  "tokens_used": 542,
  "tokens_remaining": 9458,
  "jobs_total": 5,
  "created_at": "2025-10-03T00:00:00Z",
  "last_activity": "2025-10-03T00:05:00Z"
}
```

### [AAPI-15022] Delete Session
`DELETE /v2/sessions/{id}`

Response:
```json
{
  "session_id": "sess-abc",
  "deleted": true
}
```

---

## 4. Catalog Endpoints

### [AAPI-15030] List Models
`GET /v2/catalog/models`

Response:
```json
{
  "models": [
    {
      "model_id": "llama-3.1-8b",
      "name": "Llama 3.1 8B Instruct",
      "context_length": 8192,
      "vocab_size": 128256,
      "lifecycle_state": "Active",
      "size_bytes": 8000000000
    }
  ]
}
```

### [AAPI-15031] Get Model
`GET /v2/catalog/models/{id}`

Response:
```json
{
  "model_id": "llama-3.1-8b",
  "name": "Llama 3.1 8B Instruct",
  "context_length": 8192,
  "vocab_size": 128256,
  "architecture": "llama",
  "quantization": "q4_0",
  "lifecycle_state": "Active",
  "size_bytes": 8000000000,
  "checksum": "sha256:abc123...",
  "created_at": "2025-10-01T00:00:00Z"
}
```

### [AAPI-15032] Register Model
`POST /v2/catalog/models`

Request:
```json
{
  "model_id": "llama-3.1-8b",
  "name": "Llama 3.1 8B Instruct",
  "context_length": 8192,
  "vocab_size": 128256,
  "file_path": "/models/llama-3.1-8b.gguf"
}
```

Response:
```json
{
  "model_id": "llama-3.1-8b",
  "status": "registered"
}
```

### [AAPI-15033] Update Model State
`POST /v2/catalog/models/{id}/state`

Request:
```json
{
  "lifecycle_state": "Retired"
}
```

Response:
```json
{
  "model_id": "llama-3.1-8b",
  "lifecycle_state": "Retired"
}
```

---

## 5. Meta Endpoints

### [AAPI-15040] Get Capabilities
`GET /v2/meta/capabilities`

Response:
```json
{
  "version": "0.1.0",
  "features": {
    "determinism": true,
    "streaming": true,
    "sessions": true,
    "cancellation": true
  },
  "limits": {
    "max_context_length": 16384,
    "max_batch_size": 1,
    "queue_capacity": 100
  }
}
```

### [AAPI-15041] Health Check
`GET /v2/health`

Response:
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "pools": [
    {
      "pool_id": "pool-1",
      "status": "healthy",
      "gpus": 4,
      "workers": 2
    }
  ]
}
```

---

## 6. Admin Endpoints

### [AAPI-15050] Drain Pool
`POST /v2/pools/{id}/drain`

Mark pool as draining (stop accepting new jobs, finish active jobs).

### [AAPI-15051] Reload Pool
`POST /v2/pools/{id}/reload`

Reload pool configuration.

---

## 7. Error Handling

### [AAPI-15060] Error Response Format
```json
{
  "error": {
    "code": "QUEUE_FULL",
    "message": "Queue capacity reached (100/100)",
    "retriable": true,
    "retry_after_ms": 500
  }
}
```

### [AAPI-15061] HTTP Status Codes
- `200` — Success
- `202` — Accepted (task queued)
- `400` — Bad request
- `404` — Not found
- `429` — Too many requests (queue full, rate limit)
- `500` — Internal error
- `503` — Service unavailable

---

## 8. SSE Event Format

### [AAPI-15070] Event Types

**queued**:
```json
{
  "event": "queued",
  "job_id": "job-xyz",
  "queue_position": 2,
  "predicted_start_ms": 50
}
```

**started**:
```json
{
  "event": "started",
  "job_id": "job-xyz",
  "worker_id": "worker-abc",
  "started_at": "2025-10-03T00:00:00Z"
}
```

**token**:
```json
{
  "event": "token",
  "t": "Hello",
  "i": 0
}
```

**metrics**:
```json
{
  "event": "metrics",
  "tokens_out": 42,
  "decode_time_ms": 1234
}
```

**end**:
```json
{
  "event": "end",
  "job_id": "job-xyz",
  "tokens_out": 100,
  "decode_time_ms": 2500
}
```

**error**:
```json
{
  "event": "error",
  "code": "WORKER_TIMEOUT",
  "message": "Worker did not respond within timeout"
}
```

---

## 9. Authentication

### [AAPI-15080] Bearer Token Auth
The crate MAY support Bearer token authentication:
```
Authorization: Bearer <token>
```

### [AAPI-15081] Home Profile
For home use, auth MAY be disabled (localhost only).

---

## 10. Dependencies

### [AAPI-15090] Required Crates
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

## 11. Traceability

**Code**: `bin/orchestratord-crates/agentic-api/src/`  
**Tests**: `bin/orchestratord-crates/agentic-api/tests/`  
**Parent**: `bin/orchestratord/.specs/00_orchestratord.md`  
**Contracts**: `/contracts/openapi/*.yaml`  
**Used by**: `orchestratord` (default/home mode)  
**Spec IDs**: AAPI-15001 to AAPI-15090

---

**End of Specification**
