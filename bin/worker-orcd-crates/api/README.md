# worker-api

**HTTP/RPC server for worker-orcd — Plan/Commit/Ready/Execute endpoints**

`bin/worker-orcd-crates/api` — Exposes the worker-orcd HTTP API for pool-managerd and orchestratord. Handles request validation, authentication, and coordinates with other worker crates to provide Plan (feasibility), Commit (model sealing), Ready (health), and Execute (inference) endpoints.

---

## What This Crate Offers

`worker-api` is the **HTTP/RPC server** for worker-orcd. Here's what we offer to other services and crates:

### 🔒 Core Capabilities

**1. Four RPC Endpoints**
- **Plan** — Check model feasibility (VRAM capacity, MCD/ECP compatibility)
- **Commit** — Load model into VRAM and seal (cryptographic attestation)
- **Ready** — Health check with sealed shard attestation
- **Execute** — Run inference with SSE token streaming

**2. Security-First Design**
- Bearer token authentication (via `secrets-management`)
- Input validation on all requests (via `input-validation`)
- Audit logging for all security events (via `audit-logging`)
- Timing-safe token comparison (prevents timing attacks)
- No sensitive data in errors or logs

**3. Request Validation**
- Prompt length limits (max 100,000 chars)
- Max tokens validation (max 4096)
- Null byte rejection
- Path traversal prevention
- Hash format validation
- GPU device range validation

**4. SSE Streaming**
- Server-Sent Events for token streaming
- Event format: `started → token* → metrics? → end`
- Race-free cancellation (no tokens after cancel)
- Error events with structured error codes

**5. Integration Orchestration**
- Delegates to `model-loader` for validation
- Delegates to `vram-residency` for sealing
- Delegates to `capability-matcher` for MCD/ECP checking
- Delegates to `scheduler` for job tracking
- No business logic in API layer (thin coordination)

---

## What You Get

### For pool-managerd (Model Staging)

```rust
use reqwest::Client;

// 1. Plan: Check if model fits in VRAM
let plan_response = client
    .post("http://worker-gpu-0:9300/worker/plan")
    .bearer_auth(&worker_token)
    .json(&PlanRequest {
        model_ref: "hf:meta-llama/Llama-3.1-8B".to_string(),
        shard_layout: ShardLayout::Single,
        tp_degree: None,
    })
    .send()
    .await?
    .json::<PlanResponse>()
    .await?;

if !plan_response.feasible {
    return Err(PoolError::ModelTooLarge);
}

// 2. Commit: Load model into VRAM and seal
let model_bytes = download_model(&model_ref).await?;

let commit_response = client
    .post("http://worker-gpu-0:9300/worker/commit")
    .bearer_auth(&worker_token)
    .json(&CommitRequest {
        model_ref: "hf:meta-llama/Llama-3.1-8B".to_string(),
        shard_id: "shard-abc123".to_string(),
        shard_index: 0,
        model_bytes: Some(model_bytes),
        expected_digest: Some("a3f2c1...".to_string()),
    })
    .send()
    .await?
    .json::<CommitResponse>()
    .await?;

assert!(commit_response.sealed);
println!("Model sealed: {}", commit_response.handle.digest);

// 3. Ready: Verify worker is ready
let ready_response = client
    .get("http://worker-gpu-0:9300/worker/ready")
    .send()  // No auth required (health check)
    .await?
    .json::<ReadyResponse>()
    .await?;

if ready_response.ready {
    println!("Worker ready with {} shards", ready_response.handles.len());
}
```

**What you get**:
- ✅ Feasibility check before staging (Plan)
- ✅ Cryptographically sealed model in VRAM (Commit)
- ✅ Health check with seal attestation (Ready)
- ✅ Structured error responses (400, 401, 507)
- ✅ Audit trail for all operations

---

### For orchestratord (Inference Execution, Future)

```rust
use reqwest::Client;
use futures::StreamExt;

// Execute: Run inference with SSE streaming
let response = client
    .post("http://worker-gpu-0:9300/worker/execute")
    .bearer_auth(&worker_token)
    .json(&ExecuteRequest {
        handle_id: "shard-abc123".to_string(),
        prompt: "Once upon a time".to_string(),
        params: InferenceParams {
            max_tokens: 100,
            temperature: 0.7,
            seed: Some(42),
        },
    })
    .send()
    .await?;

// Stream SSE events
let mut stream = response.bytes_stream();
while let Some(chunk) = stream.next().await {
    let event = parse_sse_event(&chunk?)?;
    
    match event.event_type.as_str() {
        "started" => {
            println!("Job started: {}", event.data.job_id);
        }
        "token" => {
            print!("{}", event.data.t);  // Stream token
        }
        "end" => {
            println!("\nDone: {} tokens in {}ms", 
                event.data.tokens_out, 
                event.data.decode_time_ms
            );
            break;
        }
        "error" => {
            eprintln!("Error: {}", event.data.message);
            return Err(event.data.code);
        }
        _ => {}
    }
}
```

**What you get**:
- ✅ SSE token streaming (incremental output)
- ✅ Structured events (`started`, `token`, `metrics`, `end`, `error`)
- ✅ Race-free cancellation
- ✅ Seal re-verification before execution
- ✅ Audit trail for inference operations

---

### For worker-orcd binary (Embedding)

```rust
use worker_api::create_router;
use axum::Router;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize dependencies
    let audit_logger = AuditLogger::new(audit_config)?;
    let model_loader = ModelLoader::new();
    let vram_manager = VramManager::new(vram_config)?;
    let capability_matcher = CapabilityMatcher::new();
    let scheduler = Scheduler::new();
    
    // Create app state
    let state = Arc::new(WorkerState {
        worker_id: config.worker_id,
        audit_logger,
        model_loader,
        vram_manager,
        capability_matcher,
        scheduler,
    });
    
    // Create router with all endpoints
    let app = create_router()
        .with_state(state);
    
    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:9300").await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}
```

**What you get**:
- ✅ Complete HTTP server (Axum-based)
- ✅ All endpoints wired and ready
- ✅ Authentication middleware included
- ✅ Error handling and logging
- ✅ Graceful shutdown support

---

## API Reference

### Endpoint Summary

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/worker/plan` | POST | ✅ Required | Check model feasibility |
| `/worker/commit` | POST | ✅ Required | Load and seal model |
| `/worker/ready` | GET | ⚙️ Configurable | Health check |
| `/worker/execute` | POST | ✅ Required | Run inference (SSE) |
| `/worker/capacity` | GET | ✅ Required | Report VRAM/slots capacity |
| `/worker/drain` | POST | ✅ Required | Toggle admission (drain/undrain) |
| `/worker/evict` | POST | ✅ Required | Evict shard(s) to free VRAM |

---

### 1. Plan Endpoint

**POST /worker/plan**

Check if a model can be loaded given VRAM constraints and capability compatibility.

**Request**:
```json
{
  "model_ref": "hf:meta-llama/Llama-3.1-8B",
  "shard_layout": "single",
  "tp_degree": null
}
```

**Response (200 OK)**:
```json
{
  "feasible": true,
  "vram_required": 8589934592,
  "shard_plan": [
    {
      "shard_index": 0,
      "vram_bytes": 8589934592,
      "gpu_device": 0
    }
  ]
}
```

**Response (200 OK - Not Feasible)**:
```json
{
  "feasible": false,
  "vram_required": 85899345920,
  "shard_plan": [],
  "incompatibility_reason": "Model requires rope_llama, worker only supports rope_neox"
}
```

**Errors**:
- `401 Unauthorized` — Invalid Bearer token
- `400 Bad Request` — Invalid model_ref or parameters

**What it does**:
1. Validates request (model_ref, shard_layout, tp_degree)
2. Checks MCD vs ECP compatibility (via capability-matcher)
3. Checks VRAM availability (via vram-residency)
4. Returns feasibility verdict with shard plan

**Performance**: < 100ms (no heavy computation)

---

### 2. Commit Endpoint

**POST /worker/commit**

Load model bytes into VRAM and seal with cryptographic attestation.

**Request**:
```json
{
  "model_ref": "hf:meta-llama/Llama-3.1-8B",
  "shard_id": "shard-abc123",
  "shard_index": 0,
  "model_bytes": "<base64-encoded-bytes>",
  "expected_digest": "a3f2c1..."
}
```

**Alternative (file path)**:
```json
{
  "model_ref": "hf:meta-llama/Llama-3.1-8B",
  "shard_id": "shard-abc123",
  "model_path": "/var/lib/llorch/models/staging/model.gguf",
  "expected_digest": "a3f2c1..."
}
```

**Response (200 OK)**:
```json
{
  "handle": {
    "shard_id": "shard-abc123",
    "gpu_device": 0,
    "vram_bytes": 8589934592,
    "digest": "a3f2c1...",
    "sealed_at": "2025-10-02T21:54:00Z"
  },
  "sealed": true
}
```

**Errors**:
- `401 Unauthorized` — Invalid Bearer token
- `400 Bad Request` — Validation failed (bad GGUF, hash mismatch, invalid shard_id)
- `507 Insufficient Storage` — VRAM allocation failed (OOM)
- `500 Internal Server Error` — Seal verification failed
- `503 Service Unavailable` — Admission closed (draining)

**What it does**:
1. Validates request (shard_id, model_ref, digest format)
2. Validates model bytes (via model-loader)
3. Seals model in VRAM (via vram-residency)
4. Emits audit events (VramSealed, SealVerificationFailed)
5. Returns sealed handle

**Performance**: I/O bound (model loading + VRAM copy)

---

### 3. Ready Endpoint

**GET /worker/ready**

Health check with sealed shard attestation.

**Response (200 OK)**:
```json
{
  "ready": true,
  "handles": [
    {
      "shard_id": "shard-abc123",
      "gpu_device": 0,
      "vram_bytes": 8589934592,
      "digest": "a3f2c1...",
      "sealed_at": "2025-10-02T21:54:00Z"
    }
  ],
  "nccl_group_id": null
}
```

**Response (200 OK - Not Ready)**:
```json
{
  "ready": false,
  "handles": [],
  "nccl_group_id": null
}
```

**Errors**: None (always returns 200)

**What it does**:
1. Checks if worker has sealed shards
2. Verifies seal integrity
3. Returns ready status with handles

**Performance**: < 10ms (fast health check)

**Authentication**: Configurable (default: no auth for health checks)

---

### 4. Execute Endpoint

**POST /worker/execute**

Run inference with SSE token streaming.

**Request**:
```json
{
  "handle_id": "shard-abc123",
  "prompt": "Once upon a time",
  "params": {
    "max_tokens": 100,
    "temperature": 0.7,
    "seed": 42
  }
}
```

**Response (200 OK, text/event-stream)**:
```
event: started
data: {"job_id": "job-xyz789", "queue_position": 0, "predicted_start_ms": 50}

event: token
data: {"t": "Once", "i": 0}

event: token
data: {"t": " upon", "i": 1}

event: metrics
data: {"queue_depth": 0, "on_time_probability": 0.95}

event: end
data: {"tokens_out": 100, "decode_time_ms": 1500}
```

**Error Event**:
```
event: error
data: {"code": "SEAL_VERIFICATION_FAILED", "message": "Seal signature mismatch", "retriable": false}
```

**Errors**:
- `401 Unauthorized` — Invalid Bearer token
- `400 Bad Request` — Validation failed (prompt too long, null bytes, invalid max_tokens)
- `404 Not Found` — Handle not found
- `500 Internal Server Error` — Inference error (emitted as SSE error event)

**What it does**:
1. Validates request (handle_id, prompt, max_tokens)
2. Re-verifies seal signature (CRITICAL)
3. Runs inference (via scheduler)
4. Streams tokens via SSE
5. Emits audit events

**Performance**: Depends on inference (streaming)

---

### 5. Capacity Endpoint

**GET /worker/capacity**

Report current VRAM and slot capacity so pool-managerd can plan and control staging.

**Response (200 OK)**:
```json
{
  "worker_id": "worker-gpu-0",
  "gpu_device": 0,
  "vram_total_bytes": 25769803776,
  "vram_used_bytes": 8589934592,
  "vram_free_bytes": 17179869184,
  "slots_total": 1,
  "slots_free": 1,
  "draining": false
}
```

**Errors**:
- `401 Unauthorized` — Invalid Bearer token

**What it does**:
1. Queries `vram-residency` for VRAM metrics
2. Reports slot capacity (from scheduler)
3. Indicates whether admission is currently drained

**Performance**: < 10ms (cache device list, query VRAM on-demand)

---

### 6. Drain Control Endpoint

**POST /worker/drain**

Toggle admission of new Commit/Execute requests. Pool-managerd uses this to coordinate draining, upgrades, or eviction.

**Request**:
```json
{ "drain": true, "reason": "rolling-upgrade" }
```

**Response (200 OK)**:
```json
{ "draining": true }
```

**Effects**:
- While `draining = true`:
  - Plan remains available (for planning), but Commit returns `503 Service Unavailable`.
  - Execute returns `503` when no in-flight job is owned by the caller.

**Errors**:
- `401 Unauthorized` — Invalid Bearer token

---

### 7. Evict Endpoint

**POST /worker/evict**

Proactively free VRAM by unsealing one or more shards. Only callable by pool-managerd.

**Request**:
```json
{ "shard_ids": ["shard-abc123", "shard-def456"] }
```

**Response (200 OK)**:
```json
{ "evicted": 2, "not_found": ["shard-def456"], "busy": [] }
```

**Errors**:
- `401 Unauthorized` — Invalid Bearer token
- `409 Conflict` — Shard currently executing; cannot evict

**What it does**:
1. Validates identifiers
2. Attempts to drop sealed shards via `vram-residency`
3. Returns per-shard outcome summary

---

## Security Guarantees

### TIER 2 Security Configuration

```rust
// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::integer_arithmetic)]
#![warn(clippy::missing_errors_doc)]
```

**What this means**:
- ✅ Never panics in request handlers (DoS prevention)
- ✅ No unwrap/expect (explicit error handling)
- ✅ Careful with arithmetic (overflow checks)
- ✅ All errors documented

---

### Authentication

**Bearer Token Authentication** (via `secrets-management`):
```rust
use secrets_management::Secret;

// Load token from file
let expected_token = Secret::load_from_file("/etc/llorch/secrets/worker-token")?;

// Timing-safe verification
if !expected_token.verify(bearer_token) {
    return Err(StatusCode::UNAUTHORIZED);
}
```

**Properties**:
- ✅ Timing-safe comparison (prevents timing attacks)
- ✅ Token never logged (use fingerprints)
- ✅ Audit events emitted (AuthSuccess, AuthFailure)
- ✅ Configurable bypass for `/ready` endpoint

---

### Input Validation

**All user inputs validated** (via `input-validation`):
```rust
use input_validation::{validate_identifier, validate_prompt, validate_range, validate_hex_string};

// Validate shard_id
validate_identifier(&body.shard_id, 256)?;

// Validate prompt (SECURITY_AUDIT #12)
validate_prompt(&body.prompt, 100_000)?;

// Validate max_tokens
validate_range(body.params.max_tokens, 1, 4096)?;

// Validate digest format
validate_hex_string(&body.expected_digest, 64)?;
```

**Prevents**:
- ❌ Path traversal: `"shard-../etc/passwd"`
- ❌ Null byte injection: `"prompt\0null"`
- ❌ VRAM exhaustion: 10MB prompt
- ❌ Integer overflow: `max_tokens: usize::MAX`

---

### Audit Logging

**All security events logged** (via `audit-logging`):
```rust
use audit_logging::{AuditLogger, AuditEvent};

// Authentication events
audit_logger.emit(AuditEvent::AuthSuccess { ... }).await.ok();
audit_logger.emit(AuditEvent::AuthFailure { ... }).await.ok();

// VRAM operations (CRITICAL)
audit_logger.emit(AuditEvent::VramSealed { ... }).await?;
audit_logger.emit(AuditEvent::SealVerificationFailed { ... }).await?;

// Security incidents
audit_logger.emit(AuditEvent::PolicyViolation { ... }).await?;
```

**Properties**:
- ✅ Tamper-evident audit trail
- ✅ Critical events propagate errors
- ✅ Non-critical events log but continue
- ✅ No sensitive data in logs

---

## Error Handling

### WorkerError Enum

```rust
pub enum WorkerError {
    InvalidInput(String),
    Unauthorized,
    ModelNotFound(String),
    IntegrityViolation(String),
    ModelTooLarge { actual: usize, max: usize },
    InvalidModel(String),
    SecurityViolation(String),
    VramAllocationFailed(String),
    SealVerificationFailed(String),
    InferenceError(String),
    InternalError(String),
}
```

**HTTP status mapping**:
- `InvalidInput` → 400 Bad Request
- `Unauthorized` → 401 Unauthorized
- `ModelNotFound` → 404 Not Found
- `IntegrityViolation` → 400 Bad Request
- `ModelTooLarge` → 413 Payload Too Large
- `InvalidModel` → 400 Bad Request
- `SecurityViolation` → 403 Forbidden
- `VramAllocationFailed` → 507 Insufficient Storage
- `SealVerificationFailed` → 500 Internal Server Error
- `InferenceError` → 500 Internal Server Error
- `InternalError` → 500 Internal Server Error

---

### Error Response Format

**JSON error response**:
```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Prompt length exceeds maximum (100,000 chars)",
    "details": {
      "field": "prompt",
      "actual_length": 150000,
      "max_length": 100000
    }
  }
}
```

**Properties**:
- ✅ Structured (not plain text)
- ✅ Stable error codes (for client parsing)
- ✅ Actionable messages
- ✅ No sensitive data (no tokens, VRAM pointers)

---

## Dependencies

### Production Dependencies

```toml
[dependencies]
# HTTP server
axum = { workspace = true }
tokio = { workspace = true }

# Serialization
serde = { workspace = true }

# Shared crates
input-validation = { path = "../../shared-crates/input-validation" }
secrets-management = { path = "../../shared-crates/secrets-management" }
audit-logging = { path = "../../shared-crates/audit-logging" }

# Worker crates
vram-residency = { path = "../vram-residency" }
model-loader = { path = "../model-loader" }
capability-matcher = { path = "../capability-matcher" }
scheduler = { path = "../scheduler" }

# Logging
tracing.workspace = true
```

**Why these dependencies?**
- `axum` — Modern, ergonomic HTTP framework (Tokio ecosystem)
- `input-validation` — Centralized security boundary (TIER 2)
- `secrets-management` — Token authentication (TIER 1)
- `audit-logging` — Security audit trail (TIER 1)
- Worker crates — Business logic delegation

---

## What This Crate Does NOT Do

### NOT a Model Loader

**worker-api does NOT**:
- ❌ Validate GGUF format (delegates to model-loader)
- ❌ Compute SHA-256 hashes (delegates to model-loader)
- ❌ Load models from disk (delegates to model-loader)

**That's the job of**: `model-loader`

---

### NOT a VRAM Manager

**worker-api does NOT**:
- ❌ Allocate VRAM (delegates to vram-residency)
- ❌ Seal shards (delegates to vram-residency)
- ❌ Verify seal signatures (delegates to vram-residency)

**That's the job of**: `vram-residency`

---

### NOT a Scheduler

**worker-api does NOT**:
- ❌ Implement scheduling logic (delegates to scheduler)
- ❌ Manage job queues (delegates to scheduler)
- ❌ Decide execution order (delegates to scheduler)

**That's the job of**: `scheduler`

---

## Specifications

Implements requirements from:
- **WORKER-4200 to WORKER-4203**: Endpoint authentication
- **WORKER-4210 to WORKER-4214**: Plan endpoint
- **WORKER-4220 to WORKER-4227**: Commit endpoint
- **WORKER-4230 to WORKER-4233**: Ready endpoint
- **WORKER-4240 to WORKER-4248**: Execute endpoint
- **WORKER-4250 to WORKER-4253**: SSE streaming security

See `.specs/` for full requirements:
- `00_api.md` — Functional specification (WORKER-4xxx)
- `10_expectations.md` — Consumer expectations

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p worker-api

# Run specific test
cargo test -p worker-api test_plan_endpoint
```

**Test coverage**:
- ✅ Request validation (all fields)
- ✅ Authentication (valid/invalid tokens)
- ✅ Error handling (all error variants)
- ✅ SSE event formatting
- ✅ Response serialization

---

### Integration Tests

```bash
# Run integration tests
cargo test -p worker-api --test integration
```

**Test coverage**:
- ✅ End-to-end Plan → Commit → Ready → Execute flow
- ✅ Authentication middleware integration
- ✅ Audit logging integration
- ✅ Error propagation from dependent crates

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Security Tier**: TIER 2 (High-importance)
- **Priority**: P0 (blocking for worker-orcd)

---

## Roadmap

### Phase 1: M0 Essentials (Week 1)
- ✅ Basic router with `/ready` stub
- ⬜ Request/response types (Plan, Commit, Ready, Execute)
- ⬜ Input validation integration
- ⬜ Error types and HTTP mapping
- ⬜ Basic unit tests

### Phase 2: Core Endpoints (Week 1-2)
- ⬜ Plan endpoint implementation
- ⬜ Commit endpoint implementation
- ⬜ Ready endpoint implementation
- ⬜ Execute endpoint implementation (stub)
- ⬜ Integration with model-loader

### Phase 3: Security (Week 2)
- ⬜ Bearer token authentication middleware
- ⬜ Audit logging integration
- ⬜ Integration with vram-residency
- ⬜ Integration with capability-matcher

### Phase 4: Production Hardening (Week 3+)
- ⬜ SSE streaming implementation
- ⬜ Integration with scheduler
- ⬜ Error handling polish
- ⬜ Integration tests
- ⬜ Performance optimization

---

## Contributing

**Before implementing**:
1. Read `.specs/00_api.md` — Functional specification
2. Read `.specs/10_expectations.md` — Consumer expectations
3. Follow TIER 2 Clippy configuration (no panics, no unwrap in handlers)

**Testing requirements**:
- Unit tests for all endpoints
- Security tests for authentication and validation
- Integration tests for end-to-end flows
- Error handling tests for all error variants

---

## For Questions

See:
- `.specs/` — Complete specifications
- `.docs/ARCHITECTURE_CHANGE_PLAN.md` — Phase 3, Task Group 1
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Issue #1 (worker-orcd endpoint auth)
- `bin/worker-orcd/.specs/00_worker-orcd.md` — Parent specification