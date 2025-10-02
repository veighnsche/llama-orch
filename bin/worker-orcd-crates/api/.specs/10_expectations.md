# Worker API — Consumer Expectations

**Status**: Draft  
**Purpose**: Documents what other crates expect from `worker-api`  
**Last Updated**: 2025-10-02

---

## 0. Overview

This document catalogs the expectations and dependencies that other worker-orcd crates and services have on `worker-api`. The worker-api crate implements the HTTP/RPC server for worker-orcd, exposing Plan/Commit/Ready/Execute endpoints.

**Core responsibility**: Accept HTTP requests from pool-managerd and orchestratord, validate inputs, coordinate with other worker crates, return responses.

**Consuming services**:
- `pool-managerd` — Calls Plan/Commit/Ready endpoints for model staging
- `orchestratord` — Calls Execute endpoint for inference (future, via pool-managerd proxy)
- `worker-orcd` binary — Embeds this crate as HTTP server

**Dependent crates** (worker-api calls these):
- `vram-residency` — For model sealing and VRAM operations
- `model-loader` — For model validation before commit
- `capability-matcher` — For MCD/ECP checking in Plan
- `scheduler` — For job state tracking
- `input-validation` — For request validation
- `secrets-management` — For Bearer token authentication
- `audit-logging` — For security event emission

---

## 1. Core Principle: RPC Server Only

### 1.1 What worker-api IS (EXP-API-1001)

**An HTTP/RPC server**:
- Exposes Plan/Commit/Ready/Execute endpoints
- Validates all incoming requests
- Authenticates requests via Bearer token
- Coordinates with other worker crates
- Returns structured responses (JSON, SSE)

**No business logic**:
- Does NOT implement model loading (delegates to model-loader)
- Does NOT implement VRAM sealing (delegates to vram-residency)
- Does NOT implement capability matching (delegates to capability-matcher)
- Does NOT implement job scheduling (delegates to scheduler)

### 1.2 What worker-api IS NOT (EXP-API-1002)

**NOT a model manager**:
- Does NOT decide which models to load
- Does NOT track model lifecycle
- Does NOT manage model cache

**NOT a VRAM manager**:
- Does NOT allocate VRAM directly
- Does NOT seal shards directly
- Does NOT verify residency directly

**NOT a scheduler**:
- Does NOT implement scheduling logic
- Does NOT manage job queues
- Does NOT decide execution order

---

## 2. pool-managerd Expectations

### 2.1 Plan Endpoint (EXP-POOL-2001)

**Required by**: `pool-managerd` (model staging phase)

**Expected usage**:
```rust
// pool-managerd side
let client = reqwest::Client::new();
let response = client
    .post("http://worker-gpu-0:9300/worker/plan")
    .bearer_auth(&worker_token)
    .json(&PlanRequest {
        model_ref: "hf:meta-llama/Llama-3.1-8B".to_string(),
        shard_layout: ShardLayout::Single,
        tp_degree: None,
    })
    .send()
    .await?;

let plan: PlanResponse = response.json().await?;

if !plan.feasible {
    return Err(PoolError::ModelTooLarge);
}

// Proceed with commit...
```

**Expectations**:
- **EXP-POOL-2001-R1**: Accept `model_ref`, `shard_layout`, `tp_degree`
- **EXP-POOL-2001-R2**: Return `feasible: bool`, `vram_required: usize`, `shard_plan: Vec<ShardPlan>`
- **EXP-POOL-2001-R3**: Check MCD vs ECP compatibility (via capability-matcher)
- **EXP-POOL-2001-R4**: Check VRAM availability (via vram-residency)
- **EXP-POOL-2001-R5**: Return 401 if Bearer token invalid
- **EXP-POOL-2001-R6**: Return 400 if request validation fails
- **EXP-POOL-2001-R7**: Response time < 100ms (no heavy computation)

---

### 2.2 Commit Endpoint (EXP-POOL-2002)

**Required by**: `pool-managerd` (model staging phase)

**Expected usage**:
```rust
// pool-managerd side
let model_bytes = download_model(&model_ref).await?;

let response = client
    .post("http://worker-gpu-0:9300/worker/commit")
    .bearer_auth(&worker_token)
    .json(&CommitRequest {
        model_ref: "hf:meta-llama/Llama-3.1-8B".to_string(),
        shard_id: "shard-abc123".to_string(),
        shard_index: 0,
        model_bytes: Some(model_bytes),  // Or model_path
        expected_digest: Some("a3f2c1...".to_string()),
    })
    .send()
    .await?;

let commit: CommitResponse = response.json().await?;

assert!(commit.sealed);
assert_eq!(commit.handle.digest, "a3f2c1...");
```

**Expectations**:
- **EXP-POOL-2002-R1**: Accept `model_ref`, `shard_id`, `shard_index`, `model_bytes` OR `model_path`, `expected_digest`
- **EXP-POOL-2002-R2**: Validate model bytes (via model-loader)
- **EXP-POOL-2002-R3**: Seal model in VRAM (via vram-residency)
- **EXP-POOL-2002-R4**: Return sealed `ModelShardHandle` with `sealed: true`
- **EXP-POOL-2002-R5**: Return 401 if Bearer token invalid
- **EXP-POOL-2002-R6**: Return 400 if validation fails (bad GGUF, hash mismatch)
- **EXP-POOL-2002-R7**: Return 507 if VRAM allocation fails (OOM)
- **EXP-POOL-2002-R8**: Emit audit events (VramSealed, SealVerificationFailed)
- **EXP-POOL-2002-R9**: Return 503 Service Unavailable with stable code `ADMISSION_CLOSED` when worker is draining

---

### 2.3 Ready Endpoint (EXP-POOL-2003)

**Required by**: `pool-managerd` (health checks)

**Expected usage**:
```rust
// pool-managerd side
let response = client
    .get("http://worker-gpu-0:9300/worker/ready")
    .send()  // No auth required (health check)
    .await?;

let ready: ReadyResponse = response.json().await?;

if ready.ready {
    println!("Worker ready with {} shards", ready.handles.len());
    for handle in ready.handles {
        println!("  - Shard {}: {} bytes", handle.shard_id, handle.vram_bytes);
    }
}
```

**Expectations**:
- **EXP-POOL-2003-R1**: Return `ready: bool`, `handles: Vec<ModelShardHandle>`, `nccl_group_id: Option<String>`
- **EXP-POOL-2003-R2**: Return `ready: false` if no model loaded or seal verification fails
- **EXP-POOL-2003-R3**: MAY be unauthenticated (configurable, default: no auth for health checks)
- **EXP-POOL-2003-R4**: Response time < 10ms (fast health check)
- **EXP-POOL-2003-R5**: Never return 5xx (always return 200 with ready: false)

---

### 2.4 Capacity Endpoint (EXP-POOL-2004)

**Purpose**: Allow pool-managerd to query VRAM and slot capacity to plan staging/eviction.

**Usage**:
```rust
let cap: CapacityResponse = client
    .get("http://worker-gpu-0:9300/worker/capacity")
    .bearer_auth(&worker_token)
    .send()
    .await?
    .json()
    .await?;
```

**Expectations**:
- **EXP-POOL-2004-R1**: Response includes `worker_id`, `gpu_device`, `vram_total_bytes`, `vram_used_bytes`, `vram_free_bytes`, `slots_total`, `slots_free`, `draining`.
- **EXP-POOL-2004-R2**: Auth required; 401 on invalid token.
- **EXP-POOL-2004-R3**: Fast (< 10ms), suitable for periodic polling.

---

### 2.5 Drain Control (EXP-POOL-2005)

**Purpose**: Pool-managerd toggles admission for maintenance or rolling upgrades.

**Usage**:
```rust
let resp: DrainResponse = client
    .post("http://worker-gpu-0:9300/worker/drain")
    .bearer_auth(&worker_token)
    .json(&DrainRequest { drain: true, reason: Some("rolling-upgrade".into()) })
    .send().await?
    .json().await?;
```

**Expectations**:
- **EXP-POOL-2005-R1**: When `drain = true`, Commit/Execute MUST return `503` with `ADMISSION_CLOSED`.
- **EXP-POOL-2005-R2**: Plan and Capacity remain available while draining.
- **EXP-POOL-2005-R3**: When `drain = false`, admission reopens immediately.

---

### 2.6 Evict (EXP-POOL-2006)

**Purpose**: Free VRAM by unsealing shards under pool-managerd control.

**Usage**:
```rust
let ev: EvictResponse = client
    .post("http://worker-gpu-0:9300/worker/evict")
    .bearer_auth(&worker_token)
    .json(&EvictRequest { shard_ids: vec!["shard-abc123".into()] })
    .send().await?
    .json().await?;
```

**Expectations**:
- **EXP-POOL-2006-R1**: Returns per-shard outcomes; non-existent shards in `not_found`.
- **EXP-POOL-2006-R2**: Shards currently executing MUST cause `409 Conflict` and be listed in `busy`.
- **EXP-POOL-2006-R3**: Auth required; 401 on invalid token.

---

## 3. orchestratord Expectations (Future)

### 3.1 Execute Endpoint (EXP-ORCH-3001)

**Required by**: `orchestratord` (via pool-managerd proxy, future)

**Expected usage**:
```rust
// orchestratord side (via pool-managerd)
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
        "token" => println!("Token: {}", event.data.t),
        "end" => println!("Done: {} tokens", event.data.tokens_out),
        "error" => return Err(event.data.message),
        _ => {}
    }
}
```

**Expectations**:
- **EXP-ORCH-3001-R1**: Accept `handle_id`, `prompt`, `params`
- **EXP-ORCH-3001-R2**: Validate prompt length (max 100,000 chars)
- **EXP-ORCH-3001-R3**: Validate max_tokens (max 4096)
- **EXP-ORCH-3001-R4**: Reject prompts with null bytes
- **EXP-ORCH-3001-R5**: Re-verify seal signature before execution
- **EXP-ORCH-3001-R6**: Stream SSE events: `started`, `token`, `metrics`, `end`, `error`
- **EXP-ORCH-3001-R7**: Return 401 if Bearer token invalid
- **EXP-ORCH-3001-R8**: Return 400 if validation fails
- **EXP-ORCH-3001-R9**: Emit `event: error` if inference fails (don't close stream abruptly)
- **EXP-ORCH-3001-R10**: Race-free cancellation (no tokens after cancel)

---

## 4. input-validation Integration

### 4.1 Request Validation (EXP-VALID-4001)

**Required by**: All endpoints

**Expected usage**:
```rust
use input_validation::{validate_identifier, validate_prompt, validate_range, validate_hex_string};

async fn commit_handler(
    Json(body): Json<CommitRequest>
) -> Result<Json<CommitResponse>> {
    // Validate shard_id
    validate_identifier(&body.shard_id, 256)
        .map_err(|e| WorkerError::InvalidInput(e.to_string()))?;
    
    // Validate expected_digest (if provided)
    if let Some(digest) = &body.expected_digest {
        validate_hex_string(digest, 64)
            .map_err(|e| WorkerError::InvalidInput(e.to_string()))?;
    }
    
    // Validate shard_index (if TP)
    if let Some(index) = body.shard_index {
        validate_range(index, 0, 100)  // Max 100 shards
            .map_err(|e| WorkerError::InvalidInput(e.to_string()))?;
    }
    
    // Proceed...
}

async fn execute_handler(
    Json(body): Json<ExecuteRequest>
) -> Result<impl IntoResponse> {
    // Validate handle_id
    validate_identifier(&body.handle_id, 256)
        .map_err(|e| WorkerError::InvalidInput(e.to_string()))?;
    
    // Validate prompt (SECURITY_AUDIT #12)
    validate_prompt(&body.prompt, 100_000)
        .map_err(|e| WorkerError::InvalidInput(e.to_string()))?;
    
    // Validate max_tokens
    validate_range(body.params.max_tokens, 1, 4096)
        .map_err(|e| WorkerError::InvalidInput(e.to_string()))?;
    
    // Proceed...
}
```

**Expectations**:
- **EXP-VALID-4001-R1**: Validate all user-controlled inputs
- **EXP-VALID-4001-R2**: Fail fast on invalid input (before calling other crates)
- **EXP-VALID-4001-R3**: Return 400 with specific error message
- **EXP-VALID-4001-R4**: Never pass unvalidated input to other crates

---

## 5. secrets-management Integration

### 5.1 Bearer Token Authentication (EXP-AUTH-5001)

**Required by**: All endpoints except `/ready` (configurable)

**Expected usage**:
```rust
use secrets_management::Secret;
use axum::{middleware::Next, http::Request, response::Response};

async fn auth_middleware<B>(
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    // Load expected token
    let expected_token = Secret::load_from_file("/etc/llorch/secrets/worker-token")
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Extract Bearer token from header
    let auth_header = req.headers()
        .get("Authorization")
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    let bearer_token = auth_header
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    // Timing-safe verification
    if !expected_token.verify(bearer_token) {
        // Audit failure
        emit_audit_event(AuditEvent::AuthFailure { ... }).await.ok();
        return Err(StatusCode::UNAUTHORIZED);
    }
    
    // Audit success
    emit_audit_event(AuditEvent::AuthSuccess { ... }).await.ok();
    
    Ok(next.run(req).await)
}
```

**Expectations**:
- **EXP-AUTH-5001-R1**: Use `secrets-management::Secret` for token loading
- **EXP-AUTH-5001-R2**: Use timing-safe verification (`Secret::verify()`)
- **EXP-AUTH-5001-R3**: Never log full tokens (use fingerprints)
- **EXP-AUTH-5001-R4**: Emit audit events (AuthSuccess, AuthFailure)
- **EXP-AUTH-5001-R5**: Return 401 on auth failure (no details in response)
- **EXP-AUTH-5001-R6**: Support configurable auth bypass for `/ready` endpoint

---

## 6. audit-logging Integration

### 6.1 Security Event Emission (EXP-AUDIT-6001)

**Required by**: All endpoints

**Expected usage**:
```rust
use audit_logging::{AuditLogger, AuditEvent};

async fn commit_handler(
    State(state): State<Arc<WorkerState>>,
    Json(body): Json<CommitRequest>
) -> Result<Json<CommitResponse>> {
    // Validate and load model
    let model_bytes = model_loader.load_and_validate(...)?;
    
    // Seal in VRAM
    let shard = vram_manager.seal_model(&model_bytes, gpu_device)?;
    
    // Audit VRAM seal (CRITICAL)
    state.audit_logger.emit(AuditEvent::VramSealed {
        timestamp: Utc::now(),
        shard_id: shard.shard_id.clone(),
        gpu_device: shard.gpu_device,
        vram_bytes: shard.vram_bytes,
        digest: shard.digest.clone(),
        worker_id: state.worker_id.clone(),
    }).await?;  // Propagate error (critical event)
    
    Ok(Json(CommitResponse {
        handle: shard.into(),
        sealed: true,
    }))
}
```

**Expectations**:
- **EXP-AUDIT-6001-R1**: Emit `AuthSuccess` and `AuthFailure` from auth middleware
- **EXP-AUDIT-6001-R2**: Emit `VramSealed` on successful commit
- **EXP-AUDIT-6001-R3**: Emit `SealVerificationFailed` on seal verification failure (CRITICAL)
- **EXP-AUDIT-6001-R4**: Emit `PolicyViolation` if security policy violated
- **EXP-AUDIT-6001-R5**: Propagate audit errors for critical events (seal verification)
- **EXP-AUDIT-6001-R6**: Log but continue for non-critical events (auth success)

---

### 7. vram-residency Integration

### 7.1 Model Sealing (EXP-VRAM-7001)

**Required by**: Commit endpoint

{{ ... }}
**Expected usage**:
```rust
use vram_residency::VramManager;

async fn commit_handler(
    State(state): State<Arc<WorkerState>>,
    Json(body): Json<CommitRequest>
) -> Result<Json<CommitResponse>> {
    // Validate model bytes (via model-loader)
    let model_bytes = if let Some(path) = body.model_path {
        state.model_loader.load_and_validate(LoadRequest {
            model_path: &path,
            expected_hash: body.expected_digest.as_deref(),
            max_size: 100_000_000_000,
        })?
    } else if let Some(bytes) = body.model_bytes {
        state.model_loader.validate_bytes(&bytes, body.expected_digest.as_deref())?;
        bytes
    } else {
        return Err(WorkerError::InvalidRequest("No model source".into()));
    };
    
    // Seal in VRAM (via vram-residency). GPU device is implied by the worker process.
    let shard = state.vram_manager.seal_model(
        body.shard_id,
        &model_bytes,
    ).await?;
    
    // Return sealed handle
    Ok(Json(CommitResponse {
        handle: shard.into(),
        sealed: true,
    }))
}
```

**Expectations**:
- **EXP-VRAM-7001-R1**: Pass validated model bytes to vram-residency
- **EXP-VRAM-7001-R2**: Receive sealed `ModelShardHandle` with `sealed: true`
- **EXP-VRAM-7001-R3**: Handle VRAM allocation failures (return 507)
- **EXP-VRAM-7001-R4**: Handle seal verification failures (return 500, emit audit event)

---

### 7.2 Seal Verification (EXP-VRAM-7002)

**Required by**: Execute endpoint

**Expected usage**:
```rust
async fn execute_handler(
    State(state): State<Arc<WorkerState>>,
    Json(body): Json<ExecuteRequest>
) -> Result<impl IntoResponse> {
    // Get sealed shard
    let shard = state.vram_manager.get_shard(&body.handle_id)?;
    
    // Re-verify seal before execution (CRITICAL)
    state.vram_manager.verify_seal(&shard).await?;
    
    // Proceed with inference...
}
```

**Expectations**:
- **EXP-VRAM-7002-R1**: Re-verify seal before every execution
- **EXP-VRAM-7002-R2**: Fail fast if seal verification fails
- **EXP-VRAM-7002-R3**: Emit audit event on verification failure

---

## 8. model-loader Integration

### 8.1 Model Validation (EXP-LOADER-8001)

**Required by**: Commit endpoint

**Expected usage**:
```rust
use model_loader::{ModelLoader, LoadRequest};

async fn commit_handler(
    State(state): State<Arc<WorkerState>>,
    Json(body): Json<CommitRequest>
) -> Result<Json<CommitResponse>> {
    let loader = ModelLoader::new();
    
    // Two paths: file or bytes
    let model_bytes = if let Some(path) = body.model_path {
        // Load from filesystem
        loader.load_and_validate(LoadRequest {
            model_path: &path,
            expected_hash: body.expected_digest.as_deref(),
            max_size: 100_000_000_000,
        })?
    } else if let Some(bytes) = body.model_bytes {
        // Validate from memory
        loader.validate_bytes(&bytes, body.expected_digest.as_deref())?;
        bytes
    } else {
        return Err(WorkerError::InvalidRequest("No model source".into()));
    };
    
    // Pass validated bytes to vram-residency...
}
```

**Expectations**:
- **EXP-LOADER-8001-R1**: Use model-loader for all model validation
- **EXP-LOADER-8001-R2**: Support both file-based and memory-based loading
- **EXP-LOADER-8001-R3**: Handle validation errors (hash mismatch, invalid GGUF)
- **EXP-LOADER-8001-R4**: Return 400 on validation failure with specific error

---

## 9. capability-matcher Integration

### 9.1 MCD/ECP Checking (EXP-CAP-9001)

**Required by**: Plan endpoint

**Expected usage**:
```rust
use capability_matcher::{CapabilityMatcher, ModelCapabilityDescriptor, EngineCapabilityProfile};

async fn plan_handler(
    State(state): State<Arc<WorkerState>>,
    Json(body): Json<PlanRequest>
) -> Result<Json<PlanResponse>> {
    // Extract MCD from model metadata (future)
    let mcd = state.capability_matcher.extract_mcd(&body.model_ref).await?;
    
    // Get worker ECP
    let ecp = state.capability_matcher.get_ecp();
    
    // Check compatibility
    let compatible = state.capability_matcher.check_compatibility(&mcd, &ecp)?;
    
    if !compatible.is_compatible {
        return Ok(Json(PlanResponse {
            feasible: false,
            vram_required: 0,
            shard_plan: vec![],
            incompatibility_reason: Some(compatible.reason),
        }));
    }
    
    // Check VRAM availability...
}
```

**Expectations**:
- **EXP-CAP-9001-R1**: Check MCD vs ECP before planning
- **EXP-CAP-9001-R2**: Return `feasible: false` if incompatible
- **EXP-CAP-9001-R3**: Include incompatibility reason in response
- **EXP-CAP-9001-R4**: Fast check (< 10ms)

---

## 10. scheduler Integration

### 10.1 Job State Tracking (EXP-SCHED-10001)

**Required by**: Execute endpoint

**Expected usage**:
```rust
use scheduler::{Scheduler, JobState};

async fn execute_handler(
    State(state): State<Arc<WorkerState>>,
    Json(body): Json<ExecuteRequest>
) -> Result<impl IntoResponse> {
    // Create job
    let job_id = state.scheduler.create_job(body.handle_id.clone()).await?;
    
    // Update state: Pending → Executing
    state.scheduler.update_state(&job_id, JobState::Executing).await?;
    
    // Run inference...
    let result = run_inference(...).await;
    
    // Update state: Executing → Completed
    state.scheduler.update_state(&job_id, JobState::Completed).await?;
    
    Ok(result)
}
```

**Expectations**:
- **EXP-SCHED-10001-R1**: Track job state (Pending → Executing → Completed)
- **EXP-SCHED-10001-R2**: Support cancellation (Executing → Canceled)
- **EXP-SCHED-10001-R3**: Provide job status queries
- **EXP-SCHED-10001-R4**: Clean up completed jobs after TTL

---

## 11. Error Handling Expectations

### 11.1 WorkerError Enum (EXP-ERROR-11001)

**Expected error variants**:
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

**HTTP status code mapping**:
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

### 11.2 Error Response Format (EXP-ERROR-11002)

**Expected JSON error response**:
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

**Expectations**:
- **EXP-ERROR-11002-R1**: Structured error responses (not plain text)
- **EXP-ERROR-11002-R2**: Stable error codes (for client parsing)
- **EXP-ERROR-11002-R3**: Actionable error messages
- **EXP-ERROR-11002-R4**: No sensitive data in errors (no tokens, VRAM pointers)

---

## 12. SSE Streaming Expectations

### 12.1 Event Format (EXP-SSE-12001)

**Expected SSE events**:
```
event: started
data: {"job_id": "job-abc123", "queue_position": 0, "predicted_start_ms": 50}

event: token
data: {"t": "Hello", "i": 0}

event: token
data: {"t": " world", "i": 1}

event: metrics
data: {"queue_depth": 0, "on_time_probability": 0.95}

event: end
data: {"tokens_out": 2, "decode_time_ms": 45}
```

**Expectations**:
- **EXP-SSE-12001-R1**: Follow orchestratord SSE format (compatibility)
- **EXP-SSE-12001-R2**: Event ordering: `started → token* → metrics? → end`
- **EXP-SSE-12001-R3**: Well-formed JSON in data field
- **EXP-SSE-12001-R4**: Terminate stream after `event: end` or `event: error`

---

### 12.2 Error Handling in Streams (EXP-SSE-12002)

**Expected error event**:
```
event: error
data: {"code": "SEAL_VERIFICATION_FAILED", "message": "Seal signature mismatch", "retriable": false}
```

**Expectations**:
- **EXP-SSE-12002-R1**: Emit `event: error` on inference failure
- **EXP-SSE-12002-R2**: Include error code, message, retriable flag
- **EXP-SSE-12002-R3**: Terminate stream after error event
- **EXP-SSE-12002-R4**: No tokens after error event (race-free)

---

## 13. Performance Expectations

### 13.1 Endpoint Latency (EXP-PERF-13001)

**Expected latency targets**:
- **Plan**: < 100ms (no heavy computation)
- **Commit**: I/O bound (model loading + VRAM copy)
- **Ready**: < 10ms (fast health check)
- **Execute**: Depends on inference (streaming)

**Optimization expectations**:
- **EXP-PERF-13001-R1**: Non-blocking I/O (async/await)
- **EXP-PERF-13001-R2**: Parallel validation where possible
- **EXP-PERF-13001-R3**: Early termination on validation failure
- **EXP-PERF-13001-R4**: Minimal allocations in hot paths

---

### 13.2 Concurrency (EXP-PERF-13002)

**Expected concurrency support**:
- Multiple Plan requests in parallel (read-only)
- Single Commit at a time (VRAM allocation is exclusive)
- Multiple Ready checks in parallel (read-only)
- Single Execute at a time (M0, single-slot scheduler)

**Expectations**:
- **EXP-PERF-13002-R1**: Use async/await for all handlers
- **EXP-PERF-13002-R2**: Use locks only where necessary (minimize contention)
- **EXP-PERF-13002-R3**: Return 503 if worker busy (don't queue indefinitely)

---

## 14. Configuration Expectations

### 14.1 Environment Variables (EXP-CONFIG-14001)

**Expected environment variables**:
```bash
# Worker bind address
WORKER_BIND_ADDR=0.0.0.0:9300

# Worker ID
WORKER_ID=worker-gpu-0

# GPU device index
WORKER_GPU_DEVICE=0

# Bearer token file
WORKER_TOKEN_FILE=/etc/llorch/secrets/worker-token

# Max prompt length
WORKER_MAX_PROMPT_LEN=100000

# Max tokens per request
WORKER_MAX_TOKENS=4096

# Audit log directory
LLORCH_AUDIT_DIR=/var/lib/llorch/audit/worker-orcd
```

---

### 14.2 Service Initialization (EXP-CONFIG-14002)

**Expected initialization pattern**:
```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Load config from env
    let config = WorkerConfig::from_env()?;
    
    // Initialize dependencies
    let audit_logger = AuditLogger::new(config.audit_config)?;
    let model_loader = ModelLoader::new();
    let vram_manager = VramManager::new(config.gpu_device)?;
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
    
    // Create router
    let app = create_router()
        .with_state(state)
        .layer(auth_middleware);
    
    // Start server
    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}
```

---

## 15. Testing Expectations

### 15.1 Unit Tests (EXP-TEST-15001)

**Expected test coverage**:
- Request validation (all fields)
- Authentication (valid/invalid tokens)
- Error handling (all error variants)
- SSE event formatting
- Response serialization

**Example tests**:
```rust
#[tokio::test]
async fn test_plan_rejects_invalid_model_ref() {
    let response = plan_handler(PlanRequest {
        model_ref: "model; rm -rf /".to_string(),
        shard_layout: ShardLayout::Single,
        tp_degree: None,
    }).await;
    
    assert!(response.is_err());
}

#[tokio::test]
async fn test_commit_rejects_hash_mismatch() {
    let response = commit_handler(CommitRequest {
        model_ref: "hf:model".to_string(),
        shard_id: "shard-1".to_string(),
        model_bytes: Some(vec![1, 2, 3]),
        expected_digest: Some("wrong_hash".to_string()),
        ..Default::default()
    }).await;
    
    assert!(matches!(response, Err(WorkerError::IntegrityViolation(_))));
}
```

---

### 15.2 Integration Tests (EXP-TEST-15002)

**Expected integration tests**:
- End-to-end Plan → Commit → Ready → Execute flow
- Authentication middleware integration
- Audit logging integration
- Error propagation from dependent crates

---

## 16. Security Expectations

### 16.1 Input Validation (EXP-SEC-16001)

**Critical requirement**: Validate ALL user inputs before processing.

**Validation points**:
- All string fields (shard_id, model_ref, handle_id, prompt)
- All numeric fields (shard_index, gpu_device, max_tokens)
- All optional fields (expected_digest, tp_degree)

**See**: `bin/shared-crates/input-validation/.specs/10_expectations.md`

---

### 16.2 Authentication (EXP-SEC-16002)

**Critical requirement**: Authenticate ALL requests except `/ready` (configurable).

**Requirements**:
- Use `secrets-management` for token loading
- Use timing-safe comparison
- Emit audit events
- Never log full tokens

**See**: `bin/shared-crates/secrets-management/.specs/10_expectations.md`

---

### 16.3 Audit Logging (EXP-SEC-16003)

**Critical requirement**: Emit audit events for all security-relevant operations.

**Required events**:
- `AuthSuccess`, `AuthFailure` — Authentication
- `VramSealed` — Model committed
- `SealVerificationFailed` — Seal verification failure (CRITICAL)
- `PolicyViolation` — Security policy violation

**See**: `bin/shared-crates/audit-logging/.specs/10_expectations.md`

---

## 17. Implementation Priority

### Phase 1: M0 Essentials (Week 1)
1. ✅ Basic router with `/ready` stub
2. ⬜ Request/response types (Plan, Commit, Ready, Execute)
3. ⬜ Input validation integration
4. ⬜ Error types and HTTP mapping
5. ⬜ Basic unit tests

### Phase 2: Core Endpoints (Week 1-2)
6. ⬜ Plan endpoint implementation
7. ⬜ Commit endpoint implementation
8. ⬜ Ready endpoint implementation
9. ⬜ Execute endpoint implementation (stub)
10. ⬜ Integration with model-loader

### Phase 3: Security (Week 2)
11. ⬜ Bearer token authentication middleware
12. ⬜ Audit logging integration
13. ⬜ Integration with vram-residency
14. ⬜ Integration with capability-matcher

### Phase 4: Production Hardening (Week 3+)
15. ⬜ SSE streaming implementation
16. ⬜ Integration with scheduler
17. ⬜ Error handling polish
18. ⬜ Integration tests
19. ⬜ Performance optimization

---

## 18. Open Questions

**Q1**: Should Plan endpoint cache feasibility results?  
**A**: No for M0. Plan is fast enough (< 100ms). Consider caching post-M0.

**Q2**: Should Commit endpoint support streaming uploads?  
**A**: No for M0. Accept entire model bytes in request. Consider streaming post-M0.

**Q3**: Should Execute endpoint support multiple concurrent jobs?  
**A**: No for M0. Single-slot scheduler. Multi-slot post-M0.

**Q4**: Should Ready endpoint require authentication?  
**A**: Configurable. Default: no auth (health check). Allow auth via config.

---

## 19. References

**Specifications**:
- `bin/worker-orcd-crates/api/.specs/00_api.md` — Main spec (WORKER-4xxx)
- `bin/worker-orcd/.specs/00_worker-orcd.md` — Parent spec
- `.docs/ARCHITECTURE_CHANGE_PLAN.md` — Phase 3, Task Group 1

**Dependencies**:
- `bin/worker-orcd-crates/vram-residency/.specs/10_expectations.md`
- `bin/worker-orcd-crates/model-loader/.specs/10_expectations.md`
- `bin/shared-crates/input-validation/.specs/10_expectations.md`
- `bin/shared-crates/secrets-management/.specs/10_expectations.md`
- `bin/shared-crates/audit-logging/.specs/10_expectations.md`

**Security**:
- `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Issue #1 (worker-orcd endpoint auth)

---

## Refinement Opportunities

- Define rate limiting strategy for Plan/Commit endpoints (prevent abuse)
- Specify retry/backoff behavior for transient errors (VRAM allocation failures)
- Define correlation ID propagation across worker crates (for distributed tracing)
- Specify graceful shutdown behavior (drain in-flight requests, flush audit logs)
- Define health check levels (liveness vs readiness) for Ready endpoint
- Specify request timeout policies (max request duration per endpoint)
- Define SSE keepalive/heartbeat strategy for long-running Execute requests

---

**End of Expectations Document**
