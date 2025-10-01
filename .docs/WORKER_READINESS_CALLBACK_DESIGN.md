# Worker Readiness Callback Design

**Date**: 2025-10-01  
**Status**: Proposed  
**Purpose**: Replace filesystem-based handoff-watcher with direct HTTP callbacks

---

## Problem Statement

The current `handoff-watcher` crate uses **filesystem polling** to detect when workers are ready:

```
pool-managerd spawns worker
         ↓
worker writes: .runtime/engines/gpu-0-r0.json
         ↓
handoff-watcher POLLS filesystem every 1000ms
         ↓
handoff-watcher finds file, parses it, invokes callback
         ↓
pool-managerd: "Worker is ready!"
```

### Issues with Current Design

1. **Wasteful polling**:
   - Constantly checks disk every 1000ms
   - Latency: Up to 1 second detection delay
   - Unnecessary I/O overhead

2. **Race conditions**:
   - File might be partially written when detected
   - No atomic "file is complete" signal
   - Cleanup conflicts (who deletes the file?)

3. **Indirect communication**:
   - File-based IPC is outdated (2025 standard is HTTP)
   - No acknowledgment mechanism
   - Poor error handling (silent failures)

4. **Not scalable**:
   - 100 workers starting = 100 files to poll
   - Filesystem becomes bottleneck

5. **Legacy design**:
   - Built for external engines (llama.cpp, vLLM) we couldn't control
   - worker-orcd is our code—we can do better

---

## Proposed Solution: Direct HTTP Callbacks

**Simple, clean, industry-standard approach:**

```
pool-managerd spawns worker-orcd
         ↓
pool-managerd passes callback URL: "http://localhost:9200/v2/internal/workers/ready"
         ↓
worker-orcd starts, loads model
         ↓
worker-orcd: POST to callback URL
{
  "worker_id": "gpu-0",
  "pool_id": "pool-0", 
  "status": "ready",
  "endpoint": "http://localhost:8001",
  "vram_used_bytes": 24000000000
}
         ↓
pool-managerd receives callback IMMEDIATELY
         ↓
pool-managerd responds: 200 OK
         ↓
pool-managerd updates registry, marks ready
```

---

## Architecture

### Worker-orcd Side (Callback Sender)

**Startup sequence:**

```rust
// bin/worker-orcd/src/main.rs
pub async fn worker_main(config: WorkerConfig) -> Result<()> {
    // 1. Initialize worker
    let worker = Worker::new(config.clone())?;
    
    // 2. Load model into VRAM
    let model_handle = worker.load_model(&config.model_path).await?;
    
    // 3. Notify pool-managerd we're ready
    notify_ready(&config).await?;
    
    // 4. Start RPC server
    worker.serve_rpc().await
}

async fn notify_ready(config: &WorkerConfig) -> Result<()> {
    let notification = ReadinessNotification {
        worker_id: config.worker_id.clone(),
        pool_id: config.pool_id.clone(),
        status: WorkerStatus::Ready,
        endpoint: format!("http://localhost:{}", config.rpc_port),
        vram_used_bytes: config.model_vram_bytes,
        capabilities: config.capabilities.clone(),
    };
    
    let client = reqwest::Client::new();
    let response = client
        .post(&config.readiness_callback_url)
        .json(&notification)
        .timeout(Duration::from_secs(5))
        .send()
        .await?;
    
    if !response.status().is_success() {
        return Err(anyhow!("Pool-managerd rejected readiness: {}", response.status()));
    }
    
    tracing::info!(
        worker_id = %config.worker_id,
        pool_id = %config.pool_id,
        "Successfully notified pool-managerd of readiness"
    );
    
    Ok(())
}
```

### Pool-managerd Side (Callback Receiver)

**API endpoint:**

```rust
// bin/pool-managerd/src/api/routes.rs
pub fn internal_routes() -> Router<Arc<PoolManagerState>> {
    Router::new()
        .route("/v2/internal/workers/ready", post(handle_worker_ready))
        .route("/v2/internal/workers/:id/failed", post(handle_worker_failed))
}

// bin/pool-managerd/src/api/workers.rs
#[axum::debug_handler]
pub async fn handle_worker_ready(
    State(state): State<Arc<PoolManagerState>>,
    Json(notification): Json<ReadinessNotification>,
) -> Result<StatusCode, AppError> {
    tracing::info!(
        worker_id = %notification.worker_id,
        pool_id = %notification.pool_id,
        endpoint = %notification.endpoint,
        vram_used_bytes = notification.vram_used_bytes,
        "Received worker readiness notification"
    );
    
    // Update registry
    state.registry.register_ready_from_handoff(
        &notification.pool_id,
        notification.endpoint.clone(),
        notification.vram_used_bytes,
        notification.capabilities,
    )?;
    
    // Emit metrics
    state.metrics.worker_ready_total
        .with_label_values(&[&notification.worker_id])
        .inc();
    
    Ok(StatusCode::OK)
}
```

### Shared Types

```rust
// contracts/api-types/src/internal.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessNotification {
    pub worker_id: String,
    pub pool_id: String,
    pub status: WorkerStatus,
    pub endpoint: String,
    pub vram_used_bytes: u64,
    pub capabilities: WorkerCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerStatus {
    Ready,
    Failed,
    Draining,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    pub gpu_device: u32,
    pub slots_total: usize,
    pub model_ref: String,
    pub engine_version: String,
}
```

---

## Worker Spawn Flow

### Pool-managerd spawns worker

```rust
// bin/pool-managerd/src/lifecycle/worker.rs
pub async fn spawn_worker(config: &PoolConfig) -> Result<WorkerHandle> {
    let callback_url = format!(
        "http://localhost:{}/v2/internal/workers/ready",
        config.pool_manager_port
    );
    
    let worker_config = WorkerConfig {
        worker_id: format!("gpu-{}", config.gpu_device),
        pool_id: config.pool_id.clone(),
        model_path: config.model_path.clone(),
        rpc_port: config.worker_rpc_port,
        readiness_callback_url: callback_url,
        gpu_device: config.gpu_device,
        model_vram_bytes: config.estimated_vram_bytes,
        capabilities: config.capabilities.clone(),
    };
    
    // Spawn worker process
    let child = Command::new("worker-orcd")
        .arg("--config")
        .arg(serde_json::to_string(&worker_config)?)
        .env("CUDA_VISIBLE_DEVICES", config.gpu_device.to_string())
        .spawn()?;
    
    tracing::info!(
        worker_id = %worker_config.worker_id,
        pid = child.id(),
        callback_url = %callback_url,
        "Spawned worker process"
    );
    
    Ok(WorkerHandle {
        worker_id: worker_config.worker_id,
        child,
        config: worker_config,
    })
}
```

---

## Benefits of HTTP Callbacks

✅ **Immediate notification**
- Zero polling delay
- Worker notifies as soon as ready
- Typical latency: <10ms

✅ **Reliable acknowledgment**
- HTTP 200 OK confirms receipt
- Worker knows pool-managerd received notification
- Retry logic is trivial (if 5xx, retry)

✅ **Scalable**
- No filesystem bottleneck
- 1000 workers? No problem
- Each worker makes one HTTP call

✅ **Debuggable**
- HTTP logs show exact request/response
- Trace correlation IDs through the stack
- Easy to test with curl/Postman

✅ **Industry standard**
- Webhooks, health checks, service mesh
- Everyone understands POST callbacks
- No custom protocols to explain

✅ **Error handling**
- HTTP status codes are clear
- Worker can retry on 5xx
- Pool-managerd can reject with 4xx + reason

---

## What to Delete

### Crates to Remove

1. **`bin/worker-orcd-crates/handoff-watcher/`**
   - Entire crate is obsolete
   - Replaced by simple HTTP POST

### Code to Remove

2. **Handoff file writing** (if any in pool-managerd)
   - No more `.runtime/engines/*.json` files
   - Clean up `.runtime/` directory structure

3. **Handoff watcher usage** in pool-managerd
   - Remove `handoff_watcher::spawn()` calls
   - Remove file polling logic

### Specs to Update

4. **`.specs/` references to handoff files**
   - Update to describe HTTP callback pattern
   - Remove filesystem polling requirements

---

## What to Create

### New Code

1. **Worker readiness endpoint** in pool-managerd
   - `POST /v2/internal/workers/ready`
   - `POST /v2/internal/workers/:id/failed` (for error reporting)

2. **Callback client** in worker-orcd
   - `notify_ready()` function
   - Retry logic for transient failures

3. **Shared types** in `contracts/api-types`
   - `ReadinessNotification`
   - `WorkerStatus` enum
   - `WorkerCapabilities`

### New Specs

4. **Worker lifecycle spec**
   - Document spawn → callback → ready flow
   - Error handling (what if callback fails?)
   - Timeout behavior

---

## Error Handling

### Worker-side Errors

**Callback fails (network error, 5xx)**:
```rust
// Retry with exponential backoff
let mut retry_count = 0;
loop {
    match notify_ready(config).await {
        Ok(_) => break,
        Err(e) if retry_count < 3 => {
            let backoff = Duration::from_millis(100 * 2_u64.pow(retry_count));
            tracing::warn!(
                error = %e,
                retry_count = retry_count,
                backoff_ms = backoff.as_millis(),
                "Readiness callback failed, retrying"
            );
            tokio::time::sleep(backoff).await;
            retry_count += 1;
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to notify readiness after retries");
            return Err(e);
        }
    }
}
```

**Pool-managerd rejects (4xx)**:
```rust
// Worker should fail fast and exit
if response.status().is_client_error() {
    let body = response.text().await?;
    return Err(anyhow!(
        "Pool-managerd rejected readiness: {} - {}",
        response.status(),
        body
    ));
}
```

### Pool-managerd Errors

**Invalid notification**:
```rust
// Return 400 Bad Request
if notification.vram_used_bytes == 0 {
    return Err(AppError::BadRequest(
        "vram_used_bytes must be non-zero".into()
    ));
}
```

**Worker ID conflict**:
```rust
// Return 409 Conflict
if state.registry.has_worker(&notification.worker_id) {
    return Err(AppError::Conflict(
        format!("Worker {} already registered", notification.worker_id)
    ));
}
```

---

## Migration Strategy

### Phase 1: Implement HTTP Callbacks (worker-orcd)

1. Add `ReadinessNotification` types to `contracts/api-types`
2. Implement `notify_ready()` in worker-orcd
3. Add `/v2/internal/workers/ready` endpoint to pool-managerd
4. Write integration tests

### Phase 2: Update Spawn Logic

1. Modify pool-managerd spawn to pass callback URL
2. Remove handoff file writing (if present)
3. Test spawn → callback → ready flow

### Phase 3: Deprecate handoff-watcher

1. Mark `handoff-watcher` crate as deprecated
2. Remove from workspace `Cargo.toml`
3. Delete crate directory
4. Update docs to reflect HTTP callback pattern

### Phase 4: Cleanup

1. Remove `.runtime/engines/` directory usage
2. Clean up filesystem polling references
3. Update specs and architecture docs

---

## Testing

### Unit Tests

**Worker callback client**:
```rust
#[tokio::test]
async fn test_notify_ready_success() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/internal/workers/ready"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock_server)
        .await;
    
    let config = WorkerConfig {
        readiness_callback_url: mock_server.uri(),
        // ... other fields
    };
    
    let result = notify_ready(&config).await;
    assert!(result.is_ok());
}
```

**Pool-managerd endpoint**:
```rust
#[tokio::test]
async fn test_handle_worker_ready() {
    let state = setup_test_state();
    let notification = ReadinessNotification {
        worker_id: "gpu-0".into(),
        pool_id: "pool-0".into(),
        status: WorkerStatus::Ready,
        endpoint: "http://localhost:8001".into(),
        vram_used_bytes: 24_000_000_000,
        capabilities: default_capabilities(),
    };
    
    let response = handle_worker_ready(
        State(state.clone()),
        Json(notification.clone())
    ).await;
    
    assert!(response.is_ok());
    assert_eq!(response.unwrap(), StatusCode::OK);
    
    // Verify registry was updated
    let pool = state.registry.get_pool("pool-0").unwrap();
    assert!(pool.ready);
}
```

### Integration Tests

**End-to-end spawn flow**:
```rust
#[tokio::test]
async fn test_worker_spawn_and_callback() {
    // 1. Start pool-managerd
    let pool_manager = start_pool_manager().await;
    
    // 2. Spawn worker
    let worker = pool_manager.spawn_worker(test_config()).await?;
    
    // 3. Wait for readiness callback
    tokio::time::timeout(
        Duration::from_secs(5),
        pool_manager.wait_for_ready("pool-0")
    ).await??;
    
    // 4. Verify worker is in registry
    let pool = pool_manager.registry.get_pool("pool-0")?;
    assert!(pool.ready);
    assert_eq!(pool.endpoint, "http://localhost:8001");
}
```

---

## Security Considerations

### Authentication

**Internal API should use Bearer token**:
```rust
// Worker sends token in callback
let response = client
    .post(&config.readiness_callback_url)
    .header("Authorization", format!("Bearer {}", config.internal_token))
    .json(&notification)
    .send()
    .await?;
```

**Pool-managerd validates token**:
```rust
pub async fn handle_worker_ready(
    State(state): State<Arc<PoolManagerState>>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
    Json(notification): Json<ReadinessNotification>,
) -> Result<StatusCode, AppError> {
    // Validate internal token
    if !auth_min::timing_safe_eq(
        auth.token().as_bytes(),
        state.config.internal_token.as_bytes()
    ) {
        return Err(AppError::Unauthorized);
    }
    
    // ... proceed with notification
}
```

### Network Binding

**Internal endpoint should bind to localhost only**:
```rust
// pool-managerd config
bind_internal = "127.0.0.1:9200"  // Localhost only
bind_public = "0.0.0.0:9201"      // External API
```

---

## Observability

### Metrics

**Worker-side**:
- `worker_readiness_callbacks_total{status}` - Total callbacks (success/failure)
- `worker_readiness_callback_duration_seconds` - Callback latency

**Pool-managerd-side**:
- `workers_ready_total{pool_id}` - Total workers that became ready
- `workers_failed_total{pool_id}` - Total workers that failed
- `readiness_callback_duration_seconds` - Endpoint latency

### Logging

**Worker logs**:
```
INFO worker_orcd: Model loaded successfully vram_bytes=24000000000
INFO worker_orcd: Notifying pool-managerd of readiness callback_url=http://localhost:9200/v2/internal/workers/ready
INFO worker_orcd: Pool-managerd acknowledged readiness response_status=200
INFO worker_orcd: RPC server listening endpoint=http://localhost:8001
```

**Pool-managerd logs**:
```
INFO pool_managerd: Spawning worker worker_id=gpu-0 pool_id=pool-0 pid=12345
INFO pool_managerd: Received readiness notification worker_id=gpu-0 endpoint=http://localhost:8001
INFO pool_managerd: Worker registered and ready worker_id=gpu-0 pool_id=pool-0
```

---

## Conclusion

Replacing filesystem-based `handoff-watcher` with **direct HTTP callbacks** provides:

✅ **Immediate notification** (no polling delay)
✅ **Reliable acknowledgment** (HTTP 200 OK)
✅ **Better scalability** (no filesystem bottleneck)
✅ **Superior debugging** (HTTP logs, traces)
✅ **Industry-standard pattern** (webhooks)
✅ **Simpler codebase** (delete entire crate)

**Recommended action**: Implement HTTP callbacks and deprecate `handoff-watcher` before worker-orcd implementation begins.

---

**Status**: Ready for review and implementation
