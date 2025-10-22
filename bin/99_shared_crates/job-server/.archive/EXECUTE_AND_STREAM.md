# Execute and Stream Helper

**Team:** TEAM-186  
**Date:** 2025-10-21  
**Status:** ✅ Complete

## Summary

Added `execute_and_stream()` helper function to job-registry shared crate, providing a reusable pattern for deferred job execution with SSE streaming.

## Function Signature

```rust
pub async fn execute_and_stream<T, F, Exec>(
    job_id: String,
    registry: Arc<JobRegistry<T>>,
    executor: Exec,
) -> impl Stream<Item = String>
where
    T: ToString + Send + 'static,
    F: std::future::Future<Output = Result<(), anyhow::Error>> + Send + 'static,
    Exec: FnOnce(String, serde_json::Value) -> F + Send + 'static,
```

## What It Does

1. **Retrieves payload** from job registry
2. **Spawns execution** in background task
3. **Handles errors** with narration events
4. **Returns stream** of results for SSE

## Usage Example

### Queen-rbee

```rust
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Use shared helper
    let token_stream = job_registry::execute_and_stream(
        job_id,
        state.registry.clone(),
        move |_job_id, payload| {
            let router_state = JobRouterState {
                registry: state.registry.clone(),
                hive_catalog: state.hive_catalog.clone(),
            };
            async move {
                route_job(router_state, payload)
                    .await
                    .map(|_| ())
                    .map_err(|e| anyhow::anyhow!(e))
            }
        },
    ).await;

    // Convert to SSE events
    let event_stream = token_stream.map(|data| Ok(Event::default().data(data)));
    Sse::new(event_stream)
}
```

### Before (50+ lines)

```rust
pub async fn handle_stream_job(...) {
    // Retrieve payload
    let payload = state.registry.take_payload(&job_id);
    
    if let Some(payload) = payload {
        // Create router state
        let router_state = ...;
        
        // Spawn execution
        tokio::spawn(async move {
            // Narration
            Narration::new(...).emit();
            
            // Execute
            if let Err(e) = route_job(...).await {
                // Error narration
                Narration::new(...).emit();
            }
        });
    } else {
        // Warning narration
        Narration::new(...).emit();
    }
    
    // Get receiver
    let receiver = state.registry.take_token_receiver(&job_id);
    
    // Create stream
    let stream = stream::unfold(receiver, |rx_opt| async move {
        match rx_opt {
            Some(mut rx) => match rx.recv().await {
                Some(token) => {
                    let event = Event::default().data(token);
                    Some((Ok(event), Some(rx)))
                }
                None => None,
            },
            None => None,
        }
    });
    
    Sse::new(stream)
}
```

### After (20 lines)

```rust
pub async fn handle_stream_job(...) {
    let token_stream = job_registry::execute_and_stream(
        job_id,
        state.registry.clone(),
        move |_job_id, payload| {
            async move {
                route_job(state, payload)
                    .await
                    .map(|_| ())
                    .map_err(|e| anyhow::anyhow!(e))
            }
        },
    ).await;
    
    let event_stream = token_stream.map(|data| Ok(Event::default().data(data)));
    Sse::new(event_stream)
}
```

**Savings:** ~30 lines per endpoint!

## Features

### 1. Automatic Payload Retrieval
```rust
let payload = registry.take_payload(&job_id);
```
No need to manually call `take_payload()`

### 2. Background Execution
```rust
tokio::spawn(async move {
    executor(job_id, payload).await;
});
```
Execution happens in background, doesn't block stream setup

### 3. Error Handling
```rust
if let Err(e) = executor(...).await {
    Narration::new(ACTOR, "job_error", &job_id)
        .human(format!("Job {} failed: {}", job_id, e))
        .error_kind("job_execution_failed")
        .emit();
}
```
Automatic error narration

### 4. Stream Creation
```rust
let receiver = registry.take_token_receiver(&job_id);
stream::unfold(receiver, |rx_opt| async move { ... })
```
Automatic stream setup from receiver

### 5. Narration Events
```rust
Narration::new("📋 job-executor", "job_execute", &job_id)
    .human(format!("Executing job {}", job_id))
    .emit();
```
Built-in observability

## Dependencies Added

**File:** `bin/99_shared_crates/job-registry/Cargo.toml`

```toml
# TEAM-186: For execute_and_stream helper
futures = "0.3"  # Stream utilities
observability-narration-core = { path = "../narration-core" }  # Narration events
anyhow = "1.0"  # Error handling
```

## Type Parameters

### `T: ToString + Send + 'static`
The token type stored in the registry. Must be convertible to String for streaming.

**Examples:**
- `String` - Direct string tokens
- `TokenResponse` - Custom token type with ToString impl
- `serde_json::Value` - JSON tokens

### `F: Future<Output = Result<(), anyhow::Error>>`
The future returned by the executor function. Must resolve to `Result<(), Error>`.

### `Exec: FnOnce(String, serde_json::Value) -> F`
The executor function that takes job_id and payload, returns a Future.

**Signature:**
```rust
|job_id: String, payload: serde_json::Value| -> Future<Output = Result<(), Error>>
```

## Narration Events

### Job Execution Start
```
Actor: 📋 job-executor
Action: job_execute
Target: job-123
Message: "Executing job job-123"
```

### Job Execution Error
```
Actor: 📋 job-executor
Action: job_error
Target: job-123
Message: "Job job-123 failed: <error>"
Error Kind: job_execution_failed
```

### No Payload Warning
```
Actor: 📋 job-executor
Action: job_no_payload
Target: job-123
Message: "Warning: No payload found for job job-123"
```

## Benefits

✅ **Reusable** - Available to all services (queen, hive, worker)  
✅ **Consistent** - Same pattern everywhere  
✅ **Less Code** - ~30 lines saved per endpoint  
✅ **Better Errors** - Automatic error handling and narration  
✅ **Type Safe** - Generic over token type  
✅ **Observable** - Built-in narration events  
✅ **Testable** - Can mock executor function  

## Use Cases

### 1. Queen-rbee (Job Routing)
```rust
execute_and_stream(job_id, registry, |_, payload| async {
    route_job(state, payload).await.map(|_| ())
})
```

### 2. Rbee-hive (Worker Management)
```rust
execute_and_stream(job_id, registry, |_, payload| async {
    manage_worker(state, payload).await
})
```

### 3. LLM Worker (Inference)
```rust
execute_and_stream(job_id, registry, |_, payload| async {
    run_inference(model, payload).await
})
```

## Future Enhancements

### 1. Timeout Support
```rust
pub async fn execute_and_stream_with_timeout<T, F, Exec>(
    job_id: String,
    registry: Arc<JobRegistry<T>>,
    executor: Exec,
    timeout: Duration,
) -> impl Stream<Item = String>
```

### 2. Retry Support
```rust
pub async fn execute_and_stream_with_retry<T, F, Exec>(
    job_id: String,
    registry: Arc<JobRegistry<T>>,
    executor: Exec,
    max_retries: u32,
) -> impl Stream<Item = String>
```

### 3. Progress Tracking
```rust
pub async fn execute_and_stream_with_progress<T, F, Exec>(
    job_id: String,
    registry: Arc<JobRegistry<T>>,
    executor: Exec,
    progress_callback: impl Fn(f32),
) -> impl Stream<Item = String>
```

## Testing

```bash
# Test job-registry
cargo test -p job-registry

# Test queen-rbee integration
cargo check -p queen-rbee

# Test endpoint
curl -X POST http://localhost:8500/v1/jobs \
  -d '{"operation": "hive_list"}'

curl http://localhost:8500/v1/jobs/{job_id}/stream
```

## Files Modified

### Shared Crate
- `bin/99_shared_crates/job-registry/src/lib.rs` - Added execute_and_stream()
- `bin/99_shared_crates/job-registry/Cargo.toml` - Added dependencies

### Queen-rbee
- `bin/10_queen_rbee/src/http.rs` - Use execute_and_stream()

## Comparison

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| Lines per endpoint | ~50 | ~20 | 60% |
| Boilerplate | High | Low | - |
| Error handling | Manual | Automatic | - |
| Narration | Manual | Automatic | - |
| Stream setup | Manual | Automatic | - |
| Reusability | None | All services | - |

## Conclusion

The `execute_and_stream()` helper provides a clean, reusable pattern for deferred job execution with SSE streaming. It's now available in the shared job-registry crate for use across all rbee services.

**Key Achievement:** Reduced endpoint code from ~50 lines to ~20 lines while adding automatic error handling and narration!
