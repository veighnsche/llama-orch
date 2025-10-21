# Job Lifecycle: Deferred Execution Pattern

**Team:** TEAM-186  
**Date:** 2025-10-21  
**Status:** ✅ Complete

## Summary

Implemented deferred execution pattern where jobs are created immediately but only start executing when the client connects to the SSE stream. This ensures no events are missed.

## Problem

**Before:** Jobs would execute immediately when created, potentially missing events if the client wasn't connected to the SSE stream yet.

**After:** Jobs are created and stored, then execute only when the client connects, guaranteeing all events are captured.

## Architecture

```
Client                    Queen-rbee
  |                           |
  |  POST /v1/jobs           |
  |------------------------->|
  |                          | 1. Create job_id
  |                          | 2. Store payload
  |                          | 3. Return job_id + sse_url
  |<-------------------------|
  |  {job_id, sse_url}       |
  |                          |
  |  GET /v1/jobs/{id}/stream|
  |------------------------->|
  |                          | 4. Retrieve payload
  |                          | 5. Start execution (async)
  |                          | 6. Stream events via SSE
  |<-------------------------|
  |  SSE: event 1            |
  |<-------------------------|
  |  SSE: event 2            |
  |<-------------------------|
  |  SSE: done               |
```

## Implementation

### 1. Pending Jobs Storage

**File:** `src/http.rs`

```rust
// TEAM-186: Pending jobs storage
struct PendingJob {
    payload: serde_json::Value,
    router_state: crate::job_router::JobRouterState,
}

static PENDING_JOBS: Lazy<Mutex<HashMap<String, PendingJob>>> = 
    Lazy::new(|| Mutex::new(HashMap::new()));
```

### 2. Job Creation (POST /v1/jobs)

**Before:**
```rust
pub async fn handle_create_job(...) {
    // Execute immediately
    let response = crate::job_router::route_job(state, payload).await?;
    Ok(Json(response))
}
```

**After:**
```rust
pub async fn handle_create_job(...) {
    // Create job but DON'T execute
    let job_id = state.registry.create_job();
    
    // Store payload for later
    PENDING_JOBS.lock().unwrap().insert(
        job_id.clone(),
        PendingJob { payload, router_state },
    );
    
    // Return immediately
    Ok(Json(HttpJobResponse { job_id, sse_url }))
}
```

### 3. Job Execution (GET /v1/jobs/{id}/stream)

**Before:**
```rust
pub async fn handle_stream_job(...) {
    // Just stream whatever is in the registry
    let receiver = registry.take_token_receiver(&job_id);
    Sse::new(stream)
}
```

**After:**
```rust
pub async fn handle_stream_job(...) {
    // Retrieve pending job
    let pending_job = PENDING_JOBS.lock().unwrap().remove(&job_id);
    
    if let Some(pending) = pending_job {
        // Start execution in background
        tokio::spawn(async move {
            crate::job_router::route_job(
                pending.router_state,
                pending.payload
            ).await;
        });
    }
    
    // Stream results
    let receiver = registry.take_token_receiver(&job_id);
    Sse::new(stream)
}
```

## Job Lifecycle States

### State 1: Created (POST)
```
Job ID: job-123
Status: Pending
Payload: Stored in PENDING_JOBS
Registry: Job exists, no receiver yet
```

### State 2: Connected (GET)
```
Job ID: job-123
Status: Executing
Payload: Retrieved from PENDING_JOBS, removed
Registry: Receiver taken, execution started
```

### State 3: Streaming
```
Job ID: job-123
Status: Running
Events: Flowing through SSE stream
Registry: Receiver active
```

### State 4: Complete
```
Job ID: job-123
Status: Done
Events: Stream closed
Registry: Job can be cleaned up
```

## Benefits

✅ **No Missed Events** - Client connects before execution starts  
✅ **Clean Separation** - Creation and execution are separate concerns  
✅ **Better UX** - Client gets job_id immediately, can connect at leisure  
✅ **Fault Tolerance** - If client never connects, job never executes (saves resources)  
✅ **Observability** - Clear narration events for each lifecycle stage  

## Narration Events

### Job Creation
```
Actor: 👑 queen-http
Action: job_create
Target: job-123
Message: "Job job-123 created, waiting for client connection"
```

### Client Connection
```
Actor: 👑 queen-http
Action: job_stream
Target: job-123
Message: "Client connected, starting job job-123"
```

### Job Execution
```
Actor: 👑 queen-http
Action: job_execute
Target: job-123
Message: "Executing job job-123"
```

### Job Error
```
Actor: 👑 queen-http
Action: job_error
Target: job-123
Message: "Job job-123 failed: <error>"
Error Kind: job_execution_failed
```

## Example Flow

### 1. Client Creates Job
```bash
curl -X POST http://localhost:8500/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "hive_list"}'
```

**Response:**
```json
{
  "job_id": "job-550e8400-e29b-41d4-a716-446655440000",
  "sse_url": "/v1/jobs/job-550e8400-e29b-41d4-a716-446655440000/stream"
}
```

**Server State:**
- Job created in registry
- Payload stored in PENDING_JOBS
- No execution yet

### 2. Client Connects to Stream
```bash
curl http://localhost:8500/v1/jobs/job-550e8400-e29b-41d4-a716-446655440000/stream
```

**Server Actions:**
1. Retrieve pending job from PENDING_JOBS
2. Remove from PENDING_JOBS (one-time execution)
3. Spawn async task to execute job
4. Start streaming SSE events

**SSE Stream:**
```
data: {"event": "start", "job_id": "job-123"}

data: {"event": "progress", "message": "Querying hive catalog..."}

data: {"event": "result", "hives": [...]}

data: {"event": "done"}
```

## Edge Cases

### Client Never Connects
- Job remains in PENDING_JOBS
- No execution occurs
- Memory leak potential (TODO: add TTL/cleanup)

### Client Disconnects During Execution
- Execution continues in background
- Events are lost (receiver dropped)
- Job completes normally

### Multiple Clients Connect
- First client triggers execution
- Subsequent clients get empty stream (receiver already taken)
- TODO: Consider supporting multiple subscribers

### Job Already Executed
- PENDING_JOBS.remove() returns None
- Warning narration event emitted
- Stream returns empty (no receiver)

## Future Enhancements

### 1. TTL for Pending Jobs
```rust
struct PendingJob {
    payload: serde_json::Value,
    router_state: crate::job_router::JobRouterState,
    created_at: std::time::Instant,  // NEW
}

// Cleanup task
tokio::spawn(async {
    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
        let mut jobs = PENDING_JOBS.lock().unwrap();
        jobs.retain(|_, job| {
            job.created_at.elapsed() < Duration::from_secs(300)
        });
    }
});
```

### 2. Multiple Subscribers
Use broadcast channel instead of unbounded channel:
```rust
let (tx, _rx) = tokio::sync::broadcast::channel(100);
// Multiple clients can subscribe
let mut rx = tx.subscribe();
```

### 3. Job Status Endpoint
```
GET /v1/jobs/{id}/status
Response: {"status": "pending" | "running" | "completed" | "failed"}
```

### 4. Job Cancellation
```
DELETE /v1/jobs/{id}
- Remove from PENDING_JOBS if not started
- Send cancel signal if running
```

## Dependencies Added

**File:** `Cargo.toml`
```toml
once_cell = "1.19"  # TEAM-186: For PENDING_JOBS static map
```

## Testing

```bash
# Build
cargo build -p queen-rbee

# Test job creation
curl -X POST http://localhost:8500/v1/jobs \
  -d '{"operation": "hive_list"}'

# Test streaming (triggers execution)
curl http://localhost:8500/v1/jobs/{job_id}/stream

# Verify narration events
curl http://localhost:8500/narration/stream
```

## Related Files

- `bin/10_queen_rbee/src/http.rs` - Job lifecycle implementation
- `bin/10_queen_rbee/src/job_router.rs` - Job execution logic
- `bin/10_queen_rbee/Cargo.toml` - Dependencies
- `bin/99_shared_crates/job-registry/src/lib.rs` - Job registry

## Design Rationale

**Why not execute immediately?**
- SSE streams can miss events if client connects after execution starts
- Client needs time to establish SSE connection
- Better UX: client gets job_id first, then connects

**Why use static HashMap?**
- Simple, thread-safe storage
- No need for complex state management
- Easy to implement and understand
- Can be replaced with database later if needed

**Why spawn async task?**
- Don't block SSE stream setup
- Execution happens in parallel with streaming
- Better responsiveness

**Why remove from PENDING_JOBS?**
- Prevent double execution
- Free memory
- Clear lifecycle (pending → executing → done)
