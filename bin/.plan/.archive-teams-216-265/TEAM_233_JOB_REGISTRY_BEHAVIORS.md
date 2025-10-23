# TEAM-233: Job Registry Behavior Inventory

**Date:** Oct 22, 2025  
**Crate:** `job-registry`  
**Complexity:** Medium  
**Status:** ✅ COMPLETE

// TEAM-233: Investigated

---

## Executive Summary

In-memory job state management for dual-call pattern: POST creates job + returns job_id, GET streams results via SSE. Generic over token type to support different response formats across binaries.

**Key Behaviors:**
- Server-generated job IDs (UUID v4)
- Token receiver storage (not sender - TEAM-154 fix)
- Deferred execution pattern (payload storage)
- Generic over token type T
- execute_and_stream helper for SSE streaming

---

## 1. Core Architecture

### 1.1 Dual-Call Pattern

**Flow:**
```text
1. POST /v1/inference
   → Create job
   → Store payload
   → Return { job_id, sse_url }

2. GET /v1/inference/{job_id}/stream
   → Retrieve payload
   → Execute operation
   → Stream results via SSE
```

**Why Dual-Call:**
- Client gets job_id immediately (can cancel, check status)
- Execution happens when client connects to stream
- No wasted work if client never connects

### 1.2 Job Structure

```rust
pub struct Job<T> {
    pub job_id: String,
    pub state: JobState,
    pub created_at: chrono::DateTime<Utc>,
    pub token_receiver: Option<TokenReceiver<T>>,
    pub payload: Option<serde_json::Value>,
}

pub enum JobState {
    Queued,
    Running,
    Completed,
    Failed(String),
}
```

**Generic Type T:**
- Worker: `TokenResponse` (Token/Error/Done)
- Queen: Could be different format
- Hive: Could be different format

**TEAM-154 FIX:** Store receiver, not sender!
- POST creates channel, stores receiver, passes sender to execution engine
- GET retrieves receiver and streams tokens
- Prevents dual-call pattern bugs

---

## 2. JobRegistry

### 2.1 Core Type

```rust
pub struct JobRegistry<T> {
    jobs: Arc<Mutex<HashMap<String, Job<T>>>>,
}
```

**Thread Safety:**
- Arc<Mutex<>> for shared access
- Safe for concurrent POST/GET requests
- Lock held only during HashMap operations

### 2.2 Key Methods

**`create_job() -> String`**
- Server generates job_id (UUID v4)
- Format: `"job-{uuid}"`
- Initial state: Queued
- Returns job_id to client

**`set_payload(job_id, payload)`**
- Store operation payload for deferred execution
- Called after job creation, before execution
- Payload is serde_json::Value (flexible)

**`take_payload(job_id) -> Option<Value>`**
- Retrieve and remove payload
- Can only be called once per job
- Used by GET endpoint to start execution

**`set_token_receiver(job_id, receiver)`**
- Store receiver for streaming
- Called after channel creation
- Receiver is moved into registry

**`take_token_receiver(job_id) -> Option<Receiver<T>>`**
- Retrieve and remove receiver
- Can only be called once per job
- Used by GET endpoint to stream results

**`update_state(job_id, state)`**
- Update job state (Queued → Running → Completed/Failed)
- Thread-safe state transitions

**`remove_job(job_id) -> Option<Job<T>>`**
- Remove job from registry (cleanup)
- Returns job if it existed

**`has_job(job_id) -> bool`**
- Check if job exists
- Used for validation

**`get_job_state(job_id) -> Option<JobState>`**
- Get current job state
- Returns clone of state

**`job_count() -> usize`**
- Count active jobs
- Useful for monitoring

**`job_ids() -> Vec<String>`**
- Get all job IDs
- Useful for debugging/admin

### 2.3 Ownership Semantics

**Take Methods (Consume):**
- `take_payload()` - Can only call once
- `take_token_receiver()` - Can only call once
- `remove_job()` - Removes from registry

**Get Methods (Borrow):**
- `get_job_state()` - Returns clone
- `has_job()` - Read-only check

**Why Take:**
- Prevents double-consumption bugs
- Clear ownership transfer
- Compiler enforces single use

---

## 3. Deferred Execution Pattern

### 3.1 execute_and_stream Helper

**Purpose:** Reusable helper for deferred execution + SSE streaming

```rust
pub async fn execute_and_stream<T, F, Exec>(
    job_id: String,
    registry: Arc<JobRegistry<T>>,
    executor: Exec,
) -> impl Stream<Item = String>
where
    T: ToString + Send + 'static,
    F: Future<Output = Result<(), anyhow::Error>> + Send + 'static,
    Exec: FnOnce(String, serde_json::Value) -> F + Send + 'static,
```

**Flow:**
1. Retrieve payload from registry (take_payload)
2. Spawn async execution in background (tokio::spawn)
3. Return stream of results for SSE
4. Stream reads from token_receiver

**Usage:**
```rust
let stream = execute_and_stream(
    job_id,
    registry,
    |job_id, payload| async move {
        route_job(state, payload).await
    }
).await;
```

### 3.2 Narration Integration

**Actor:** `"job-exec"` (8 chars, ≤10 limit)

**Actions:**
- `execute` - Job execution started
- `failed` - Job execution failed
- `no_payload` - Warning: no payload found

**Pattern:**
```rust
NARRATE.action("execute")
    .job_id(&job_id)
    .context(job_id.clone())
    .human("Executing job {}")
    .emit();
```

---

## 4. Integration Points

### 4.1 Used By

**queen-rbee:**
- Job creation in POST /v1/jobs
- Job execution in GET /v1/jobs/{job_id}/stream
- Usage: 6 imports in product code

**Usage Pattern:**
```rust
// 1. Create job
let job_id = registry.create_job();

// 2. Store payload
registry.set_payload(&job_id, serde_json::to_value(operation)?);

// 3. Create SSE channel
sse_sink::create_job_channel(job_id.clone(), 1000);

// 4. Return job_id to client
Ok(Json(CreateJobResponse { job_id, sse_url }))

// ... later, in GET endpoint ...

// 5. Stream results
let stream = execute_and_stream(
    job_id,
    registry,
    |job_id, payload| async move {
        route_job(state, payload).await
    }
).await;
```

### 4.2 NOT Used By (Yet)

**rbee-hive:**
- Could use for worker inference jobs
- Currently has custom implementation

**Consolidation Opportunity:** Hive could use job-registry for worker inference

---

## 5. Token Types

### 5.1 Generic Design

**Why Generic:**
- Worker: Different token format than queen
- Hive: Different token format than worker
- Each binary defines its own token type

**Type Constraints:**
```rust
T: Send + 'static  // For JobRegistry
T: ToString + Send + 'static  // For execute_and_stream
```

### 5.2 Example Token Types

**Worker (hypothetical):**
```rust
enum TokenResponse {
    Token(String),
    Error(String),
    Done,
}

impl ToString for TokenResponse {
    fn to_string(&self) -> String {
        match self {
            TokenResponse::Token(s) => s.clone(),
            TokenResponse::Error(e) => format!("ERROR: {}", e),
            TokenResponse::Done => "DONE".to_string(),
        }
    }
}
```

**Queen (current):**
- Uses String directly
- Simple SSE events

---

## 6. Error Handling

### 6.1 Error Cases

**Job Not Found:**
- `take_payload()` returns None
- `take_token_receiver()` returns None
- Caller must handle gracefully

**Double Take:**
- Second call to `take_payload()` returns None
- Second call to `take_token_receiver()` returns None
- Compiler prevents bugs via ownership

**Execution Failure:**
- Caught in execute_and_stream
- Emits error narration
- Job state updated to Failed

### 6.2 Cleanup

**When to Remove:**
- After job completes (Completed/Failed)
- After stream closes
- On timeout (if implemented)

**Memory Leak Prevention:**
- Must call `remove_job()` after completion
- Otherwise jobs accumulate in memory

---

## 7. Test Coverage

### 7.1 Existing Tests

**Unit Tests:**
- ✅ Create job (generates UUID)
- ✅ Get job state
- ✅ Update state transitions
- ✅ Token receiver storage and retrieval
- ✅ Remove job
- ✅ Job IDs listing

**Async Tests:**
- ✅ Token receiver streaming (tokio::test)

### 7.2 Test Gaps

**Missing Tests:**
- ❌ Concurrent job creation (race conditions)
- ❌ Concurrent payload/receiver access
- ❌ Memory leak tests (jobs not removed)
- ❌ execute_and_stream with actual execution
- ❌ Stream cancellation (client disconnects)
- ❌ Job state transitions (Queued → Running → Completed)
- ❌ Payload serialization edge cases
- ❌ Large payload handling
- ❌ Job cleanup on error

---

## 8. Performance Characteristics

**Job Creation:**
- UUID generation: ~100ns
- HashMap insert: O(1)
- Lock contention: Minimal (short critical section)

**Job Lookup:**
- HashMap lookup: O(1)
- Lock held only during lookup

**Memory:**
- Job struct: ~200 bytes
- Payload: Variable (JSON)
- Receiver: ~100 bytes
- Total: ~300 bytes + payload size

**Cleanup:**
- Must manually remove jobs
- No automatic cleanup (by design)
- Potential memory leak if not removed

---

## 9. Dependencies

**Core:**
- `tokio` - Async runtime, MPSC channels
- `uuid` - Job ID generation
- `chrono` - Timestamps
- `serde_json` - Payload serialization
- `futures` - Stream utilities
- `anyhow` - Error handling
- `observability-narration-core` - Narration

**Standard Library:**
- `std::collections::HashMap` - Job storage
- `std::sync::{Arc, Mutex}` - Thread safety

---

## 10. Critical Behaviors Summary

1. **Server-generated job IDs** - Client doesn't provide, server generates UUID
2. **Store receiver, not sender** - TEAM-154 fix prevents dual-call bugs
3. **Deferred execution** - Payload stored, execution happens on GET
4. **Take semantics** - Can only retrieve payload/receiver once
5. **Generic over token type** - Flexible for different binaries
6. **Manual cleanup** - Must call remove_job() to prevent leaks
7. **Thread-safe** - Arc<Mutex<>> for concurrent access

---

## 11. Design Patterns

**Pattern:** Registry + Deferred Execution

**Registry:**
- Central storage for job state
- Thread-safe access
- CRUD operations

**Deferred Execution:**
- POST stores payload
- GET retrieves and executes
- Prevents wasted work

**Streaming:**
- MPSC channel for token streaming
- Receiver stored in registry
- Stream reads from receiver

---

## 12. Future Enhancements

**Potential Features:**
- ❌ Automatic cleanup (TTL-based)
- ❌ Job cancellation
- ❌ Job progress tracking
- ❌ Job history/audit log
- ❌ Job priority queue
- ❌ Job retry logic

**Why Not Implemented:**
- Keep it simple
- Each binary has different needs
- Can be added later if needed

---

**Handoff:** Ready for Phase 5 integration analysis  
**Next:** TEAM-234 (rbee-heartbeat + timeout-enforcer)
