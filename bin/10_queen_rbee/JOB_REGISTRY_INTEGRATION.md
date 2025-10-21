# Job Registry Integration for Deferred Execution

**Team:** TEAM-186  
**Date:** 2025-10-21  
**Status:** ✅ Complete

## Summary

Integrated payload storage into the existing `JobRegistry` instead of using a separate `PENDING_JOBS` static map. This provides a cleaner, more maintainable solution with all job data centralized in one place.

## Changes Made

### 1. Extended JobRegistry (Shared Crate)

**File:** `bin/99_shared_crates/job-registry/src/lib.rs`

**Added to `Job<T>` struct:**
```rust
pub struct Job<T> {
    pub job_id: String,
    pub state: JobState,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub token_receiver: Option<TokenReceiver<T>>,
    pub payload: Option<serde_json::Value>,  // TEAM-186: NEW
}
```

**Added methods:**
```rust
// Store payload for deferred execution
pub fn set_payload(&self, job_id: &str, payload: serde_json::Value)

// Retrieve and remove payload (one-time use)
pub fn take_payload(&self, job_id: &str) -> Option<serde_json::Value>
```

**Added dependency:**
```toml
serde_json = "1.0"  # TEAM-186: For payload storage
```

### 2. Updated handle_create_job

**File:** `bin/10_queen_rbee/src/http.rs`

**Before:**
```rust
// Used separate PENDING_JOBS static map
PENDING_JOBS.lock().unwrap().insert(
    job_id.clone(),
    PendingJob { payload, router_state },
);
```

**After:**
```rust
// Use JobRegistry directly
state.registry.set_payload(&job_id, payload);
```

**Benefits:**
- ✅ Simpler code (3 lines → 1 line)
- ✅ No separate static map needed
- ✅ No once_cell dependency needed
- ✅ All job data in one place

### 3. Updated handle_stream_job

**File:** `bin/10_queen_rbee/src/http.rs`

**Before:**
```rust
// Retrieved from separate PENDING_JOBS map
let pending_job = PENDING_JOBS.lock().unwrap().remove(&job_id);
if let Some(pending) = pending_job {
    // Execute with stored router_state
    crate::job_router::route_job(pending.router_state, pending.payload).await;
}
```

**After:**
```rust
// Retrieve from JobRegistry
let payload = state.registry.take_payload(&job_id);
if let Some(payload) = payload {
    // Create router_state on demand
    let router_state = crate::job_router::JobRouterState {
        registry: state.registry.clone(),
        hive_catalog: state.hive_catalog.clone(),
    };
    crate::job_router::route_job(router_state, payload).await;
}
```

**Benefits:**
- ✅ Single source of truth
- ✅ Router state created on demand (not stored)
- ✅ Consistent with JobRegistry patterns

### 4. Updated Router State

**File:** `bin/10_queen_rbee/src/main.rs`

**Changed:**
```rust
// Pass full SchedulerState instead of just registry
.route("/v1/jobs/:job_id/stream", get(http::handle_stream_job))
.with_state(job_state.clone())  // Was: job_state.registry
```

### 5. Removed Dependencies

**File:** `bin/10_queen_rbee/Cargo.toml`

**Removed:**
```toml
once_cell = "1.19"  # No longer needed
```

**File:** `bin/10_queen_rbee/src/http.rs`

**Removed:**
```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;

struct PendingJob { ... }
static PENDING_JOBS: Lazy<...> = ...;
```

## Architecture Comparison

### Before (Separate Map)
```
JobRegistry                PENDING_JOBS (static)
├─ job_id                  ├─ job_id
├─ state                   ├─ payload
├─ created_at              └─ router_state
└─ token_receiver

Two separate data structures!
```

### After (Integrated)
```
JobRegistry
├─ job_id
├─ state
├─ created_at
├─ token_receiver
└─ payload          ← NEW

Single source of truth!
```

## Benefits

### 1. Single Source of Truth
- All job data in `JobRegistry`
- No synchronization issues
- Easier to query and debug

### 2. Simpler Code
- Removed ~30 lines of boilerplate
- No static map management
- No once_cell dependency

### 3. Better Consistency
- Uses existing JobRegistry patterns
- Follows `take_token_receiver()` pattern
- Same thread-safety guarantees

### 4. Easier Testing
- Can mock JobRegistry
- No global state to manage
- Cleaner test setup

### 5. Better Lifecycle Management
- Can use `remove_job()` for cleanup
- TTL can be based on `created_at`
- State transitions tracked properly

## Job Lifecycle (Final)

### 1. POST /v1/jobs
```rust
let job_id = registry.create_job();           // Create job
registry.set_payload(&job_id, payload);       // Store payload
return HttpJobResponse { job_id, sse_url };   // Return immediately
```

**Registry State:**
```
Job {
    job_id: "job-123",
    state: Queued,
    created_at: 2025-10-21T03:40:00Z,
    token_receiver: None,
    payload: Some({...}),  ← Stored
}
```

### 2. GET /v1/jobs/{id}/stream
```rust
let payload = registry.take_payload(&job_id);  // Retrieve (one-time)
if let Some(payload) = payload {
    tokio::spawn(async {
        route_job(router_state, payload).await;  // Execute
    });
}
let receiver = registry.take_token_receiver(&job_id);  // Stream
```

**Registry State After:**
```
Job {
    job_id: "job-123",
    state: Running,  ← Updated by job execution
    created_at: 2025-10-21T03:40:00Z,
    token_receiver: None,  ← Taken for streaming
    payload: None,  ← Consumed
}
```

## API Consistency

JobRegistry now has symmetric operations:

| Operation | Receiver | Payload |
|-----------|----------|---------|
| **Set** | `set_token_receiver()` | `set_payload()` |
| **Take** | `take_token_receiver()` | `take_payload()` |
| **Pattern** | One-time use | One-time use |

Both follow the same pattern:
1. Set during job creation
2. Take when needed (consumes)
3. Can only be called once

## Testing

```bash
# Compile check
cargo check -p job-registry
cargo check -p queen-rbee

# Test job creation
curl -X POST http://localhost:8500/v1/jobs \
  -d '{"operation": "hive_list"}'

# Response:
{
  "job_id": "job-550e8400-e29b-41d4-a716-446655440000",
  "sse_url": "/v1/jobs/job-550e8400-e29b-41d4-a716-446655440000/stream"
}

# Test streaming (triggers execution)
curl http://localhost:8500/v1/jobs/job-550e8400-e29b-41d4-a716-446655440000/stream
```

## Files Modified

### Shared Crate
- `bin/99_shared_crates/job-registry/src/lib.rs` - Added payload field and methods
- `bin/99_shared_crates/job-registry/Cargo.toml` - Added serde_json dependency

### Queen-rbee
- `bin/10_queen_rbee/src/http.rs` - Use JobRegistry instead of PENDING_JOBS
- `bin/10_queen_rbee/src/main.rs` - Pass full SchedulerState
- `bin/10_queen_rbee/Cargo.toml` - Removed once_cell dependency

## Migration Notes

### For Other Services

If other services (rbee-hive, llm-worker) need deferred execution:

1. **Already available:** `set_payload()` and `take_payload()` methods
2. **No changes needed:** JobRegistry is a shared crate
3. **Same pattern:** Create job → Set payload → Take on connect → Execute

### Future Enhancements

The JobRegistry can be further extended:

```rust
pub struct Job<T> {
    // ... existing fields ...
    pub metadata: HashMap<String, String>,  // Generic metadata
    pub timeout: Option<Duration>,          // Job timeout
    pub retry_count: u32,                   // Retry attempts
    pub priority: u8,                       // Job priority
}
```

## Comparison: Before vs After

### Lines of Code
- **Before:** ~60 lines (PENDING_JOBS + PendingJob struct + imports)
- **After:** ~20 lines (JobRegistry methods)
- **Savings:** ~40 lines removed

### Dependencies
- **Before:** once_cell, std::collections::HashMap, std::sync::Mutex
- **After:** None (all in JobRegistry)

### Complexity
- **Before:** Two separate data structures to manage
- **After:** One centralized registry

### Maintainability
- **Before:** Need to keep PENDING_JOBS and JobRegistry in sync
- **After:** Single source of truth

## Conclusion

By integrating payload storage into the existing `JobRegistry`, we achieved:

✅ **Cleaner code** - Removed ~40 lines of boilerplate  
✅ **Single source of truth** - All job data in one place  
✅ **Better consistency** - Follows existing patterns  
✅ **Easier testing** - No global state  
✅ **Simpler dependencies** - Removed once_cell  

The deferred execution pattern is now fully integrated into the shared job-registry crate, making it available for all services in the rbee ecosystem.
