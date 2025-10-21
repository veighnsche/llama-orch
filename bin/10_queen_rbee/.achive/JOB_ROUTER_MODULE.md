# Job Router Module

**Team:** TEAM-186  
**Date:** 2025-10-21  
**Status:** ✅ Complete

## Summary

Created dedicated `job_router` module to handle all job routing and operation dispatch logic, separating concerns from the HTTP layer.

## Architecture

```
POST /v1/jobs (JSON payload)
    ↓
http::handle_create_job()
    ↓
job_router::route_job()
    ↓
Parse into Operation enum
    ↓
Match and dispatch to handler
    ↓
Execute async in background
    ↓
Stream results via SSE
```

## Files Created

### `src/job_router.rs` (New Module)

**Purpose:** Centralized job routing and operation dispatch

**Responsibilities:**
1. Parse JSON payloads into typed `Operation` enum
2. Create jobs in registry
3. Route operations to appropriate handlers
4. Execute handlers asynchronously
5. Stream results to job registry

**Key Components:**

```rust
pub struct JobRouterState {
    pub registry: Arc<JobRegistry<String>>,
    pub hive_catalog: Arc<HiveCatalog>,
}

pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

pub async fn route_job(
    state: JobRouterState,
    payload: serde_json::Value,
) -> Result<JobResponse>
```

## Operation Handlers

All operation handlers are implemented as stub functions with TODO markers:

### Hive Operations
- `handle_hive_start_job()`
- `handle_hive_stop_job()`
- `handle_hive_list_job()`
- `handle_hive_get_job()`
- `handle_hive_create_job()`
- `handle_hive_update_job()`
- `handle_hive_delete_job()`

### Worker Operations
- `handle_worker_spawn_job()`
- `handle_worker_list_job()`
- `handle_worker_get_job()`
- `handle_worker_delete_job()`

### Model Operations
- `handle_model_download_job()`
- `handle_model_list_job()`
- `handle_model_get_job()`
- `handle_model_delete_job()`

### Inference
- `handle_infer_job()`

## Changes Made

### 1. Created `src/job_router.rs`
- 400+ lines of routing logic
- Exhaustive pattern matching on `Operation` enum
- Stub implementations for all handlers
- Narration events for observability

### 2. Updated `src/lib.rs`
```rust
pub mod job_router;  // TEAM-186: Job routing and operation dispatch
```

### 3. Updated `src/main.rs`
```rust
mod job_router;  // TEAM-186: Job routing and operation dispatch
```

### 4. Updated `src/http.rs`
**Before:**
```rust
pub async fn handle_create_job(...) {
    // Extract operation field
    let operation = payload["operation"].as_str()?;
    
    // TODO: Route to appropriate handler
    let job_id = state.registry.create_job();
    // ...
}
```

**After:**
```rust
pub async fn handle_create_job(...) {
    // Delegate to job_router
    let router_state = crate::job_router::JobRouterState {
        registry: state.registry.clone(),
        hive_catalog: state.hive_catalog.clone(),
    };
    
    let response = crate::job_router::route_job(router_state, payload).await?;
    
    Ok(Json(HttpJobResponse {
        job_id: response.job_id,
        sse_url: response.sse_url,
    }))
}
```

### 5. Updated `Cargo.toml`
```toml
rbee-operations = { path = "../99_shared_crates/rbee-operations" }
```

## Benefits

✅ **Separation of Concerns** - HTTP layer only handles HTTP, routing logic is separate  
✅ **Type Safety** - Uses typed `Operation` enum instead of strings  
✅ **Exhaustive Matching** - Compiler ensures all operations are handled  
✅ **Testability** - Router logic can be tested independently of HTTP  
✅ **Maintainability** - All routing logic in one place  
✅ **Extensibility** - Easy to add new operations  

## Flow Example

### Request
```bash
curl -X POST http://localhost:8500/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "hive_list"}'
```

### Processing
1. **HTTP Layer** (`http.rs`)
   - Receives JSON payload
   - Extracts state
   - Calls `job_router::route_job()`

2. **Router Layer** (`job_router.rs`)
   - Parses JSON into `Operation::HiveList`
   - Creates job in registry
   - Matches operation and calls `handle_hive_list_job()`
   - Returns `JobResponse { job_id, sse_url }`

3. **Handler** (`handle_hive_list_job()`)
   - TODO: Query hive catalog
   - TODO: Stream results to job registry
   - TODO: Emit completion event

4. **Response**
   ```json
   {
     "job_id": "job-550e8400-e29b-41d4-a716-446655440000",
     "sse_url": "/v1/jobs/job-550e8400-e29b-41d4-a716-446655440000/stream"
   }
   ```

## Narration Events

The router emits narration events for observability:

```rust
const ACTOR_QUEEN_ROUTER: &str = "👑 queen-router";
const ACTION_ROUTE_JOB: &str = "route_job";
const ACTION_PARSE_OPERATION: &str = "parse_operation";
```

**Example Events:**
- Parse operation: `👑 queen-router → parse_operation → parsing`
- Route job: `👑 queen-router → route_job → hive_list`
- Job created: `👑 queen-router → route_job → job-123`

## Next Steps

### Phase 1: Implement Hive Operations
- [ ] `handle_hive_list_job()` - Query hive catalog
- [ ] `handle_hive_get_job()` - Get specific hive
- [ ] `handle_hive_start_job()` - Start hive daemon
- [ ] `handle_hive_stop_job()` - Stop hive daemon
- [ ] `handle_hive_create_job()` - Register new hive
- [ ] `handle_hive_update_job()` - Update hive config
- [ ] `handle_hive_delete_job()` - Remove hive

### Phase 2: Implement Worker Operations
- [ ] `handle_worker_spawn_job()` - Spawn worker on hive
- [ ] `handle_worker_list_job()` - List workers on hive
- [ ] `handle_worker_get_job()` - Get worker details
- [ ] `handle_worker_delete_job()` - Delete worker

### Phase 3: Implement Model Operations
- [ ] `handle_model_download_job()` - Download model to hive
- [ ] `handle_model_list_job()` - List models on hive
- [ ] `handle_model_get_job()` - Get model details
- [ ] `handle_model_delete_job()` - Delete model

### Phase 4: Implement Inference
- [ ] `handle_infer_job()` - Run inference on hive

### Phase 5: Add Tests
- [ ] Unit tests for router logic
- [ ] Integration tests for each operation
- [ ] Error handling tests

## Testing

```bash
# Compile check
cargo check -p queen-rbee

# Build
cargo build -p queen-rbee

# Test (when implemented)
cargo test -p queen-rbee job_router
```

## Related Files

- `bin/10_queen_rbee/src/job_router.rs` - Router implementation
- `bin/10_queen_rbee/src/http.rs` - HTTP endpoints
- `bin/10_queen_rbee/src/main.rs` - Module declarations
- `bin/10_queen_rbee/src/lib.rs` - Library exports
- `bin/99_shared_crates/rbee-operations/src/lib.rs` - Operation types
- `bin/API_REFERENCE.md` - API documentation

## Code Organization

```
bin/10_queen_rbee/
├── src/
│   ├── main.rs              # Binary entry point, module declarations
│   ├── lib.rs               # Library exports
│   ├── http.rs              # HTTP endpoints (thin wrappers)
│   ├── job_router.rs        # Job routing logic (NEW)
│   ├── operations.rs        # Operation constants
│   ├── health.rs            # Health endpoint
│   └── heartbeat.rs         # Heartbeat endpoint
```

## Design Principles

1. **Single Responsibility** - Each module has one clear purpose
2. **Dependency Inversion** - HTTP depends on router, not vice versa
3. **Type Safety** - Use enums instead of strings
4. **Exhaustive Matching** - Compiler checks all cases
5. **Async by Default** - All handlers are async
6. **Observable** - Narration events for all operations
7. **Testable** - Pure functions, no hidden dependencies
