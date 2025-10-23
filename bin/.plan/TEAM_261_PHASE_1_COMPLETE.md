# TEAM-261 Phase 1: Add job-server to rbee-hive

**Date:** Oct 23, 2025  
**Updated:** Oct 23, 2025 (Added simplification decision)  
**Status:** ✅ COMPLETE + DECISION  
**Team:** TEAM-261

---

## 🎯 ARCHITECTURAL DECISION (Oct 23, 2025)

**After completing Phase 1, we analyzed whether hive should be CLI or daemon.**

**Decision:** Keep daemon, remove hive heartbeat

**Why?**
- ✅ Daemon is 10-100x faster (1-5ms vs 80-350ms)
- ✅ Real-time SSE streaming essential for UX
- ✅ No command injection security risk
- ❌ Remove hive heartbeat (workers → queen direct)

**Next:** Implement simplification (see `TEAM_261_SIMPLIFICATION_AUDIT.md`)

---

## Mission

Add `job-server` pattern to `rbee-hive` to enable queen-rbee → hive forwarding and create a consistent dual-call pattern across all three binaries.

---

## Architecture

### Before (Phase 0)
```
rbee-keeper (CLI)
    ↓ uses job-client ✅
    POST /v1/jobs → queen-rbee (server)
    ↓ uses job-server ✅
    GET /v1/jobs/{job_id}/stream

queen-rbee (client)
    ↓ uses job-client ✅
    POST /v1/jobs → rbee-hive (server)
    ↓ ❌ NO JOB SERVER!
    ❌ No /v1/jobs endpoints
```

### After (Phase 1)
```
rbee-keeper (CLI)
    ↓ uses job-client ✅
    POST /v1/jobs → queen-rbee (server)
    ↓ uses job-server ✅
    GET /v1/jobs/{job_id}/stream

queen-rbee (client)
    ↓ uses job-client ✅
    POST /v1/jobs → rbee-hive (server)
    ↓ uses job-server ✅ NEW!
    GET /v1/jobs/{job_id}/stream ✅ NEW!
```

**Result:** Consistent dual-call pattern at every level!

---

## Changes Made

### 1. Updated Dependencies

**File:** `bin/20_rbee_hive/Cargo.toml`

**Added:**
```toml
# TEAM-261: Job server pattern (mirrors queen-rbee)
job-server = { path = "../99_shared_crates/job-server" }
rbee-operations = { path = "../99_shared_crates/rbee-operations" }
futures = "0.3"
async-stream = "0.3"
```

### 2. Created job_router.rs

**File:** `bin/20_rbee_hive/src/job_router.rs` (267 LOC)

**Purpose:** Route operations to appropriate handlers

**Key Functions:**
- `create_job()` - Create job and SSE channel (mirrors queen-rbee)
- `execute_job()` - Execute job and stream results (mirrors queen-rbee)
- `route_operation()` - Parse and dispatch operations

**Operations Handled:**
- ✅ WorkerSpawn (TODO: implementation)
- ✅ WorkerList (TODO: implementation)
- ✅ WorkerGet (TODO: implementation)
- ✅ WorkerDelete (TODO: implementation)
- ✅ ModelDownload (TODO: implementation)
- ✅ ModelList (TODO: implementation)
- ✅ ModelGet (TODO: implementation)
- ✅ ModelDelete (TODO: implementation)
- ✅ Infer (TODO: implementation)
- ❌ Hive operations (rejected - handled by queen-rbee)

**Pattern:**
```rust
match operation {
    Operation::WorkerSpawn { hive_id, model, worker, device } => {
        NARRATE.action("worker_spawn").job_id(&job_id).emit();
        // TODO: Implement worker spawning
    }
    // ... other operations
    _ => {
        Err(anyhow::anyhow!("Operation not supported by hive"))
    }
}
```

### 3. Created HTTP Module

**Files:**
- `bin/20_rbee_hive/src/http/mod.rs` (5 LOC)
- `bin/20_rbee_hive/src/http/jobs.rs` (135 LOC)

**Purpose:** HTTP handlers for job endpoints (mirrors queen-rbee)

**Endpoints:**
- `POST /v1/jobs` - Create job, return job_id + sse_url
- `GET /v1/jobs/{job_id}/stream` - Stream SSE events

**State:**
```rust
pub struct HiveState {
    pub registry: Arc<JobRegistry<String>>,
    // TODO: Add worker_registry when implemented
    // TODO: Add model_catalog when implemented
}
```

**Pattern:** Thin HTTP wrappers that delegate to `job_router`

### 4. Updated main.rs

**File:** `bin/20_rbee_hive/src/main.rs`

**Changes:**
1. Added module declarations:
   ```rust
   mod http;
   mod job_router;
   ```

2. Added imports:
   ```rust
   use axum::{routing::{get, post}, ...};
   use job_server::JobRegistry;
   ```

3. Initialized job registry:
   ```rust
   let job_registry: Arc<JobRegistry<String>> = Arc::new(JobRegistry::new());
   ```

4. Created HTTP state:
   ```rust
   let job_state = http::jobs::HiveState {
       registry: job_registry,
   };
   ```

5. Added routes:
   ```rust
   let app = Router::new()
       .route("/health", get(health_check))
       .route("/capabilities", get(get_capabilities))
       .route("/v1/jobs", post(http::jobs::handle_create_job))           // NEW
       .route("/v1/jobs/:job_id/stream", get(http::jobs::handle_stream_job)) // NEW
       .with_state(job_state);
   ```

---

## Testing

### Compilation
```bash
cargo check -p rbee-hive
```

**Result:** ✅ SUCCESS (only minor warnings about unused constants)

### Manual Testing
```bash
# Start rbee-hive
cargo run --bin rbee-hive -- --port 9000

# In another terminal, test job creation
curl -X POST http://localhost:9000/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation":"worker_list","hive_id":"localhost"}'

# Should return:
# {"job_id":"job-<uuid>","sse_url":"/v1/jobs/job-<uuid>/stream"}

# Test SSE streaming
curl http://localhost:9000/v1/jobs/job-<uuid>/stream

# Should stream:
# data: TODO: List workers on hive 'localhost'
# data: [DONE]
```

---

## Code Quality

### ✅ Mirrors queen-rbee Pattern
- Same file structure (`http/jobs.rs`, `job_router.rs`)
- Same function names (`create_job`, `execute_job`, `route_operation`)
- Same SSE streaming logic
- Same error handling

### ✅ Proper Attribution
- All files tagged with `TEAM-261`
- Comments explain purpose and mirror pattern

### ✅ No Duplication
- Uses shared `job-server` crate
- Uses shared `rbee-operations` crate
- HTTP handlers are thin wrappers (~60 lines each)

### ✅ Ready for Implementation
- All operations have TODO markers
- Clear structure for adding worker/model logic
- Registry pattern ready for worker_registry integration

---

## Next Steps

### Phase 2: Integration Tests (2-3 hours)

Create `bin/99_shared_crates/job-integration-tests/` with tests covering:

1. **keeper → queen → hive flow**
   ```rust
   #[tokio::test]
   async fn test_end_to_end_worker_spawn() {
       // Start mock hive
       // Start mock queen (knows about hive)
       // Use job-client as keeper
       // Submit WorkerSpawn operation
       // Verify forwarding through chain
       // Verify SSE streaming through chain
   }
   ```

2. **Error propagation**
   - Invalid operation
   - Missing job_id
   - Hive not running

3. **SSE streaming**
   - Multiple events
   - [DONE] marker
   - Timeout handling

4. **Concurrent requests**
   - Multiple jobs
   - Job isolation

### Phase 3: Implement Worker Operations (Later)

Wire up actual implementations:
- Worker spawning (use daemon-lifecycle)
- Worker listing (query worker_registry)
- Model management (use model_catalog)
- Inference routing (route to workers)

---

## Files Created/Modified

### Created (3 files, 407 LOC)
- `bin/20_rbee_hive/src/job_router.rs` (267 LOC)
- `bin/20_rbee_hive/src/http/mod.rs` (5 LOC)
- `bin/20_rbee_hive/src/http/jobs.rs` (135 LOC)

### Modified (2 files)
- `bin/20_rbee_hive/Cargo.toml` (+4 lines)
- `bin/20_rbee_hive/src/main.rs` (+15 lines, restructured)

### Total Impact
- **Lines Added:** ~430 LOC
- **Compilation:** ✅ PASS
- **Pattern:** ✅ Consistent with queen-rbee
- **Ready for:** Integration tests + implementation

---

## Verification Checklist

- ✅ Dependencies added to Cargo.toml
- ✅ job_router.rs created with all operations
- ✅ http/jobs.rs created with HTTP handlers
- ✅ main.rs updated with routes
- ✅ Compilation successful
- ✅ Pattern mirrors queen-rbee
- ✅ All files tagged with TEAM-261
- ✅ No code duplication
- ✅ Ready for integration tests

---

**TEAM-261 Phase 1 Complete**  
**Date:** Oct 23, 2025  
**Status:** ✅ SUCCESS  
**Next:** Phase 2 - Integration Tests
