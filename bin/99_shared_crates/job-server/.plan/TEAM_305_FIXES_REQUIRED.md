# TEAM-305 Critical & Major Fixes Required

**Date:** October 26, 2025  
**Status:** Implementation Plan  
**Priority:** HIGH

---

## Critical Issues (Must Fix)

### 🔴 CRITICAL #1: String-Based Error Detection

**Problem:** Using `error_msg.contains("cancelled by user")` is fragile and error-prone.

**Fix:** Create proper `JobError` enum

**Files to Change:**
- `job-server/src/lib.rs` - Add JobError enum
- `job-server/src/lib.rs` - Update execute_and_stream_with_timeout to use JobError

**Code:**
```rust
#[derive(Debug)]
pub enum JobError {
    Cancelled,
    Timeout(Duration),
    ExecutionFailed(String),
}
```

---

### 🔴 CRITICAL #2: job-client Doesn't Handle [CANCELLED]

**Problem:** Added [CANCELLED] signal but client doesn't check for it.

**Fix:** Add [CANCELLED] handling to job-client

**Files to Change:**
- `job-client/src/lib.rs` - Add check for [CANCELLED] signal

**Code:**
```rust
// Check for [CANCELLED] marker
if data.contains("[CANCELLED]") {
    return Err(anyhow::anyhow!("Job was cancelled"));
}
```

---

### 🔴 CRITICAL #3: No HTTP Endpoint to Cancel Jobs

**Problem:** Implemented cancel_job() but no HTTP endpoint to call it.

**Fix:** Add DELETE /v1/jobs/{job_id} endpoint

**Files to Change:**
- `queen-rbee/src/http/jobs.rs` - Add cancel_job handler
- `rbee-hive/src/http/jobs.rs` - Add cancel_job handler

**Code:**
```rust
/// DELETE /v1/jobs/{job_id} - Cancel a job
async fn cancel_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let cancelled = state.registry.cancel_job(&job_id);
    
    if cancelled {
        Ok(Json(serde_json::json!({
            "job_id": job_id,
            "status": "cancelled"
        })))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            format!("Job {} not found or cannot be cancelled", job_id)
        ))
    }
}
```

---

## Major Issues (Should Fix)

### 🟡 MAJOR #4: Race Condition in State Updates

**Problem:** Stream may check state before executor updates it.

**Fix:** Use oneshot channel to synchronize state updates

**Files to Change:**
- `job-server/src/lib.rs` - Add oneshot channel for result communication

**Code:**
```rust
let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();

// In executor:
let _ = result_tx.send(final_state);

// In stream:
let final_state = result_rx.try_recv().ok();
```

---

### 🟡 MAJOR #5: Timeout Not Configurable per Job Type

**Problem:** Same timeout for all jobs.

**Fix:** Add timeout field to Job struct

**Files to Change:**
- `job-server/src/lib.rs` - Add timeout field to Job
- `job-server/src/lib.rs` - Add create_job_with_timeout method
- `job-server/src/lib.rs` - Read timeout from job in execute_and_stream_with_timeout

**Code:**
```rust
pub struct Job<T> {
    // ... existing fields ...
    pub timeout: Option<Duration>,
}

pub fn create_job_with_timeout(&self, timeout: Option<Duration>) -> String {
    // ...
}
```

---

## Implementation Order

1. ✅ **CRITICAL #1** - JobError enum (30 min)
2. ✅ **CRITICAL #2** - job-client [CANCELLED] handling (10 min)
3. ✅ **CRITICAL #3** - HTTP cancel endpoint (30 min)
4. ✅ **MAJOR #4** - Fix race condition (45 min)
5. ✅ **MAJOR #5** - Configurable timeout (30 min)

**Total Time:** ~2.5 hours

---

## Testing Plan

1. Update existing tests for JobError
2. Add test for job-client [CANCELLED] handling
3. Add integration test for HTTP cancel endpoint
4. Add test for race condition fix
5. Add test for per-job timeout

---

## Success Criteria

- ✅ All tests pass
- ✅ No string-based error detection
- ✅ job-client handles all signals ([DONE], [ERROR], [CANCELLED])
- ✅ Can cancel jobs via HTTP
- ✅ No race conditions in state updates
- ✅ Timeout configurable per job

---

**Let's implement these fixes now.**
