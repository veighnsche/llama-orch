# FT-R001: Cancellation Endpoint

**Team**: Foundation-Alpha  
**Sprint**: Sprint 2 - FFI Layer  
**Size**: S (1 day)  
**Type**: Retroactive (added post-Sprint 1)  
**Spec Ref**: M0-W-1330, WORK-3044, SYS-6.3.5

---

## Story Description

Implement POST /cancel endpoint for idempotent job cancellation. This enables clients to stop running inference jobs, free resources, and receive proper error events via SSE.

---

## Acceptance Criteria

- [ ] POST /cancel endpoint accepts `{"job_id": "..."}` request
- [ ] Returns HTTP 202 Accepted immediately
- [ ] Cancellation is idempotent (repeated cancels are safe)
- [ ] Stops decoding within 100ms of receiving cancel request
- [ ] Frees VRAM buffers and resources
- [ ] Emits SSE `error` event with code `CANCELLED`
- [ ] Cancellation completes within 5s deadline (per spec)
- [ ] Unit tests validate cancellation logic
- [ ] Integration tests validate end-to-end cancellation flow
- [ ] Cancellation works for both active and completed jobs

---

## Dependencies

### Upstream (Blocks This Story)
- FT-006: FFI interface definition (need cancellation FFI function)
- FT-010: CUDA context init (need active job tracking)

### Downstream (This Story Blocks)
- FT-024: HTTP-FFI-CUDA integration test (needs cancellation for timeout tests)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/src/http/cancel.rs` - Cancellation handler
- `bin/worker-orcd/src/http/routes.rs` - Add cancel route
- `bin/worker-orcd/src/http/mod.rs` - Export cancel module
- `bin/worker-orcd/tests/cancel_endpoint_integration.rs` - Integration tests

### Key Interfaces

#### Request Format (M0-W-1330)
```json
{
  "job_id": "job-xyz"
}
```

#### Response Format
```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{
  "status": "cancelling",
  "job_id": "job-xyz"
}
```

#### SSE Error Event (emitted to active stream)
```json
event: error
data: {
  "code": "CANCELLED",
  "message": "Job cancelled by client request"
}
```

### Implementation Notes

#### 1. Job State Tracking
```rust
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Active job tracker
pub struct JobTracker {
    /// Map of job_id -> cancellation flag
    active_jobs: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

impl JobTracker {
    pub fn register(&self, job_id: String) -> Arc<AtomicBool> {
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.active_jobs.lock().unwrap().insert(job_id, cancel_flag.clone());
        cancel_flag
    }
    
    pub fn cancel(&self, job_id: &str) -> bool {
        if let Some(flag) = self.active_jobs.lock().unwrap().get(job_id) {
            flag.store(true, Ordering::SeqCst);
            true
        } else {
            false // Job not found or already completed
        }
    }
    
    pub fn unregister(&self, job_id: &str) {
        self.active_jobs.lock().unwrap().remove(job_id);
    }
}
```

#### 2. Cancel Handler
```rust
use crate::http::routes::AppState;
use axum::{extract::{Extension, State}, Json};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD};

#[derive(Debug, Deserialize)]
pub struct CancelRequest {
    pub job_id: String,
}

#[derive(Debug, Serialize)]
pub struct CancelResponse {
    pub status: String,
    pub job_id: String,
}

/// Handle POST /cancel
///
/// Cancels an active inference job. This is idempotent - repeated
/// cancels for the same job_id are safe.
///
/// # Spec References
/// - M0-W-1330: Cancellation endpoint
/// - WORK-3044: Cancellation semantics
pub async fn handle_cancel(
    Extension(correlation_id): Extension<String>,
    State(state): State<AppState>,
    Json(req): Json<CancelRequest>,
) -> Json<CancelResponse> {
    info!(
        correlation_id = %correlation_id,
        job_id = %req.job_id,
        "Cancellation requested"
    );
    
    // Attempt to cancel the job
    let cancelled = state.job_tracker.cancel(&req.job_id);
    
    if cancelled {
        // Job was active and is now being cancelled
        Narration::new(ACTOR_WORKER_ORCD, "cancel", &req.job_id)
            .human(format!("Cancelling job {}", req.job_id))
            .correlation_id(&correlation_id)
            .job_id(&req.job_id)
            .emit();
    } else {
        // Job not found or already completed (idempotent)
        warn!(
            correlation_id = %correlation_id,
            job_id = %req.job_id,
            "Cancel requested for unknown or completed job"
        );
        
        Narration::new(ACTOR_WORKER_ORCD, "cancel", &req.job_id)
            .human(format!("Cancel requested for unknown job {}", req.job_id))
            .correlation_id(&correlation_id)
            .job_id(&req.job_id)
            .emit_warn();
    }
    
    // Always return 202 Accepted (idempotent)
    Json(CancelResponse {
        status: "cancelling".to_string(),
        job_id: req.job_id,
    })
}
```

#### 3. Inference Loop Integration
```rust
// In execute handler, check cancellation flag periodically
let cancel_flag = state.job_tracker.register(req.job_id.clone());

// During token generation loop
while let Some(token) = inference.next_token()? {
    // Check cancellation flag
    if cancel_flag.load(Ordering::SeqCst) {
        // Emit cancellation error event
        let event = InferenceEvent::Error {
            code: error_codes::CANCELLED.to_string(),
            message: "Job cancelled by client request".to_string(),
        };
        
        // Send event and break
        tx.send(event).await?;
        break;
    }
    
    // Emit token event
    // ...
}

// Cleanup
state.job_tracker.unregister(&req.job_id);
```

#### 4. Idempotency
- Cancelling a non-existent job returns 202 (not 404)
- Cancelling an already-completed job returns 202
- Cancelling an already-cancelled job returns 202
- This matches spec requirement for idempotent cancellation

#### 5. Resource Cleanup
```rust
// When cancellation detected:
// 1. Stop token generation loop
// 2. Free CUDA inference handle
// 3. Free KV cache buffers
// 4. Emit SSE error event
// 5. Close SSE stream
// 6. Unregister job from tracker
```

---

## Testing Strategy

### Unit Tests
- Test CancelRequest deserialization
- Test CancelResponse serialization
- Test JobTracker::register() creates cancel flag
- Test JobTracker::cancel() sets flag to true
- Test JobTracker::cancel() returns false for unknown job
- Test JobTracker::unregister() removes job
- Test idempotency: multiple cancels are safe

### Integration Tests
- Test POST /cancel with active job returns 202
- Test POST /cancel with unknown job returns 202 (idempotent)
- Test POST /cancel with completed job returns 202 (idempotent)
- Test cancellation stops token generation
- Test SSE stream receives error event with CANCELLED code
- Test resources are freed after cancellation
- Test cancellation completes within 5s deadline
- Test correlation ID propagates through cancellation

### Manual Verification
1. Start server: `cargo run -- --port 8080 --model test.gguf --gpu-device 0`
2. Start inference: `curl -X POST http://localhost:8080/execute -d '{"job_id":"test-123","prompt":"Write a long story...","max_tokens":1000,"temperature":0.7,"seed":42}'`
3. Cancel job: `curl -X POST http://localhost:8080/cancel -d '{"job_id":"test-123"}'`
4. Verify SSE stream receives error event with code "CANCELLED"
5. Verify 202 response
6. Cancel again: `curl -X POST http://localhost:8080/cancel -d '{"job_id":"test-123"}'`
7. Verify still 202 (idempotent)

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (7+ tests)
- [ ] Integration tests passing (8+ tests)
- [ ] Documentation updated (cancel handler docs, JobTracker docs)
- [ ] Narration integration complete
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §7.4 Cancellation Endpoint (M0-W-1330)
- Related Stories: FT-006 (FFI layer), FT-010 (CUDA context)
- Parent Requirement: SYS-6.3.5 (Cancellation Handling)

---

**Status**: 📋 Ready for execution (after FT-006, FT-010)  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04  
**Type**: Retroactive (identified in Sprint 1 retrospective)

---
Planned by Project Management Team 📋  
*Added retroactively to ensure M0-W-1330 compliance*

---

## 🎀 Narration Opportunities (v0.2.0)

**From**: Narration-Core Team  
**Updated**: 2025-10-04 (v0.2.0)

### Critical Events to Narrate

#### 1. Cancellation Request Received (INFO level) ℹ️
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD};

Narration::new(ACTOR_WORKER_ORCD, "cancel", &req.job_id)
    .human(format!("Cancelling job {}", req.job_id))
    .correlation_id(&correlation_id)
    .job_id(&req.job_id)
    .emit();
```

#### 2. Cancellation Completed (INFO level) ℹ️
```rust
Narration::new(ACTOR_WORKER_ORCD, "cancel_complete", &job_id)
    .human(format!("Job {} cancelled successfully", job_id))
    .correlation_id(&correlation_id)
    .job_id(&job_id)
    .emit();
```

#### 3. Cancel for Unknown Job (WARN level) ⚠️
```rust
Narration::new(ACTOR_WORKER_ORCD, "cancel", &req.job_id)
    .human(format!("Cancel requested for unknown job {}", req.job_id))
    .correlation_id(&correlation_id)
    .job_id(&req.job_id)
    .emit_warn();
```

### Why This Matters

**Cancellation events** are critical for:
- 🔗 **Request tracing** (cancel → inference stop)
- 🐛 **Debugging** timeout scenarios
- 📊 **Metrics** (cancellation rate, reasons)
- 🚨 **Alerting** on excessive cancellations
- 📈 **SLO tracking** (cancellation latency)

---

## 🔍 Testing Team Requirements

**From**: Testing Team (Pre-Development Audit)

### Unit Testing Requirements
- **Test CancelRequest deserialization** (JSON parsing)
- **Test CancelResponse serialization** (JSON output)
- **Test JobTracker::register() creates cancel flag** (state management)
- **Test JobTracker::cancel() sets flag** (atomic operation)
- **Test JobTracker::cancel() for unknown job** (idempotency)
- **Test JobTracker::unregister() removes job** (cleanup)
- **Test multiple cancels are idempotent** (repeated calls)

### Integration Testing Requirements
- **Test POST /cancel with active job returns 202** (happy path)
- **Test POST /cancel with unknown job returns 202** (idempotency)
- **Test POST /cancel with completed job returns 202** (idempotency)
- **Test cancellation stops token generation** (inference loop)
- **Test SSE stream receives CANCELLED error event** (event emission)
- **Test resources freed after cancellation** (VRAM cleanup)
- **Test cancellation completes within 5s** (deadline)
- **Test correlation ID in cancel logs** (observability)

### BDD Testing Requirements (VERY IMPORTANT)
- **Scenario**: Cancel active inference job
  - Given an active inference job with job_id "test-123"
  - When I POST to /cancel with {"job_id": "test-123"}
  - Then I should receive 202 Accepted
  - And the SSE stream should receive error event with code "CANCELLED"
  - And token generation should stop within 100ms
- **Scenario**: Cancel unknown job (idempotent)
  - Given no active job with job_id "unknown-123"
  - When I POST to /cancel with {"job_id": "unknown-123"}
  - Then I should receive 202 Accepted
  - And no error should occur
- **Scenario**: Cancel already-completed job (idempotent)
  - Given a completed job with job_id "completed-123"
  - When I POST to /cancel with {"job_id": "completed-123"}
  - Then I should receive 202 Accepted
  - And no error should occur

### Critical Paths to Test
- Cancellation flag propagation to inference loop
- SSE error event emission
- Resource cleanup (VRAM, handles)
- Idempotency for all job states
- Deadline compliance (5s max)

### Edge Cases
- Cancel during model loading
- Cancel during first token generation
- Cancel during last token generation
- Cancel after job completed naturally
- Cancel after job failed with error
- Multiple simultaneous cancels for same job
- Cancel with empty job_id
- Cancel with invalid job_id format

---

## Implementation Plan

### Phase 1: Job Tracking Infrastructure
1. Create `JobTracker` struct with HashMap<String, Arc<AtomicBool>>
2. Add `job_tracker` to `AppState`
3. Register jobs in execute handler
4. Unregister jobs on completion/error

### Phase 2: Cancel Handler
1. Create `cancel.rs` with handler function
2. Extract job_id from request
3. Call `job_tracker.cancel()`
4. Return 202 Accepted (always)
5. Add narration events

### Phase 3: Inference Loop Integration
1. Pass cancel flag to inference loop
2. Check flag periodically (every token or every 100ms)
3. Emit CANCELLED error event when flag set
4. Break loop and cleanup resources

### Phase 4: Testing
1. Unit tests for JobTracker
2. Unit tests for cancel handler
3. Integration tests for end-to-end flow
4. Idempotency tests

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (7+ tests)
- [ ] Integration tests passing (8+ tests)
- [ ] Documentation updated (cancel handler docs, JobTracker docs)
- [ ] Narration integration complete
- [ ] M0-W-1330 requirement satisfied
- [ ] Story marked complete in day-tracker.md

---

## M0 Specification Requirements

### M0-W-1330: POST /cancel

**From M0 Spec §7.4**:

Worker-orcd MUST expose cancellation endpoint:

**Request**:
```json
{
  "job_id": "job-xyz"
}
```

**Response**: HTTP 202 Accepted

**Semantics**:
- ✅ Idempotent: Repeated cancels for same `job_id` are safe
- ✅ Stop decoding promptly (within 100ms)
- ✅ Free resources (VRAM buffers)
- ✅ Emit SSE `error` event with code `CANCELLED`

**Deadline**: Cancellation MUST complete within 5s.

**Spec Reference**: WORK-3044, SYS-6.3.5

---

## Why This Story is Retroactive

**Identified During**: Sprint 1 retrospective (2025-10-04)

**Reason**: M0-W-1330 (POST /cancel) is a **required M0 feature** but was not included in the original Sprint 2 plan. The Sprint 2 stories focused on FFI infrastructure (FT-006 through FT-010) but did not include the HTTP endpoint for cancellation.

**Impact**: Without this story, M0 would be incomplete (missing required endpoint).

**Priority**: HIGH - Required for M0 compliance

---

## Integration with Existing Work

### Builds On (Sprint 1)
- ✅ HTTP server infrastructure (FT-001)
- ✅ Correlation ID middleware (FT-004)
- ✅ SSE event types (FT-003) - uses `InferenceEvent::Error` with `CANCELLED` code

### Requires (Sprint 2)
- ⏳ FFI interface (FT-006) - needs CUDA cancellation function
- ⏳ CUDA context (FT-010) - needs active job state

### Enables
- FT-024: HTTP-FFI-CUDA integration test (can test cancellation scenarios)
- M0 completion (satisfies M0-W-1330 requirement)

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04  
**Type**: Retroactive Addition

---
Planned by Project Management Team 📋  
*Added retroactively to ensure M0 compliance*

---
Built by Foundation-Alpha 🏗️

---

## 📋 Project Management Review (2025-10-04)

**Reviewed by**: Project Management Team  
**Finding**: **This story is NOT retroactive work**

### Analysis

After reviewing the M0 specification (`bin/.specs/01_M0_worker_orcd.md`), the cancellation endpoint was **always a required M0 feature**:

- **M0-W-1330** (§7.4, lines 1262-1282): POST /cancel endpoint is **✅ Required** for M0
- References parent requirement **M0-SYS-6.3.5** (Cancellation Handling)
- Full specification with request/response format, semantics, and 5s deadline
- Marked as "✅ Required" in the endpoint summary table (line 3043)

### Correct Characterization

This story should be characterized as:
- ✅ **Planned M0 work** (not discovered work)
- ✅ **Correctly deferred** from Sprint 1 to Sprint 2 (foundation before cancellation)
- ✅ **Part of original Sprint 2 scope** (Day 18 in Sprint 2 plan)

### Why the Confusion?

The Sprint 1 retrospective incorrectly stated:
- ❌ "M0-W-1330 (POST /cancel) is required for M0 but was **missing from Sprint 2 plan**"
- **Reality**: It was always in Sprint 2 plan as FT-R001 (Day 18)

The "retroactive" label appears to be a documentation error, not a reflection of actual project history.

### Recommendation

Remove the "Retroactive" designation from this story. It is **planned Sprint 2 work** for a **known M0 requirement** that was **correctly sequenced** after HTTP foundation (Sprint 1) and FFI layer (Sprint 2 Days 10-17).

---

**Review Status**: ✅ Complete  
**Spec Compliance**: ✅ M0-W-1330 verified in spec  
**Recommendation**: Remove "Retroactive" label

---
Coordinated by Project Management Team 📋
