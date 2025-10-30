# TEAM-353: Worker Job-Based Architecture Migration Plan

**Date:** Oct 30, 2025  
**Status:** PLAN  
**Current:** Worker already uses dual-call pattern (TEAM-154)  
**Goal:** Align with Hive/Queen pattern using operations-contract

---

## Current State Analysis

### What Worker Already Has ✅

**Dual-Call Pattern (TEAM-154):**
```
POST /v1/inference → returns {job_id, sse_url}
GET /v1/inference/{job_id}/stream → SSE stream
```

**Components:**
- `JobRegistry<TokenResponse>` - Job lifecycle management
- `RequestQueue` - Inference request queue
- `WorkerState` - Combines queue + registry
- Token streaming via MPSC channels

### What's Different from Hive/Queen ❌

**Hive/Queen Pattern:**
```rust
POST /v1/jobs → accepts Operation enum from operations-contract
GET /v1/jobs/{job_id}/stream → SSE stream
```

**Key Differences:**
1. Worker uses `/v1/inference` - should be `/v1/jobs`
2. Worker uses custom `ExecuteRequest` - should use `Operation` enum
3. Worker doesn't parse operations - should use `operations-contract`
4. Worker uses `JobRegistry<TokenResponse>` - should use `JobRegistry<String>` (narration)

---

## Migration Plan

### Phase 1: Add Operations Support

**Goal:** Accept `Operation::Infer` from operations-contract

**Files to Create:**
1. `src/job_router.rs` - Parse operations and route to handlers
   - Parse `Operation::Infer` from JSON payload
   - Extract prompt, model, sampling params
   - Call existing inference logic
   - Return `JobResponse {job_id, sse_url}`

**Files to Modify:**
1. `src/http/routes.rs`
   - Add route: `POST /v1/jobs` → `handle_create_job`
   - Keep `/v1/inference` for backwards compatibility (deprecated)
   - Add `operations-contract` dependency

2. `src/http/execute.rs`
   - Rename to `src/http/jobs.rs`
   - Split into:
     - `handle_create_job()` - thin HTTP wrapper
     - Delegate to `job_router::create_job()`

**Dependencies to Add:**
```toml
operations-contract = { path = "../../97_contracts/operations-contract" }
```

---

### Phase 2: Align with Hive Pattern

**Goal:** Match Hive's job_router structure

**Create `src/job_router.rs`:**
```rust
// TEAM-353: Worker job router - mirrors Hive/Queen pattern

use operations_contract::Operation;
use job_server::JobRegistry;
use crate::backend::request_queue::RequestQueue;

pub struct JobState {
    pub registry: Arc<JobRegistry<String>>, // Changed from TokenResponse
    pub queue: Arc<RequestQueue>,
}

pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

/// Create job from operation
pub async fn create_job(
    state: JobState,
    payload: serde_json::Value,
) -> Result<JobResponse> {
    // Parse operation
    let operation: Operation = serde_json::from_value(payload)?;
    
    // Route to handler
    match operation {
        Operation::Infer(req) => execute_infer(state, req).await,
        _ => Err(anyhow!("Unsupported operation")),
    }
}

/// Execute inference operation
async fn execute_infer(
    state: JobState,
    req: InferRequest,
) -> Result<JobResponse> {
    // Create job
    let job_id = state.registry.create_job();
    
    // Create SSE channel for narration
    observability_narration_core::sse_sink::create_job_channel(job_id.clone(), 1000);
    
    // Convert to GenerationRequest
    let gen_req = GenerationRequest {
        request_id: job_id.clone(),
        prompt: req.prompt,
        config: SamplingConfig::from(req),
        response_tx, // Token channel
    };
    
    // Add to queue
    state.queue.add_request(gen_req)?;
    
    Ok(JobResponse {
        job_id,
        sse_url: format!("/v1/jobs/{}/stream", job_id),
    })
}
```

---

### Phase 3: Update Registry Type

**Goal:** Change from `JobRegistry<TokenResponse>` to `JobRegistry<String>`

**Problem:** Worker needs BOTH:
- Token streaming (for inference output)
- Narration streaming (for job status)

**Solution:** Dual channels (like Hive does)

**Modify `src/http/stream.rs`:**
```rust
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<WorkerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Take narration receiver (for job status)
    let sse_rx_opt = sse_sink::take_job_receiver(&job_id);
    
    // Take token receiver (for inference output)
    let token_rx_opt = state.registry.take_token_receiver(&job_id);
    
    // Merge both streams
    let combined_stream = async_stream::stream! {
        // Stream narration events
        while let Some(event) = sse_rx.recv().await {
            yield Ok(Event::default().data(&event.formatted));
        }
        
        // Stream tokens
        while let Some(token) = token_rx.recv().await {
            yield Ok(Event::default().data(&token.text));
        }
        
        yield Ok(Event::default().data("[DONE]"));
    };
    
    Sse::new(combined_stream)
}
```

---

### Phase 4: Backwards Compatibility

**Goal:** Support both old and new endpoints during transition

**Routes:**
```rust
// New (job-based)
.route("/v1/jobs", post(jobs::handle_create_job))
.route("/v1/jobs/{job_id}/stream", get(stream::handle_stream_job))

// Old (deprecated, for backwards compatibility)
.route("/v1/inference", post(jobs::handle_create_job)) // Same handler!
.route("/v1/inference/{job_id}/stream", get(stream::handle_stream_job))
```

**Deprecation Plan:**
1. Add both endpoints (v0.2.0)
2. Mark `/v1/inference` as deprecated (v0.3.0)
3. Remove `/v1/inference` (v1.0.0)

---

## Implementation Steps

### Step 1: Add operations-contract ✅
```bash
# Add to Cargo.toml
operations-contract = { path = "../../97_contracts/operations-contract" }
```

### Step 2: Create job_router.rs
- Parse `Operation::Infer` from JSON
- Extract inference parameters
- Create job in registry
- Add to request queue
- Return `JobResponse`

### Step 3: Update HTTP layer
- Rename `execute.rs` → `jobs.rs`
- Add `POST /v1/jobs` route
- Keep `/v1/inference` for backwards compat
- Update `WorkerState` to match Hive pattern

### Step 4: Update streaming
- Change registry type to `JobRegistry<String>`
- Add dual-channel support (narration + tokens)
- Merge streams in SSE handler

### Step 5: Testing
- Test with new `/v1/jobs` endpoint
- Test with old `/v1/inference` endpoint
- Verify both work identically
- Test with Worker UI (uses WASM SDK)

---

## File Changes Summary

### New Files
- `src/job_router.rs` (~200 LOC) - Operation routing

### Modified Files
- `src/http/routes.rs` - Add `/v1/jobs` routes
- `src/http/execute.rs` → `src/http/jobs.rs` - Thin HTTP wrapper
- `src/http/stream.rs` - Dual-channel streaming
- `Cargo.toml` - Add operations-contract dependency

### Deleted Files
- None (backwards compatible)

---

## Benefits

1. **Consistency** - Same pattern as Hive/Queen
2. **Type Safety** - Uses operations-contract enum
3. **Backwards Compatible** - Old endpoints still work
4. **Future Proof** - Ready for more operations (health, status, etc.)
5. **UI Ready** - Worker UI already uses WASM SDK with operations

---

## Risks & Mitigations

### Risk 1: Breaking existing clients
**Mitigation:** Keep `/v1/inference` endpoint, mark as deprecated

### Risk 2: Registry type change
**Mitigation:** Add dual-channel support gradually, test thoroughly

### Risk 3: Token streaming complexity
**Mitigation:** Keep existing token channel, add narration channel separately

---

## Timeline

**Estimated:** 2-3 hours

- Step 1 (dependencies): 5 min
- Step 2 (job_router): 45 min
- Step 3 (HTTP layer): 30 min
- Step 4 (streaming): 45 min
- Step 5 (testing): 30 min

---

## Success Criteria

✅ Worker accepts `Operation::Infer` via `/v1/jobs`  
✅ Old `/v1/inference` endpoint still works  
✅ SSE streaming includes both narration and tokens  
✅ Worker UI works with new endpoints  
✅ All tests pass  
✅ No breaking changes for existing clients  

---

**Ready to implement?** This plan maintains backwards compatibility while aligning with the job-based architecture.
