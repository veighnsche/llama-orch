# Narration, Job IDs, and Correlation IDs: Architecture Decision

**Created:** 2025-10-22 03:33 AM  
**Context:** Resolving confusion about job_id vs correlation_id in narration

---

## TL;DR

**Keep BOTH `job_id` and `correlation_id` - they serve different purposes:**

- **`job_id`**: Local to each service, used for SSE channel isolation (security)
- **`correlation_id`**: Global, flows end-to-end for request tracing

**Current narration pattern is CORRECT:**
```rust
NARRATE
    .action("hive_start")
    .job_id(&job_id)           // ← For SSE routing (local)
    .correlation_id(&corr_id)  // ← For end-to-end tracing (global)
    .human("Starting hive")
    .emit();
```

---

## The Dual-Call Pattern (Job-Based Architecture)

All operations in rbee use this pattern:

```
1. Client: POST /v1/jobs
   → Payload: { operation: "hive_start", hive_id: "localhost", ... }
   ← Response: { job_id: "job-abc123", sse_url: "/v1/jobs/job-abc123/stream" }

2. Client: GET /v1/jobs/job-abc123/stream
   → Streams: SSE events with narration
   ← Stream: data: [qn-router] hive_start: Starting hive
             data: [qn-router] hive_check: Checking if already running
             data: [DONE]
```

**Why this pattern?**
- Decouples job creation from execution
- Allows async/background processing
- Client connects when ready to receive stream
- Supports multiple concurrent jobs

---

## Job ID vs Correlation ID

### Job ID (`job_id`)

**Purpose:** Local SSE channel routing and security isolation

**Scope:** One service (e.g., queen-rbee)

**Generated:** By each service when it creates a job

**Used for:**
- SSE channel routing (`sse_sink` uses it to route narration events)
- Job-specific isolation (prevents job A from seeing job B's narration)
- Local job tracking within a service

**Example Flow:**
```
keeper: (no job_id)
  ↓
queen: job-abc123  (creates job_id for its work)
  ↓
hive: job-def456   (creates its own job_id for its work)
  ↓
worker: job-ghi789 (creates its own job_id for its work)
```

**Why needed in narration?**
```rust
NARRATE.action("hive_start").job_id(&job_id).emit();
//                            ^^^^^^^^^^^^^^^^
// This routes the narration event to the correct SSE channel
// so the keeper receives it on /v1/jobs/job-abc123/stream
```

**CANNOT be removed** because:
1. SSE sink requires it for channel routing (`sse_sink::send()` drops events without job_id)
2. Security: Prevents job cross-contamination
3. Multiple concurrent jobs need isolation

---

### Correlation ID (`correlation_id`)

**Purpose:** End-to-end request tracing across services

**Scope:** Entire request chain (keeper → queen → hive → worker)

**Generated:** Once at queen (earliest backend), flows downstream

**Used for:**
- Tracing a request through multiple services
- Debugging distributed operations
- Logs aggregation (search for correlation_id to see full request flow)

**Example Flow:**
```
keeper: (generates nothing)
  ↓ POST /v1/jobs { operation: "hive_start" }
queen: correlation_id = "corr-xyz123" (generated here)
  ↓ Injects into payload
hive: correlation_id = "corr-xyz123" (same ID)
  ↓
worker: correlation_id = "corr-xyz123" (same ID)
```

**Why generated at queen, not client?**
- Multiple clients (CLI, web UI, API) shouldn't all need to know about it
- Backend controls consistency
- Clients CAN optionally provide one (for advanced use cases)

---

## Current Implementation (Correct)

### 1. Queen generates correlation_id

```rust
// bin/10_queen_rbee/src/job_router.rs
pub async fn create_job(state: JobState, mut payload: serde_json::Value) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    
    // Generate correlation_id if client didn't provide one
    let correlation_id = payload.get("correlation_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(generate_correlation_id);
    
    // Inject into payload for downstream propagation
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("correlation_id".to_string(), serde_json::Value::String(correlation_id));
    }
    
    // ... rest of job creation
}
```

### 2. Operations extract and use both IDs

```rust
// bin/10_queen_rbee/src/job_router.rs
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    ...
) -> Result<()> {
    // Extract correlation_id from payload
    let correlation_id = payload.get("correlation_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    
    // Use BOTH in narration
    NARRATE
        .action("hive_start")
        .job_id(&job_id)                    // ← SSE routing
        .correlation_id(correlation_id)     // ← End-to-end tracing
        .human("Starting hive")
        .emit();
}
```

---

## What About the Stream Endpoint?

### Current: `/v1/jobs/{job_id}/stream`

**Should we change it to `/v1/jobs/{correlation_id}/stream`?**

**NO - Keep using job_id**

**Reasons:**
1. **SSE channels are indexed by job_id** (internal implementation detail)
2. **correlation_id spans multiple services** - which job's stream would you get?
3. **job_id is unique per service** - no ambiguity
4. **Backward compatibility** - changing would break existing clients

**The flow is correct:**
```
Client → POST /v1/jobs { operation: "...", correlation_id: "corr-123" (optional) }
       ← { job_id: "job-abc", sse_url: "/v1/jobs/job-abc/stream" }
       
Client → GET /v1/jobs/job-abc/stream
       ← SSE stream (events include both job_id and correlation_id)
```

---

## Can We Remove job_id from Narration?

**NO - job_id is REQUIRED for narration to work**

### Why it's needed:

```rust
// bin/99_shared_crates/narration-core/src/sse_sink.rs
pub fn send(fields: &NarrationFields) {
    let event = NarrationEvent::from(fields.clone());

    // SECURITY: Only send if we have a job_id
    if let Some(job_id) = &fields.job_id {
        SSE_BROADCASTER.send_to_job(job_id, event);
    }
    // If no job_id: DROP (fail-fast, prevent privacy leaks)
}
```

**Without job_id:**
- Narration events would be dropped
- Client would never receive them
- No output in CLI/Web UI

**This was the bug we just fixed!**
- Operations weren't including `.job_id(&job_id)`
- Events were being dropped
- `./rbee hive start` hung forever waiting for events

---

## Best Practices

### 1. Use both IDs in all narration

```rust
NARRATE
    .action("hive_start")
    .job_id(&job_id)           // ← REQUIRED for SSE routing
    .correlation_id(&corr_id)  // ← Optional but recommended for tracing
    .human("Starting hive")
    .emit();
```

### 2. Generate correlation_id at queen (done ✅)

```rust
// Queen creates it if client doesn't provide
let correlation_id = payload.get("correlation_id")
    .and_then(|v| v.as_str())
    .map(|s| s.to_string())
    .unwrap_or_else(generate_correlation_id);
```

### 3. Propagate correlation_id downstream

When queen calls hive, include it:
```rust
let payload = json!({
    "operation": "worker_spawn",
    "correlation_id": correlation_id,  // ← Pass it along
    "model": "...",
    ...
});
```

### 4. Extract and use in all services

```rust
// In hive, worker, etc.
let correlation_id = payload.get("correlation_id")
    .and_then(|v| v.as_str())
    .unwrap_or("unknown");

NARRATE
    .action("worker_spawn")
    .job_id(&local_job_id)            // ← Local to this service
    .correlation_id(correlation_id)    // ← Flows from keeper
    .human("Spawning worker")
    .emit();
```

---

## Visualization

```
┌──────────────────────────────────────────────────────────────┐
│ Request Flow: keeper → queen → hive → worker                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  KEEPER (CLI)                                                │
│  ┌─────────────────────────────────────────┐                │
│  │ No job_id, no correlation_id yet        │                │
│  │ POST /v1/jobs { operation: "..." }      │                │
│  └────────────────┬────────────────────────┘                │
│                   │                                          │
│                   ↓                                          │
│  QUEEN                                                       │
│  ┌─────────────────────────────────────────┐                │
│  │ job_id = "job-abc123" (generated)       │                │
│  │ correlation_id = "corr-xyz" (generated) │                │
│  │                                          │                │
│  │ NARRATE.action("route_job")              │                │
│  │   .job_id("job-abc123")      ← SSE route │                │
│  │   .correlation_id("corr-xyz") ← Tracing  │                │
│  │   .emit()                                │                │
│  └────────────────┬────────────────────────┘                │
│                   │                                          │
│                   ↓ POST /v1/jobs { correlation_id: "..." } │
│  HIVE                                                        │
│  ┌─────────────────────────────────────────┐                │
│  │ job_id = "job-def456" (new, local)      │                │
│  │ correlation_id = "corr-xyz" (from queen)│                │
│  │                                          │                │
│  │ NARRATE.action("worker_spawn")           │                │
│  │   .job_id("job-def456")      ← SSE route │                │
│  │   .correlation_id("corr-xyz") ← Tracing  │                │
│  │   .emit()                                │                │
│  └────────────────┬────────────────────────┘                │
│                   │                                          │
│                   ↓ POST /v1/jobs { correlation_id: "..." } │
│  WORKER                                                      │
│  ┌─────────────────────────────────────────┐                │
│  │ job_id = "job-ghi789" (new, local)      │                │
│  │ correlation_id = "corr-xyz" (from queen)│                │
│  │                                          │                │
│  │ NARRATE.action("inference")              │                │
│  │   .job_id("job-ghi789")      ← SSE route │                │
│  │   .correlation_id("corr-xyz") ← Tracing  │                │
│  │   .emit()                                │                │
│  └──────────────────────────────────────────┘                │
│                                                              │
└──────────────────────────────────────────────────────────────┘

Logs aggregated by correlation_id:
  corr-xyz: [qn-router] route_job: Executing hive_start
  corr-xyz: [hive-mgr] worker_spawn: Starting worker on GPU0
  corr-xyz: [llm-worker] inference: Processing prompt
```

---

## Future Improvements

### Task-Local Context (eliminates repetition)

Instead of `.job_id(&job_id)` everywhere:

```rust
// Set once at task start
NARRATION_CONTEXT.set(|ctx| {
    ctx.job_id = job_id;
    ctx.correlation_id = correlation_id;
});

// Then just use NARRATE - context is automatic!
NARRATE.action("hive_start").human("Starting hive").emit();
// job_id and correlation_id are automatically included
```

**Implementation:** Use tokio `task_local!` macro

**Benefit:** 
- No more `.job_id(&job_id)` repetition
- Cleaner code
- Less chance of forgetting to add it

**Complexity:** Medium (need tokio task-local storage)

---

## Summary

✅ **Keep current implementation:**
- `job_id`: Required for SSE routing, local to each service
- `correlation_id`: Optional but recommended for end-to-end tracing
- Both should be in narration events

✅ **Stream endpoint stays:** `/v1/jobs/{job_id}/stream`

✅ **correlation_id generated at queen** (earliest backend)

✅ **Clients don't need to know about correlation_id** (queen handles it)

❌ **Don't remove job_id from narration** (events would be dropped)

❌ **Don't change stream endpoint to use correlation_id** (breaks SSE routing)

---

## Good night! 😴

This document has everything you need. The current implementation is correct. The only thing left to do is actually USE the correlation_id in narration events (extract it from payload and add it). Sleep well!
