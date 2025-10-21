# ISOLATION ANALYSIS: Are We Guaranteed Isolated Channels?

**Created by:** TEAM-204  
**Date:** 2025-10-22  
**Question:** Are we guaranteed to have isolated channels now?

---

## TL;DR

**YES, we're guaranteed isolation... but there's a gap.**

**The Good:** Job-scoped narration is isolated  
**The Gap:** System narration (no job_id) is DROPPED  

---

## Current Architecture

### How It Works

```rust
// When narration is emitted:
pub fn send(fields: &NarrationFields) {
    let event = NarrationEvent::from(fields.clone());
    
    // SECURITY: Only send if we have a job_id
    if let Some(job_id) = &fields.job_id {
        SSE_BROADCASTER.send_to_job(job_id, event);
    }
    // If no job_id: DROP (fail-fast, prevent privacy leaks)
}
```

### Channel Creation

```rust
// In job_router.rs create_job():
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    
    // TEAM-200: Create job-specific SSE channel for isolation
    observability_narration_core::sse_sink::create_job_channel(job_id.clone(), 1000);
    
    NARRATE
        .action("job_create")
        .context(&job_id)
        .job_id(&job_id)  // ← Sets job_id on narration
        .human("Job {} created")
        .emit();
    
    Ok(JobResponse { job_id, sse_url })
}
```

---

## Isolation Guarantee

### ✅ YES: Job-Scoped Narration is Isolated

**Flow:**
```
1. Client: POST /v1/jobs {"operation": "hive_status"}
2. Queen: create_job() → job_id="job-abc123"
3. Queen: create_job_channel("job-abc123", 1000)
4. Client: GET /v1/jobs/job-abc123/stream
5. Queen: subscribe_to_job("job-abc123")
6. Narration: NARRATE.job_id("job-abc123").emit()
7. SSE: send() → send_to_job("job-abc123", event)
8. Client: Receives ONLY job-abc123's narration
```

**Isolation Properties:**
- ✅ Job A can't see Job B's narration
- ✅ Each job has separate channel
- ✅ Channels cleaned up after job completes
- ✅ No cross-contamination possible

---

## The Gap: System Narration

### What Happens to Narration WITHOUT job_id?

**Examples from queen-rbee/src/main.rs:**
```rust
// Queen startup (NO job_id)
NARRATE
    .action("start")
    .context(args.port.to_string())
    .human("Queen-rbee starting on port {}")
    .emit();  // ← No job_id!

// Queen ready (NO job_id)
NARRATE
    .action("ready")
    .human("Ready to accept connections")
    .emit();  // ← No job_id!
```

**What happens:**
```rust
pub fn send(fields: &NarrationFields) {
    if let Some(job_id) = &fields.job_id {
        SSE_BROADCASTER.send_to_job(job_id, event);
    }
    // ← Falls through, event is DROPPED
}
```

**Result:** System narration is DROPPED (not sent via SSE)

---

## Why Was Global Channel Created?

### The Original Reason

**From TEAM-198's proposal:**
> "Global channel for non-job narration (queen startup, etc.)"

**The thinking:**
1. Queen emits system narration (startup, shutdown, etc.)
2. This narration has no job_id
3. Need somewhere to send it → global channel
4. Keeper could subscribe to global to see system events

### Why This Was Wrong

**Problem 1: Privacy Violation**
- Global channel mixed system events with job events
- Race condition: job narration could fall back to global
- User A sees User B's data

**Problem 2: Wrong Use Case**
- System narration is for OPERATORS, not users
- Should go to logs/monitoring, not SSE
- SSE is for job-specific narration

---

## Current State Analysis

### Where Does System Narration Go?

**Answer:** stderr only (via `narrate_at_level()`)

```rust
// In lib.rs narrate_at_level():
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    // 1. Always output to stderr
    eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, human);
    
    // 2. Send to SSE if enabled
    if sse_sink::is_enabled() {
        sse_sink::send(&fields);  // ← Drops if no job_id
    }
    
    // 3. Emit to tracing (if configured)
    // ...
}
```

**Flow:**
```
Queen startup narration (no job_id)
  ↓
narrate_at_level()
  ↓
eprintln!() → stderr ✅
  ↓
sse_sink::send() → DROPPED ❌
  ↓
tracing::event!() → logs ✅
```

---

## Is This a Problem?

### For Job Narration: NO

**Guaranteed isolation:**
- ✅ Job channels created before execution
- ✅ All job narration has job_id
- ✅ No cross-contamination possible

### For System Narration: DEPENDS

**Current behavior:**
- ✅ Goes to stderr (operators can see it)
- ✅ Goes to tracing/logs (monitoring systems)
- ❌ NOT in SSE (web UI won't see it)

**Is this correct?**

**YES, because:**
1. System narration is for operators, not users
2. Users shouldn't see "Queen starting" (not their concern)
3. Operators have access to logs/stderr
4. Web UI should only show job-specific narration

---

## Race Condition Analysis

### Was Global Channel Created for Race Conditions?

**NO.** Let me trace the actual race condition:

**The Race:**
```
Thread 1 (create_job):
  1. job_id = create_job()
  2. create_job_channel(job_id)  ← Channel created
  3. NARRATE.job_id(job_id).emit()  ← Narration emitted

Thread 2 (execute_job):
  4. execute_job(job_id)
  5. NARRATE.job_id(job_id).emit()  ← Might emit before step 2!
```

**But this race doesn't happen because:**
```rust
// In job_router.rs:
pub async fn create_job(...) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    
    // Channel created BEFORE returning
    create_job_channel(job_id.clone(), 1000);
    
    // Narration emitted AFTER channel created
    NARRATE.job_id(&job_id).emit();
    
    Ok(JobResponse { job_id, sse_url })
}

// Client can't call execute_job until they have job_id
// They get job_id from create_job response
// By that time, channel already exists
```

**Sequence:**
```
1. Client: POST /v1/jobs
2. Queen: create_job()
3. Queen: create_job_channel()  ← Channel exists
4. Queen: Returns job_id to client
5. Client: GET /v1/jobs/{job_id}/stream
6. Queen: execute_job()
7. Queen: NARRATE.job_id().emit()  ← Channel already exists
```

**No race condition possible!**

---

## What About Background Tasks?

### Could Background Tasks Emit Before Channel Created?

**Example:**
```rust
// What if we did this?
tokio::spawn(async move {
    NARRATE.job_id(&job_id).emit();  // ← Background task
});
```

**Answer:** Yes, this could race!

**But we don't do this.** All narration happens in:
1. `create_job()` - After channel created
2. `execute_job()` - After client subscribes
3. Operation handlers - During execution

**All happen AFTER channel creation.**

---

## Conclusion

### Are We Guaranteed Isolation?

**YES.**

**Guarantees:**
1. ✅ Job channels created before any job narration
2. ✅ Each job has separate channel
3. ✅ No global channel (no cross-contamination)
4. ✅ Fail-fast (drop instead of leak)

### What About System Narration?

**By design:**
- System narration (no job_id) → stderr + logs
- Job narration (with job_id) → SSE + stderr + logs

**This is correct:**
- Users see their job narration (SSE)
- Operators see everything (logs)
- No privacy violations

### Why Was Global Channel Created?

**NOT for race conditions.**

**Actual reason:**
- TEAM-198 wanted system narration in SSE
- Didn't realize system narration ≠ job narration
- Created global channel for "convenience"
- Didn't consider privacy implications

**The fix:**
- Remove global channel
- System narration → logs only
- Job narration → SSE (isolated)

---

## Recommendations

### Current Architecture is Correct

**No changes needed:**
- Job isolation works
- System narration in logs (correct place)
- No privacy violations possible

### If You Want System Narration in Web UI

**DON'T use global channel!**

**Instead:**
1. Create separate `/v1/system/events` endpoint
2. Require operator authentication
3. Separate from job narration
4. Different security model

**Or:**
- Just use logs/monitoring dashboards
- System events aren't for end users

---

**END OF ANALYSIS**

**Answer:** YES, we're guaranteed isolation. Global channel was NOT for race conditions—it was a misguided attempt to put system narration in SSE.
