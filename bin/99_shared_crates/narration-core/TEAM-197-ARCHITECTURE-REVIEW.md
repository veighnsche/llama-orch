# TEAM-197 Critical Review of TEAM-198's Architecture

**TEAM-197** | **Date:** 2025-10-22 | **Status:** CRITICAL ANALYSIS

---

## Executive Summary

**TEAM-198's proposal is EXCELLENT** and addresses the core issue correctly. However, there are **3 critical flaws** and **2 missed opportunities** that need addressing.

**Verdict:** ⚠️ **MOSTLY CORRECT** but needs refinement before implementation.

---

## What TEAM-198 Got Right ✅

### 1. Correctly Identified the Core Problem
- ✅ Decentralized formatting creates maintenance burden
- ✅ SSE consumers must manually format events
- ✅ Format changes require updating all consumers
- ✅ This is exactly what I found during debugging

### 2. Correct Solution Direction
- ✅ Centralize formatting in narration-core
- ✅ Add `formatted` field to `NarrationEvent`
- ✅ Keep existing fields for backward compatibility
- ✅ This matches my original recommendation (before I removed it)

### 3. Correctly Understood the Use Case
- ✅ Daemons can't use stdout (remote machines)
- ✅ Queen is natural aggregation point
- ✅ All narration must flow through SSE for web-UI
- ✅ Keeper is the only CLI tool that can stdout

### 4. Good Implementation Plan
- ✅ Phased rollout (narration-core → queen → hive/worker)
- ✅ Non-breaking changes (adds field, doesn't remove)
- ✅ Clear verification checklist
- ✅ Reasonable code size impact (~112 lines)

---

## Critical Flaws 🚨

### FLAW 1: Missing Redaction in SSE Path

**Location:** TEAM-198's proposed `From<NarrationFields>` implementation (line 204-226)

```rust
// TEAM-198's CODE:
impl From<NarrationFields> for NarrationEvent {
    fn from(fields: NarrationFields) -> Self {
        // Apply redaction
        let human = redact_secrets(&fields.human, RedactionPolicy::default());
        
        // PRE-FORMAT the text (same as stderr)
        let formatted = format!("[{:<10}] {:<15}: {}", fields.actor, fields.action, human);
        // ...
    }
}
```

**The Problem:** Only `human` is redacted, but `formatted` uses the redacted `human`!

**What's Missing:**
- ❌ `cute` is NOT redacted before adding to event
- ❌ `story` is NOT redacted before adding to event
- ❌ `target` is NOT redacted before adding to event

**Impact:** **HIGH SEVERITY** - Secrets could leak through SSE stream!

**The Fix:**
```rust
impl From<NarrationFields> for NarrationEvent {
    fn from(fields: NarrationFields) -> Self {
        // TEAM-197: Redact ALL fields, not just human
        let human = redact_secrets(&fields.human, RedactionPolicy::default());
        let cute = fields.cute.as_ref()
            .map(|c| redact_secrets(c, RedactionPolicy::default()));
        let story = fields.story.as_ref()
            .map(|s| redact_secrets(s, RedactionPolicy::default()));
        let target = redact_secrets(&fields.target, RedactionPolicy::default());
        
        // Pre-format with redacted human
        let formatted = format!("[{:<10}] {:<15}: {}", fields.actor, fields.action, human);
        
        Self {
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            target,  // Use redacted target
            human: human.to_string(),
            formatted,
            cute: cute.map(|c| c.to_string()),
            story: story.map(|s| s.to_string()),
            correlation_id: fields.correlation_id,
            job_id: fields.job_id,
            emitted_by: fields.emitted_by,
            emitted_at_ms: fields.emitted_at_ms,
        }
    }
}
```

**Why This Matters:**
- stderr output goes through `narrate_at_level()` which redacts everything (line 433-440 in lib.rs)
- SSE output must have IDENTICAL redaction
- Otherwise SSE could leak secrets that stderr doesn't

---

### FLAW 2: The Queen Ingestion Endpoint is WRONG

**Location:** TEAM-198's proposed `handle_ingest_narration()` (line 419-428)

**TEAM-198's Approach:**
```rust
pub async fn handle_ingest_narration(
    Json(req): Json<NarrationIngestionRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    // Broadcast to global SSE sink
    sse_sink::send_formatted(&req.formatted, req.job_id, req.correlation_id);
    Ok(StatusCode::OK)
}
```

**The Problems:**

#### Problem 2a: No Job Context
When hive/worker send narration, which job does it belong to?
- ❌ No `job_id` in the request path
- ❌ Narration broadcasts to ALL SSE subscribers
- ❌ Keeper might see narration from other users' jobs!

**Example of the Bug:**
```
User A runs: ./rbee hive status
User B runs: ./rbee infer "hello"

User A sees:
[job-exec  ] execute        : Executing job A
[hive      ] spawn          : Spawning worker  ← From User B's job!
```

#### Problem 2b: No Authentication
- ❌ Anyone can POST to `/v1/narration`
- ❌ Malicious actor could inject fake narration
- ❌ No way to verify narration came from legitimate hive/worker

#### Problem 2c: Wrong Endpoint Pattern
The ingestion endpoint should be **job-specific**, not global:

**CORRECT Approach:**
```rust
// POST /v1/jobs/{job_id}/narration
pub async fn handle_ingest_narration(
    Path(job_id): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<NarrationIngestionRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    // 1. Verify job exists
    let job_state = state.registry.get_state(&job_id)
        .ok_or((StatusCode::NOT_FOUND, "Job not found".to_string()))?;
    
    // 2. Verify job is active
    if !matches!(job_state, JobState::Running) {
        return Err((StatusCode::CONFLICT, "Job not running".to_string()));
    }
    
    // 3. Send to job-specific SSE stream
    // (This requires refactoring SSE broadcaster to support job-specific channels)
    sse_sink::send_to_job(&job_id, &req.formatted);
    
    Ok(StatusCode::OK)
}
```

**Why This Matters:**
- Narration is per-job, not global
- Different jobs should have isolated SSE streams
- Security: Only authenticated requests should inject narration

---

### FLAW 3: The `format_narration()` Helper is Incomplete

**Location:** TEAM-198's proposed helper (line 378-383)

**TEAM-198's Proposal:**
```rust
pub fn format_narration(actor: &str, action: &str, human: &str) -> String {
    format!("[{:<10}] {:<15}: {}", actor, action, human)
}
```

**The Problem:** This helper doesn't handle redaction!

**Current Code Path:**
```
narrate_at_level() in lib.rs (line 425-479)
  ├─ Redacts human, cute, story (line 433-440)
  ├─ Formats with eprintln! (line 449)
  └─ Sends to SSE via sse_sink::send() (line 452-454)
      └─ Converts to NarrationEvent (no formatting, no redaction!)
```

**If we extract format logic, we MUST also extract redaction logic!**

**CORRECT Approach:**
```rust
/// Format narration with redaction applied
/// 
/// TEAM-197: This is the SINGLE source of truth for formatting.
/// Used by both stderr output and SSE events.
pub fn format_narration_with_redaction(
    actor: &str,
    action: &str,
    human: &str,
) -> String {
    // CRITICAL: Apply redaction BEFORE formatting
    let redacted_human = redact_secrets(human, RedactionPolicy::default());
    format!("[{:<10}] {:<15}: {}", actor, action, redacted_human)
}

// Or keep the current approach:
// Apply redaction in narrate_at_level(), then pass redacted strings to format helper
```

**Why This Matters:**
- Redaction and formatting are coupled
- If we separate them, we risk forgetting redaction
- Must maintain security properties

---

## Missed Opportunities 💡

### OPPORTUNITY 1: SSE Broadcaster Needs Refactoring

**Current Architecture:**
- One global SSE broadcaster
- All narration goes to all subscribers
- No job-specific isolation

**Problem:**
When multiple jobs run concurrently:
```
Job A: ./rbee hive status (keeper subscribes)
Job B: ./rbee infer "hello" (web-UI subscribes)

Job A SSE stream sees:
[job-exec  ] execute        : Executing job A
[job-exec  ] execute        : Executing job B  ← WRONG! This is Job B
[worker    ] inference      : Generating tokens ← From Job B
```

**Solution:** Refactor SSE broadcaster to support **job-scoped channels**

```rust
// NEW: Job-scoped SSE broadcaster
pub struct SseBroadcaster {
    // Global channel for non-job narration (queen startup, etc.)
    global: Arc<Mutex<Option<broadcast::Sender<NarrationEvent>>>>,
    
    // Per-job channels (keyed by job_id)
    jobs: Arc<Mutex<HashMap<String, broadcast::Sender<NarrationEvent>>>>,
}

impl SseBroadcaster {
    /// Send narration to a specific job's SSE stream
    pub fn send_to_job(&self, job_id: &str, event: NarrationEvent) {
        let jobs = self.jobs.lock().unwrap();
        if let Some(tx) = jobs.get(job_id) {
            let _ = tx.send(event);
        }
    }
    
    /// Subscribe to a specific job's SSE stream
    pub fn subscribe_to_job(&self, job_id: &str) -> Option<broadcast::Receiver<NarrationEvent>> {
        let jobs = self.jobs.lock().unwrap();
        jobs.get(job_id).map(|tx| tx.subscribe())
    }
    
    /// Create a new job channel
    pub fn create_job_channel(&self, job_id: String, capacity: usize) {
        let (tx, _) = broadcast::channel(capacity);
        self.jobs.lock().unwrap().insert(job_id, tx);
    }
    
    /// Remove a job channel when job completes
    pub fn remove_job_channel(&self, job_id: &str) {
        self.jobs.lock().unwrap().remove(job_id);
    }
}
```

**Why This Matters:**
- Proper isolation between jobs
- Each keeper sees only their job's narration
- Web-UI can subscribe to specific jobs
- No cross-contamination

---

### OPPORTUNITY 2: The Worker Already Has the Answer!

**TEAM-198 Missed This:**

Look at the worker's `narrate_dual()` implementation (line 117-135 in `worker/narration.rs`):

```rust
pub fn narrate_dual(fields: NarrationFields) {
    // 1. ALWAYS emit to tracing (for operators/developers)
    observability_narration_core::narrate(fields.clone());

    // 2. IF in HTTP request context, ALSO emit to SSE (for users)
    let sse_event = InferenceEvent::Narration { /* ... */ };
    let _ = narration_channel::send_narration(sse_event);
}
```

**Key Insight:** Worker uses a **thread-local channel** to send narration to the HTTP request's SSE stream!

**This is BETTER than queen ingestion endpoint because:**
1. ✅ **No network round-trip** - narration goes directly into SSE stream
2. ✅ **Automatically scoped to request** - thread-local channel
3. ✅ **No authentication needed** - internal mechanism
4. ✅ **No routing confusion** - narration belongs to current request

**TEAM-198's Proposed Approach:**
```
Worker → NARRATE.emit() → HTTP POST to queen → Queen broadcasts → SSE stream
   └─ Extra network hop
   └─ Needs authentication
   └─ Needs job_id routing
```

**Worker's EXISTING Approach:**
```
Worker → narrate_dual() → thread-local channel → SSE stream (directly)
   └─ No network hop
   └─ No authentication needed
   └─ Automatically scoped to request
```

**Recommendation for Hive:**
Hive should use the SAME pattern as worker:
1. When hive spawns worker (HTTP request to queen), use thread-local channel
2. Narration goes directly into that HTTP request's SSE stream
3. No separate ingestion endpoint needed!

**How to Adapt for Hive:**
```rust
// In hive's HTTP client (when calling queen endpoints)
pub async fn narrate_to_request_stream(fields: NarrationFields) {
    // 1. Always emit to stderr (for daemon logs)
    observability_narration_core::narrate(fields.clone());
    
    // 2. If in HTTP request context, also emit to request's SSE stream
    // (This requires queen to support SSE for hive management endpoints)
    if let Some(tx) = REQUEST_SSE_CHANNEL.with(|c| c.borrow().clone()) {
        let event = /* convert to SSE event */;
        let _ = tx.send(event).await;
    }
}
```

**Why This Matters:**
- Simpler architecture (no separate ingestion endpoint)
- Better performance (no extra network hops)
- More secure (no external API to protect)
- Already proven pattern (worker uses it)

---

## Code Flow Comparison

### TEAM-198's Proposed Flow

```
[Hive Lifecycle Event]
    ↓
NARRATE.action().emit()
    ↓
narration-core::narrate() → stderr (daemon logs)
    ↓
[Separate Code Path]
    ↓
narrate_to_queen() → HTTP POST /v1/narration
    ↓
Queen receives, broadcasts via global SSE
    ↓
Keeper's job SSE stream picks it up
    ↓
Keeper stdout
```

**Issues:**
- ❌ Extra network hop (POST to queen)
- ❌ Needs authentication
- ❌ Needs job_id routing
- ❌ Two separate code paths (stderr vs SSE)

### TEAM-197's Recommended Flow (Based on Worker Pattern)

```
[Hive Lifecycle Event]
    ↓
NARRATE.action().emit()
    ↓
narration-core::narrate_at_level()
    ├─ stderr (daemon logs)
    └─ sse_sink::send()
        ├─ If thread-local channel exists → send to request's SSE stream
        └─ Otherwise → send to global broadcaster
```

**Benefits:**
- ✅ Single code path (one `.emit()` call)
- ✅ No extra network hops
- ✅ Automatically scoped to request
- ✅ No authentication needed
- ✅ Consistent with worker pattern

---

## Specific Code Issues

### Issue 1: TEAM-198's `send_formatted()` is Wrong

**Location:** Line 439-454 in TEAM-198's proposal

```rust
pub fn send_formatted(formatted: &str, job_id: Option<String>, correlation_id: Option<String>) {
    SSE_BROADCASTER.send(NarrationEvent {
        formatted: formatted.to_string(),
        job_id,
        correlation_id,
        // Other fields can be empty for ingested events
        actor: String::new(),  // ← BAD! Empty actor
        action: String::new(), // ← BAD! Empty action
        target: String::new(),
        human: String::new(),
        // ...
    });
}
```

**Problems:**
1. ❌ `actor` and `action` are empty strings
2. ❌ These fields are PUBLIC and serialized to JSON
3. ❌ Consumers might check `if event.actor == "hive"` and get empty string
4. ❌ Breaks backward compatibility

**Solution:** Don't add `send_formatted()` at all! Just use the normal `From<NarrationFields>` path.

---

### Issue 2: TEAM-198's Hive narration.rs has Fire-and-Forget Bug

**Location:** Line 498-502 in TEAM-198's proposal

```rust
// Fire and forget (don't block on queen response)
let _ = client
    .post(format!("{}/v1/narration", queen_url))
    .json(&payload)
    .send()
    .await;
```

**The Bug:** Using `let _ =` discards errors, but we still `.await` the future!

This means:
- ✅ We wait for the HTTP request to complete
- ❌ But we ignore if it failed
- ❌ Hive will block on slow/dead queen

**Better Approach (True Fire-and-Forget):**
```rust
// Spawn task to avoid blocking
tokio::spawn(async move {
    let result = client
        .post(format!("{}/v1/narration", queen_url))
        .json(&payload)
        .timeout(Duration::from_millis(100))  // Fast timeout
        .send()
        .await;
    
    if let Err(e) = result {
        eprintln!("[hive      ] narration_send_failed: {}", e);
    }
});
```

**Why This Matters:**
- Hive lifecycle shouldn't block on queen availability
- If queen is down, hive should still function
- Narration is best-effort, not critical path

---

## Architecture Recommendation

### Keep What TEAM-198 Got Right:
1. ✅ Add `formatted` field to `NarrationEvent`
2. ✅ Pre-format in `From<NarrationFields>` impl
3. ✅ Update queen SSE consumer to use `event.formatted`
4. ✅ Phased rollout approach

### Fix Critical Flaws:
1. 🚨 **Redact ALL fields** in `From<NarrationFields>` (not just `human`)
2. 🚨 **Make SSE job-scoped**, not global
3. 🚨 **Ingestion endpoint must be per-job**: `POST /v1/jobs/{job_id}/narration`
4. 🚨 **Add authentication** to ingestion endpoint

### Adopt Worker Pattern:
1. 💡 **Use thread-local channels** instead of HTTP ingestion
2. 💡 **Hive narration in request context** → thread-local SSE channel
3. 💡 **Hive lifecycle narration** → global SSE (queen startup, etc.)
4. 💡 **No separate ingestion endpoint needed**

### Simplified Architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│ NARRATION FLOW (TEAM-197 RECOMMENDED)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. ANY Component calls NARRATE.action().emit()                │
│     ↓                                                            │
│  2. narration-core::narrate_at_level()                         │
│     ├─ Redacts secrets                                         │
│     ├─ Formats: "[actor     ] action         : message"        │
│     ├─ Outputs to stderr (daemon logs)                         │
│     └─ Sends to SSE via sse_sink::send()                       │
│         ↓                                                       │
│  3. SSE Broadcaster (job-scoped)                               │
│     ├─ If thread-local channel exists:                         │
│     │   └─ Send to current request's SSE stream               │
│     └─ Otherwise:                                               │
│         └─ Send to job-specific channel (if job_id present)   │
│             └─ Or send to global channel                       │
│                                                                 │
│  4. Keeper subscribes to job-specific SSE stream               │
│     ├─ Receives pre-formatted strings                          │
│     └─ println!() to stdout                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Differences from TEAM-198:**
- ✅ No separate HTTP ingestion endpoint
- ✅ Thread-local channels for request-scoped narration
- ✅ Job-specific SSE channels for isolation
- ✅ Single code path (one `.emit()` call)

---

## Implementation Priority

### Phase 0: Critical Fixes (MUST DO)
1. **Fix redaction in SSE path** - Security issue
2. **Add job-scoped SSE broadcaster** - Isolation issue
3. **Update queen consumer to use `event.formatted`** - Simplification

### Phase 1: Centralized Formatting (TEAM-198 Phase 1)
1. Add `formatted` field to `NarrationEvent`
2. Pre-format in `From<NarrationFields>`
3. Keep existing fields for backward compatibility

### Phase 2: Thread-Local SSE (Like Worker)
1. Add thread-local SSE channel support to narration-core
2. Update hive to use thread-local channel (like worker)
3. No HTTP ingestion endpoint needed

### Phase 3: Hive Narration
1. Replace `println!()` with `NARRATE.action().emit()`
2. Narration automatically goes to request's SSE stream
3. Test with remote hive

### Phase 4: Worker Lifecycle (Already Mostly Done)
1. Worker inference already uses `narrate_dual()` (works)
2. Worker lifecycle should use same pattern
3. Test with remote worker

---

## Final Verdict

### TEAM-198's Score: 7/10

**Strengths:**
- ✅ Identified the core problem correctly
- ✅ Proposed the right solution direction
- ✅ Comprehensive analysis of current architecture
- ✅ Good implementation plan

**Weaknesses:**
- 🚨 Missing redaction in SSE path (security issue)
- 🚨 Wrong ingestion endpoint design (no job isolation)
- 🚨 Didn't leverage worker's existing pattern
- 💡 Missed opportunity for job-scoped SSE

### TEAM-197's Recommendation:

**Adopt TEAM-198's core idea (centralized formatting) but:**
1. Fix the security issue (redaction)
2. Add job-scoped SSE broadcaster
3. Use worker's thread-local channel pattern
4. Skip the HTTP ingestion endpoint

**Result:** Simpler, more secure, better isolated, proven pattern.

---

**TEAM-197** | **2025-10-22** | **Critical Review Complete**

**Bottom Line:** TEAM-198 did excellent analysis but missed critical security and architectural issues. Implementation should proceed with fixes noted above.
