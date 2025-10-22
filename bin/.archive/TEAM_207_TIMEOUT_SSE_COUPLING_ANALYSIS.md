# TEAM-207: TimeoutEnforcer SSE Coupling Analysis

**Date**: 2025-10-22  
**Status**: 🔴 CRITICAL BUG FOUND  
**Priority**: P0 - Breaks SSE streaming

---

## The Problem

**TimeoutEnforcer narration has NO `job_id`**, so it **NEVER reaches the SSE stream**.

### Evidence

**TimeoutEnforcer emits without job_id**:
```rust
// bin/99_shared_crates/timeout-enforcer/src/lib.rs:207
NARRATE
    .action("start")
    .context(label.clone())
    .context(total_secs.to_string())
    .human("⏱️  {0} (timeout: {1}s)")
    .emit();  // ❌ NO .job_id() - goes to stdout ONLY!
```

**SSE sink REQUIRES job_id**:
```rust
// bin/99_shared_crates/narration-core/src/sse_sink.rs:126
pub fn send_to_job(&self, job_id: &str, event: NarrationEvent) {
    let senders = self.senders.lock().unwrap();
    if let Some(tx) = senders.get(job_id) {
        let _ = tx.try_send(event);
    }
    // If channel doesn't exist: DROP THE EVENT (fail-fast)
}
```

**Narration routing logic** (from memory):
```rust
// If narration has job_id → send to SSE channel
// If narration has NO job_id → print to stdout
// If SSE channel doesn't exist → DROP (fail-fast)
```

---

## Why It Works in rbee-keeper But Not queen-rbee

### rbee-keeper (Client Side)
- ✅ Prints to STDOUT directly
- ✅ User sees timeout in terminal
- ✅ Not trying to send over SSE

### queen-rbee (Server Side)
- ❌ Tries to send over SSE
- ❌ NO job_id in TimeoutEnforcer narration
- ❌ SSE sink drops the event
- ❌ User NEVER sees timeout indicator

---

## The Flow

```
[keeper] Submits job, gets job_id, starts SSE stream
    ↓
[queen] Creates job_id channel in SSE sink
    ↓
[queen] Executes operation in background
    ↓
[queen] TimeoutEnforcer.enforce() wraps fetch_capabilities
    ↓
[TimeoutEnforcer] NARRATE.action("start")...emit()  ← NO job_id!
    ↓
[narration-core] Checks: has job_id? NO
    ↓
[narration-core] Routes to: STDOUT (not SSE)
    ↓
[SSE sink] Never receives the event
    ↓
[keeper] Never sees timeout indicator in stream
```

---

## The Solution

TimeoutEnforcer needs to be **job_id-aware** so its narration flows through SSE.

### Option 1: Add job_id Parameter (RECOMMENDED)

```rust
impl TimeoutEnforcer {
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }
}

// Usage in queen-rbee:
let result = TimeoutEnforcer::new(Duration::from_secs(15))
    .with_label("Fetching device capabilities")
    .with_job_id(&job_id)  // ← ADD THIS
    .with_countdown()
    .enforce(async {
        fetch_hive_capabilities(&endpoint).await
    })
    .await;
```

**Changes needed**:
1. Add `job_id: Option<String>` field to `TimeoutEnforcer` struct (~1 LOC)
2. Add `with_job_id()` method (~4 LOC)
3. Pass `job_id` to narration in `enforce_with_countdown()` (~2 LOC)
4. Update all call sites to pass job_id (~10 call sites × 1 LOC = 10 LOC)

**Total**: ~17 LOC

---

### Option 2: Use Thread-Local job_id Context (COMPLEX)

Store job_id in thread-local storage, TimeoutEnforcer reads it automatically.

**Problems**:
- Async tasks switch threads
- Thread-locals don't work across await points
- Would need tokio::task_local! which is more complex

**NOT RECOMMENDED**

---

### Option 3: Make TimeoutEnforcer Generic Over Context (OVERKILL)

```rust
impl TimeoutEnforcer<JobContext> { ... }
```

**Problems**:
- Way too complex
- Breaks existing API
- Not worth it

**NOT RECOMMENDED**

---

## Current State

### Where TimeoutEnforcer is Used

1. **rbee-keeper** (WORKS - goes to stdout):
   - `queen_lifecycle.rs` - Starting queen-rbee
   - `job_client.rs` - SSE streaming (outer wrapper)

2. **queen-rbee** (BROKEN - needs job_id):
   - `job_router.rs` - Fetching device capabilities

### What Needs job_id

- ✅ `rbee-keeper`: NO (prints to stdout, correct)
- ❌ `queen-rbee`: YES (sends over SSE, broken)

---

## Implementation Plan

### Phase 1: Add job_id Support to TimeoutEnforcer

**File**: `bin/99_shared_crates/timeout-enforcer/src/lib.rs`

**Changes**:
1. Add field to struct:
```rust
pub struct TimeoutEnforcer {
    duration: Duration,
    label: Option<String>,
    show_countdown: bool,
    job_id: Option<String>,  // ← NEW
}
```

2. Add builder method:
```rust
pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
    self.job_id = Some(job_id.into());
    self
}
```

3. Update `enforce_with_countdown()`:
```rust
// Line 207
let mut narration = NARRATE
    .action("start")
    .context(label.clone())
    .context(total_secs.to_string())
    .human("⏱️  {0} (timeout: {1}s)");

if let Some(ref job_id) = self.job_id {
    narration = narration.job_id(job_id);
}

narration.emit();
```

4. Update timeout error narration (line 255):
```rust
let mut narration = NARRATE
    .action("timeout")
    .context(label.clone())
    .context(total_secs.to_string())
    .human("❌ {0} TIMED OUT after {1}s")
    .error_kind("operation_timeout");

if let Some(ref job_id) = self.job_id {
    narration = narration.job_id(job_id);
}

narration.emit_error();
```

---

### Phase 2: Update queen-rbee Call Sites

**File**: `bin/10_queen_rbee/src/job_router.rs`

**Change**:
```rust
let caps_result = TimeoutEnforcer::new(Duration::from_secs(15))
    .with_label("Fetching device capabilities")
    .with_job_id(&job_id)  // ← ADD THIS
    .with_countdown()
    .enforce(async {
        // ...
    })
    .await;
```

---

## Expected Behavior After Fix

### Before (Current - BROKEN)
```
[keeper    ] job_stream     : 📡 Streaming results...
[qn-router ] hive_caps      : 📊 Fetching device capabilities...
← MISSING TIMEOUT INDICATOR ←
[qn-router ] hive_caps_ok   : ✅ Discovered 1 device(s)
```

### After (Fixed - WORKS)
```
[keeper    ] job_stream     : 📡 Streaming results...
[timeout   ] start          : ⏱️  Fetching device capabilities (timeout: 15s)
⠙ [███▒                                    ] 3/15s - Fetching device capabilities
[qn-router ] hive_caps_http : 🌐 GET http://127.0.0.1:9000/capabilities
[hive      ] caps_request   : 📡 Received capabilities request from queen
[hive      ] caps_gpu_check : 🔍 Detecting GPUs via nvidia-smi...
⠹ [██████▒                                 ] 6/15s - Fetching device capabilities
[qn-router ] hive_caps_ok   : ✅ Discovered 1 device(s)
```

---

## Testing Checklist

After implementing fix:

- [ ] `./rbee hive start` shows timeout indicator in SSE stream
- [ ] Timeout countdown updates every second
- [ ] Timeout narration includes job_id
- [ ] SSE sink receives timeout events
- [ ] Keeper displays timeout in stream
- [ ] rbee-keeper still works (stdout path)
- [ ] Build succeeds for both packages

---

## Summary

**Root Cause**: TimeoutEnforcer emits narration without `job_id`, so SSE sink drops the events

**Impact**: Users never see timeout indicators when operations run in queen-rbee

**Fix**: Add `job_id` parameter to TimeoutEnforcer, pass it in queen-rbee

**Effort**: ~20 LOC

**Priority**: P0 - Breaks observability

---

**END OF ANALYSIS**
