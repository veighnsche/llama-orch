# [DONE] SIGNAL INVESTIGATION & IMPLEMENTATION PLAN

**Status:** ⚠️ CRITICAL ARCHITECTURAL ISSUE  
**Severity:** HIGH  
**Created:** October 26, 2025  
**Owner:** TEAM-304 (must fix immediately)

---

## The Problem

### Current Broken State

**[DONE] is being emitted by narration-core**, which is WRONG:

```rust
// narration-core/tests/e2e_job_client_integration.rs
n!("done", "[DONE]");  // ❌ WRONG! Narration emitting [DONE]

// narration-core/tests/bin/fake_queen.rs
n!("done", "[DONE]");  // ❌ WRONG! Narration emitting [DONE]

// narration-core/tests/bin/fake_hive.rs
n!("done", "[DONE]");  // ❌ WRONG! Narration emitting [DONE]
```

**job-server does NOT emit [DONE]:**

```rust
// job-server/src/lib.rs:execute_and_stream()
stream::unfold(receiver, |rx_opt| async move {
    match rx_opt {
        Some(mut rx) => match rx.recv().await {
            Some(token) => {
                let data = token.to_string();
                Some((data, Some(rx)))  // ❌ No [DONE] marker!
            }
            None => None,  // ❌ Just ends stream, no explicit marker
        },
        None => None,
    }
})
```

**job-client expects [DONE]:**

```rust
// job-client/src/lib.rs
if data.contains("[DONE]") {
    return Ok(job_id);  // ✅ Correctly looks for [DONE]
}
```

---

## Why This Is Critical

### 1. Architectural Violation

**[DONE] is a job lifecycle signal, NOT a narration event.**

- ✅ **Job lifecycle:** Job created → Running → Complete → [DONE]
- ❌ **Narration:** Observability events about what's happening

**Mixing these concerns is a fundamental architecture violation.**

### 2. Broken Separation of Concerns

**What should happen:**
```
job-server manages job lifecycle → sends [DONE] when job complete
narration-core emits observability events → never sends [DONE]
```

**What actually happens:**
```
narration-core emits [DONE] as a narration event ❌
job-server doesn't send [DONE] at all ❌
```

### 3. Production Impact

**In production:**
- Workers complete jobs
- job-server should send [DONE]
- Clients wait forever because no [DONE] is sent
- **Streams never close properly**

---

## Root Cause Analysis

### How Did This Happen?

1. **job-server was created** without [DONE] signal
2. **job-client was created** expecting [DONE] signal
3. **Tests started failing** because streams never closed
4. **Quick fix:** Emit [DONE] via narration to make tests pass
5. **Problem hidden:** Tests pass but architecture is broken

### Why It Wasn't Caught

- ❌ No architectural review
- ❌ No separation of concerns enforcement
- ❌ Tests passing gave false confidence
- ❌ [DONE] treated as "just another string" not a lifecycle signal

---

## The Proper Architecture

### Job Lifecycle Signals (job-server responsibility)

```rust
// job-server should send these:
[START]    - Job execution started (optional)
[PROGRESS] - Job progress updates (optional)
[DONE]     - Job completed successfully (REQUIRED)
[ERROR]    - Job failed (REQUIRED)
```

### Narration Events (narration-core responsibility)

```rust
// narration-core should send these:
{"actor": "worker", "action": "load_model", "human": "Loading model..."}
{"actor": "worker", "action": "inference_start", "human": "Starting inference"}
{"actor": "worker", "action": "inference_complete", "human": "Inference complete"}
```

### Clear Separation

```
┌─────────────┐
│ job-server  │ → Sends: [DONE], [ERROR], [START]
│ (lifecycle) │    Purpose: Signal job state changes
└─────────────┘

┌─────────────┐
│ narration   │ → Sends: JSON narration events
│ (observ.)   │    Purpose: Observability, debugging, UX
└─────────────┘
```

---

## Implementation Plan

### Phase 1: Fix job-server (Priority 1) ⚠️

**File:** `bin/99_shared_crates/job-server/src/lib.rs`

**Changes needed:**

1. **Add [DONE] to execute_and_stream:**

```rust
pub async fn execute_and_stream<T, F, Exec>(
    job_id: String,
    registry: Arc<JobRegistry<T>>,
    executor: Exec,
) -> impl Stream<Item = String>
where
    T: ToString + Send + 'static,
    F: std::future::Future<Output = Result<(), anyhow::Error>> + Send + 'static,
    Exec: FnOnce(String, serde_json::Value) -> F + Send + 'static,
{
    // ... existing code ...

    stream::unfold((receiver, false), |(rx_opt, done_sent)| async move {
        if done_sent {
            return None;  // Already sent [DONE], stop
        }

        match rx_opt {
            Some(mut rx) => match rx.recv().await {
                Some(token) => {
                    let data = token.to_string();
                    Some((data, (Some(rx), false)))
                }
                None => {
                    // Channel closed, send [DONE]
                    Some(("[DONE]".to_string(), (None, true)))
                }
            },
            None => {
                // No receiver, send [DONE] immediately
                Some(("[DONE]".to_string(), (None, true)))
            }
        }
    })
}
```

2. **Add [ERROR] support:**

```rust
// When executor fails, send [ERROR] instead of [DONE]
if let Err(e) = executor(job_id_clone.clone(), payload).await {
    // Send error to channel
    if let Some(tx) = error_sender {
        let _ = tx.send(format!("[ERROR] {}", e));
    }
}
```

3. **Update JobState to track completion:**

```rust
pub enum JobState {
    Queued,
    Running,
    Completed,      // ← Should trigger [DONE]
    Failed(String), // ← Should trigger [ERROR]
}
```

**Effort:** ~2 hours  
**Risk:** Medium (changes core streaming logic)

### Phase 2: Remove [DONE] from narration-core (Priority 1) ⚠️

**Files to fix:**

1. `narration-core/tests/e2e_job_client_integration.rs`
2. `narration-core/tests/bin/fake_queen.rs`
3. `narration-core/tests/bin/fake_hive.rs`

**Changes:**

```rust
// BEFORE (WRONG):
n!("done", "[DONE]");

// AFTER (CORRECT):
// Remove this line entirely
// job-server will send [DONE] automatically
```

**Effort:** ~30 minutes  
**Risk:** Low (just removing incorrect code)

### Phase 3: Update job-client (Priority 2)

**File:** `bin/99_shared_crates/job-client/src/lib.rs`

**Verify it handles [DONE] and [ERROR]:**

```rust
// Current code (line 163):
if data.contains("[DONE]") {
    return Ok(job_id);
}

// Add [ERROR] handling:
if data.contains("[ERROR]") {
    return Err(anyhow::anyhow!("Job failed: {}", data));
}
```

**Effort:** ~30 minutes  
**Risk:** Low (additive change)

### Phase 4: Add Tests (Priority 2)

**File:** `bin/99_shared_crates/job-server/tests/done_signal_tests.rs` (new)

```rust
#[tokio::test]
async fn test_execute_and_stream_sends_done() {
    let registry = Arc::new(JobRegistry::new());
    let job_id = registry.create_job();
    
    // Set up channel
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);
    
    // Send some tokens
    tx.send("token1".to_string()).unwrap();
    tx.send("token2".to_string()).unwrap();
    drop(tx);  // Close channel
    
    // Stream should include [DONE]
    let stream = execute_and_stream(
        job_id.clone(),
        registry,
        |_, _| async { Ok(()) }
    ).await;
    
    let results: Vec<String> = stream.collect().await;
    
    assert_eq!(results, vec!["token1", "token2", "[DONE]"]);
}

#[tokio::test]
async fn test_execute_and_stream_sends_error_on_failure() {
    // Test that [ERROR] is sent when executor fails
}
```

**Effort:** ~1 hour  
**Risk:** Low (tests only)

### Phase 5: Update Documentation (Priority 3)

**Files:**
- `job-server/README.md` - Document [DONE] and [ERROR] signals
- `job-client/README.md` - Document expected signals
- `narration-core/README.md` - Clarify narration is NOT for lifecycle signals

**Effort:** ~30 minutes  
**Risk:** None

---

## Timeline

### Immediate (Today)
- [ ] Phase 1: Fix job-server to send [DONE]
- [ ] Phase 2: Remove [DONE] from narration-core
- [ ] Phase 3: Update job-client error handling

### Short-term (This Week)
- [ ] Phase 4: Add comprehensive tests
- [ ] Phase 5: Update documentation

**Total Effort:** ~4-5 hours  
**Priority:** CRITICAL - Must fix before any production use

---

## Acceptance Criteria

### job-server
- [ ] `execute_and_stream()` sends [DONE] when channel closes
- [ ] `execute_and_stream()` sends [ERROR] when executor fails
- [ ] Tests verify [DONE] is sent
- [ ] Tests verify [ERROR] is sent

### narration-core
- [ ] No code emits [DONE] as narration
- [ ] Tests don't rely on narration for [DONE]
- [ ] Documentation clarifies narration ≠ lifecycle

### job-client
- [ ] Handles [DONE] correctly
- [ ] Handles [ERROR] correctly
- [ ] Tests verify both signals

### Architecture
- [ ] Clear separation: job-server = lifecycle, narration = observability
- [ ] No mixing of concerns
- [ ] Documentation explains the separation

---

## Risks if Not Fixed

### Short-term
- ✅ Tests currently pass (false confidence)
- ❌ Architecture is fundamentally broken
- ❌ Production will have hanging streams

### Long-term
- ❌ Impossible to add proper job lifecycle management
- ❌ Can't distinguish between narration and lifecycle events
- ❌ Maintenance nightmare (where does [DONE] come from?)
- ❌ Can't add features like job cancellation, timeouts, etc.

---

## Comparison: Before vs. After

### Before (Current - BROKEN)

```
Worker completes job
  → narration-core emits n!("done", "[DONE]")  ❌
  → job-server streams narration
  → Client sees [DONE] in narration stream
  → Client closes connection
```

**Problems:**
- [DONE] is a narration event (wrong layer)
- job-server doesn't know job is done
- Can't track job completion properly
- Mixing concerns

### After (Proposed - CORRECT)

```
Worker completes job
  → job-server detects channel close
  → job-server sends [DONE] signal  ✅
  → narration-core emits observability events
  → Client sees [DONE] from job-server
  → Client closes connection
```

**Benefits:**
- [DONE] is a lifecycle signal (correct layer)
- job-server manages job lifecycle
- Can track job completion
- Clear separation of concerns

---

## Code Locations

### Files that need changes:
```
bin/99_shared_crates/job-server/src/lib.rs
  - execute_and_stream() function (line 282-343)
  - Add [DONE] and [ERROR] signals

bin/99_shared_crates/narration-core/tests/e2e_job_client_integration.rs
  - Remove n!("done", "[DONE]") (line 83)

bin/99_shared_crates/narration-core/tests/bin/fake_queen.rs
  - Remove n!("done", "[DONE]") (line 98)

bin/99_shared_crates/narration-core/tests/bin/fake_hive.rs
  - Remove n!("done", "[DONE]") (line 103)

bin/99_shared_crates/job-client/src/lib.rs
  - Add [ERROR] handling (after line 163)
```

### New files needed:
```
bin/99_shared_crates/job-server/tests/done_signal_tests.rs
  - Test [DONE] is sent
  - Test [ERROR] is sent
```

---

## Summary

**The Issue:** [DONE] is being emitted by narration-core instead of job-server, violating separation of concerns.

**The Fix:** Move [DONE] signal to job-server where it belongs.

**The Impact:** Critical architectural fix that enables proper job lifecycle management.

**The Urgency:** Must fix before production use.

**Owner:** TEAM-304

**Status:** UNRESOLVED - Awaiting implementation
