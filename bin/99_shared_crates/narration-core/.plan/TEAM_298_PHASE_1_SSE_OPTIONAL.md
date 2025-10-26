# TEAM-298: Phase 1 - Make SSE Optional

**Status:** READY FOR IMPLEMENTATION  
**Estimated Duration:** 1 week  
**Dependencies:** None  
**Risk Level:** Low (non-breaking changes)

---

## Mission

Make SSE delivery optional for narration events. Narration should work even if SSE channels don't exist yet, with stdout as the primary output and SSE as opportunistic enhancement.

---

## ⚠️ CRITICAL: DO YOUR RESEARCH FIRST!

**DO NOT START CODING UNTIL YOU COMPLETE THE RESEARCH PHASE!**

### Required Research (Complete ALL before coding)

#### 1. Read Current Implementation
- [ ] Read `bin/99_shared_crates/narration-core/src/sse_sink.rs` (full file)
- [ ] Read `bin/99_shared_crates/narration-core/src/lib.rs` (narrate functions)
- [ ] Read `bin/99_shared_crates/narration-core/src/builder.rs` (emit methods)
- [ ] Read `bin/10_queen_rbee/src/job_router.rs` (create_job function)
- [ ] Read `bin/20_rbee_hive/src/job_router.rs` (create_job function)

#### 2. Understand Current Flow
Document your understanding of:
- [ ] How `create_job_channel()` works
- [ ] How `send()` routes to SSE channels
- [ ] What happens if channel doesn't exist (currently)
- [ ] Where `job_id` comes from in narration
- [ ] How `SSE_CHANNEL_REGISTRY` is structured

#### 3. Find All Usage Sites
Search and document:
- [ ] All calls to `create_job_channel()` (use grep)
- [ ] All calls to `sse_sink::send()` (use grep)
- [ ] All places where `.job_id()` is manually added
- [ ] All SSE endpoint handlers

#### 4. Review Tests
- [ ] Read `bin/99_shared_crates/narration-core/tests/sse_channel_lifecycle_tests.rs`
- [ ] Read `bin/99_shared_crates/narration-core/tests/narration_job_isolation_tests.rs`
- [ ] Understand what tests currently verify

#### 5. Create Research Summary
Write a document (`.plan/TEAM_298_RESEARCH_SUMMARY.md`) with:
- Current flow diagram (text-based)
- List of all files that need changes
- List of all tests that need updates
- Potential risks you identified
- Questions for clarification

**ONLY AFTER COMPLETING RESEARCH: Proceed to implementation**

---

## Problem Statement

### Current Behavior (Fragile)

```rust
// In job_router.rs
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    
    // ← CRITICAL: Must create channel BEFORE any narration!
    sse_sink::create_job_channel(job_id.clone(), 1000);
    
    NARRATE.action("job_create")
        .job_id(&job_id)  // ← REQUIRED or narration is dropped!
        .emit();
    
    Ok(JobResponse { job_id, sse_url })
}
```

**Problems:**
1. If you forget `create_job_channel()`, narration is silently dropped
2. If narration happens before `create_job_channel()`, it's lost
3. Race conditions between channel creation and narration emission
4. No fallback mechanism

### Current SSE Sink Behavior

```rust
// In sse_sink.rs
pub fn send(fields: &NarrationFields) {
    let Some(job_id) = &fields.job_id else {
        return;  // DROP if no job_id (fail-fast)
    };
    
    let event = NarrationEvent::from(fields.clone());
    SSE_CHANNEL_REGISTRY.send_to_job(job_id, event);
    // ↑ Drops silently if channel doesn't exist
}
```

---

## Desired Behavior (Resilient)

### After Phase 1

```rust
// In job_router.rs
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    
    // Channel creation is now OPTIONAL (but still recommended)
    sse_sink::create_job_channel(job_id.clone(), 1000);
    
    NARRATE.action("job_create")
        .job_id(&job_id)
        .emit();  // ← Works even if channel doesn't exist!
    
    Ok(JobResponse { job_id, sse_url })
}

// Even better: narration BEFORE channel creation works!
pub async fn create_job_v2(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    
    NARRATE.action("job_create")
        .job_id(&job_id)
        .emit();  // ← Goes to stdout, SSE attempt fails gracefully
    
    sse_sink::create_job_channel(job_id.clone(), 1000);  // ← Created after!
    
    Ok(JobResponse { job_id, sse_url })
}
```

**Benefits:**
1. Narration always works (stdout is always available)
2. SSE is opportunistic (if channel exists, great! if not, no problem)
3. No silent failures
4. More resilient to timing issues

---

## Implementation Tasks

### Task 1: Add `try_send()` to SseChannelRegistry

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

**Changes:**
```rust
impl SseChannelRegistry {
    // EXISTING: send_to_job (keeps current behavior)
    pub fn send_to_job(&self, job_id: &str, event: NarrationEvent) {
        let senders = self.senders.lock().unwrap();
        if let Some(tx) = senders.get(job_id) {
            let _ = tx.try_send(event);
        }
        // Silently drops if channel doesn't exist
    }
    
    // NEW: try_send_to_job (returns success/failure)
    pub fn try_send_to_job(&self, job_id: &str, event: NarrationEvent) -> bool {
        let senders = self.senders.lock().unwrap();
        if let Some(tx) = senders.get(job_id) {
            match tx.try_send(event) {
                Ok(_) => return true,
                Err(_) => return false,  // Channel full or closed
            }
        }
        false  // Channel doesn't exist
    }
}
```

**Testing:**
```rust
#[test]
fn test_try_send_returns_false_when_no_channel() {
    let registry = SseChannelRegistry::new();
    let event = NarrationEvent { /* ... */ };
    
    let result = registry.try_send_to_job("nonexistent-job", event);
    assert_eq!(result, false);
}

#[test]
fn test_try_send_returns_true_when_channel_exists() {
    let registry = SseChannelRegistry::new();
    registry.create_job_channel("job-123".to_string(), 100);
    
    let event = NarrationEvent { /* ... */ };
    let result = registry.try_send_to_job("job-123", event);
    assert_eq!(result, true);
}
```

### Task 2: Refactor `send()` to Use `try_send()`

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

**Changes:**
```rust
/// Send a narration event to job-specific channel.
///
/// TEAM-298: REFACTORED - SSE is now opportunistic, not mandatory
///
/// This function:
/// 1. Checks if job_id exists (from fields or context)
/// 2. Attempts to send to SSE channel
/// 3. Returns success/failure (caller can log if needed)
///
/// # Behavior
///
/// - **No job_id**: Returns false (no SSE possible)
/// - **Channel exists**: Attempts send, returns true/false
/// - **Channel missing**: Returns false (not an error!)
///
/// Narration always goes to stdout first, so SSE failure is OK.
pub fn send(fields: &NarrationFields) -> bool {
    // Try to get job_id from fields or context
    let job_id = fields.job_id.as_ref()
        .or_else(|| {
            // TEAM-298: Check thread-local context
            crate::context::get_context()
                .and_then(|ctx| ctx.job_id.as_ref())
        });
    
    if let Some(job_id) = job_id {
        let event = NarrationEvent::from(fields.clone());
        return SSE_CHANNEL_REGISTRY.try_send_to_job(job_id, event);
    }
    
    false  // No job_id, SSE not possible
}
```

**Testing:**
```rust
#[test]
fn test_send_without_channel_returns_false() {
    let fields = NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        job_id: Some("nonexistent-job".to_string()),
        ..Default::default()
    };
    
    let result = sse_sink::send(&fields);
    assert_eq!(result, false);  // Channel doesn't exist, but that's OK!
}

#[test]
fn test_send_with_channel_returns_true() {
    sse_sink::create_job_channel("job-123".to_string(), 100);
    
    let fields = NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        job_id: Some("job-123".to_string()),
        ..Default::default()
    };
    
    let result = sse_sink::send(&fields);
    assert_eq!(result, true);
}
```

### Task 3: Update `narrate()` to Always Emit to Stdout

**File:** `bin/99_shared_crates/narration-core/src/lib.rs`

**Changes:**
```rust
/// Emit INFO-level narration (default)
///
/// TEAM-298: REFACTORED - Stdout is primary, SSE is opportunistic
///
/// This function:
/// 1. ALWAYS emits to stderr (guaranteed visibility)
/// 2. Attempts to emit to SSE (if job_id present)
/// 3. Notifies test capture adapter (if active)
pub fn narrate(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Info)
}

pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    let Some(tracing_level) = level.to_tracing_level() else {
        return; // MUTE - no output
    };

    // TEAM-298: ALWAYS output to stderr first (primary output)
    // This works whether or not SSE channel exists
    eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, fields.human);

    // TEAM-298: Attempt SSE delivery (opportunistic)
    // Returns true if sent, false if no channel (both are OK!)
    if sse_sink::is_enabled() {
        let _sse_sent = sse_sink::send(&fields);
        // We don't care if SSE failed - stdout already has it
    }

    // Emit structured event for tracing subscribers
    match tracing_level {
        Level::TRACE => emit_event!(Level::TRACE, fields),
        Level::DEBUG => emit_event!(Level::DEBUG, fields),
        Level::INFO => emit_event!(Level::INFO, fields),
        Level::WARN => emit_event!(Level::WARN, fields),
        Level::ERROR => emit_event!(Level::ERROR, fields),
    }

    // Notify capture adapter if active
    #[cfg(any(test, feature = "test-support"))]
    {
        capture::notify(fields);
    }
}
```

### Task 4: Add Mode Detection Enum

**File:** `bin/99_shared_crates/narration-core/src/lib.rs`

**Changes:**
```rust
/// Narration execution mode
///
/// TEAM-298: Determines how narration is delivered
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NarrationMode {
    /// Standalone: stdout only (no job context)
    Standalone,
    
    /// Job context: stdout + SSE (job_id known)
    Job,
}

impl NarrationMode {
    /// Auto-detect current execution mode
    ///
    /// TEAM-298: Checks thread-local context for job_id
    pub fn detect() -> Self {
        if let Some(ctx) = crate::context::get_context() {
            if ctx.job_id.is_some() {
                return Self::Job;
            }
        }
        Self::Standalone
    }
}
```

**Testing:**
```rust
#[test]
fn test_mode_detect_standalone() {
    let mode = NarrationMode::detect();
    assert_eq!(mode, NarrationMode::Standalone);
}

#[tokio::test]
async fn test_mode_detect_job() {
    let ctx = NarrationContext::new().with_job_id("job-123");
    
    crate::context::with_narration_context(ctx, async {
        let mode = NarrationMode::detect();
        assert_eq!(mode, NarrationMode::Job);
    }).await;
}
```

### Task 5: Update Tests

**Files to update:**
- `bin/99_shared_crates/narration-core/tests/sse_channel_lifecycle_tests.rs`
- `bin/99_shared_crates/narration-core/tests/narration_job_isolation_tests.rs`
- `bin/99_shared_crates/narration-core/tests/format_consistency.rs`

**New tests to add:**
```rust
// In sse_channel_lifecycle_tests.rs
#[tokio::test]
async fn test_narration_without_channel_works() {
    // TEAM-298: Narration should work even without channel
    
    let fields = NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test without channel".to_string(),
        job_id: Some("no-channel-job".to_string()),
        ..Default::default()
    };
    
    // This should NOT panic or fail
    narrate(fields);
    
    // Success! (before TEAM-298, this would fail-fast)
}

#[tokio::test]
async fn test_narration_before_channel_creation() {
    // TEAM-298: Narration can happen before channel is created
    
    let job_id = "early-narration-job";
    
    // 1. Emit narration (channel doesn't exist yet!)
    let fields = NarrationFields {
        actor: "test",
        action: "early",
        target: job_id.to_string(),
        human: "This happened before channel was created".to_string(),
        job_id: Some(job_id.to_string()),
        ..Default::default()
    };
    narrate(fields);  // Goes to stdout, SSE fails gracefully
    
    // 2. Now create the channel
    sse_sink::create_job_channel(job_id.to_string(), 100);
    let mut rx = sse_sink::take_job_receiver(job_id).unwrap();
    
    // 3. Future narration should go to SSE
    let fields2 = NarrationFields {
        actor: "test",
        action: "later",
        target: job_id.to_string(),
        human: "This happened after channel was created".to_string(),
        job_id: Some(job_id.to_string()),
        ..Default::default()
    };
    narrate(fields2);
    
    // 4. Only the second event should be in SSE channel
    let event = rx.recv().await.expect("Should receive event");
    assert_eq!(event.action, "later");
    
    // First event went to stdout only (that's correct!)
}
```

---

## Verification Checklist

Before marking this phase complete, verify:

- [ ] `try_send_to_job()` returns false when channel doesn't exist
- [ ] `try_send_to_job()` returns true when channel exists
- [ ] `send()` works without channel (returns false, no panic)
- [ ] `narrate()` always emits to stderr
- [ ] `narrate()` attempts SSE delivery (but doesn't require it)
- [ ] `NarrationMode::detect()` works correctly
- [ ] All existing tests still pass
- [ ] New tests for resilient behavior pass
- [ ] No regressions in SSE delivery (when channel exists)

---

## Success Criteria

1. **Narration works without channels** - No panics, no silent failures
2. **Stdout is always available** - Users always see narration
3. **SSE is opportunistic** - If channel exists, great! If not, no problem
4. **Backward compatible** - Existing code continues working
5. **Tests pass** - All existing + new tests green

---

## Handoff to TEAM-299

After completing Phase 1, document:
1. What you changed (file-by-file summary)
2. What tests you added
3. Any issues you encountered
4. Recommendations for Phase 2

Create: `.plan/TEAM_298_HANDOFF.md`

---

## Notes

- This phase is **non-breaking** - existing code continues working
- Focus on **resilience** - narration should never fail silently
- **Stdout is king** - SSE is a bonus, not a requirement
- Keep changes **minimal** - don't refactor more than necessary
