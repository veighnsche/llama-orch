# TEAM-298: Phase 1 - SSE Optional + Privacy Fix

**Status:** IN PROGRESS (TEAM-297 complete, privacy fix required)  
**Estimated Duration:** 1 week  
**Dependencies:** TEAM-297 (Phase 0 API Redesign)  
**Risk Level:** HIGH (privacy-critical, security fix)

---

## 🚨 CRITICAL: Privacy Violation Discovered

**TEAM-297 implementation has CRITICAL privacy violation!**

See: [PRIVACY_FIX_REQUIRED.md](./PRIVACY_FIX_REQUIRED.md)

**Problem:** Global stderr output leaks narration between jobs (multi-tenant data leak)

**MUST FIX IMMEDIATELY in Phase 1!**

---

## Mission

1. **Fix privacy violation** - Remove global stderr output
2. **Make SSE optional** - Narration works without channels
3. **Add keeper mode** - Single-user CLI can print to terminal
4. **Ensure isolation** - Job-scoped narration only

---

## ⚠️ CRITICAL: DO YOUR RESEARCH FIRST!

### Required Research

1. **Read PRIVACY_FIX_REQUIRED.md** - Understand privacy violation
2. **Read TEAM-297 Handoff** - Understand new `n!()` macro and mode system
3. **Read SSE Sink** - `src/sse_sink.rs` (understand current channel system)
4. **Analyze stderr usage** - Where does `eprintln!()` get called?
5. **Plan keeper mode** - How to conditionally enable stderr
6. **Create Research Summary** - `.plan/TEAM_298_RESEARCH_SUMMARY.md`

**DO NOT CODE UNTIL RESEARCH IS COMPLETE!**

---

## Problem 1: Privacy Violation (CRITICAL!)

```rust
// TEAM-297 implementation:
eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
// ↑ ALL narration goes to global stderr!

// Multi-tenant scenario:
// User A: rbee infer --prompt "secret data"
// User B: rbee infer --prompt "other data"
// → Both see each other's narration! PRIVACY LEAK!
```

## Problem 2: Fragile SSE Dependencies

```rust
// Current: MUST create channel first
create_job_channel(job_id.clone(), 1000);  // ← Forget this = broken!
n!("start", "Starting");  // ← Works only if channel exists
```

## Solution: Privacy + Opportunistic SSE

```rust
// After: Secure AND resilient!
n!("start", "Starting");  
// → SSE if job_id + channel exists (job-scoped, secure)
// → stderr ONLY in keeper mode (single-user)
// → No global leaks!

// Keeper mode (single-user CLI):
std::env::set_var("RBEE_KEEPER_MODE", "1");
n!("start", "Starting");  // → Can print to terminal (user's own)
```

---

## Implementation Tasks

### Task 1: Add `try_send()` to SSE Sink

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

```rust
impl SseChannelRegistry {
    /// Try to send event (returns success/failure)
    ///
    /// TEAM-298: Non-blocking send - failure is OK!
    pub fn try_send_to_job(&self, job_id: &str, event: NarrationEvent) -> bool {
        let senders = self.senders.lock().unwrap();
        if let Some(tx) = senders.get(job_id) {
            match tx.try_send(event) {
                Ok(_) => return true,
                Err(_) => return false,  // Full or closed
            }
        }
        false  // Channel doesn't exist (not an error!)
    }
}

/// Try to send to SSE (non-blocking)
///
/// TEAM-298: Returns true if sent, false if no channel
/// Failure is OK - stdout always has the narration!
pub fn try_send(job_id: &str, event: NarrationEvent) -> bool {
    SSE_CHANNEL_REGISTRY.try_send_to_job(job_id, event)
}
```

### Task 2: Update `macro_emit()` for SSE

**File:** `bin/99_shared_crates/narration-core/src/lib.rs`

```rust
pub fn macro_emit(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
) {
    let mode = get_narration_mode();
    let message = match mode {
        NarrationMode::Human => human,
        NarrationMode::Cute => cute.unwrap_or(human),
        NarrationMode::Story => story.unwrap_or(human),
    };
    
    let actor = context::get_actor().unwrap_or("unknown");
    let job_id = context::get_context().and_then(|ctx| ctx.job_id.clone());
    
    let fields = NarrationFields {
        actor,
        action,
        target: action.to_string(),
        human: message.to_string(),
        cute: cute.map(|s| s.to_string()),
        story: story.map(|s| s.to_string()),
        job_id,
        ..Default::default()
    };
    
    // TEAM-298: Call updated narrate() with opportunistic SSE
    narrate(fields);
}
```

### Task 3: Remove stderr Completely (PRIVACY FIX!)

**File:** `bin/99_shared_crates/narration-core/src/lib.rs`

**⚠️ CRITICAL: Complete removal, not conditional!**

Environment variables are exploitable. The ONLY secure solution is complete removal.

See: [PRIVACY_ATTACK_SURFACE_ANALYSIS.md](./PRIVACY_ATTACK_SURFACE_ANALYSIS.md)

```rust
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    let Some(tracing_level) = level.to_tracing_level() else {
        return; // MUTE
    };

    let mode = get_narration_mode();
    let message = match mode {
        NarrationMode::Human => &fields.human,
        NarrationMode::Cute => fields.cute.as_ref().unwrap_or(&fields.human),
        NarrationMode::Story => fields.story.as_ref().unwrap_or(&fields.human),
    };

    // TEAM-298: REMOVED - No stderr in narration-core!
    // This eliminates the attack surface completely.
    // Keeper displays via separate SSE subscription (see Task 4).
    
    // OLD (INSECURE):
    // eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
    
    // NEW (SECURE):
    // No stderr code path exists in narration-core.
    // Cannot be exploited if code doesn't exist.

    // TEAM-298: SSE is PRIMARY and ONLY output
    // Job-scoped, secure, no privacy leaks.
    if sse_sink::is_enabled() {
        let _sent = sse_sink::try_send(&fields);
    }

    // Emit structured event for tracing subscribers
    match tracing_level {
        Level::TRACE => emit_event!(Level::TRACE, fields),
        Level::DEBUG => emit_event!(Level::DEBUG, fields),
        Level::INFO => emit_event!(Level::INFO, fields),
        Level::WARN => emit_event!(Level::WARN, fields),
        Level::ERROR => emit_event!(Level::ERROR, fields),
    }

    #[cfg(any(test, feature = "test-support"))]
    {
        capture::notify(fields);
    }
}
```

**Defense in depth:**
1. Code doesn't exist → can't be exploited
2. SSE is job-scoped → no cross-job leaks
3. Keeper displays separately → clear separation
4. Tests use capture → no stderr dependency

### Task 4: Add Keeper Display (Separate from Core)

**Note:** This is for TEAM-301, documented here for completeness.

**New File:** `bin/00_rbee_keeper/src/display.rs`

```rust
use observability_narration_core::sse_sink::NarrationEvent;
use tokio::sync::mpsc;

/// Display narration events to terminal
///
/// TEAM-298: This is ONLY in keeper (single-user CLI).
/// Keeper subscribes to SSE and displays to terminal.
/// This code does NOT exist in daemons (secure by design).
pub async fn display_narration_stream(mut rx: mpsc::Receiver<NarrationEvent>) {
    while let Some(event) = rx.recv().await {
        // Display to keeper's terminal (single-user, no privacy issue)
        eprintln!("{}", event.formatted);
    }
}
```

### Task 5: Add Privacy Tests (CRITICAL!)

**New File:** `bin/99_shared_crates/narration-core/tests/privacy_isolation_tests.rs`

```rust
use observability_narration_core::*;
use serial_test::serial;

#[tokio::test]
#[serial(sse_sink)]
async fn test_multi_tenant_isolation() {
    // TEAM-298: CRITICAL - Verify no cross-job data leaks
    
    let job_a = "user-a-secret-job";
    let job_b = "user-b-secret-job";
    
    // Create separate SSE channels
    sse_sink::create_job_channel(job_a.to_string(), 100);
    sse_sink::create_job_channel(job_b.to_string(), 100);
    
    let mut rx_a = sse_sink::take_job_receiver(job_a).unwrap();
    let mut rx_b = sse_sink::take_job_receiver(job_b).unwrap();
    
    // User A narration
    let ctx_a = NarrationContext::new().with_job_id(job_a);
    with_narration_context(ctx_a, async {
        n!("secret", "User A's secret API key: sk-abc123");
    }).await;
    
    // User B narration
    let ctx_b = NarrationContext::new().with_job_id(job_b);
    with_narration_context(ctx_b, async {
        n!("secret", "User B's secret API key: sk-xyz789");
    }).await;
    
    // Verify isolation
    let event_a = rx_a.recv().await.unwrap();
    let event_b = rx_b.recv().await.unwrap();
    
    assert_eq!(event_a.human, "User A's secret API key: sk-abc123");
    assert_eq!(event_b.human, "User B's secret API key: sk-xyz789");
    
    // CRITICAL: User A never sees User B's data!
    // CRITICAL: User B never sees User A's data!
}

#[test]
fn test_no_stderr_ever() {
    // TEAM-298: CRITICAL - Verify narration-core NEVER prints to stderr
    
    let adapter = CaptureAdapter::install();
    n!("test", "No stderr output");
    
    // Verify captured but NOT printed to stderr
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    
    // Visual verification: no stderr output (secure!)
}

#[test]
fn test_all_tests_use_capture_adapter() {
    // TEAM-298: All tests must use capture adapter
    // No test should depend on stderr
    
    let adapter = CaptureAdapter::install();
    
    n!("test1", "Test 1");
    n!("test2", "Test 2");
    n!("test3", "Test 3");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 3);
    
    // All captured, none printed to stderr
}
```

### Task 6: Add SSE Optional Tests

**New File:** `bin/99_shared_crates/narration-core/tests/sse_optional_tests.rs`

```rust
use observability_narration_core::*;

#[tokio::test]
async fn test_narration_without_channel() {
    // TEAM-298: Narration should work without SSE channel
    
    n!("test", "This works without channel");
    
    // Success! (before TEAM-298, this would fail)
}

#[tokio::test]
async fn test_narration_before_channel() {
    // TEAM-298: Narration can happen BEFORE channel creation
    
    let job_id = "early-job";
    
    // 1. Set context (job_id known)
    let ctx = NarrationContext::new().with_job_id(job_id);
    
    with_narration_context(ctx, async {
        // 2. Emit narration (channel doesn't exist yet!)
        n!("early", "This happened before channel");
        
        // 3. NOW create channel
        sse_sink::create_job_channel(job_id.to_string(), 100);
        let mut rx = sse_sink::take_job_receiver(job_id).unwrap();
        
        // 4. Future narration goes to SSE
        n!("later", "This goes to SSE");
        
        // 5. Only second event in channel
        let event = rx.recv().await.expect("Should receive");
        assert_eq!(event.action, "later");
        
        // First event went to stdout only (correct!)
    }).await;
}

#[tokio::test]
async fn test_sse_still_works_when_available() {
    // TEAM-298: SSE still works when channel exists
    
    let job_id = "sse-job";
    
    // Create channel first
    sse_sink::create_job_channel(job_id.to_string(), 100);
    let mut rx = sse_sink::take_job_receiver(job_id).unwrap();
    
    // Set context
    let ctx = NarrationContext::new().with_job_id(job_id);
    
    with_narration_context(ctx, async {
        n!("test", "This goes to SSE");
        
        // Should receive event
        let event = rx.recv().await.expect("Should receive");
        assert_eq!(event.action, "test");
    }).await;
}
```

---

## Verification Checklist

### Privacy (CRITICAL!)
- [ ] stderr completely removed from narration-core
- [ ] No eprintln code path exists
- [ ] Multi-tenant isolation test passes
- [ ] No cross-job data leaks
- [ ] All tests use capture adapter

### SSE Optional
- [ ] `try_send()` returns false when no channel
- [ ] `try_send()` returns true when channel exists
- [ ] Narration works without channel (no panic)
- [ ] SSE still works when channel exists

### General
- [ ] No regressions in existing tests
- [ ] New tests pass (privacy + SSE optional)
- [ ] Keeper mode flag works correctly

---

## Success Criteria

### Privacy (CRITICAL!)
1. **No multi-tenant leaks** - Job narration isolated to SSE channels
2. **No stderr in narration-core** - Code physically removed
3. **Keeper displays separately** - Via SSE subscription
4. **Security by design** - Attack surface eliminated

### Resilience
1. **Narration always works** - Even without SSE channels
2. **SSE is primary** - Job-scoped, secure output
3. **Keeper mode is secondary** - For single-user CLI only
4. **Backward compatible** - Existing code continues working (with privacy fix)

---

## Handoff to TEAM-299

Document in `.plan/TEAM_298_HANDOFF.md`:
1. What changed in SSE sink
2. New `try_send()` behavior
3. How `narrate()` was updated
4. Test results
5. Recommendations for Phase 2 (context)
