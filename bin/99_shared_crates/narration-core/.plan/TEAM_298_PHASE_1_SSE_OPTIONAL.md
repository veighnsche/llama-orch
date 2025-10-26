# TEAM-298: Phase 1 - Make SSE Optional

**Status:** BLOCKED (Requires TEAM-297 completion)  
**Estimated Duration:** 1 week  
**Dependencies:** TEAM-297 (Phase 0 API Redesign)  
**Risk Level:** Low (non-breaking changes)

---

## Mission

Make SSE delivery optional for narration events. Narration works even if SSE channels don't exist, with stdout as primary output and SSE as opportunistic enhancement.

---

## ⚠️ CRITICAL: DO YOUR RESEARCH FIRST!

### Required Research

1. **Read TEAM-297 Handoff** - Understand new `n!()` macro and mode system
2. **Read SSE Sink** - `src/sse_sink.rs` (understand current channel system)
3. **Find All create_job_channel() Calls** - Grep and document all locations
4. **Understand Failure Mode** - What happens when channel doesn't exist now
5. **Create Research Summary** - `.plan/TEAM_298_RESEARCH_SUMMARY.md`

**DO NOT CODE UNTIL RESEARCH IS COMPLETE!**

---

## Problem: Fragile SSE Dependencies

```rust
// Current: MUST create channel first
create_job_channel(job_id.clone(), 1000);  // ← Forget this = broken!
n!("start", "Starting");  // ← Works only if channel exists

// If you reverse the order:
n!("start", "Starting");  // ← DROPPED! (no channel yet)
create_job_channel(job_id.clone(), 1000);  // ← Too late!
```

## Solution: Opportunistic SSE

```rust
// After: Works regardless!
n!("start", "Starting");  // → stdout always works
create_job_channel(job_id.clone(), 1000);  // ← Optional, for SSE
n!("progress", "Still working");  // → Now goes to SSE too!
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

### Task 3: Update `narrate()` Function

**File:** `bin/99_shared_crates/narration-core/src/lib.rs`

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

    // TEAM-298: ALWAYS emit to stderr (primary output)
    eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);

    // TEAM-298: Try SSE (opportunistic - failure OK!)
    if sse_sink::is_enabled() {
        if let Some(ref job_id) = fields.job_id {
            let event = NarrationEvent::from(fields.clone());
            let _sent = sse_sink::try_send(job_id, event);
            // Don't care if it failed - stdout already has it!
        }
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

### Task 4: Add Tests

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

- [ ] `try_send()` returns false when no channel
- [ ] `try_send()` returns true when channel exists
- [ ] Narration works without channel (no panic)
- [ ] `narrate()` always emits to stderr
- [ ] SSE still works when channel exists
- [ ] No regressions in existing tests
- [ ] New tests pass

---

## Success Criteria

1. **Narration always works** - Even without SSE channels
2. **Stdout is primary** - Always available
3. **SSE is bonus** - If channel exists, great! If not, no problem
4. **Backward compatible** - Existing code continues working

---

## Handoff to TEAM-299

Document in `.plan/TEAM_298_HANDOFF.md`:
1. What changed in SSE sink
2. New `try_send()` behavior
3. How `narrate()` was updated
4. Test results
5. Recommendations for Phase 2 (context)
