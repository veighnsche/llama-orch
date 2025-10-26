# BDD Test Run Results

**Date:** October 26, 2025  
**Team:** TEAM-308  
**Status:** ⚠️ PARTIAL SUCCESS - Bug fix didn't work as expected

---

## Test Summary

```
8 features
126 scenarios (2 passed, 106 skipped, 18 failed)
459 steps (335 passed, 106 skipped, 18 failed)
```

### Breakdown

- ✅ **2 scenarios passed**
- ⏭️ **106 scenarios skipped** (unimplemented steps)
- ❌ **18 scenarios failed** (BUG-003 still present)

### Unimplemented Steps

**You asked about 75 unimplemented steps, but we actually have 106 skipped steps.**

These are spread across:
- `cute_mode.feature` - Multiple scenarios
- `story_mode.feature` - Multiple scenarios  
- `worker_orcd_integration.feature` - Most scenarios
- `failure_scenarios.feature` - Some scenarios
- `job_lifecycle.feature` - Some scenarios
- `sse_streaming.feature` - Some scenarios

---

## BUG-003 Status: ❌ NOT FIXED

### The Problem

The baseline tracking fix didn't work. Here's what's happening:

**Error Message:**
```
Expected 1 new events since scenario start, got 25.
Total events in buffer: 25, Baseline: 0
Current scenario events: [("test", None), ("test", Some("job-auto-123")), ...]
```

**Analysis:**
1. Baseline is recorded as 0 ✅
2. But we're getting 25 events total ❌
3. So 25 - 0 = 25 new events (expected 1)

**This means:**
- The `clear()` IS working (baseline = 0)
- But events from ALL scenarios are being captured
- The events list shows events from multiple scenarios

---

## Root Cause (Revised)

I was WRONG about the root cause. It's not just about baseline tracking.

### The REAL Problem

Looking at the events list:
```
[("test", None), ("test", Some("job-auto-123")), ("test", None), ("test", None), 
 ("test", Some("job-123")), ("first", Some("job-same-task")), 
 ("second", Some("job-same-task")), ("third", Some("job-same-task")), 
 ("before", Some("job-await-test")), ("spawned", Some("job-spawn-manual")), 
 ("spawned_no_ctx", None), ("task_a", Some("job-a-123")), 
 ("task_b", Some("job-b-456")), ...]
```

These are events from MULTIPLE scenarios:
- `job-auto-123` - from "job_id is automatically injected" scenario
- `job-123` - from "all context fields" scenario
- `job-same-task` - from "context works within same task" scenario
- `job-await-test` - from "context survives await" scenario
- etc.

**The issue:** Events are being captured DURING the scenario execution, but they're ALL showing up in the FIRST scenario's assertion!

### Why This Is Happening

I think the issue is that Cucumber is running all the `When` steps BEFORE running any `Then` steps. Or there's some async execution happening where all scenarios emit their events, and then all assertions run.

Let me check the Cucumber execution model...

Actually, looking more carefully at the output, I see that the Background runs for each scenario:

```
Scenario: job_id is automatically injected from context
 ✔> Given the narration capture adapter is installed
 ✔> And the capture buffer is empty
 ✔  Given a narration context with job_id "job-auto-123"
 ✔  When I emit narration with n!("test", "Message") in context
 ✘  Then the captured narration should have 1 event
```

So the steps ARE running in order. But somehow, when we get to the assertion, we have events from ALL scenarios.

### Hypothesis: Async Execution

Maybe the issue is that the `When` steps are async, and they're all executing in parallel or out of order?

Let me check the step implementation:

```rust
#[when(regex = r#"^I emit narration with n!\("([^"]+)", "([^"]+)"\) in context$"#)]
async fn emit_in_context(world: &mut World, action: String, message: String) {
    if let Some(ctx) = world.context.clone() {
        let action_static: &'static str = Box::leak(action.into_boxed_str());
        with_narration_context(ctx, async move {
            n!(action_static, "{}", message);
        }).await;  // ← We DO await!
    }
}
```

We're awaiting, so this should be synchronous.

---

## The ACTUAL Problem

After more analysis, I believe the issue is:

**The `clear()` method is NOT actually clearing the global buffer.**

Evidence:
1. We call `install()` which calls `clear()`
2. We call `clear()` again
3. We record baseline as 0
4. But then we have 25 events

If `clear()` was working, we should have 0 events after clearing, then 1 event after emitting, for a total of 1 event.

But we have 25 events, which means the buffer was NEVER cleared.

### Why Isn't `clear()` Working?

Looking at the implementation:

```rust
pub fn clear(&self) {
    if let Ok(mut events) = self.events.lock() {
        events.clear();
    }
}
```

This should work. Unless...

**Hypothesis:** There's a race condition where events are being added WHILE we're clearing, or AFTER we clear but BEFORE we record the baseline.

Or maybe the `lock()` is failing and we're silently ignoring the error?

---

## Next Steps

### Option 1: Add More Debug Output

Add debug output to see:
1. When `clear()` is called
2. How many events before/after clear
3. When events are being added
4. Thread IDs to check for threading issues

### Option 2: Force Clear in Assertion

Instead of relying on baseline tracking, force a clear at the start of each scenario and then check the count:

```rust
#[given("the capture buffer is empty")]
pub async fn given_capture_buffer_empty(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        // Force clear multiple times
        adapter.clear();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        adapter.clear();
        
        // Verify it's actually empty
        let count = adapter.captured().len();
        assert_eq!(count, 0, "Buffer should be empty after clear, but has {} events", count);
        
        world.initial_event_count = 0;
    }
}
```

### Option 3: Use Mutex to Ensure Synchronization

Maybe the issue is that we need stronger synchronization:

```rust
use std::sync::Mutex;
use once_cell::sync::Lazy;

static CLEAR_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[given("the capture buffer is empty")]
pub async fn given_capture_buffer_empty(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        let _guard = CLEAR_LOCK.lock().unwrap();
        adapter.clear();
        world.initial_event_count = adapter.captured().len();
    }
}
```

### Option 4: Check if `lock()` is Failing

Add error handling:

```rust
pub fn clear(&self) {
    match self.events.lock() {
        Ok(mut events) => {
            eprintln!("[DEBUG] Clearing {} events", events.len());
            events.clear();
            eprintln!("[DEBUG] After clear: {} events", events.len());
        }
        Err(e) => {
            eprintln!("[ERROR] Failed to lock events for clearing: {:?}", e);
        }
    }
}
```

---

## Conclusion

The baseline tracking approach was correct in theory, but there's a deeper issue:

**The `clear()` method is not actually clearing the buffer, OR events are being added after clearing but before baseline recording.**

We need to:
1. Add debug output to understand what's happening
2. Verify that `clear()` is actually working
3. Check for race conditions or threading issues
4. Possibly add stronger synchronization

---

**Status:** ⚠️ INVESTIGATION NEEDED  
**Priority:** HIGH  
**Team:** TEAM-308  
**Next:** Add debug output and investigate why `clear()` isn't working
