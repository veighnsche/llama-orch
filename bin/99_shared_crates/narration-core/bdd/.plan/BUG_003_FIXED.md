# BUG #003: PREMATURE "FIX" ❌

**Date:** October 26, 2025  
**Team:** TEAM-308  
**Status:** ⚠️ THIS DOCUMENT IS OUTDATED - SEE ACTUAL FIX BELOW

**WARNING:** This document describes TEAM-308's FAILED attempt (baseline tracking).  
**ACTUAL FIX:** See TEAM_308_FINAL_SUMMARY.md or BUG_003_BREAKTHROUGH.md

---

## Summary

**Problem:** BDD scenarios were failing due to global CaptureAdapter accumulating events across scenarios.

**Root Cause:** The CaptureAdapter uses a global singleton (`OnceLock`) that persists for the entire test run, causing events to accumulate across scenarios.

**Wrong Fix:** The proposed fix used job_id filtering, which:
- ❌ Defeats the purpose of testing context-free narration
- ❌ Breaks scenarios that deliberately don't use job_id
- ❌ Uses magic numbers and arbitrary fallbacks
- ❌ Doesn't fix the underlying architecture issue

**Correct Fix:** Track baseline event count per scenario and count only NEW events.

---

## What Was Changed

### 1. Added Event Tracking to World

**File:** `src/steps/world.rs`

```rust
#[derive(cucumber::World, Default)]
pub struct World {
    // ... existing fields ...
    
    // TEAM-308: Per-scenario event tracking (fixes BUG-003)
    pub initial_event_count: usize,
}
```

**Why:** Each scenario needs to track its own baseline to isolate from global state.

---

### 2. Record Baseline Instead of Clearing

**File:** `src/steps/test_capture.rs`

**Before:**
```rust
#[given("the capture buffer is empty")]
pub async fn given_capture_buffer_empty(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        adapter.clear();  // ← Unreliable with global state
    }
}
```

**After:**
```rust
#[given("the capture buffer is empty")]
pub async fn given_capture_buffer_empty(world: &mut World) {
    // TEAM-308: Record baseline instead of clearing (fixes BUG-003)
    // This allows proper test isolation with global CaptureAdapter
    if let Some(adapter) = &world.adapter {
        world.initial_event_count = adapter.captured().len();
    }
}
```

**Why:** Don't rely on `clear()` to work. Instead, record the current count as a baseline.

---

### 3. Count Only New Events

**File:** `src/steps/test_capture.rs`

**Before:** (94-120 lines of job_id filtering logic with magic numbers)
```rust
let relevant_events: Vec<_> = if let Some(ctx) = &world.context {
    if let Some(ref ctx_job_id) = ctx.job_id {
        captured.iter()
            .filter(|e| e.job_id.as_deref() == Some(ctx_job_id.as_str()))
            .collect()
    } else {
        // Take last N+10 events?! Magic number!
        let start = captured.len().saturating_sub(count + 10);
        captured[start..].iter().collect()
    }
}
// ... more fallback logic ...
```

**After:** (15 lines, no filtering, no magic numbers)
```rust
#[then(regex = r#"^the captured narration should have (\d+) events?$"#)]
pub async fn then_captured_has_n_events(world: &mut World, count: usize) {
    // TEAM-308: Count only NEW events since scenario baseline (fixes BUG-003)
    // This works for ALL scenarios, including those without job_id
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let new_events = captured.len().saturating_sub(world.initial_event_count);
        
        assert_eq!(new_events, count, 
            "Expected {} new events since scenario start, got {}. \n\
             Total events in buffer: {}, Baseline: {}, Current scenario events: {:?}", 
            count, new_events, captured.len(), world.initial_event_count,
            captured.get(world.initial_event_count..).unwrap_or(&[])
                .iter()
                .map(|e| (&e.action, &e.job_id))
                .collect::<Vec<_>>()
        );
    }
}
```

**Why:** 
- Works for ALL scenarios (with or without job_id)
- No filtering needed
- No magic numbers
- Clear error messages showing exactly what happened

---

## Why This Fix Is Correct

### ✅ Supports Context-Free Narration

These scenarios now work correctly:

```gherkin
Scenario: narration without context works normally
  When I emit narration with n!("test", "No context") without context
  Then the captured narration should have 1 event
  # ✅ No job_id filtering needed!

Scenario: empty context (no fields set)
  Given an empty narration context
  When I emit narration with n!("test", "Empty context") in context
  Then the captured narration should have 1 event
  # ✅ Works without job_id!
```

### ✅ Handles Global State Correctly

```
Scenario 1:
  Baseline: 0 events
  Emit: 1 event
  Total: 1 event
  New since baseline: 1 - 0 = 1 ✅

Scenario 2:
  Baseline: 1 event (from Scenario 1 still in buffer)
  Emit: 1 event
  Total: 2 events
  New since baseline: 2 - 1 = 1 ✅
```

### ✅ Clear Error Messages

When a test fails, you see:
```
Expected 1 new events since scenario start, got 3.
Total events in buffer: 28, Baseline: 25
Current scenario events: [("test", Some("job-123")), ("other", None), ("third", Some("job-456"))]
```

You can immediately see:
- How many events were expected
- How many were actually emitted
- The total buffer size
- The baseline count
- Exactly which events were emitted in this scenario

---

## Test Results

**Before Fix:**
- ❌ 18 scenarios failing
- ❌ All context_propagation tests broken
- ❌ "without context" scenarios broken

**After Fix:**
- ✅ All scenarios should pass
- ✅ Context-free narration works
- ✅ Context-based narration works
- ✅ No job_id filtering needed

---

## Why The User Was Right

### 1. "Global state sounds like a privacy issue"

**USER WAS RIGHT.** Not about privacy (test-only), but about **test isolation**.

Global singletons in test code violate test isolation principles. Each test should be independent.

### 2. "Please look at how we try to prevent using job_id"

**USER WAS RIGHT.** Many scenarios deliberately test WITHOUT job_id:
- "narration without context" - NO job_id
- "empty context" - NO job_id  
- "NOT inherited by tokio::spawn" - NO job_id

Forcing job_id filtering defeats the entire purpose of these tests.

### 3. "Find the real solution"

**USER WAS RIGHT.** The real solution is NOT to work around the problem with filtering, but to properly handle the global state architecture.

---

## Long-Term Recommendation

Consider redesigning CaptureAdapter to NOT use global state:

**Option 1: Thread-Local Storage**
```rust
thread_local! {
    static CAPTURE: RefCell<Vec<CapturedNarration>> = RefCell::new(Vec::new());
}
```

**Option 2: Per-World Instance**
```rust
impl CaptureAdapter {
    pub fn install() -> Self {
        Self::new()  // New instance each time, not singleton
    }
}
```

But for now, the baseline tracking solution is pragmatic and correct.

---

## Files Changed

1. ✅ `src/steps/world.rs` - Added `initial_event_count` field
2. ✅ `src/steps/test_capture.rs` - Record baseline, count new events
3. ✅ Compilation verified: `cargo check -p observability-narration-core-bdd`

**Lines Changed:**
- Added: ~10 lines
- Removed: ~30 lines of job_id filtering
- **Net result: Simpler, clearer, more correct**

---

## Lessons Learned

### 1. Be Skeptical of Quick Fixes

The initial analysis correctly identified the problem (global state), but proposed the wrong solution (job_id filtering).

**Always ask:** "Does this fix respect the design intent of the tests?"

### 2. Listen to Users

The user's skepticism was well-founded:
- "Global state sounds like a privacy issue" → Test isolation issue
- "How we try to prevent using job_id" → Tests are DESIGNED to work without it
- "Find the real solution" → Don't work around, fix properly

### 3. Simplicity Wins

**Bad fix:** 94 lines of filtering logic with magic numbers  
**Good fix:** 15 lines counting new events

The simpler solution is often the correct one.

---

## Next Steps

1. ✅ **DONE:** Implement baseline tracking
2. ✅ **DONE:** Verify compilation
3. ⏳ **TODO:** Run full BDD test suite
4. ⏳ **TODO:** Verify all 18 scenarios pass
5. ⏳ **TODO:** Update documentation

---

**Status:** ✅ IMPLEMENTATION COMPLETE  
**Verification:** ✅ COMPILES SUCCESSFULLY  
**Status:** ❌ FAILED - This approach didn't work  
**Team:** TEAM-308  
**Signature:** Investigation documented, but fix was unsuccessful

---

## ⚠️ REDIRECT TO ACTUAL FIX

This document describes a fix that **DID NOT WORK**.

**The ACTUAL fix is in:**
1. **TEAM_308_FINAL_SUMMARY.md** - Complete story with results
2. **BUG_003_BREAKTHROUGH.md** - The smoking gun discovery
3. **BUG_003_DEEP_INVESTIGATION.md** - Full investigation process

**What actually worked:**
- Root cause: Cucumber's `--concurrency 64` (parallel execution)
- Fix: `.max_concurrent_scenarios(1)` in `src/main.rs`
- Result: 83% improvement (18 failures → 2 failures)

**This document is kept for archaeological purposes only.**
