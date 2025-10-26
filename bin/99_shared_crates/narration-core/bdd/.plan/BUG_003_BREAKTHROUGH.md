# BUG-003: BREAKTHROUGH - Root Cause Found!

**Date:** October 26, 2025  
**Team:** TEAM-308  
**Status:** ✅ ROOT CAUSE CONFIRMED

---

## 🎯 SMOKING GUN: Parallel Execution

### The Discovery

```bash
# Default (concurrency=64)
126 scenarios (2 passed, 106 skipped, 18 failed)

# Sequential (concurrency=1)  
126 scenarios (17 passed, 107 skipped, 2 failed)
```

**83% of failures were caused by parallel execution!**

### The Proof (VERIFIED)

**Test 1 - WITHOUT fix:**
```bash
# Comment out .max_concurrent_scenarios(1)
cargo build && cargo run --bin bdd-runner
Result: 2 passed, 106 skipped, 18 failed ❌
```

**Test 2 - WITH fix:**
```bash
# Restore .max_concurrent_scenarios(1)  
cargo build && cargo run --bin bdd-runner
Result: 17 passed, 107 skipped, 2 failed ✅
```

**Verification:** Toggled fix on/off to prove causation.

**83% improvement (15/18 failures fixed) by forcing sequential execution!**

### The Real Root Cause

**Cucumber runs scenarios with `--concurrency 64` by default.**

With 64 concurrent scenarios ALL sharing the SAME global `CaptureAdapter` singleton:
- Scenario A clears the buffer
- Scenario B clears the buffer
- Scenario A emits 1 event → buffer has 1
- Scenario C emits 1 event → buffer has 2  
- Scenario A asserts (expects 1, gets 2+) ❌
- Scenarios D, E, F, G... all emit → buffer grows to 25+
- All scenarios fail with "Expected 1, got 25+"

**This explains EVERYTHING:**
1. ✅ Why baseline is 0 (each scenario's clear() works)
2. ✅ Why we get 25 events (from ~18 concurrent scenarios)
3. ✅ Why events have different job_ids (from different scenarios)
4. ✅ Why TEAM-307's job_id filtering seemed to "work" partially
5. ✅ Why TEAM-308's baseline tracking failed

---

## Failed Attempts Explained

### TEAM-307: job_id Filtering ❌

**Why it seemed to help:**
- Filtered out events from OTHER concurrent scenarios
- Reduced noise from parallel execution

**Why it was still wrong:**
- Doesn't fix the root cause (parallel access to global state)
- Defeats the purpose of context-free narration tests
- Race conditions still present

### TEAM-308: Baseline Tracking ❌

**Why it failed:**
- Baseline correctly recorded as 0
- But between baseline recording and assertion, OTHER concurrent scenarios added events
- Race condition: read baseline (0) → other scenarios emit → assertion fails

---

## The Solution

### Option 1: Force Sequential Execution (Quick Fix)

**File:** `src/main.rs`

```rust
#[tokio::main]
async fn main() {
    World::cucumber()
        .max_concurrent_scenarios(1)  // Force sequential
        .run_and_exit("features").await;
}
```

**Pros:**
- ✅ Simple 1-line fix
- ✅ Eliminates race conditions
- ✅ Tests will pass

**Cons:**
- ❌ Slower test execution (sequential vs parallel)
- ❌ Doesn't scale for large test suites
- ❌ Doesn't fix the underlying architecture issue

### Option 2: Thread-Local CaptureAdapter (Proper Fix)

**File:** `bin/99_shared_crates/narration-core/src/output/capture.rs`

```rust
use std::cell::RefCell;

thread_local! {
    static THREAD_CAPTURE: RefCell<Vec<CapturedNarration>> = RefCell::new(Vec::new());
}

impl CaptureAdapter {
    pub fn install() -> Self {
        // Clear thread-local storage
        THREAD_CAPTURE.with(|events| events.borrow_mut().clear());
        Self
    }
    
    pub(crate) fn capture(&self, event: CapturedNarration) {
        THREAD_CAPTURE.with(|events| {
            events.borrow_mut().push(event);
        });
    }
    
    pub fn captured(&self) -> Vec<CapturedNarration> {
        THREAD_CAPTURE.with(|events| events.borrow().clone())
    }
    
    pub fn clear(&self) {
        THREAD_CAPTURE.with(|events| events.borrow_mut().clear());
    }
}
```

**Pros:**
- ✅ Each thread/scenario has isolated storage
- ✅ No race conditions
- ✅ Supports parallel execution
- ✅ Proper architecture

**Cons:**
- ❌ More invasive change
- ❌ Requires testing across all scenarios

### Option 3: Scenario-Specific CaptureAdapter (Alternative)

**File:** `src/steps/world.rs`

Instead of global singleton, create new instance per scenario:

```rust
#[derive(cucumber::World, Default)]
pub struct World {
    pub adapter: Option<CaptureAdapter>,  // Not a singleton!
    // ...
}
```

**File:** `src/output/capture.rs`

```rust
impl CaptureAdapter {
    pub fn install() -> Self {
        // Don't use OnceLock - create new instance
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }
}
```

**Pros:**
- ✅ True isolation per scenario
- ✅ No global state
- ✅ Supports parallel execution

**Cons:**
- ❌ Requires updating ALL capture points in production code
- ❌ May break existing patterns

---

## Remaining 2 Failures

With sequential execution, only 2 scenarios still fail:

```
✘ Then the captured narration should have 1 event
✘ Then the captured narration should have 5 events
```

**These might be LEGITIMATE bugs** now that we've removed the parallel execution noise.

Need to investigate these separately:
1. Which scenarios are failing?
2. What's the actual vs expected count?
3. Is this a real bug or a test issue?

---

## Recommended Action Plan

### Phase 1: Quick Win (5 minutes) ✅ DO THIS FIRST

Force sequential execution in `main.rs`:

```rust
#[tokio::main]
async fn main() {
    World::cucumber()
        .max_concurrent_scenarios(1)
        .run_and_exit("features").await;
}
```

**Result:** 17/18 scenarios will pass immediately.

### Phase 2: Fix Remaining 2 Failures (30 minutes)

Investigate the 2 scenarios that still fail with sequential execution.

### Phase 3: Long-Term Fix (2 hours)

Implement thread-local CaptureAdapter for proper parallel test support.

---

## Lessons Learned

### 1. Question Default Assumptions

We assumed Cucumber ran scenarios sequentially. It doesn't (default concurrency=64).

**Learning:** Always check framework defaults, especially for testing frameworks.

### 2. Parallel Execution + Global State = Bad Time

The global singleton pattern is fundamentally incompatible with parallel test execution.

**Learning:** Test fixtures should be isolated per test, not global.

### 3. Symptoms vs Root Cause

Both TEAM-307 and TEAM-308 treated SYMPTOMS:
- TEAM-307: "Events are leaking" → Filter by job_id
- TEAM-308: "Buffer not clearing" → Track baseline

Neither addressed ROOT CAUSE:
- **Parallel scenarios accessing shared global state**

**Learning:** Before implementing a fix, understand the root cause.

### 4. Experimental Validation Is Critical

The breakthrough came from running a simple experiment:
- "What if I run with --concurrency 1?"
- Result: 83% of failures disappeared

**Learning:** Test your hypotheses with experiments, not just theory.

---

## Summary

**Root Cause:** Cucumber's default concurrency=64 causes parallel scenarios to race on the global CaptureAdapter singleton.

**Proof:** With `--concurrency 1`, 15/18 failures disappear (83% improvement).

**Quick Fix:** Force sequential execution (1 line change).

**Proper Fix:** Implement thread-local CaptureAdapter.

**Next Steps:**
1. ✅ Add max_concurrent_scenarios(1) to main.rs
2. ⏳ Investigate remaining 2 failures
3. ⏳ Implement thread-local CaptureAdapter for long-term

---

**Status:** ✅ ROOT CAUSE IDENTIFIED AND CONFIRMED  
**Priority:** HIGH → MEDIUM (quick fix available)  
**Team:** TEAM-308  
**Breakthrough:** Parallel execution was the problem all along
