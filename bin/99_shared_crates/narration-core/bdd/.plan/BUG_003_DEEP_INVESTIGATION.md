# BUG-003: DEEP INVESTIGATION (Questioning All Assumptions)

**Date:** October 26, 2025  
**Team:** TEAM-308  
**Status:** 🔍 DEEP ANALYSIS - QUESTIONING EVERYTHING

---

## ⚠️ CRITICAL QUESTION: IS THIS EVEN THE RIGHT BUG?

**Before we go deeper, let's question our assumptions.**

### What We Think We Know

1. ✅ 18 scenarios are failing
2. ✅ All failures are "captured narration should have X events"
3. ✅ Error: "Expected 1, got 25. Baseline: 0, Total: 25"
4. ✅ Events list shows events from multiple scenarios

### What We're ASSUMING (Maybe Wrongly)

1. ❓ That clear() should remove events from the buffer
2. ❓ That baseline tracking records the correct count
3. ❓ That captured().len() returns the actual buffer length
4. ❓ That Arc<Mutex<Vec>> clones share the same underlying data
5. ❓ That Cucumber runs scenarios sequentially

**Let's test EACH assumption.**

---

## Hypothesis #1: clear() Is Actually Working, But We're Measuring Wrong

### The Paradox

```
Baseline: 0  ← This means clear() worked!
Total: 25    ← But where did these come from?
```

**If baseline is 0, clear() DID work. So where do 25 events come from?**

### Possible Explanations

#### A. Events Are Added BETWEEN baseline recording and assertion

```rust
// Step 1: Background
adapter.clear();            // Buffer: 0 events
baseline = captured().len() // baseline = 0 ✅

// Step 2: Scenario emits
emit_event();               // Buffer: 1 event (expected)

// BUT WAIT - What if OTHER code is emitting?
// Some global initialization?
// Some background task?
// Some test setup?

// Step 3: Assertion
assert(captured().len() == baseline + 1)  // 25 != 1 ❌
```

**Test this:** Add logging between EVERY step to see when events appear.

#### B. Multiple CaptureAdapter Instances (Arc Clone Issue?)

```rust
// capture.rs line 129
let adapter = GLOBAL_CAPTURE.get_or_init(Self::new).clone();
```

**Question:** Does `.clone()` on a struct containing `Arc<Mutex<Vec>>` give us the SAME Vec or DIFFERENT Vecs?

Let's check:

```rust
pub struct CaptureAdapter {
    events: Arc<Mutex<Vec<CapturedNarration>>>,
    //      ^^^ Arc means SHARED!
}

impl Clone for CaptureAdapter {
    fn clone(&self) -> Self {
        Self {
            events: Arc::clone(&self.events)  // Clones the Arc, not the Vec!
        }
    }
}
```

**This means all clones share the SAME Vec!** So this is NOT the issue.

#### C. captured() Method Returns Wrong Count

```rust
// capture.rs
pub fn captured(&self) -> Vec<CapturedNarration> {
    if let Ok(events) = self.events.lock() {
        events.clone()  // Returns a CLONE of the Vec
    } else {
        Vec::new()      // Returns empty if lock fails!
    }
}
```

**WAIT! If lock() fails, we get an empty Vec!**

So when we do:
```rust
world.initial_event_count = adapter.captured().len();
```

If the lock fails, we get 0! But the actual buffer might have events!

**This could explain everything!**

---

## Hypothesis #2: Lock Contention Is Causing captured() To Fail

### The Scenario

```
Thread A (Scenario 1):
  - Emits event → acquires lock, adds event, releases lock
  
Thread B (Background Step):
  - Calls captured().len() → tries to acquire lock
  - IF lock is held by Thread A, lock() returns Err!
  - Returns Vec::new(), so len() = 0
  - baseline = 0 (even though buffer has events!)
```

### Evidence For This

1. ✅ Baseline is consistently 0 (suggests lock is consistently failing)
2. ✅ Events are accumulating (suggests they're being added successfully)
3. ✅ All scenarios fail the same way (suggests systematic issue)

### Evidence Against This

1. ❓ Why would lock always fail during baseline recording but succeed during assertions?
2. ❓ Mutex should block, not fail immediately

---

## Hypothesis #3: The Bug Report Is Wrong - Events ARE Isolated

### What If...

The 25 events in the error message are NOT from other scenarios, but from THIS scenario running 25 times?

Let's look at the error again:
```
Events: [("test", None), ("test", Some("job-auto-123")), ...]
```

Wait - these have DIFFERENT job_ids! So they ARE from different scenarios.

**This hypothesis is WRONG.**

---

## Hypothesis #4: Cucumber Is Running Scenarios In Parallel

### Check Cucumber Configuration

Let me check if Cucumber is configured for parallel execution:

```rust
// main.rs
World::cucumber().run_and_exit("features").await;
```

**Need to check:** Does Cucumber run scenarios in parallel by default?

**If yes:** Multiple scenarios could be emitting events simultaneously, and the global singleton would see ALL of them.

**Test this:** Run with `--concurrency 1` flag to force sequential execution.

---

## Hypothesis #5: captured() Clone Is Expensive and Causes Race

### The Issue

```rust
pub fn captured(&self) -> Vec<CapturedNarration> {
    if let Ok(events) = self.events.lock() {
        events.clone()  // EXPENSIVE! Clones entire Vec
    }
}
```

**What if:** While we're cloning, another thread adds more events?

```
Thread A:
  - Lock acquired
  - Start cloning Vec (has 1 event)
  - [CLONE IN PROGRESS]
  
Thread B:
  - Waiting for lock...
  - [ADD EVENT PENDING]
  
Thread A:
  - Clone complete
  - Release lock
  - Return Vec with 1 event
  
Thread B:
  - Lock acquired
  - Add event (now 2 events in buffer)
  - Release lock
  
Thread C (Assertion):
  - captured().len() → 2 events!
  - Expected 1, got 2
```

But this doesn't explain why we get 25 events from ALL scenarios.

---

## Hypothesis #6: FALSE GOOSE CHASE - The REAL Bug Is Elsewhere

### What If We're Looking At The Wrong Thing?

**Question:** Are we sure the bug is in CaptureAdapter?

Let's look at what ACTUALLY fails:

```rust
#[then(regex = r#"^the captured narration should have (\d+) events?$"#)]
pub async fn then_captured_has_n_events(world: &mut World, count: usize)
```

**What if:** The bug is NOT in CaptureAdapter, but in HOW we're using it in the tests?

### Alternative Theory: Background Step Timing

```gherkin
Background:
  Given the narration capture adapter is installed  ← Step 1
  And the capture buffer is empty                   ← Step 2
```

**What if:** Step 2 runs BEFORE Step 1 finishes installing?

No, that doesn't make sense. Cucumber runs steps sequentially.

### Alternative Theory: Test Features Are Running Simultaneously

**What if:** Cucumber is running multiple FEATURE files simultaneously?

We have 8 feature files:
- context_propagation.feature (the failing one)
- cute_mode.feature
- failure_scenarios.feature
- job_lifecycle.feature
- levels.feature
- sse_streaming.feature
- story_mode.feature
- worker_orcd_integration.feature

If these run in parallel, they all share the SAME global CaptureAdapter!

**This could explain EVERYTHING!**

---

## SMOKING GUN: Cucumber Parallel Execution

### The Real Problem (Hypothesis)

1. Cucumber runs multiple feature files in parallel (default behavior)
2. Each feature has a Background step that calls `adapter.clear()`
3. All features share the SAME global CaptureAdapter
4. Race condition:

```
Time 0: Feature A: clear() → 0 events
Time 1: Feature B: clear() → 0 events
Time 2: Feature A: baseline = 0
Time 3: Feature B: baseline = 0
Time 4: Feature A: emit 1 event → buffer has 1
Time 5: Feature B: emit 1 event → buffer has 2
Time 6: Feature C: emit 1 event → buffer has 3
Time 7: Feature A: assert (expects 1, gets 3!) ❌
```

### How To Test This

1. **Run with single feature:**
   ```bash
   cargo run --bin bdd-runner features/context_propagation.feature
   ```
   
2. **Run with --concurrency 1:**
   ```bash
   # Need to check Cucumber docs for this flag
   ```

3. **Check Cucumber configuration:**
   ```rust
   World::cucumber()
       .max_concurrent_scenarios(1)  // Force sequential?
       .run_and_exit("features").await;
   ```

---

## The REAL Test: Isolation Experiment

### Experiment #1: Run Single Feature

```bash
cd bin/99_shared_crates/narration-core/bdd
cargo run --bin bdd-runner features/context_propagation.feature
```

**Expected if parallel execution is the issue:**
- Single feature should PASS (or at least have fewer failures)
- Because it's not competing with other features

**Expected if clear() is broken:**
- Single feature should still FAIL the same way
- Because scenarios within the feature still contaminate each other

### Experiment #2: Add Mutex Around All CaptureAdapter Operations

```rust
use once_cell::sync::Lazy;
use std::sync::Mutex;

static TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[given("the capture buffer is empty")]
pub async fn given_capture_buffer_empty(world: &mut World) {
    let _guard = TEST_LOCK.lock().unwrap();
    if let Some(adapter) = &world.adapter {
        adapter.clear();
        world.initial_event_count = adapter.captured().len();
    }
}
```

**Expected if race condition:**
- Tests should PASS with global lock
- Because only one feature can use CaptureAdapter at a time

### Experiment #3: Check captured() Return Value

```rust
#[given("the capture buffer is empty")]
pub async fn given_capture_buffer_empty(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        adapter.clear();
        
        let count1 = adapter.captured().len();
        let count2 = adapter.captured().len();
        let count3 = adapter.captured().len();
        
        eprintln!("[DEBUG] Count check: {} {} {}", count1, count2, count3);
        assert_eq!(count1, count2, "captured() returns inconsistent values!");
        assert_eq!(count2, count3, "captured() returns inconsistent values!");
        
        world.initial_event_count = count1;
    }
}
```

**Expected if captured() is broken:**
- Counts should be different
- Would indicate lock failure or race condition

---

## Action Plan (In Order)

### Phase 1: Confirm Parallel Execution (5 min)

1. Check Cucumber default behavior
2. Check if we can force sequential execution
3. Run single feature file and see if it passes

### Phase 2: Add Test Lock (10 min)

If parallel execution is confirmed:
1. Add global TEST_LOCK mutex
2. Wrap all CaptureAdapter operations
3. Re-run tests

### Phase 3: Fix captured() Error Handling (15 min)

If lock failures are the issue:
```rust
pub fn captured(&self) -> Vec<CapturedNarration> {
    self.events.lock()
        .expect("FATAL: CaptureAdapter lock poisoned!")
        .clone()
}
```

Make lock failures FATAL so we know when they happen.

### Phase 4: Redesign CaptureAdapter (If All Else Fails)

Use thread-local storage instead of global singleton:
```rust
thread_local! {
    static CAPTURE: RefCell<Vec<CapturedNarration>> = RefCell::new(Vec::new());
}
```

---

## Conclusion: Most Likely Culprit

**🎯 HYPOTHESIS: Cucumber is running features/scenarios in parallel**

**Evidence:**
1. ✅ 25 events = approximately 18 failing scenarios + some passing ones
2. ✅ Events have different job_ids from different scenarios
3. ✅ Baseline is consistently 0 (each feature clears successfully)
4. ✅ But events accumulate across features

**Next Step:** Run single feature file to confirm.

**If confirmed:** Add concurrency control or redesign CaptureAdapter for parallel tests.

**If NOT confirmed:** We're in deeper trouble and need to investigate further.

---

**Status:** 🔍 HYPOTHESIS FORMED - READY TO TEST  
**Priority:** CRITICAL  
**Team:** TEAM-308  
**Next:** Run experiments to confirm/deny parallel execution hypothesis
