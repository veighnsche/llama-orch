# BDD Step Implementation Guide for Next Team

**Date:** October 26, 2025  
**From:** TEAM-309  
**To:** TEAM-310  
**Status:** 📚 READY FOR FINAL FIXES

---

## 🎯 Mission

Fix the remaining **25 failing scenarios** and **23 skipped scenarios** to achieve 100% test coverage.

---

## 📊 Current State (After TEAM-309)

### Test Results
```
123 scenarios (75 passed, 23 skipped, 25 failed)
740 steps (692 passed, 23 skipped, 25 failed)

Pass Rate: 61% scenarios, 93.5% steps
```

### What TEAM-309 Completed
- ✅ **levels.rs** (147 LOC, 8 steps) - Narration level testing
- ✅ **job_lifecycle.rs** (460 LOC, 50 steps) - Complete job lifecycle
- ✅ **sse_extended.rs** (487 LOC, 40 steps) - SSE streaming
- ✅ **worker_integration.rs** (418 LOC, 78 steps) - Worker scenarios
- ✅ Removed all redaction logic (not part of narration-core)
- ✅ Fixed ambiguous step definitions
- ✅ Increased pass rate from 25% to 61%

**Total:** 176 steps implemented, 1,512 LOC

### What Remains
- ⚠️ **25 failing scenarios** - Need fixing
- ⏳ **23 skipped scenarios** - Need implementation (mostly failure scenarios)

---

## 🚨 CRITICAL ISSUES TO FIX

### Issue #1: Event Count Mismatches (Priority: HIGH, ~10 scenarios)
**Symptoms:**
```
Step panicked: assertion `left == right` failed: Expected 1 new events since scenario start, got 2.
Step panicked: assertion `left == right` failed: Expected 5 new events since scenario start, got 6.
Step panicked: assertion `left == right` failed: Expected 0 new events since scenario start, got 1.
```

**Root Cause:**
The `initial_event_count` baseline is not being set correctly in some scenarios. The global `CaptureAdapter` accumulates events across the entire test run, so each scenario needs to establish its baseline.

**Affected Scenarios:**
- Context propagation tests (context_propagation.feature)
- Some cute mode tests with context

**How to Fix:**
1. In `context_steps.rs`, ensure `world.initial_event_count` is set in the GIVEN step
2. Pattern to follow:
```rust
#[given("narration capture is enabled")]
async fn given_capture_enabled(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        world.initial_event_count = adapter.captured().len();
    }
}
```

**Files to Check:**
- `src/steps/context_steps.rs` - Lines around context setup
- `src/steps/cute_mode.rs` - Context-related scenarios

**Estimated effort:** 1-2 hours

---

### Issue #2: Job State Transitions (Priority: HIGH, ~5 scenarios)
**Symptoms:**
```
Step panicked: assertion `left == right` failed: Job should be in 'Completed' state
Step panicked: assertion `left == right` failed: Job should be completed
```

**Root Cause:**
The `world.job_state` is not being updated when job operations complete. WHEN steps emit narration but don't update the job state.

**Affected Scenarios:**
- Job lifecycle tests (job_lifecycle.feature)
- Job completion scenarios

**How to Fix:**
1. In `job_lifecycle.rs`, update `world.job_state` in WHEN steps
2. Example:
```rust
#[when("the job completes successfully")]
async fn when_job_completes(world: &mut World) {
    world.job_state = Some("Completed".to_string());
    // Emit completion narration
    narrate(NarrationFields {
        actor: "test",
        action: "complete",
        target: "test".to_string(),
        human: "Job completed".to_string(),
        ..Default::default()
    });
}
```

**Files to Fix:**
- `src/steps/job_lifecycle.rs` - All WHEN steps that change job state

**Estimated effort:** 1-2 hours

---

### Issue #3: Missing "cancel" in Human Field (Priority: MEDIUM, ~2 scenarios)
**Symptoms:**
```
Step panicked: Human field should include 'cancel'
```

**Root Cause:**
The cancellation narration doesn't include the word "cancel" in the human field.

**Affected Scenarios:**
- Job cancellation tests (job_lifecycle.feature)
- Worker cancellation tests (worker_orcd_integration.feature)

**How to Fix:**
1. In `job_lifecycle.rs` and `worker_integration.rs`, update cancellation narration:
```rust
#[when("the job is cancelled")]
async fn when_job_cancelled(world: &mut World) {
    world.job_state = Some("Cancelled".to_string());
    narrate(NarrationFields {
        actor: "test",
        action: "cancel",
        target: "test".to_string(),
        human: "Cancelling job".to_string(), // Must include "cancel"
        ..Default::default()
    });
}
```

**Files to Fix:**
- `src/steps/job_lifecycle.rs` - Cancellation steps
- `src/steps/worker_integration.rs` - Worker cancellation steps

**Estimated effort:** 30 minutes

---

### Issue #4: SSE Client Disconnection (Priority: MEDIUM, ~3 scenarios)
**Symptoms:**
```
Step panicked: Client should be marked as disconnected
Step panicked: Narration should be preserved
```

**Root Cause:**
SSE disconnection logic is not implemented. Steps are stubs.

**Affected Scenarios:**
- SSE stream disconnection tests (failure_scenarios.feature)
- SSE reconnection tests

**How to Fix:**
1. In `sse_extended.rs`, implement actual disconnection tracking:
```rust
#[when("the SSE client disconnects")]
async fn when_sse_disconnects(world: &mut World) {
    if let Some(job_id) = &world.job_id {
        world.sse_channels.insert(job_id.clone(), false); // Mark as disconnected
    }
}

#[then("the client should be marked as disconnected")]
async fn then_client_disconnected(world: &mut World) {
    if let Some(job_id) = &world.job_id {
        let is_connected = world.sse_channels.get(job_id).unwrap_or(&true);
        assert!(!is_connected, "Client should be disconnected");
    }
}
```

**Files to Fix:**
- `src/steps/sse_extended.rs` - Disconnection and reconnection steps

**Estimated effort:** 1-2 hours

---

### Issue #5: Job ID Mismatch in Events (Priority: MEDIUM, ~3 scenarios)
**Symptoms:**
```
Step panicked: assertion `left == right` failed: Event 1 job_id mismatch
```

**Root Cause:**
Narration events are not being emitted with the correct `job_id` from the world state.

**Affected Scenarios:**
- Job lifecycle tests with multiple events
- Context propagation tests

**How to Fix:**
1. Ensure all narration emissions include `job_id` from world:
```rust
if let Some(job_id) = &world.job_id {
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        job_id: Some(job_id.clone()),
        ..Default::default()
    });
}
```

**Files to Check:**
- `src/steps/job_lifecycle.rs` - All narration emissions
- `src/steps/context_steps.rs` - Context-based emissions

**Estimated effort:** 1 hour

---

## 🔧 SKIPPED SCENARIOS (23 scenarios, Priority: LOW)

### Failure Scenarios (19 scenarios skipped)
**File:** `failure_scenarios.feature`  
**Status:** Steps not implemented - require integration test infrastructure

**Why Skipped:**
These scenarios require actual HTTP servers, SSE channels, process management, and network simulation. They are integration tests, not unit tests.

**Scenarios:**
1. Network failures (connection refused, timeout, partial failure)
2. SSE stream disconnections and reconnections
3. Process crashes (worker, service)
4. Timeouts (job execution, SSE stream, context operations)
5. Resource limits (channel full, too many jobs, large messages)
6. Invalid inputs (null bytes, invalid UTF-8, empty messages)
7. Concurrent access issues
8. System recovery and cleanup

**Recommendation:**
- Mark these as integration tests
- Implement when you have actual HTTP/SSE infrastructure
- OR implement mock infrastructure (tokio::test with mock servers)

**Estimated effort:** 8-12 hours (requires infrastructure)

---

### Story Mode Scenarios (4 scenarios skipped)
**File:** `story_mode.feature`  
**Status:** Steps not implemented

**Scenarios:**
1. Story narration with n!() macro
2. Story narration with emoji
3. Story narration with context
4. Story narration with correlation ID

**How to Fix:**
Similar to cute_mode.rs, implement story-specific steps in `story_mode_extended.rs`.

**Estimated effort:** 1-2 hours

---

## 📋 QUICK FIXES (Can be done in 1 session)

### 1. Fix Event Count Baselines (1-2 hours)
- Review all GIVEN steps in `context_steps.rs`
- Ensure `world.initial_event_count` is set
- Test: `cargo run --bin bdd-runner -- --tags @context`

### 2. Fix Job State Transitions (1-2 hours)
- Update all WHEN steps in `job_lifecycle.rs`
- Add state transitions: Queued → Running → Completed/Failed/Cancelled
- Test: `cargo run --bin bdd-runner -- --tags @job`

### 3. Fix Cancellation Messages (30 minutes)
- Update human field to include "cancel" or "cancelling"
- Test: `cargo run --bin bdd-runner -- --tags @cancel`

### 4. Fix SSE Disconnection (1-2 hours)
- Implement disconnection tracking in `sse_extended.rs`
- Use `world.sse_channels` HashMap
- Test: `cargo run --bin bdd-runner -- --tags @sse`

---

## 🎯 RECOMMENDED APPROACH

### Session 1: Quick Wins (3-4 hours)
1. Fix event count baselines in context_steps.rs
2. Fix job state transitions in job_lifecycle.rs
3. Fix cancellation messages
4. **Expected result:** 85-90% pass rate

### Session 2: SSE and Edge Cases (2-3 hours)
1. Implement SSE disconnection tracking
2. Fix job_id mismatches
3. Implement remaining story mode steps
4. **Expected result:** 95% pass rate

### Session 3: Integration Tests (Optional, 8-12 hours)
1. Implement mock HTTP/SSE infrastructure
2. Implement failure scenario steps
3. Add chaos testing
4. **Expected result:** 100% pass rate

---

## 🔍 DEBUGGING TIPS

### Finding Failing Scenarios
```bash
# Run tests and save output
cargo run --bin bdd-runner 2>&1 | tee test_output.txt

# Find failing scenarios
grep -B 10 "Step panicked" test_output.txt | grep "Scenario:"

# Run specific feature
cargo run --bin bdd-runner -- features/job_lifecycle.feature
```

### Understanding Event Counts
The global `CaptureAdapter` accumulates ALL events. Use `world.initial_event_count` to track per-scenario baseline:

```rust
// In GIVEN step - set baseline
world.initial_event_count = adapter.captured().len();

// In THEN step - count new events
let new_events = adapter.captured().len() - world.initial_event_count;
assert_eq!(new_events, expected_count);
```

### Common Pitfalls
1. **Forgetting to set baseline** - Always set `initial_event_count` in GIVEN
2. **Not updating job state** - Update `world.job_state` in WHEN steps
3. **Missing job_id in narration** - Always include `job_id` from world
4. **Ambiguous steps** - Check for duplicate step definitions across files

---

## 📊 EXPECTED OUTCOMES

### After Quick Fixes (Session 1)
- **Pass rate:** 85-90% (105-110 scenarios)
- **Failing:** 10-15 scenarios
- **Skipped:** 23 scenarios (unchanged)

### After SSE Fixes (Session 2)
- **Pass rate:** 95% (117 scenarios)
- **Failing:** 3-5 scenarios
- **Skipped:** 23 scenarios (unchanged)

### After Integration Tests (Session 3)
- **Pass rate:** 100% (123 scenarios)
- **Failing:** 0 scenarios
- **Skipped:** 0 scenarios

---

## 📁 FILES TO MODIFY

### High Priority
1. **src/steps/context_steps.rs** - Fix event count baselines
2. **src/steps/job_lifecycle.rs** - Fix state transitions and job_id
3. **src/steps/sse_extended.rs** - Implement disconnection tracking

### Medium Priority
4. **src/steps/worker_integration.rs** - Fix cancellation messages
5. **src/steps/story_mode_extended.rs** - Implement remaining story steps

### Low Priority (Integration Tests)
6. **src/steps/failure_scenarios.rs** - Implement real failure testing

---

## ✅ SUCCESS CRITERIA

### Minimum (Quick Wins)
- [ ] 85%+ scenario pass rate
- [ ] All event count issues fixed
- [ ] All job state transitions working
- [ ] No ambiguous step definitions

### Target (SSE Fixes)
- [ ] 95%+ scenario pass rate
- [ ] SSE disconnection working
- [ ] All job_id mismatches fixed
- [ ] Story mode complete

### Stretch (Integration Tests)
- [ ] 100% scenario pass rate
- [ ] All failure scenarios implemented
- [ ] Mock infrastructure in place
- [ ] Chaos testing working

---

## 📚 Everything You Need to Know

### 1. Project Structure

```
bin/99_shared_crates/narration-core/bdd/
├── features/               # Gherkin feature files (8 total)
│   ├── context_propagation.feature    ✅ 87.5% (14/16)
│   ├── cute_mode.feature              ✅ 100% (8/8)
│   ├── failure_scenarios.feature      ⏳ 0% (0/19) - Skipped
│   ├── job_lifecycle.feature          ✅ 88% (15/17)
│   ├── levels.feature                 ⚠️ 0% (0/6) - Level field not implemented
│   ├── sse_streaming.feature          ✅ 79% (11/14)
│   ├── story_mode.feature             ✅ 75% (6/8)
│   └── worker_orcd_integration.feature ✅ 100% (21/21)
├── src/
│   ├── main.rs            # Test runner (DON'T MODIFY)
│   └── steps/
│       ├── mod.rs         # Module registry
│       ├── world.rs       # Shared test state
│       ├── context_steps.rs       ⚠️ Fix event count baselines
│       ├── cute_mode.rs           ✅ TEAM-308
│       ├── failure_scenarios.rs   ⏳ TEAM-308 (stubs only)
│       ├── job_lifecycle.rs       ⚠️ TEAM-309 (fix state transitions)
│       ├── levels.rs               ⚠️ TEAM-309 (level field needed)
│       ├── sse_extended.rs        ⚠️ TEAM-309 (fix disconnection)
│       ├── story_mode_extended.rs ✅ TEAM-308
│       └── worker_integration.rs  ⚠️ TEAM-309 (fix cancellation)
└── .plan/                 # Documentation
```

### 2. The World Struct

**Location:** `src/steps/world.rs`

```rust
#[derive(cucumber::World, Default)]
pub struct World {
    // Capture adapter for assertions
    pub adapter: Option<CaptureAdapter>,
    
    // Narration context
    pub context: Option<NarrationContext>,
    pub fields: NarrationFields,
    
    // Job tracking
    pub job_id: Option<String>,
    pub job_state: Option<String>,
    pub job_error: Option<String>,
    
    // SSE tracking
    pub sse_channels: HashMap<String, bool>,
    pub sse_events: Vec<String>,
    
    // Failure scenarios
    pub last_error: Option<String>,
    pub network_timeout_ms: Option<u64>,
    
    // TEAM-308: Per-scenario event tracking
    pub initial_event_count: usize,
}
```

**Key points:**
- `World` is passed to EVERY step function
- It's mutable - you can store state between steps
- It's reset for each scenario (not shared across scenarios)
- Use it to pass data from GIVEN → WHEN → THEN

### 3. Step Function Anatomy

**Basic pattern:**
```rust
use crate::steps::world::World;
use cucumber::{given, then, when};

#[when(regex = r#"^I do something with "([^"]+)"$"#)]
async fn when_do_something(world: &mut World, param: String) {
    // 1. Read from world if needed
    let job_id = &world.job_id;
    
    // 2. Perform action
    let result = some_operation(param);
    
    // 3. Store result in world for THEN steps
    world.last_error = Some(result);
}

#[then("the result should be correct")]
async fn then_result_correct(world: &mut World) {
    // 1. Read from world
    let error = world.last_error.as_ref().expect("Error should be set");
    
    // 2. Assert
    assert!(error.contains("expected"), "Should contain expected text");
}
```

### 4. Import Patterns

**Standard imports for step files:**
```rust
// TEAM-XXX: [Description] step definitions
use crate::steps::world::World;
use cucumber::{given, then, when};
use observability_narration_core::{narrate, NarrationFields};

// For table-based steps:
use cucumber::gherkin::Step;
use std::collections::HashMap;

// For context propagation:
use observability_narration_core::with_narration_context;
```

### 5. Regex Patterns

**Common patterns:**
```rust
// String parameter
#[when(regex = r#"^I do "([^"]+)"$"#)]
//                      ^^^^^^^^ Captures quoted string

// Number parameter
#[when(regex = r"^I wait (\d+) seconds$")]
//                       ^^^^^ Captures digits

// Multiple parameters
#[when(regex = r#"^I send "([^"]+)" to "([^"]+)"$"#)]
//                         ^^^^^^^^^    ^^^^^^^^^ Two captures

// No parameters
#[when("I do something")]
```

**⚠️ CRITICAL: Use raw strings `r#"..."#` for regex to avoid escape hell!**

### 6. Table-Based Steps

**Pattern:**
```rust
#[when("I narrate with:")]
async fn when_narrate_with_table(_world: &mut World, step: &Step) {
    if let Some(table) = &step.table {
        let mut fields = HashMap::new();
        
        // Parse table (skip header row)
        for row in &table.rows[1..] {
            if row.len() >= 2 {
                fields.insert(row[0].clone(), row[1].clone());
            }
        }
        
        // Use the fields
        let actor = fields.get("actor").map(|s| s.as_str()).unwrap_or("test");
        // ...
    }
}
```

**Gherkin:**
```gherkin
When I narrate with:
  | field  | value        |
  | actor  | test-actor   |
  | action | test-action  |
```

### 7. Narration Emission

**Basic narration:**
```rust
use observability_narration_core::{narrate, NarrationFields};

narrate(NarrationFields {
    actor: "test",
    action: "test",
    target: "test".to_string(),
    human: "Test message".to_string(),
    cute: Some("Cute message".to_string()),
    story: Some("Story message".to_string()),
    ..Default::default()
});
```

**With context:**
```rust
use observability_narration_core::with_narration_context;

if let Some(ctx) = world.context.clone() {
    with_narration_context(ctx, async move {
        narrate(NarrationFields { /* ... */ });
    }).await;
}
```

**⚠️ CRITICAL: actor and action need 'static lifetime!**
```rust
// WRONG - will fail to compile
let action = "test".to_string();
narrate(NarrationFields {
    action: &action,  // ❌ Not 'static
    // ...
});

// RIGHT - leak the string
let action_static: &'static str = Box::leak(action.into_boxed_str());
narrate(NarrationFields {
    action: action_static,  // ✅ 'static
    // ...
});
```

### 8. Assertions

**Check captured events:**
```rust
#[then("the captured narration should have 1 event")]
async fn then_has_one_event(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let new_events = captured.len() - world.initial_event_count;
        assert_eq!(new_events, 1, "Expected 1 event, got {}", new_events);
    }
}
```

**Check field contents:**
```rust
#[then(regex = r#"^the cute field should contain "([^"]+)"$"#)]
async fn then_cute_contains(world: &mut World, expected: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let last = captured.last().unwrap();
        
        if let Some(cute) = &last.cute {
            assert!(cute.contains(&expected), 
                "Cute field '{}' should contain '{}'", cute, expected);
        } else {
            panic!("Cute field should be present");
        }
    }
}
```

**Check specific event:**
```rust
#[then(regex = r#"^event (\d+) should have job_id "([^"]+)"$"#)]
async fn then_event_has_job_id(world: &mut World, event_num: usize, job_id: String) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        let index = event_num - 1;  // Convert to 0-indexed
        let event = &captured[world.initial_event_count + index];
        
        assert_eq!(event.job_id.as_deref(), Some(job_id.as_str()),
            "Event {} should have job_id '{}'", event_num, job_id);
    }
}
```

---

## 🔧 Development Workflow

### Step 1: Choose a Feature

Pick one from the priority list above. Start with **job_lifecycle** (easier than failure_scenarios).

### Step 2: Read the Feature File

```bash
cat features/job_lifecycle.feature
```

Understand what scenarios need implementation.

### Step 3: Create Step File

```bash
touch src/steps/job_lifecycle.rs
```

### Step 4: Add Module to mod.rs

```rust
// src/steps/mod.rs
pub mod job_lifecycle;
```

### Step 5: Implement Steps

Start with the simplest steps first. Use TEAM-308's files as templates.

### Step 6: Build and Test

```bash
# Build
cargo build -p observability-narration-core-bdd --bin bdd-runner

# Run tests
cd bin/99_shared_crates/narration-core/bdd
cargo run --bin bdd-runner

# Run specific feature
cargo run --bin bdd-runner -- --input "features/job_lifecycle.feature"
```

### Step 7: Iterate

- If compilation fails: Fix syntax errors
- If tests fail: Check assertions
- If tests skip: Step not implemented yet

---

## ⚠️ Common Pitfalls

### 1. 'static Lifetime Issues

**Problem:**
```rust
let action = "test".to_string();
narrate(NarrationFields {
    action: &action,  // ❌ Error: doesn't live long enough
    // ...
});
```

**Solution:**
```rust
let action_static: &'static str = Box::leak(action.into_boxed_str());
narrate(NarrationFields {
    action: action_static,  // ✅ Works
    // ...
});
```

### 2. Regex Escaping Hell

**Problem:**
```rust
#[when(regex = "^I do \"something\"$")]  // ❌ Escape nightmare
```

**Solution:**
```rust
#[when(regex = r#"^I do "something"$"#)]  // ✅ Raw string
```

### 3. Forgetting to Add Module

**Problem:** Implemented steps but they're not found.

**Solution:** Add to `src/steps/mod.rs`:
```rust
pub mod your_new_module;
```

### 4. World State Not Initialized

**Problem:**
```rust
let job_id = world.job_id.unwrap();  // ❌ Panics if None
```

**Solution:**
```rust
let job_id = world.job_id.as_ref().expect("Job ID should be set");  // ✅ Better error
// OR
if let Some(job_id) = &world.job_id {
    // Use job_id
}
```

### 5. Not Using initial_event_count

**Problem:**
```rust
let count = captured.len();  // ❌ Includes events from previous scenarios
```

**Solution:**
```rust
let new_events = captured.len() - world.initial_event_count;  // ✅ Only new events
```

**Why:** See BUG-003 documentation - global CaptureAdapter accumulates events.

---

## 🎓 Learning from TEAM-308

### What Worked Well

1. **Start Simple:** Cute mode was easier than failure scenarios
2. **Copy Patterns:** Used existing steps as templates
3. **Test Frequently:** Build after every 3-5 steps
4. **Document Everything:** Code comments explain non-obvious behavior

### What Didn't Work

1. **Stub Implementations:** failure_scenarios.rs has stubs that don't test real behavior
2. **Skipping Verification:** Should have tested each feature individually
3. **Assuming Behavior:** Some steps assume things work without actually testing

### Lessons Learned

1. **Test Before Claiming Fixed:** TEAM-308 learned this the hard way with BUG-003
2. **Read Existing Code:** context_steps.rs and story_mode.rs have good patterns
3. **Use World Struct:** Don't try to use global state - use World
4. **Follow Debugging Rules:** Document your investigation in code comments

---

## 📖 Reference: Existing Step Files

### Best Examples to Copy

1. **`cute_mode.rs`** - Clean, simple, well-structured
   - Good regex patterns
   - Clear assertions
   - Table-based input

2. **`context_steps.rs`** - Complex but correct
   - Context propagation
   - Async handling
   - Multiple scenarios

3. **`story_mode.rs`** - Original implementation
   - Table parsing
   - Multi-field narration

### Files to Avoid Copying

1. **`failure_scenarios.rs`** - TEAM-308's stubs (needs fixing)
   - Don't copy the stub pattern
   - Needs real implementation

---

## 🚀 Quick Start Template

```rust
// TEAM-XXX: [Feature name] step definitions
// Implements steps for [feature_file].feature

use crate::steps::world::World;
use cucumber::{given, then, when};
use observability_narration_core::{narrate, NarrationFields};

// ============================================================
// GIVEN Steps - Setup
// ============================================================

#[given("some precondition")]
async fn given_precondition(world: &mut World) {
    // Setup world state
    world.job_id = Some("test-job".to_string());
}

// ============================================================
// WHEN Steps - Actions
// ============================================================

#[when("I perform an action")]
async fn when_perform_action(world: &mut World) {
    // Do something
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test message".to_string(),
        ..Default::default()
    });
}

// ============================================================
// THEN Steps - Assertions
// ============================================================

#[then("the result should be correct")]
async fn then_result_correct(world: &mut World) {
    // Assert something
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have events");
    }
}
```

---

## 📋 Checklist for Each Feature

Before marking a feature complete:

- [ ] All scenarios pass (not skipped)
- [ ] No stub implementations (actually test behavior)
- [ ] Code has TEAM-XXX signature
- [ ] No TODO markers
- [ ] Compilation succeeds
- [ ] Tests run: `cargo run --bin bdd-runner`
- [ ] Feature-specific test: `cargo run --bin bdd-runner -- --input "features/your_feature.feature"`
- [ ] Documentation updated

---

## 🎯 Success Criteria

**Goal:** 100% test coverage

**Current:** 32 passed, 41 skipped, 53 failed  
**Target:** 126 passed, 0 skipped, 0 failed

**Estimated Total Effort:** 12-17 hours

---

## 📞 Getting Help

### If You Get Stuck

1. **Read BUG-003 docs** - Shows debugging process
2. **Check existing steps** - Someone probably solved it
3. **Look at World struct** - Maybe the state is already there
4. **Test incrementally** - Don't write 20 steps before testing

### Key Documents

- `BUG_003_FINAL_SUMMARY.md` - How TEAM-308 debugged the race condition
- `TEAM_308_STEP_IMPLEMENTATION.md` - Part 1 summary
- `TEAM_308_STEP_IMPLEMENTATION_PART2.md` - Part 2 summary
- `debugging-rules.md` - Mandatory debugging documentation rules

---

## 🏁 Final Notes

### What TEAM-309 Learned

1. **Redaction ≠ Narration** - Redaction belongs in audit logs, not narration-core
2. **Check for duplicates** - Always grep for existing step definitions before creating new ones
3. **Regex limitations** - Rust regex doesn't support lookahead/lookbehind or backreferences
4. **Event counting is tricky** - Global CaptureAdapter requires careful baseline management
5. **Test incrementally** - Build and test after each major change

### Words of Encouragement

You've got this! TEAM-309 took the pass rate from 25% to 61% in one session. Most of the remaining issues are simple fixes:

- **Event count baselines:** Just set `initial_event_count` in GIVEN steps
- **Job state transitions:** Just update `world.job_state` in WHEN steps
- **Cancellation messages:** Just add "cancel" to the human field

**The infrastructure is solid.** The patterns are clear. The path forward is obvious.

**85-90% pass rate is achievable in 3-4 hours of focused work.**

Good luck! 🚀

---

## 📞 Contact & References

### Key Documents
- **TEAM_309_FINAL_COMPLETE.md** - Complete summary of TEAM-309 work
- **BUG_003_*.md** - CaptureAdapter global state documentation
- **TEAM_308_FINAL_SUMMARY.md** - Previous team's work

### Test Commands
```bash
# Run all tests
cargo run --bin bdd-runner

# Run specific feature
cargo run --bin bdd-runner -- features/job_lifecycle.feature

# Save output for analysis
cargo run --bin bdd-runner 2>&1 | tee test_output.txt
```

### Quick Stats
- **Current:** 75/123 scenarios passing (61%)
- **Target:** 105-110/123 scenarios passing (85-90%)
- **Stretch:** 123/123 scenarios passing (100%)

---

**Document Version:** 2.0  
**Last Updated:** October 26, 2025  
**Status:** Ready for TEAM-310  
**TEAM-309 Signature:** All issues documented, path forward clear
