# BDD Step Implementation Guide for Next Team

**Date:** October 26, 2025  
**From:** TEAM-308  
**To:** Next Implementation Team  
**Status:** 📚 COMPREHENSIVE GUIDE

---

## 🎯 Mission

Implement the remaining **41 unimplemented BDD steps** to achieve 100% test coverage.

---

## 📊 Current State

### Test Results (After TEAM-308)
```
126 scenarios (32 passed, 41 skipped, 53 failed)
488 steps (394 passed, 41 skipped, 53 failed)
```

### What TEAM-308 Completed
- ✅ **cute_mode.rs** (16 steps, 250 LOC) - Cute narration feature
- ✅ **story_mode_extended.rs** (14 steps, 270 LOC) - Story narration feature
- ✅ **failure_scenarios.rs** (29 steps, 330 LOC) - Basic failure handling (NEEDS WORK)

**Total:** 59 steps implemented, 850 LOC

### What Remains
- ⏳ **41 skipped steps** - Need implementation
- ⚠️ **53 failing steps** - Need fixing (mostly in failure_scenarios)

---

## 🗺️ Roadmap: Features to Implement

### Priority 1: Fix Failure Scenarios (HIGH)
**File:** `src/steps/failure_scenarios.rs` (already exists, needs fixing)  
**Status:** 29 steps implemented but FAILING  
**Problem:** Steps are stubs - they don't actually test real behavior

**What needs fixing:**
1. Network failure tests need actual HTTP mocking
2. SSE stream tests need real channel management
3. Process crash tests need actual process monitoring
4. Timeout tests need real async timeout handling

**Estimated effort:** 4-6 hours

### Priority 2: Job Lifecycle (~20 steps)
**File:** `src/steps/job_lifecycle.rs` (NEW)  
**Feature:** `features/job_lifecycle.feature`  
**Status:** Not started

**What to implement:**
- Job state transitions (pending → running → complete)
- Job cancellation
- Job timeout handling
- Job error states

**Estimated effort:** 3-4 hours

### Priority 3: SSE Streaming (~15 steps)
**File:** `src/steps/sse_extended.rs` (NEW)  
**Feature:** `features/sse_streaming.feature`  
**Status:** Partially implemented in `sse_steps.rs`

**What to implement:**
- SSE channel lifecycle
- Event streaming verification
- Channel cleanup
- Multiple subscriber handling

**Estimated effort:** 2-3 hours

### Priority 4: Worker Integration (~16 steps)
**File:** `src/steps/worker_integration.rs` (NEW)  
**Feature:** `features/worker_orcd_integration.feature`  
**Status:** Not started

**What to implement:**
- Worker lifecycle events
- Model management
- Performance metrics
- Error handling

**Estimated effort:** 3-4 hours

---

## 📚 Everything You Need to Know

### 1. Project Structure

```
bin/99_shared_crates/narration-core/bdd/
├── features/               # Gherkin feature files
│   ├── cute_mode.feature          ✅ COMPLETE
│   ├── story_mode.feature         ✅ COMPLETE
│   ├── failure_scenarios.feature  ⚠️ FAILING
│   ├── job_lifecycle.feature      ⏳ TODO
│   ├── sse_streaming.feature      ⏳ TODO
│   └── worker_orcd_integration.feature ⏳ TODO
├── src/
│   ├── main.rs            # Test runner (DON'T MODIFY)
│   └── steps/
│       ├── mod.rs         # Module registry (ADD NEW MODULES HERE)
│       ├── world.rs       # Shared test state
│       ├── cute_mode.rs           ✅ TEAM-308
│       ├── story_mode_extended.rs ✅ TEAM-308
│       ├── failure_scenarios.rs   ⚠️ TEAM-308 (needs fixing)
│       └── [YOUR NEW FILES HERE]
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

### What TEAM-308 Wishes They Knew

1. **Test each feature individually** - Don't run all tests until feature is done
2. **Stub implementations are worse than no implementation** - They hide problems
3. **The World struct is your friend** - Use it liberally
4. **Regex raw strings save hours** - Always use `r#"..."#`
5. **'static lifetime is annoying but necessary** - Use Box::leak()

### Words of Encouragement

You've got this! TEAM-308 implemented 59 steps in a few hours. The patterns are established, the infrastructure is there, you just need to fill in the gaps.

**The hardest part is already done** (BUG-003 fix, infrastructure setup).

**The easiest part is left** (following established patterns).

Good luck! 🚀

---

**Document Version:** 1.0  
**Last Updated:** October 26, 2025  
**Status:** Ready for Next Team  
**TEAM-308 Signature:** Comprehensive guide complete
