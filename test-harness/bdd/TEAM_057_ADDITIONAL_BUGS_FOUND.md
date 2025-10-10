# ADDITIONAL WORK ITEMS FOUND (Mid-Development Analysis)

**Created by:** TEAM-057 (The Thinking Team)  
**Date:** 2025-10-10  
**Status:** 📋 WORK IN PROGRESS - Additional TODO items discovered  
**Context:** rbee v0.1.0 is ~68% complete (early development)  
**Purpose:** Document incomplete work beyond the original 5 contradictions

---

## Executive Summary

**IMPORTANT:** This is mid-development analysis, NOT production code review.

During final verification research, discovered **6 additional categories of incomplete work** that need implementation:

1. **TODO/Stub Epidemic** - 5+ step definitions with unimplemented TODO comments
2. **Duplicate Step Definitions** - Conflicting implementations across files
3. **Mock-Only Steps** - 200+ lines of steps that only log, never verify
4. **Missing Assertions** - Steps that should verify but only log
5. **Inconsistent Exit Code Handling** - Some set exit codes, most don't
6. **LLORCH_BDD_FEATURE_PATH Bug** - Can't run individual scenarios by line number

---

## Work Category 1: TODO Comments (Implementation Pending) 📋 IN PROGRESS

### Evidence from Code

**File:** `test-harness/bdd/src/steps/registry.rs`

```rust
// Line 78-79
#[when(expr = "queen-rbee queries {string}")]
pub async fn when_query_url(world: &mut World, url: String) {
    // TODO: Perform HTTP query
    tracing::debug!("Querying: {}", url);
}

// Line 84-85
#[when(expr = "rbee-keeper queries the worker registry")]
pub async fn when_query_worker_registry(world: &mut World) {
    // TODO: Query registry
    tracing::debug!("Querying worker registry");
}

// Line 93-94
#[then(expr = "the response is:")]
pub async fn then_response_is(world: &mut World, step: &cucumber::gherkin::Step) {
    // TODO: Verify response matches expected JSON
    tracing::debug!("Expected response: {}", expected_json);
}

// Line 114-115
#[then(expr = "the registry returns worker {string} with state {string}")]
pub async fn then_registry_returns_worker(world: &mut World, worker_id: String, state: String) {
    // TODO: Verify registry response
    tracing::debug!("Registry should return worker {} with state {}", worker_id, state);
}

// Line 125-126
#[then(expr = "rbee-keeper sends inference request directly to {string}")]
pub async fn then_send_inference_direct(world: &mut World, url: String) {
    // TODO: Send inference request
    tracing::debug!("Sending inference request to: {}", url);
}

// Line 136-137
#[then(expr = "the total latency is under {int} seconds")]
pub async fn then_latency_under(world: &mut World, seconds: u64) {
    // TODO: Verify latency
    tracing::debug!("Latency should be under {} seconds", seconds);
}
```

### Status

**6 TODO comments in registry.rs** - Planned work, not yet implemented.

**Scenarios awaiting implementation:**
- "Worker registry returns empty list" (line 264)
- "Worker registry returns matching idle worker" (line 275)
- "Worker registry returns matching busy worker" (line 280)
- "Warm start - reuse existing idle worker" (line 230)

**Development status:**
- Step definitions created (scaffolding complete)
- Implementation pending (normal for ~68% complete project)
- These are on the TODO list for completion
- Part of planned BDD test completion work

---

## Work Category 2: Duplicate Step Definitions (Resolved) ✅ CLEANED UP

### Evidence from Comments

**File:** `test-harness/bdd/src/steps/happy_path.rs:43-44`
```rust
// TEAM-044: Removed duplicate "I run:" step - real implementation is in cli_commands.rs
// TEAM-042 had created a mock version here that conflicted with the real execution
```

**File:** `test-harness/bdd/src/steps/worker_startup.rs:12`
```rust
// TEAM-045: Removed duplicate step - defined in lifecycle.rs
```

**File:** `test-harness/bdd/src/steps/edge_cases.rs:106`
```rust
// TEAM-042: Removed duplicate step definition - now in beehive_registry.rs
```

### The Pattern

Multiple teams (TEAM-040, TEAM-042, TEAM-044, TEAM-045) created step definitions without checking if they already existed.

**Why this happened:**
1. No central registry of step definitions
2. Teams worked in parallel
3. Similar step text → duplicate implementations
4. Some were mocks, some were real → conflicts

**Current state:**
- Duplicates have been removed (good!)
- But comments remain showing the problem existed
- Risk of reintroduction if pattern continues

### Recommendation

Create a step definition index or use cucumber's built-in duplicate detection.

---

## Work Category 3: Mock-Only Steps (Implementation Pending) 📋 IN PROGRESS

### Evidence: Massive Mock Infrastructure

**File sizes (lines of code):**
```
lifecycle.rs:     331 lines (64 tracing::debug calls)
happy_path.rs:    367 lines (5 tracing::debug calls)  
edge_cases.rs:    199 lines (34 tracing::debug calls)
registry.rs:      153 lines (18 tracing::debug calls)
inference_execution.rs: 73 lines (13 tracing::debug calls)
worker_startup.rs: 70 lines (12 tracing::debug calls)
```

**Total:** ~1,200 lines of step definitions, **~160 are just `tracing::debug` calls**

### Example: lifecycle.rs (ALL STUBS)

```rust
#[given(expr = "rbee-hive is started as HTTP daemon on port {int}")]
pub async fn given_hive_started_daemon(world: &mut World, port: u16) {
    tracing::debug!("rbee-hive started as daemon on port {}", port);
}

#[given(expr = "rbee-hive spawned a worker")]
pub async fn given_hive_spawned_worker(world: &mut World) {
    tracing::debug!("rbee-hive spawned a worker");
}

#[given(expr = "rbee-hive is running as persistent daemon")]
pub async fn given_hive_running_persistent(world: &mut World) {
    tracing::debug!("rbee-hive running as persistent daemon");
}

// ... 64 more just like this!
```

### Development Status

**lifecycle.rs has 331 lines of scaffolding** - Step definitions created, implementation pending.

**Scenarios using these steps:**
- "Rbee-hive remains running as persistent HTTP daemon" (line 786)
- "Rbee-hive monitors worker health" (line 797)
- "Rbee-hive enforces idle timeout" (line 808)
- "Cascading shutdown when rbee-hive receives SIGTERM" (line 818)
- "rbee-keeper exits after inference" (line 830)
- "Ephemeral mode - rbee-keeper spawns rbee-hive" (line 840)
- "Persistent mode - rbee-hive pre-started" (line 854)

**All 7 lifecycle scenarios have scaffolding** - Awaiting implementation (normal for mid-development).

---

## Work Category 4: Assertions Pending Implementation 📋 TODO

### The Pattern

Steps that SHOULD verify behavior but only log:

**Example 1: worker_startup.rs**
```rust
#[then(expr = "the worker HTTP server binds to port {int}")]
pub async fn then_worker_binds_to_port(world: &mut World, port: u16) {
    tracing::debug!("Worker should bind to port {}", port);
    // NO VERIFICATION! Should check if port is actually bound
}
```

**Example 2: inference_execution.rs**
```rust
#[then(expr = "rbee-keeper streams tokens to stdout in real-time")]
pub async fn then_stream_tokens_stdout(world: &mut World) {
    tracing::debug!("Should stream tokens to stdout");
    // NO VERIFICATION! Should check if tokens were actually streamed
}
```

**Example 3: happy_path.rs**
```rust
#[then(expr = "the health check returns version {string} and status {string}")]
pub async fn then_health_check_response(world: &mut World, version: String, status: String) {
    tracing::info!("✅ Health check returned version={}, status={}", version, status);
    // NO VERIFICATION! Should assert world.last_http_response matches
}
```

### Development Status

**~50+ "then" steps with scaffolding** - Step definitions exist, assertions pending.

These are verification steps (Then clauses) that will assert once implementation is complete. Currently in TODO state.

---

## Work Category 5: Exit Code Handling (Partial Implementation) 📋 IN PROGRESS

### The Pattern

**Some steps set exit codes:**
```rust
// edge_cases.rs:80
world.last_exit_code = Some(1);

// edge_cases.rs:87
world.last_exit_code = Some(130);

// worker_preflight.rs:76
world.last_exit_code = Some(1);
```

**Most steps don't:**
```rust
// All of lifecycle.rs (331 lines) - never sets exit code
// All of worker_startup.rs (70 lines) - never sets exit code
// All of inference_execution.rs (73 lines) - never sets exit code
// Most of registry.rs (153 lines) - never sets exit code
```

### The Problem

**Inconsistent behavior:** Some scenarios can verify exit codes, most can't.

**Why it matters:**
- "Then the exit code is 0" assertions fail with "got None"
- Only 3-4 step definitions actually set exit codes
- Rest rely on real command execution (which works)
- But mock/stub steps never set exit codes

### Example Failure

```
Scenario: EC1 - Connection timeout with retry and backoff
  ...
  And the exit code is 1

Step failed: assertion failed: Expected exit code 1, got None
```

**Why:** The "when_attempt_connection" step doesn't set `world.last_exit_code`.

---

## Work Category 6: LLORCH_BDD_FEATURE_PATH Enhancement Needed 📋 TODO

### Current Limitation

**Environment variable doesn't yet support line numbers:**

```bash
# This FAILS:
LLORCH_BDD_FEATURE_PATH="tests/features/test-001.feature:176" cargo run --bin bdd-runner

# Error: "Could not read path: /home/vince/Projects/llama-orch/tests/features/test-001.feature:176"
```

### Evidence from Testing

```
Failed to parse: Failed to parse feature: Could not read path: /home/vince/Projects/llama-orch/tests/features/test-001.feature:176
```

### Impact

**Can't run individual scenarios for debugging!**

**Workarounds:**
1. Run entire feature file (slow, 62 scenarios)
2. Comment out scenarios you don't want (manual, error-prone)
3. Use tags (requires adding @tags to scenarios)

### Implementation Status

**File:** `test-harness/bdd/src/main.rs:26-38`

```rust
let features = match std::env::var("LLORCH_BDD_FEATURE_PATH").ok() {
    Some(p) => {
        let pb = PathBuf::from(&p);
        if pb.is_absolute() {
            pb
        } else {
            let workspace_root = root.parent().unwrap().parent().unwrap();
            workspace_root.join(pb)
        }
    }
    None => root.join("tests/features"),
};
```

**Current:** Treats the entire string as a file path, doesn't parse `file:line` syntax.

**TODO:** Parse line number and pass to cucumber runner (enhancement for debugging).

---

## Work Category 7: Port Conflict Handling (Enhancement) 📋 TODO

### Current Implementation

**File:** `test-harness/bdd/src/steps/global_queen.rs:72`

```rust
let binary_path = workspace_dir.join("target/debug/queen-rbee");

tracing::info!("🐝 Starting GLOBAL queen-rbee process at {:?}...", binary_path);

let child = {
    let mut child = tokio::process::Command::new(&binary_path)
            .args(["--port", "8080", "--database"])
            .arg(&db_path)
            .env("MOCK_SSH", "true")
            // ...
```

**Current:** Always uses port 8080, no fallback (simple implementation for early development).

**What happens if port 8080 is already in use:**
1. Global queen-rbee fails to start
2. Tests detect this: "queen-rbee exited during startup with status: X (likely port 8080 already in use)"
3. Tests panic and abort

**Evidence from code (line 92-93):**
```rust
Ok(Some(status)) => {
    panic!("Global queen-rbee exited during startup with status: {} (likely port 8080 already in use)", status);
}
```

### Impact

**Tests are NOT idempotent!**

If you:
1. Run tests
2. Tests fail mid-way
3. queen-rbee still running on port 8080
4. Run tests again → PANIC (port in use)

**Workaround:** Manually kill queen-rbee between test runs.

**Enhancement TODO:** 
- Use random port
- OR: Check if port is in use and kill existing process
- OR: Reuse existing queen-rbee if already running

**Note:** This is a test infrastructure enhancement, not a critical bug. Current implementation works for single test runs.

---

## Summary Table (Development Status)

| Work Category | Status | Files Affected | Lines Affected | Development Phase |
|--------------|----------|----------------|----------------|-------------------|
| 1. TODO Comments | 📋 TODO | registry.rs | 6 TODOs | Implementation pending |
| 2. Duplicates | ✅ DONE | 3 files | N/A (resolved) | Cleanup complete |
| 3. Mock-Only Steps | 📋 TODO | 6 files | ~200 lines | Scaffolding done, impl pending |
| 4. Assertions | 📋 TODO | 5 files | ~50 steps | Step defs exist, assertions TODO |
| 5. Exit Code Handling | 📋 PARTIAL | All files | ~90% of steps | Partial implementation |
| 6. LLORCH_BDD_FEATURE_PATH | 📋 ENHANCEMENT | main.rs | 1 feature | Line number parsing TODO |
| 7. Port Conflict | 📋 ENHANCEMENT | global_queen.rs | 1 feature | Fallback handling TODO |

---

## Development Progress Analysis

### Original Analysis

**TEAM-057 initially identified:**
- 20 scenarios pending completion
- 5 architectural decisions needed
- Primary work: Registration model implementation

### Updated Analysis After Deep Research

**Additional work items found:**
- 6+ TODO comments in core functionality (not edge cases!)
- 200+ lines of mock-only steps (lifecycle, worker startup, etc.)
- 50+ "then" steps that should assert but only log
- LLORCH_BDD_FEATURE_PATH doesn't work for individual scenarios
- Port 8080 conflict makes tests non-idempotent

**Impact on completion estimate:**
- **Original:** "42 complete, 20 pending"
- **Reality:** "42 complete, 20 pending + 7-12 with scaffolding only"
- **Fully implemented:** ~30-35 scenarios with complete verification
- **Scaffolding only:** 7-12 scenarios need implementation (not just passing)

---

## Development Recommendations

### Immediate Actions (P0)

1. **Implement LLORCH_BDD_FEATURE_PATH line parsing** - Debugging enhancement
2. **Complete TODO steps in registry.rs** - 6 core functionality steps
3. **Add port conflict handling** - Test reliability enhancement

### Short-term Actions (P1)

4. **Implement lifecycle.rs steps** - 331 lines of scaffolding, 7 scenarios
5. **Add assertions to "then" steps** - 50+ steps need verification logic
6. **Document completion status** - Track which scenarios are fully implemented

### Long-term Actions (P2)

7. **Create step definition registry** - Prevent duplicate work
8. **Standardize exit code handling** - Consistent pattern across all steps
9. **Add @wip tags to incomplete scenarios** - Clear status tracking

---

## Impact on Original Plan

### Phase 1: Explicit Node Registration

**Original estimate:** 42 → 45-47 passing

**Revised estimate:** 42 → 43-45 passing (some scenarios are false positives)

### Phase 2: Implement Edge Cases

**Original estimate:** 45-47 → 54-58 passing

**Revised estimate:** 43-45 → 48-52 passing (need to implement TODOs too, not just edge cases)

### Phase 3: Fix HTTP Issues

**Original estimate:** 54-58 → 58-62 passing

**Revised estimate:** 48-52 → 54-58 passing

### NEW Phase 4: Implement Mock-Only Steps

**Estimate:** 54-58 → 60-62 passing

**What:** Implement lifecycle.rs, worker_startup.rs, inference_execution.rs steps

**Impact:** 7+ scenarios currently pass but test nothing

---

## Files Requiring Immediate Attention

1. **`test-harness/bdd/src/main.rs`** - Fix LLORCH_BDD_FEATURE_PATH parsing
2. **`test-harness/bdd/src/steps/registry.rs`** - Implement 6 TODO comments
3. **`test-harness/bdd/src/steps/lifecycle.rs`** - Implement 64 stub steps
4. **`test-harness/bdd/src/steps/global_queen.rs`** - Add port conflict handling
5. **`test-harness/bdd/src/steps/worker_startup.rs`** - Add assertions
6. **`test-harness/bdd/src/steps/inference_execution.rs`** - Add assertions

---

**TEAM-057 signing off on additional work items analysis.**

**Status:** 6 additional work categories identified  
**Development Phase:** Mid-development (~68% complete)  
**Impact:** More TODO items than originally estimated  
**Confidence:** VERY HIGH - All work items verified through code inspection  
**Recommendation:** Update implementation plan to include these TODO items

**The path to 100% completion is clear - we now have a complete TODO list for the remaining 32%.**
