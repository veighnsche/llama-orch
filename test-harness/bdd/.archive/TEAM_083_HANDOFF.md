# TEAM-083 HANDOFF - BDD Remaining Wiring & Integration Tests

**From:** TEAM-082  
**Date:** 2025-10-11  
**Status:** ✅ All stub assertions fixed, ready for wiring

---

## What TEAM-082 Accomplished

### ✅ Priority 4: Fixed All Stub Assertions (COMPLETE)
- **41 functions fixed** with meaningful assertions
- **0 stub assertions remaining**
- **82 TEAM-082 signatures** added
- **Compilation passes** (0 errors)

### ✅ Bonus: World Struct Improvements
- Added `reset_for_scenario()` helper for state cleanup
- Prevents state leakage between test scenarios
- Improves test reliability and isolation

---

## Current State

### Wiring Progress
- **Total functions:** ~139 step definitions
- **Wired to real APIs:** 117 (84.2%)
- **Remaining stubs:** 22 (15.8%)
- **Stub assertions:** 0 ✅

### What TEAM-082 Did NOT Complete (Deferred Work)

**From original TEAM-082 handoff, these priorities were NOT started:**

1. ❌ **Priority 1: Wire Remaining Stub Functions** (6 hours) - DEFERRED
   - Inference execution steps (3 hours)
   - Worker registry HTTP endpoints (2 hours)
   - Model provisioning steps (1 hour)

2. ❌ **Priority 2: Add Integration Tests** (4 hours) - DEFERRED
   - Create 900-integration-e2e.feature
   - Test multi-component interactions

3. ❌ **Priority 3: Improve Test Reliability** (2 hours) - DEFERRED
   - Add timeouts to async operations
   - Add cleanup between scenarios
   - Add retry logic for flaky operations

**Reason for deferral:** TEAM-082 focused exclusively on fixing stub assertions (Priority 4 from TEAM-081 handoff) and completed that work thoroughly.

---

## Priority 1: Wire Remaining Stub Functions (6 hours) 🔴 CRITICAL

**Goal:** Increase wiring from 84.2% to 95%+ (132/139 functions)

### 1.1 Inference Execution Steps (3 hours)

**File:** `test-harness/bdd/src/steps/inference_execution.rs`

**Status:** Partially wired by TEAM-076, but some functions still need work

**Functions that may need wiring:**

1. **Slot allocation race conditions**
   - Check if `when_concurrent_slot_requests_at_worker()` exists
   - Wire to `llm_worker_rbee::SlotManager` if available

2. **Request cancellation**
   - Check if `when_cancel_request()` exists
   - Wire to real HTTP POST /cancel endpoint
   - Use `reqwest` to send cancellation

3. **SSE streaming**
   - Check if `when_sse_streaming()` exists
   - Wire to real SSE client
   - Use `eventsource-client` or similar

**Product code available:**
- `llm-worker-rbee` crate has inference APIs
- `rbee-hive` has SSE streaming support
- Check `bin/llm-worker-rbee/src/` for slot management

**Verification:**
```bash
# Check what's already wired
rg "TEAM-076" test-harness/bdd/src/steps/inference_execution.rs

# Look for stub functions
rg "TODO|FIXME|Wire to" test-harness/bdd/src/steps/inference_execution.rs
```

### 1.2 Worker Registry HTTP Endpoints (2 hours)

**File:** `test-harness/bdd/src/steps/queen_rbee_registry.rs`

**Functions to wire:**

1. **HTTP endpoint testing**
   ```rust
   #[when(expr = "rbee-hive sends POST to {string}")]
   pub async fn when_post_to_endpoint(world: &mut World, endpoint: String) {
       // TEAM-083: Wire to real HTTP client
       // Use world's HTTP client factory (create_http_client)
       let client = crate::steps::world::create_http_client();
       // Send POST request to endpoint
   }
   ```

2. **Query filtering**
   ```rust
   #[when(expr = "I query workers with capability {string}")]
   pub async fn when_query_by_capability(world: &mut World, capability: String) {
       // TEAM-083: Wire to real registry query
       // Use registry.list() and filter by capability
   }
   ```

**Product code available:**
- `queen_rbee::WorkerRegistry` has query methods
- HTTP client factory in `world.rs`: `create_http_client()`

### 1.3 Model Provisioning Steps (1 hour)

**File:** `test-harness/bdd/src/steps/model_provisioning.rs`

**Functions to wire:**

1. **Download tracking**
   ```rust
   #[when(expr = "model download starts")]
   pub async fn when_download_starts(world: &mut World) {
       // TEAM-083: Wire to real DownloadTracker
       // Use rbee_hive::DownloadTracker::start_download()
   }
   ```

2. **Catalog registration**
   ```rust
   #[then(expr = "model is registered in catalog")]
   pub async fn then_model_in_catalog(world: &mut World) {
       // TEAM-083: Wire to real ModelCatalog
       // Use model_catalog::ModelCatalog::get()
   }
   ```

**Product code available:**
- `rbee_hive::DownloadTracker` for download tracking
- `model_catalog::ModelCatalog` for catalog operations

---

## Priority 2: Add Integration Tests (4 hours) 🟡 HIGH

**Goal:** Test real component interactions end-to-end

### 2.1 Create Integration Feature File

**Create:** `test-harness/bdd/tests/features/900-integration-e2e.feature`

```gherkin
Feature: End-to-End Integration Tests
  As a system integrator
  I want to test complete workflows
  So that I verify all components work together

  @integration @e2e
  Scenario: Complete inference workflow
    Given queen-rbee is running
    And rbee-hive is running on workstation
    And worker-001 is registered with model "tinyllama-q4"
    When client sends inference request via queen-rbee
    Then queen-rbee routes to worker-001
    And worker-001 processes the request
    And tokens are streamed back to client
    And worker returns to idle state
    And metrics are recorded

  @integration @e2e
  Scenario: Worker failover
    Given queen-rbee is running
    And worker-001 is processing request "req-001"
    And worker-002 is available with same model
    When worker-001 crashes unexpectedly
    Then queen-rbee detects crash within 5 seconds
    And request "req-001" can be retried on worker-002
    And user receives result without data loss

  @integration @e2e
  Scenario: Model download and registration
    Given rbee-hive is running
    And model "tinyllama-q4" is not in catalog
    When rbee-hive downloads model from HuggingFace
    Then download completes successfully
    And model is registered in catalog
    And model is available for worker startup
```

### 2.2 Implement Integration Step Definitions

**Create:** `test-harness/bdd/src/steps/integration.rs`

```rust
// Step definitions for Integration Tests
// Created by: TEAM-083
//
// ⚠️ CRITICAL: These steps test REAL component interactions
// ⚠️ Start actual processes and test end-to-end workflows

use cucumber::{given, then, when};
use crate::steps::world::World;

#[given(expr = "queen-rbee is running")]
pub async fn given_queen_rbee_running(world: &mut World) {
    // TEAM-083: Start real queen-rbee process
    // Use tokio::process::Command
    // Store process handle in world.queen_rbee_process
}

#[given(expr = "rbee-hive is running on workstation")]
pub async fn given_rbee_hive_running(world: &mut World) {
    // TEAM-083: Start real rbee-hive process
    // Use tokio::process::Command
    // Store process handle in world.rbee_hive_processes
}

#[when(expr = "client sends inference request via queen-rbee")]
pub async fn when_client_sends_request(world: &mut World) {
    // TEAM-083: Send real HTTP request
    // Use world's HTTP client factory
    // POST to http://localhost:8080/v1/inference
}

#[then(expr = "queen-rbee routes to worker-001")]
pub async fn then_routes_to_worker(world: &mut World) {
    // TEAM-083: Verify routing via registry
    // Check worker-001 state changed to Busy
}
```

### 2.3 Test Multi-Component Interactions

**Components to test:**
- queen-rbee ↔ rbee-hive (worker registration, heartbeats)
- rbee-hive ↔ llm-worker-rbee (worker startup, inference)
- queen-rbee ↔ client (request routing, SSE streaming)
- worker ↔ model-catalog (model loading, validation)

---

## Priority 3: Improve Test Reliability (2 hours) 🟡 HIGH

**Goal:** Make tests deterministic and prevent hangs

### 3.1 Add Timeouts to All Async Operations

**Problem:** Tests can hang indefinitely on network operations

**Solution:**
```rust
use tokio::time::timeout;
use std::time::Duration;

// Add to all registry operations
let result = timeout(
    Duration::from_secs(5),
    registry.get("worker-001")
).await;

assert!(result.is_ok(), "Registry operation timed out after 5s");
let worker = result.unwrap();
```

**Files to update:**
- `concurrency.rs` - Add timeouts to concurrent operations
- `failure_recovery.rs` - Add timeouts to failover tests
- `queen_rbee_registry.rs` - Add timeouts to registry calls
- `inference_execution.rs` - Add timeouts to inference requests

### 3.2 Add Cleanup Between Scenarios

**Problem:** State leaks between scenarios causing flaky tests

**Solution:** Use the `reset_for_scenario()` helper added by TEAM-082

```rust
// In background.rs or hooks
use cucumber::codegen::before;

#[before]
async fn before_scenario(world: &mut World) {
    // TEAM-083: Reset state before each scenario
    world.reset_for_scenario();
    tracing::info!("Scenario state reset");
}
```

**Files to update:**
- `background.rs` - Add before/after hooks
- Consider adding `@cleanup` tags for scenarios that need extra cleanup

### 3.3 Add Retry Logic for Flaky Operations

**Problem:** Network operations can fail transiently

**Solution:**
```rust
// Add retry helper to world.rs or error_helpers.rs
async fn retry_with_backoff<F, Fut, T>(
    mut f: F,
    max_attempts: u32,
) -> Result<T, String>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, String>>,
{
    for attempt in 1..=max_attempts {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if attempt == max_attempts => return Err(e),
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
            }
        }
    }
    unreachable!()
}
```

**Use cases:**
- HTTP requests to worker endpoints
- Registry queries during failover
- Model download operations

---

## Priority 4: Documentation & Maintenance (1 hour) 🟢 LOW

### 4.1 Update Feature File Comments

Add migration notes to feature files:

```gherkin
# MIGRATION HISTORY:
# - Originally in test-001.feature (monolithic)
# - Migrated by TEAM-079 to 200-concurrency-scenarios.feature
# - Stub assertions fixed by TEAM-082
# - Integration tests added by TEAM-083
```

### 4.2 Create Wiring Status Document

**Create:** `test-harness/bdd/WIRING_STATUS.md`

```markdown
# BDD Wiring Status

Last updated: TEAM-083

## Overall Progress
- Total functions: 139
- Wired: 132 (95%)
- Remaining: 7 (5%)

## By Feature File
| File | Wired | Total | % |
|------|-------|-------|---|
| 200-concurrency | 30 | 30 | 100% |
| 210-failure-recovery | 24 | 24 | 100% |
| 130-inference-execution | 28 | 30 | 93% |
...
```

### 4.3 Update README

Add section about running BDD tests:

```markdown
## Running BDD Tests

# Run all tests
cargo test --package test-harness-bdd

# Run specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/200-concurrency-scenarios.feature \
  cargo test --package test-harness-bdd

# Run with verbose output
cargo test --package test-harness-bdd -- --nocapture

# Run integration tests only
cargo test --package test-harness-bdd -- --tags @integration
```

---

## Checklist for TEAM-083

### Phase 1: Wiring (6 hours)
- [ ] Wire 5+ inference execution functions
- [ ] Wire 3+ worker registry HTTP endpoint functions
- [ ] Wire 2+ model provisioning functions
- [ ] Add TEAM-083 signatures to all changes
- [ ] Verify compilation passes
- [ ] Verify tests run without panics

### Phase 2: Integration Tests (4 hours)
- [ ] Create 900-integration-e2e.feature
- [ ] Implement integration step definitions
- [ ] Add 3+ end-to-end workflow tests
- [ ] Test multi-component interactions
- [ ] Verify integration tests pass

### Phase 3: Reliability (2 hours)
- [ ] Add timeouts to all async operations
- [ ] Add cleanup hooks using `reset_for_scenario()`
- [ ] Add retry logic for flaky operations
- [ ] Test for race conditions
- [ ] Verify tests are deterministic

### Phase 4: Documentation (1 hour)
- [ ] Update feature file comments
- [ ] Create WIRING_STATUS.md
- [ ] Update README with BDD instructions
- [ ] Document any new patterns
- [ ] Create handoff for TEAM-084

### Quality Gates
- [ ] Compilation: 0 errors
- [ ] Tests: All pass (or document known failures)
- [ ] Wiring: 95%+ (target: 132/139 functions)
- [ ] Integration tests: 3+ scenarios
- [ ] Code signatures: TEAM-083 on all changes
- [ ] Documentation: Complete and accurate

---

## Important Notes

### Files Already Well-Wired (Don't Redo)

These files were already wired by previous teams:
- `inference_execution.rs` - TEAM-076 did extensive wiring
- `failure_recovery.rs` - TEAM-081 wired to WorkerRegistry
- `concurrency.rs` - TEAM-080/081 wired to WorkerRegistry
- `model_catalog.rs` - TEAM-079 wired to real SQLite

**Action:** Verify these are complete, don't rewrite them.

### Stub Assertions Are DONE

TEAM-082 fixed ALL stub assertions. You should NOT find any:
```rust
assert!(world.last_action.is_some());  // ❌ None of these remain
```

If you find one, it's a bug - report it.

### Use Existing Helpers

**HTTP client:**
```rust
let client = crate::steps::world::create_http_client();
// Has 10s timeout, 5s connection timeout
```

**State cleanup:**
```rust
world.reset_for_scenario();
// Clears concurrent state, SSE events, tokens, etc.
```

---

## Anti-Patterns to Avoid

❌ **DON'T:**
- Add new `assert!(world.last_action.is_some())` - use meaningful assertions
- Leave TODO markers without implementation
- Create stub functions without wiring them
- Remove TEAM signatures from previous teams
- Skip verification steps
- Use background commands that can hang (see engineering rules)

✅ **DO:**
- Wire to real product code from `/bin/`
- Use meaningful assertions with clear error messages
- Add TEAM-083 signatures
- Test your changes
- Document progress
- Add timeouts to async operations
- Use foreground commands only

---

## Success Criteria

### Minimum Acceptable (TEAM-083)
- [ ] 10+ functions wired to real APIs
- [ ] 3+ integration tests added
- [ ] Compilation passes
- [ ] Progress documented

### Target Goal (TEAM-083)
- [ ] Wiring: 95%+ (132/139 functions)
- [ ] Integration tests: 5+ scenarios
- [ ] Reliability improvements: timeouts, cleanup, retry
- [ ] Documentation: complete and accurate

### Stretch Goal (TEAM-083)
- [ ] Wiring: 100% (139/139 functions)
- [ ] Integration tests: 10+ scenarios
- [ ] Performance tests: load testing
- [ ] CI/CD: automated BDD runs

---

## Reference Documents

**Created by TEAM-082:**
- `TEAM_082_COMPLETE.md` - Stub assertion fixes
- `TEAM_082_EXTENDED_SUMMARY.md` - Detailed summary

**Created by previous teams:**
- `TEAM_081_COMPLETE.md` - WorkerRegistry wiring
- `TEAM_080_ARCHITECTURAL_FIX_COMPLETE.md` - Architectural decisions

**Key files:**
- Feature files: `test-harness/bdd/tests/features/`
- Step definitions: `test-harness/bdd/src/steps/`
- Product code: `/bin/queen-rbee/`, `/bin/rbee-hive/`, `/bin/llm-worker-rbee/`

---

## Verification Commands

```bash
# Check compilation
cargo check --package test-harness-bdd

# Count TEAM-083 signatures (after your work)
rg "TEAM-083:" test-harness/bdd/src/steps/

# Verify no stub assertions were added
rg "assert!\(world\.last_action\.is_some\(\)\)" test-harness/bdd/src/steps/

# Check wiring progress
rg "TEAM-" test-harness/bdd/src/steps/ | wc -l

# Run tests (foreground only!)
cargo test --package test-harness-bdd -- --nocapture
```

---

## Questions?

**If stuck:**
1. Read TEAM-082 handoff documents
2. Check product code in `/bin/` directories
3. Look at already-wired functions for patterns
4. Search for migration notes in step definitions
5. Verify scenario exists in feature files before wiring

**Key insight:** The product code is MORE COMPLETE than the tests. Just connect them!

---

**Created by:** TEAM-082  
**Date:** 2025-10-11  
**Time:** 17:42  
**Next Team:** TEAM-083  
**Estimated Work:** 13 hours (1.5-2 days)  
**Priority:** P0 - Critical for production readiness
