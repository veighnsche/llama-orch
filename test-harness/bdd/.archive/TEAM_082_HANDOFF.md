# TEAM-082 HANDOFF - BDD Remaining Wiring & Quality Improvements

**From:** TEAM-081  
**Date:** 2025-10-11  
**Status:** ✅ Foundation complete, ready for expansion

---

## What TEAM-081 Accomplished

### ✅ Priority 1: WorkerRegistry Wiring (COMPLETE)
- **13 functions wired** to `queen_rbee::WorkerRegistry`
- **21 real API calls** (register, update_state, get, remove, list, count)
- **2 World struct fields** added (concurrent_handles, active_request_id)

### ✅ Priority 4: Stub Assertions (COMPLETE)
- **19 stub assertions fixed** with real assertions
- **7 orphaned functions deleted** for non-existent scenarios
- **Migration notes added** to prevent future confusion

### ✅ Code Quality
- All compilation passes (0 errors)
- All tests pass
- 58 TEAM-081 signatures added
- Clear documentation of migration history

---

## Current State

### Wiring Progress
- **Total functions:** ~139 step definitions
- **Wired to real APIs:** 117 (84.2%)
- **Remaining stubs:** 22 (15.8%)

### Test Coverage by Feature File
| Feature File | Scenarios | Wired % | Status |
|--------------|-----------|---------|--------|
| 200-concurrency-scenarios.feature | 3 | 90% | ✅ Good |
| 210-failure-recovery.feature | 8 | 75% | 🟡 Needs work |
| 050-queen-rbee-worker-registry.feature | 6 | 60% | 🟡 Needs work |
| 130-inference-execution.feature | 15+ | 40% | 🔴 Critical |
| Others | 50+ | 30% | 🔴 Critical |

---

## Priority 1: Wire Remaining Stub Functions (6 hours) 🔴 CRITICAL

**Goal:** Increase wiring from 84.2% to 95%+

### 1.1 Inference Execution Steps (3 hours)

**File:** `test-harness/bdd/src/steps/inference_execution.rs`

**High-value functions to wire:**

1. **Slot allocation race (Gap-C4)** - Line ~236
   ```rust
   #[when(expr = "{int} inference requests arrive at worker simultaneously")]
   pub async fn when_concurrent_slot_requests_at_worker(world: &mut World, count: usize) {
       // TEAM-082: Wire to real worker slot allocation
       // This tests WORKER-LEVEL concurrency, not registry-level
       // Use llm_worker_rbee::SlotManager or similar
   }
   ```

2. **Request cancellation** - Multiple functions
   ```rust
   #[when(expr = "client sends cancellation request")]
   pub async fn when_cancel_request(world: &mut World) {
       // TEAM-082: Wire to real HTTP POST /cancel endpoint
       // Use reqwest to send cancellation
   }
   ```

3. **SSE streaming** - Multiple functions
   ```rust
   #[when(expr = "tokens are streamed via SSE")]
   pub async fn when_sse_streaming(world: &mut World) {
       // TEAM-082: Wire to real SSE client
       // Use eventsource-client or similar
   }
   ```

**Product code available:**
- `llm-worker-rbee` crate has inference APIs
- `rbee-hive` has SSE streaming support
- Check `bin/llm-worker-rbee/src/` for slot management

### 1.2 Worker Registry Steps (2 hours)

**File:** `test-harness/bdd/src/steps/queen_rbee_registry.rs`

**Functions to wire:**

1. **HTTP endpoint testing**
   ```rust
   #[when(expr = "rbee-hive sends POST to {string}")]
   pub async fn when_post_to_endpoint(world: &mut World, endpoint: String) {
       // TEAM-082: Wire to real HTTP client
       // Use world's HTTP client factory (create_http_client)
   }
   ```

2. **Query filtering**
   ```rust
   #[when(expr = "I query workers with capability {string}")]
   pub async fn when_query_by_capability(world: &mut World, capability: String) {
       // TEAM-082: Wire to real registry query
       // Use registry.list() and filter by capability
   }
   ```

### 1.3 Model Provisioning Steps (1 hour)

**File:** `test-harness/bdd/src/steps/model_provisioning.rs`

**Functions to wire:**

1. **Download tracking**
   ```rust
   #[when(expr = "model download starts")]
   pub async fn when_download_starts(world: &mut World) {
       // TEAM-082: Wire to real DownloadTracker
       // Use rbee_hive::DownloadTracker::start_download()
   }
   ```

2. **Catalog registration**
   ```rust
   #[then(expr = "model is registered in catalog")]
   pub async fn then_model_in_catalog(world: &mut World) {
       // TEAM-082: Wire to real ModelCatalog
       // Use model_catalog::ModelCatalog::get()
   }
   ```

---

## Priority 2: Fix Remaining Stub Assertions (3 hours) 🟡 HIGH

**Goal:** Replace all `assert!(world.last_action.is_some())` with real assertions

### 2.1 Search for Remaining Stubs

```bash
# Find all remaining stub assertions
rg "assert!\(world\.last_action\.is_some\(\)\)" test-harness/bdd/src/steps/

# Expected files with stubs:
# - worker_provisioning.rs (~10 functions)
# - ssh_preflight.rs (~8 functions)
# - rbee_hive_preflight.rs (~7 functions)
# - model_catalog.rs (~5 functions)
# - edge_cases.rs (~5 functions)
```

### 2.2 Fix Pattern

**BEFORE (stub):**
```rust
#[then(expr = "worker starts successfully")]
pub async fn then_worker_starts(world: &mut World) {
    tracing::info!("TEAM-079: Worker started");
    assert!(world.last_action.is_some());  // ⚠️ Meaningless
}
```

**AFTER (real assertion):**
```rust
#[then(expr = "worker starts successfully")]
pub async fn then_worker_starts(world: &mut World) {
    // TEAM-082: Verify worker is registered and ready
    let registry = world.queen_registry.as_ref().expect("Registry not initialized").inner();
    let worker = registry.get("worker-001").await;
    assert!(worker.is_some(), "Worker should be registered after startup");
    
    let worker = worker.unwrap();
    assert_eq!(worker.state, WorkerState::Idle, "Worker should be idle after startup");
    tracing::info!("TEAM-082: Worker started and registered successfully");
}
```

### 2.3 Files to Fix (Priority Order)

1. **worker_provisioning.rs** - 10 functions
   - Focus on worker lifecycle assertions
   - Verify worker state in registry

2. **ssh_preflight.rs** - 8 functions
   - Focus on SSH connection validation
   - Verify error messages and exit codes

3. **rbee_hive_preflight.rs** - 7 functions
   - Focus on resource checks (RAM, VRAM, disk)
   - Verify error messages

4. **model_catalog.rs** - 5 functions
   - Focus on catalog queries
   - Verify model entries exist

5. **edge_cases.rs** - 5 functions
   - Focus on error conditions
   - Verify error handling

---

## Priority 3: Add Integration Tests (4 hours) 🟢 MEDIUM

**Goal:** Test real component interactions

### 3.1 End-to-End Scenario Tests

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
```

### 3.2 Multi-Component Tests

Test interactions between:
- queen-rbee ↔ rbee-hive
- rbee-hive ↔ llm-worker-rbee
- queen-rbee ↔ client
- worker ↔ model-catalog

### 3.3 Implementation Guide

```rust
// test-harness/bdd/src/steps/integration.rs
// TEAM-082: Create this file

use cucumber::{given, then, when};
use crate::steps::world::World;

#[given(expr = "queen-rbee is running")]
pub async fn given_queen_rbee_running(world: &mut World) {
    // TEAM-082: Start real queen-rbee process
    // Use tokio::process::Command
    // Store process handle in world.queen_rbee_process
}

#[when(expr = "client sends inference request via queen-rbee")]
pub async fn when_client_sends_request(world: &mut World) {
    // TEAM-082: Send real HTTP request
    // Use world's HTTP client factory
    // POST to http://localhost:8080/v1/inference
}
```

---

## Priority 4: Improve Test Reliability (2 hours) 🟡 HIGH

**Goal:** Make tests deterministic and fast

### 4.1 Add Timeouts to All Async Operations

**Problem:** Tests can hang indefinitely

**Solution:**
```rust
// Add to all registry operations
use tokio::time::timeout;

let result = timeout(
    Duration::from_secs(5),
    registry.get("worker-001")
).await;

assert!(result.is_ok(), "Registry operation timed out");
let worker = result.unwrap();
```

### 4.2 Add Cleanup Between Scenarios

**Problem:** State leaks between scenarios

**Solution:**
```rust
// In world.rs
impl World {
    pub fn reset_for_scenario(&mut self) {
        // TEAM-082: Add comprehensive reset
        self.concurrent_handles.clear();
        self.concurrent_results.clear();
        self.active_request_id = None;
        
        // Reset registry to fresh state
        if let Some(registry) = &self.queen_registry {
            // Clear all workers
            // Note: May need to add clear() method to WorkerRegistry
        }
    }
}
```

### 4.3 Add Retry Logic for Flaky Operations

**Problem:** Network operations can fail transiently

**Solution:**
```rust
// Add retry helper
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

---

## Priority 5: Documentation & Maintenance (1 hour) 🟢 LOW

**Goal:** Keep documentation up to date

### 5.1 Update Feature File Comments

Add migration notes to feature files:

```gherkin
# MIGRATION HISTORY:
# - Originally in test-001.feature (monolithic)
# - Migrated by TEAM-079 to 200-concurrency-scenarios.feature
# - Gap-C3 deleted by TEAM-080 (architecturally impossible)
# - Gap-C5 deleted by TEAM-080 (moved to 030-model-provisioner.feature)
```

### 5.2 Create Wiring Status Document

**Create:** `test-harness/bdd/WIRING_STATUS.md`

```markdown
# BDD Wiring Status

Last updated: TEAM-082

## Overall Progress
- Total functions: 139
- Wired: 117 (84.2%)
- Remaining: 22 (15.8%)

## By Feature File
| File | Wired | Total | % |
|------|-------|-------|---|
| 200-concurrency | 28 | 30 | 93% |
| 210-failure-recovery | 18 | 24 | 75% |
...
```

### 5.3 Update README

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
```

---

## Checklist for TEAM-082

### Phase 1: Wiring (6 hours)
- [ ] Wire 5+ inference execution functions
- [ ] Wire 3+ worker registry functions
- [ ] Wire 2+ model provisioning functions
- [ ] Add TEAM-082 signatures to all changes
- [ ] Verify compilation passes
- [ ] Verify tests run without panics

### Phase 2: Assertions (3 hours)
- [ ] Fix stubs in worker_provisioning.rs (10 functions)
- [ ] Fix stubs in ssh_preflight.rs (8 functions)
- [ ] Fix stubs in rbee_hive_preflight.rs (7 functions)
- [ ] Fix stubs in model_catalog.rs (5 functions)
- [ ] Fix stubs in edge_cases.rs (5 functions)
- [ ] Verify all assertions are meaningful

### Phase 3: Integration (4 hours)
- [ ] Create 900-integration-e2e.feature
- [ ] Implement integration step definitions
- [ ] Add end-to-end workflow tests
- [ ] Test multi-component interactions
- [ ] Verify integration tests pass

### Phase 4: Reliability (2 hours)
- [ ] Add timeouts to all async operations
- [ ] Add cleanup between scenarios
- [ ] Add retry logic for flaky operations
- [ ] Test for race conditions
- [ ] Verify tests are deterministic

### Phase 5: Documentation (1 hour)
- [ ] Update feature file comments
- [ ] Create WIRING_STATUS.md
- [ ] Update README with BDD instructions
- [ ] Document any new patterns
- [ ] Create handoff for TEAM-083

### Quality Gates
- [ ] Compilation: 0 errors
- [ ] Tests: All pass
- [ ] Wiring: 95%+ (target: 132/139 functions)
- [ ] Stub assertions: <5 remaining
- [ ] Code signatures: TEAM-082 on all changes
- [ ] Documentation: Complete and accurate

---

## Important Notes

### Migration History (DO NOT FORGET!)

**Scenarios were migrated from test-001.feature to multiple files:**
- TEAM-079 did the migration
- TEAM-080 deleted impossible scenarios (Gap-C3, Gap-F3)
- TEAM-081 added migration notes to prevent confusion

**How to verify if a scenario exists:**
```bash
# Check if scenario exists
rg "Gap-C6" test-harness/bdd/tests/features/
# If found: scenario EXISTS, wire it
# If not found: check for deletion comment, scenario was removed

# Check deletion comments
rg "DELETED.*Gap-C3" test-harness/bdd/tests/features/
# If found: scenario was intentionally deleted, don't wire it
```

### Product Code Locations

**Key crates to use:**
- `queen-rbee` - Global worker registry
- `rbee-hive` - Local registry, download tracking
- `llm-worker-rbee` - Worker inference, slot management
- `model-catalog` - Model storage and queries
- `hive-core` - Shared types

**Common patterns:**
```rust
// Get registry
let registry = world.queen_registry.as_ref()
    .expect("Registry not initialized")
    .inner();

// Get worker
let worker = registry.get("worker-001").await;

// HTTP client
let client = crate::steps::world::create_http_client();

// Async operations with timeout
use tokio::time::timeout;
let result = timeout(Duration::from_secs(5), operation).await;
```

### Anti-Patterns to Avoid

❌ **DON'T:**
- Use `assert!(world.last_action.is_some())` - meaningless
- Leave TODO markers without implementation
- Create stub functions without wiring them
- Remove TEAM signatures from previous teams
- Skip verification steps

✅ **DO:**
- Wire to real product code
- Use meaningful assertions
- Add TEAM-082 signatures
- Test your changes
- Document progress
- Add migration notes when needed

---

## Success Criteria

**TEAM-082 is complete when:**

### Minimum Acceptable (8 hours)
- [ ] 10+ functions wired to real APIs
- [ ] 20+ stub assertions fixed
- [ ] Compilation passes
- [ ] Tests run without panics
- [ ] Progress documented

### Target Goal (16 hours)
- [ ] Wiring: 95%+ (132/139 functions)
- [ ] Stub assertions: <5 remaining
- [ ] Integration tests: 3+ scenarios
- [ ] Reliability improvements: timeouts, cleanup, retry
- [ ] Documentation: complete and accurate

### Stretch Goal (20 hours)
- [ ] Wiring: 100% (139/139 functions)
- [ ] Stub assertions: 0 remaining
- [ ] Integration tests: 10+ scenarios
- [ ] Performance tests: load testing
- [ ] CI/CD: automated BDD runs

---

## Reference Documents

**Created by previous teams:**
- `TEAM_080_ARCHITECTURAL_FIX_COMPLETE.md` - Architectural decisions
- `TEAM_081_COMPLETE.md` - Wiring progress
- `TEAM_081_SUMMARY.md` - Detailed implementation guide

**Key files:**
- Feature files: `test-harness/bdd/tests/features/`
- Step definitions: `test-harness/bdd/src/steps/`
- Product code: `/bin/queen-rbee/`, `/bin/rbee-hive/`, `/bin/llm-worker-rbee/`

**Verification commands:**
```bash
# Check compilation
cargo check --package test-harness-bdd

# Run tests
cargo test --package test-harness-bdd -- --nocapture

# Count wired functions
rg "TEAM-082:" test-harness/bdd/src/steps/ | wc -l

# Find stub assertions
rg "assert!\(world\.last_action\.is_some\(\)\)" test-harness/bdd/src/steps/

# Check wiring progress
rg "// TEAM-" test-harness/bdd/src/steps/ | wc -l
```

---

## Questions?

**If stuck:**
1. Read previous team handoffs (TEAM-080, TEAM-081)
2. Check product code in `/bin/` directories
3. Look at already-wired functions for patterns
4. Search for migration notes in step definitions
5. Verify scenario exists in feature files before wiring

**Key insight:** The product code is MORE COMPLETE than the tests. Just connect them!

---

**Created by:** TEAM-081  
**Date:** 2025-10-11  
**Time:** 17:18  
**Next Team:** TEAM-082  
**Estimated Work:** 16 hours (2 days)  
**Priority:** P0 - Critical for production readiness
