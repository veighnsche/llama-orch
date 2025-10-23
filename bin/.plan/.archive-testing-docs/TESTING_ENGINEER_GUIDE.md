# Testing Engineer Guide - Start Here

**Date:** Oct 22, 2025  
**Audience:** Testing engineers starting test implementation  
**Reading Time:** 30-45 minutes  
**Scope:** NUC-friendly, reasonable scale testing

---

## Welcome!

This guide will help you understand the rbee system and start implementing tests. Follow this reading order to build context efficiently.

---

## Phase 1: System Overview (15 minutes)

### 1.1 Start with the Big Picture

**Read First:**
- 📋 `README.md` (project root) - What is rbee?
- 📋 `bin/.plan/00_INDEX.md` - Plan folder structure
- 📋 `bin/.plan/PHASE_5_COMPLETE_SUMMARY.md` - Integration flows overview

**What You'll Learn:**
- System architecture (keeper → queen → hive → worker)
- Component responsibilities
- Key patterns (dual-call, SSE streaming, narration)

**Time:** 10 minutes

### 1.2 Understand Testing Scope

**Read Next:**
- 📋 `bin/.plan/TESTING_GAPS_EXECUTIVE_SUMMARY.md` - High-level testing gaps
- 📋 `bin/.plan/TESTING_GAPS_ADDITIONAL_FINDINGS.md` - Specific gaps with reasonable scale

**What You'll Learn:**
- What's implemented vs. not implemented
- What needs tests (IMPLEMENTED code only)
- Reasonable scale for NUC (5-10 concurrent, not 100+)
- Critical gaps (SSE, concurrency, timeouts)

**Time:** 5 minutes

---

## Phase 2: Component Deep Dive (30 minutes)

### 2.1 Shared Crates (Core Infrastructure)

**Read in Order:**

1. **Narration System** (10 minutes)
   - 📋 `bin/.plan/TEAM_230_NARRATION_BEHAVIORS.md` - How narration works
   - 📋 `bin/99_shared_crates/observability-narration-core/README.md` - API docs
   - **Key Concepts:** NarrationFactory, SSE routing, job_id propagation

2. **Job Registry** (5 minutes)
   - 📋 `bin/.plan/TEAM_233_JOB_REGISTRY_BEHAVIORS.md` - Job lifecycle
   - 📋 `bin/99_shared_crates/job-registry/README.md` - API docs
   - **Key Concepts:** Dual-call pattern, deferred execution, token streaming

3. **Daemon Lifecycle** (5 minutes)
   - 📋 `bin/.plan/TEAM_231_DAEMON_LIFECYCLE_BEHAVIORS.md` - Process spawning
   - 📋 `bin/99_shared_crates/daemon-lifecycle/README.md` - API docs
   - **Key Concepts:** Stdio::null() (CRITICAL), SSH agent propagation

4. **Config & Operations** (5 minutes)
   - 📋 `bin/.plan/TEAM_232_CONFIG_OPERATIONS_BEHAVIORS.md` - Config loading
   - 📋 `bin/99_shared_crates/rbee-config/README.md` - Config format
   - **Key Concepts:** Unix-style config, SSH syntax, localhost special case

5. **Heartbeat & Timeout** (5 minutes)
   - 📋 `bin/.plan/TEAM_234_HEARTBEAT_TIMEOUT_BEHAVIORS.md` - Health monitoring
   - 📋 `bin/99_shared_crates/rbee-heartbeat/README.md` - API docs
   - **Key Concepts:** Three-tier heartbeat, staleness detection, timeout enforcement

### 2.2 Queen-Specific Crates (5 minutes)

1. **Hive Lifecycle** (3 minutes)
   - 📋 `bin/.plan/TEAM_220_HIVE_LIFECYCLE_BEHAVIORS.md` - Hive operations
   - **Key Concepts:** Binary resolution, health polling, capabilities cache

2. **Hive Registry** (2 minutes)
   - 📋 `bin/.plan/TEAM_221_HIVE_REGISTRY_BEHAVIORS.md` - Hive state
   - 📋 `bin/15_queen_rbee_crates/hive-registry/TESTING_GUIDE.md` - Testing priorities
   - **Key Concepts:** Staleness detection, worker aggregation

3. **SSH Client** (2 minutes)
   - 📋 `bin/.plan/TEAM_222_SSH_CLIENT_BEHAVIORS.md` - SSH testing
   - **Key Concepts:** 5-step connection flow, pre-flight checks

---

## Phase 3: Integration Flows (15 minutes)

### 3.1 Keeper ↔ Queen Integration

**Read:**
- 📋 `bin/.plan/TEAM_239_KEEPER_QUEEN_INTEGRATION.md` - Client-server flow

**What You'll Learn:**
- Dual-call pattern (POST creates job, GET streams results)
- Layered timeouts (HTTP 10s, SSE 30s, operation 15s)
- SSE channel lifecycle
- Error propagation (HTTP → SSE → CLI)

**Time:** 5 minutes

### 3.2 Queen ↔ Hive Integration

**Read:**
- 📋 `bin/.plan/TEAM_240_QUEEN_HIVE_INTEGRATION.md` - Server-hive flow

**What You'll Learn:**
- Hive spawn with Stdio::null()
- Heartbeat flow (hive → queen every 5s)
- Capabilities discovery
- SSH integration

**Time:** 5 minutes

### 3.3 Future Flows (NOT YET IMPLEMENTED)

**Read (for context only):**
- 📋 `bin/.plan/TEAM_241_HIVE_WORKER_INTEGRATION.md` - Hive-worker flow (NOT IMPLEMENTED)
- 📋 `bin/.plan/TEAM_242_E2E_INFERENCE_FLOWS.md` - Full inference flow (NOT IMPLEMENTED)

**What You'll Learn:**
- What's planned but not yet built
- Don't write tests for these yet

**Time:** 5 minutes

---

## Phase 4: Test Planning (30 minutes)

### 4.1 Review Test Checklists

**Read in Order:**

1. **Part 1: Shared Crates** (10 minutes)
   - 📋 `bin/.plan/TESTING_GAPS_MASTER_CHECKLIST_PART_1.md`
   - **Focus:** Narration, daemon-lifecycle, config, job-registry

2. **Part 2: Heartbeat + Binaries** (10 minutes)
   - 📋 `bin/.plan/TESTING_GAPS_MASTER_CHECKLIST_PART_2.md`
   - **Focus:** Heartbeat, timeout, rbee-keeper, queen-rbee

3. **Part 3: Integration Flows** (10 minutes)
   - 📋 `bin/.plan/TESTING_GAPS_MASTER_CHECKLIST_PART_3.md`
   - **Focus:** rbee-hive, keeper↔queen, queen↔hive

4. **Part 4: E2E + Infrastructure** (skip for now)
   - 📋 `bin/.plan/TESTING_GAPS_MASTER_CHECKLIST_PART_4.md`
   - **Note:** E2E inference not implemented yet

### 4.2 Understand Priorities

**Critical Path (Start Here):**
1. SSE channel lifecycle (memory leaks, race conditions)
2. Concurrent access patterns (job-registry, hive-registry)
3. Stdio::null() behavior (prevents E2E test hangs)
4. Timeout propagation (all layers)
5. Resource cleanup (disconnect, crash, timeout)

**Medium Priority (After Critical):**
6. SSH client (0% coverage)
7. Binary resolution (hive-lifecycle)
8. Graceful shutdown (hive-lifecycle)
9. Capabilities cache (hive-lifecycle)
10. Error propagation (all boundaries)

**Low Priority (Nice to Have):**
11. Format string edge cases (narration)
12. Table formatting edge cases (narration)
13. Config corruption handling
14. Correlation ID validation

---

## Phase 5: Test Infrastructure (15 minutes)

### 5.1 BDD Framework

**Read:**
- 📋 `xtask/README.md` - BDD runner documentation
- 📋 Look at existing `.feature` files in `bdd/tests/features/`

**What You'll Learn:**
- How to write BDD tests (Given/When/Then)
- How to run BDD tests (`cargo xtask bdd`)
- Step definition patterns

**Time:** 10 minutes

### 5.2 Test Helpers

**Explore:**
- 📂 `bin/99_shared_crates/*/tests/` - Existing test patterns
- 📂 `bin/00_rbee_keeper/tests/` - Integration test examples

**What You'll Learn:**
- Mock patterns
- Test fixtures
- Async test setup

**Time:** 5 minutes

---

## Quick Reference: Testing Principles

### DO: Reasonable Scale (NUC-Friendly)

✅ **Concurrent Operations:** 5-10 concurrent (not 100+)  
✅ **Job/Hive/Worker Count:** 100 total (not 1000+)  
✅ **Payload Size:** 1MB max (not 10MB+)  
✅ **Workers per Hive:** 5 workers (not 50+)  
✅ **SSE Channels:** 10 concurrent (not 100+)

### DON'T: Overkill Scale

❌ **100+ concurrent operations** - Too much for NUC  
❌ **1000+ jobs/hives/workers** - Unrealistic  
❌ **10MB+ payloads** - Excessive  
❌ **50+ workers per hive** - Unrealistic  
❌ **100+ concurrent SSE channels** - Overkill

### Focus on IMPLEMENTED Features

✅ **Test:** Hive operations (start, stop, status, list)  
✅ **Test:** SSE streaming (narration, job-scoped channels)  
✅ **Test:** Heartbeat flow (hive → queen)  
✅ **Test:** Config loading (SSH syntax, localhost special case)

❌ **Don't Test:** Worker operations (NOT IMPLEMENTED)  
❌ **Don't Test:** Inference flow (NOT IMPLEMENTED)  
❌ **Don't Test:** Model provisioning (NOT IMPLEMENTED)

### Critical Invariants to Test

1. **job_id MUST propagate** - Without it, narration doesn't reach SSE
2. **[DONE] marker MUST be sent** - Keeper uses it to detect completion
3. **Stdio::null() MUST be used** - Prevents pipe hangs in E2E tests
4. **Timeouts MUST fire** - Zero tolerance for hanging operations
5. **Channels MUST be cleaned up** - Prevent memory leaks

---

## Test Implementation Workflow

### Step 1: Pick a Component

**Start with HIGH priority:**
1. SSH Client (0% coverage, 15 tests)
2. Daemon Lifecycle Stdio::null() (CRITICAL, 4 tests)
3. Hive Registry concurrent access (HIGH, 5 tests)
4. Job Registry concurrent access (HIGH, 5 tests)

### Step 2: Read Component Docs

**For each component:**
1. Read behavior inventory (TEAM-XXX document)
2. Read README.md in component folder
3. Read existing tests (if any)
4. Read TESTING_GUIDE.md (if exists)

### Step 3: Write Tests

**BDD Pattern:**
```gherkin
Feature: SSH Connection Testing
  Scenario: SSH agent not running
    Given SSH_AUTH_SOCK is not set
    When I test SSH connection to "localhost"
    Then the result should be failure
    And the error should mention "SSH agent"
```

**Rust Pattern:**
```rust
#[tokio::test]
async fn test_ssh_agent_not_running() {
    // Setup
    std::env::remove_var("SSH_AUTH_SOCK");
    
    // Execute
    let config = SshConfig { /* ... */ };
    let result = test_ssh_connection(config).await.unwrap();
    
    // Assert
    assert!(!result.success);
    assert!(result.error.unwrap().contains("SSH agent"));
}
```

### Step 4: Run Tests

**Commands:**
```bash
# Run unit tests
cargo test -p <crate-name>

# Run BDD tests
cargo xtask bdd

# Run integration tests
cargo test --test <test-name>

# Run all tests
cargo test --workspace
```

### Step 5: Verify Coverage

**Check:**
- All happy paths tested
- All error paths tested
- All edge cases tested
- All concurrent scenarios tested
- All cleanup scenarios tested

---

## Common Pitfalls

### 1. Testing Unimplemented Features

❌ **Wrong:** Writing tests for worker operations (not implemented)  
✅ **Right:** Writing tests for hive operations (implemented)

### 2. Unrealistic Scale

❌ **Wrong:** Testing 1000 concurrent operations  
✅ **Right:** Testing 5-10 concurrent operations

### 3. Missing job_id Propagation

❌ **Wrong:** Narration without job_id (events are dropped)  
✅ **Right:** Narration with job_id (events reach SSE)

### 4. Forgetting Stdio::null()

❌ **Wrong:** Daemon spawn without Stdio::null() (E2E tests hang)  
✅ **Right:** Daemon spawn with Stdio::null() (E2E tests work)

### 5. Not Testing Cleanup

❌ **Wrong:** Only testing happy path  
✅ **Right:** Testing cleanup on error, timeout, disconnect

---

## Getting Help

### Documentation Locations

**Component Docs:**
- `bin/99_shared_crates/<crate>/README.md` - Shared crates
- `bin/15_queen_rbee_crates/<crate>/README.md` - Queen crates
- `bin/<binary>/README.md` - Binary docs

**Behavior Inventories:**
- `bin/.plan/TEAM_XXX_<component>_BEHAVIORS.md` - Detailed behaviors
- `bin/.plan/TEAM_XXX_NARRATION_INVENTORY.md` - Narration patterns

**Testing Guides:**
- `bin/<crate>/TESTING_GUIDE.md` - Component-specific testing
- `bin/.plan/TESTING_GAPS_*.md` - Testing gaps and priorities

### Code Examples

**Look at existing tests:**
- `bin/99_shared_crates/observability-narration-core/tests/` - Narration tests
- `bin/99_shared_crates/timeout-enforcer/tests/` - Timeout tests
- `bin/00_rbee_keeper/tests/` - Integration tests

---

## Reading Checklist

Use this to track your progress:

### Phase 1: System Overview (15 min)
- [ ] README.md (project root)
- [ ] bin/.plan/00_INDEX.md
- [ ] bin/.plan/PHASE_5_COMPLETE_SUMMARY.md
- [ ] bin/.plan/TESTING_GAPS_EXECUTIVE_SUMMARY.md
- [ ] bin/.plan/TESTING_GAPS_ADDITIONAL_FINDINGS.md

### Phase 2: Component Deep Dive (30 min)
- [ ] TEAM_230_NARRATION_BEHAVIORS.md
- [ ] TEAM_233_JOB_REGISTRY_BEHAVIORS.md
- [ ] TEAM_231_DAEMON_LIFECYCLE_BEHAVIORS.md
- [ ] TEAM_232_CONFIG_OPERATIONS_BEHAVIORS.md
- [ ] TEAM_234_HEARTBEAT_TIMEOUT_BEHAVIORS.md
- [ ] TEAM_220_HIVE_LIFECYCLE_BEHAVIORS.md
- [ ] TEAM_221_HIVE_REGISTRY_BEHAVIORS.md
- [ ] TEAM_222_SSH_CLIENT_BEHAVIORS.md

### Phase 3: Integration Flows (15 min)
- [ ] TEAM_239_KEEPER_QUEEN_INTEGRATION.md
- [ ] TEAM_240_QUEEN_HIVE_INTEGRATION.md
- [ ] TEAM_241_HIVE_WORKER_INTEGRATION.md (context only)
- [ ] TEAM_242_E2E_INFERENCE_FLOWS.md (context only)

### Phase 4: Test Planning (30 min)
- [ ] TESTING_GAPS_MASTER_CHECKLIST_PART_1.md
- [ ] TESTING_GAPS_MASTER_CHECKLIST_PART_2.md
- [ ] TESTING_GAPS_MASTER_CHECKLIST_PART_3.md

### Phase 5: Test Infrastructure (15 min)
- [ ] xtask/README.md
- [ ] Existing .feature files
- [ ] Existing test patterns

**Total Reading Time:** ~90 minutes (1.5 hours)

---

## Next Steps

After completing this guide:

1. **Pick a component** from the HIGH priority list
2. **Read component docs** (behavior inventory + README)
3. **Write tests** (start with 1-2 tests to get familiar)
4. **Run tests** (verify they pass)
5. **Iterate** (add more tests, refine)

**Good luck! 🚀**

---

**Questions?** Check the behavior inventories or existing test patterns for examples.
