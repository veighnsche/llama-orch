# Testing Quick Start - TL;DR

**For:** Testing engineers who want to start immediately  
**Reading Time:** 5 minutes

---

## Start Here (3 Steps)

### Step 1: Read the Guide (90 minutes)

📋 **`TESTING_ENGINEER_GUIDE.md`** - Complete reading guide with checklist

**What it covers:**
- System architecture
- Component behaviors
- Integration flows
- Test priorities
- Test infrastructure

### Step 2: Pick Your First Component (5 minutes)

**HIGH Priority (Start Here):**

1. **SSH Client** - 0% coverage, 15 tests, 5-7 days
   - 📋 Read: `TEAM_222_SSH_CLIENT_BEHAVIORS.md`
   - 📋 Code: `bin/15_queen_rbee_crates/ssh-client/`

2. **Daemon Lifecycle (Stdio::null())** - CRITICAL, 4 tests, 2-3 days
   - 📋 Read: `TEAM_231_DAEMON_LIFECYCLE_BEHAVIORS.md`
   - 📋 Code: `bin/99_shared_crates/daemon-lifecycle/`

3. **Hive Registry (Concurrent Access)** - HIGH, 5 tests, 3-4 days
   - 📋 Read: `TEAM_221_HIVE_REGISTRY_BEHAVIORS.md`
   - 📋 Read: `bin/15_queen_rbee_crates/hive-registry/TESTING_GUIDE.md`
   - 📋 Code: `bin/15_queen_rbee_crates/hive-registry/`

4. **Job Registry (Concurrent Access)** - HIGH, 5 tests, 2-3 days
   - 📋 Read: `TEAM_233_JOB_REGISTRY_BEHAVIORS.md`
   - 📋 Code: `bin/99_shared_crates/job-registry/`

### Step 3: Write Tests (varies)

**BDD Pattern:**
```gherkin
Feature: SSH Connection Testing
  Scenario: SSH agent not running
    Given SSH_AUTH_SOCK is not set
    When I test SSH connection to "localhost"
    Then the result should be failure
    And the error should mention "SSH agent"
```

**Run:**
```bash
cargo xtask bdd
```

---

## Critical Rules

### ✅ DO: Test IMPLEMENTED Features

- Hive operations (start, stop, status, list)
- SSE streaming (narration, job-scoped channels)
- Heartbeat flow (hive → queen)
- Config loading (SSH syntax, localhost)

### ❌ DON'T: Test UNIMPLEMENTED Features

- Worker operations (NOT IMPLEMENTED)
- Inference flow (NOT IMPLEMENTED)
- Model provisioning (NOT IMPLEMENTED)

### ✅ DO: Reasonable Scale (NUC-Friendly)

- 5-10 concurrent operations
- 100 jobs/hives/workers
- 1MB payloads
- 5 workers per hive

### ❌ DON'T: Overkill Scale

- 100+ concurrent operations
- 1000+ jobs/hives/workers
- 10MB+ payloads
- 50+ workers per hive

---

## Test Priorities

### Priority 1: Critical Path (Start Here)

1. **SSE channel lifecycle** - Memory leaks, race conditions
2. **Concurrent access** - Job-registry, hive-registry
3. **Stdio::null()** - Prevents E2E test hangs (CRITICAL)
4. **Timeout propagation** - All layers
5. **Resource cleanup** - Disconnect, crash, timeout

**Effort:** 40-60 days (1 developer) or 2-3 weeks (3 developers)

### Priority 2: Medium Priority

6. **SSH client** - 0% coverage
7. **Binary resolution** - Hive-lifecycle
8. **Graceful shutdown** - Hive-lifecycle
9. **Capabilities cache** - Hive-lifecycle
10. **Error propagation** - All boundaries

**Effort:** 30-40 days (1 developer) or 2-3 weeks (3 developers)

### Priority 3: Low Priority

11. **Format string edge cases** - Narration
12. **Table formatting edge cases** - Narration
13. **Config corruption** - Config loading
14. **Correlation ID** - Narration

**Effort:** 20-30 days (1 developer) or 1-2 weeks (3 developers)

---

## Documents to Read (in order)

### Must Read (60 minutes)

1. **System Overview:**
   - `TESTING_GAPS_EXECUTIVE_SUMMARY.md` (5 min)
   - `TESTING_GAPS_ADDITIONAL_FINDINGS.md` (5 min)
   - `PHASE_5_COMPLETE_SUMMARY.md` (10 min)

2. **Component Behaviors:**
   - `TEAM_230_NARRATION_BEHAVIORS.md` (10 min)
   - `TEAM_233_JOB_REGISTRY_BEHAVIORS.md` (5 min)
   - `TEAM_231_DAEMON_LIFECYCLE_BEHAVIORS.md` (5 min)

3. **Integration Flows:**
   - `TEAM_239_KEEPER_QUEEN_INTEGRATION.md` (10 min)
   - `TEAM_240_QUEEN_HIVE_INTEGRATION.md` (10 min)

### Should Read (30 minutes)

4. **Test Checklists:**
   - `TESTING_GAPS_MASTER_CHECKLIST_PART_1.md` (10 min)
   - `TESTING_GAPS_MASTER_CHECKLIST_PART_2.md` (10 min)
   - `TESTING_GAPS_MASTER_CHECKLIST_PART_3.md` (10 min)

### Optional (for specific components)

5. **Component-Specific:**
   - `TEAM_220_HIVE_LIFECYCLE_BEHAVIORS.md` (hive operations)
   - `TEAM_221_HIVE_REGISTRY_BEHAVIORS.md` (hive state)
   - `TEAM_222_SSH_CLIENT_BEHAVIORS.md` (SSH testing)
   - `TEAM_232_CONFIG_OPERATIONS_BEHAVIORS.md` (config loading)
   - `TEAM_234_HEARTBEAT_TIMEOUT_BEHAVIORS.md` (heartbeat)

---

## Critical Invariants (MUST Test)

1. **job_id MUST propagate** - Without it, narration doesn't reach SSE
2. **[DONE] marker MUST be sent** - Keeper uses it to detect completion
3. **Stdio::null() MUST be used** - Prevents pipe hangs in E2E tests
4. **Timeouts MUST fire** - Zero tolerance for hanging operations
5. **Channels MUST be cleaned up** - Prevent memory leaks

---

## Test Commands

```bash
# Run unit tests for a crate
cargo test -p <crate-name>

# Run BDD tests
cargo xtask bdd

# Run integration tests
cargo test --test <test-name>

# Run all tests
cargo test --workspace

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test <test-name>
```

---

## Common Pitfalls

1. ❌ Testing unimplemented features (worker ops, inference)
2. ❌ Unrealistic scale (100+ concurrent, 1000+ jobs)
3. ❌ Missing job_id propagation (events are dropped)
4. ❌ Forgetting Stdio::null() (E2E tests hang)
5. ❌ Not testing cleanup (memory leaks)

---

## Getting Help

**Component Docs:**
- `bin/99_shared_crates/<crate>/README.md`
- `bin/15_queen_rbee_crates/<crate>/README.md`

**Behavior Inventories:**
- `bin/.plan/TEAM_XXX_<component>_BEHAVIORS.md`

**Testing Guides:**
- `bin/<crate>/TESTING_GUIDE.md` (if exists)
- `bin/.plan/TESTING_GAPS_*.md`

**Code Examples:**
- `bin/99_shared_crates/*/tests/` - Existing tests
- `bin/00_rbee_keeper/tests/` - Integration tests

---

## Summary Statistics

**Total Testing Gap:** ~585 tests  
**Total Effort:** 177-248 days (1 developer) or 20-28 weeks (3 developers)

**By Priority:**
- Priority 1 (Critical): 40-60 days
- Priority 2 (Medium): 30-40 days
- Priority 3 (Low): 20-30 days
- Additional Findings: 47-68 days

**By Component:**
- Shared Crates: ~150 tests (40-60 days)
- Binaries: ~120 tests (50-70 days)
- Integration: ~100 tests (80-110 days)
- Additional: ~135 tests (47-68 days)

---

**Ready to start? Read `TESTING_ENGINEER_GUIDE.md` for the full guide!**
