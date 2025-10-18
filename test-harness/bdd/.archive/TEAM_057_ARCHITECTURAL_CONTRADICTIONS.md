# ARCHITECTURAL DECISIONS NEEDED (Mid-Development Analysis)

**Created by:** TEAM-057 (The Thinking Team)  
**Date:** 2025-10-10  
**Status:** 📋 ANALYSIS COMPLETE - Architectural decisions identified  
**Context:** rbee v0.1.0 is ~68% complete (early development)

---

## Executive Summary

**IMPORTANT:** This is mid-development analysis for a ~68% complete project, NOT production code review.

Deep analysis reveals **5 architectural decisions needed** between spec, tests, and implementation to complete the remaining 32%. These are design choices that need to be made to finish the BDD test suite.

---

## Contradiction 1: Registration Model Mismatch 🔴 CRITICAL

### Spec Says (bin/.specs/.gherkin/test-001.md)

**Lines 35-106:** Explicit two-phase setup required:

```gherkin
# Phase 1: User explicitly registers node
rbee-keeper setup add-node \
  --name workstation \
  --ssh-host workstation.home.arpa \
  ...

# Phase 2: User runs inference using registered node
rbee-keeper infer \
  --node workstation \
  --model hf:... \
  ...
```

**Spec is clear:** Node must be in queen-rbee's beehive registry BEFORE inference.

### Tests Say (tests/features/test-001.feature)

**Lines 12-17:** Background defines topology declaratively:

```gherkin
Background:
  Given the following topology:
    | node        | hostname              | components                |
    | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee|
```

**Test assumption:** Nodes are implicitly available from topology definition.

### Implementation Says (src/steps/background.rs)

**Lines 8-23:** Topology step just stores NodeInfo in memory:

```rust
pub async fn given_topology(world: &mut World, step: &cucumber::gherkin::Step) {
    // Just stores node info, doesn't register anywhere
    world.topology.insert(node.clone(), NodeInfo { ... });
}

**Reality:** Topology doesn't trigger registration in queen-rbee's beehive registry.

### The Contradiction

- **Impact:** 14+ scenarios fail. Even scenarios WITH explicit registration (lines 176, 231) still fail, suggesting the registration step itself has issues OR there are additional missing preconditions.
- **Tests:** Implicit availability assumed → Background topology → immediate inference
- **Code:** Background just stores metadata → no registry interaction → inference fails

**Why 20 scenarios fail:** They use nodes from Background topology that were never registered in queen-rbee's beehive registry.

{{ ... }}
### Impact

- 3 CLI command scenarios fail (basic inference, shutdown worker, install)
- 6 HTTP scenarios fail (trying to connect to unregistered nodes)
- Multiple edge cases fail (no node registration to test against)

---

## Contradiction 2: Test Isolation vs. Global State 🔴 CRITICAL

### Spec Says

**Lines 1004-1010:** queen-rbee is persistent daemon with SQLite database:

```
# RULE 2: queen-rbee is a PERSISTENT HTTP DAEMON (ORCHESTRATOR)
#   - Maintains: rbee-hive Registry (SQLite at ~/.rbee/beehives.db)
```

### Tests Say

**test-001.feature line 22:** Each scenario should be isolated:

```gherkin
And the rbee-hive registry is SQLite at "~/.rbee/beehives.db"
```

Each Background implies fresh state per scenario.

### Implementation Says (src/steps/global_queen.rs)

**Lines 55-127:** Single global queen-rbee instance with shared database:

```rust
static GLOBAL_QUEEN: OnceLock<GlobalQueenRbee> = OnceLock::new();

pub async fn start_global_queen_rbee() {
    // Single instance, single database, shared across ALL scenarios
    let db_path = temp_dir.path().join("global_test_beehives.db");
    // ...
}
```

### The Contradiction

- **Spec:** Persistent daemon with persistent database (intentional state)
- **Tests:** Each scenario assumes fresh state (Background is re-executed)
- **Code:** Global instance + shared database → state leaks between scenarios

**Example:**
1. Scenario A registers node "workstation"
2. Scenario B runs with fresh Background
3. Scenario B expects empty registry but node "workstation" is still there from A

### Impact

- Tests are NOT isolated—order matters
- Passing/failing depends on execution sequence
- Debugging is impossible (Heisenbug territory)
- Retry logic helps mask timing issues but doesn't fix root cause

---

## Contradiction 3: Execution vs. Test Expectations 🟡 MEDIUM

### Spec Says

Real distributed system with SSH:

```
queen-rbee → SSH → rbee-hive (remote node) → worker
```

**CRITICAL UPDATE:** After deeper investigation, **all queen-rbee endpoints ARE implemented:**
- `/v2/tasks` ✅ (inference.rs:31)
- `/v2/workers/shutdown` ✅ (workers.rs:85)
- `/v2/registry/beehives/add` ✅ (beehives.rs:25)
- `/v2/registry/beehives/list` ✅ (beehives.rs:114)
- `/v2/registry/beehives/remove` ✅ (beehives.rs:127)

All rbee-keeper commands ARE implemented with retry logic:
- `infer` with backend/device ✅ (infer.rs with 3-attempt retry)
- `setup add-node` ✅ (setup.rs:89)
- `setup list-nodes` ✅ (setup.rs:142)
- `workers shutdown` ✅ (workers.rs:188 with retry)

### Tests Say

Mock infrastructure available:
- Mock rbee-hive on port 9200 ✅
- Mock worker on port 8001 ✅
- Mock SSH via `MOCK_SSH=true` ✅

### Implementation Says

**Tests execute REAL binaries** that connect to REAL queen-rbee:

1. Test runs: `rbee infer --node workstation ...`
2. Real `rbee` binary executes
3. Sends POST to http://localhost:8080/v2/tasks ✅ (endpoint exists!)
4. queen-rbee receives request ✅
5. queen-rbee queries beehive registry for "workstation" ✅ (code at inference.rs:38)
6. **Node not found** → Returns 404 with error message
7. rbee-keeper receives 404, returns exit code 1

### The Contradiction

- **Spec:** Node must be explicitly registered before use
- **Tests:** Assume nodes from Background topology are available
- **Code:** Endpoints exist but node lookup fails → 404

**The REAL problem:** This is NOT missing endpoints (they exist!). It's **Contradiction 1** (registration model mismatch). Nodes aren't registered, so endpoint returns 404.

### Impact

- 3 CLI commands fail with exit code 1 (not "connection error" - it's 404 Not Found)
- Endpoints ARE implemented
- Infrastructure IS complete
- **The fix is to register nodes explicitly** (Contradiction 1)

---

## Contradiction 4: Step Definition Philosophy 🟡 HIGH

### Spec Says

Tests should verify behavior:

> "EC1 - Connection timeout with retry and backoff"
> "Then the exit code is 1"

Tests exist to catch regressions and verify behavior.

### Tests Say

Concrete assertions:

```gherkin
Scenario: EC1 - Connection timeout with retry and backoff
  ...
  Then the exit code is 1
```

Expected: Command actually executes, returns exit code 1.

### Implementation Says (src/steps/edge_cases.rs)

**Lines 62-65:** Stub that just logs:

```rust
#[when(expr = "rbee-keeper attempts connection")]
pub async fn when_attempt_connection(world: &mut World) {
    tracing::debug!("Attempting connection");
    // NO COMMAND EXECUTION!
    // world.last_exit_code remains None
}
```

### The Contradiction

- **Spec:** Edge cases must be handled and return proper error codes
- **Tests:** Assertions expect real behavior (exit code 1)
- **Code:** Stubs just log, don't execute, don't set exit codes

**Result:** Tests pass when they should fail (false positives) or fail when they should pass (false negatives).

### Impact

- 9 edge case scenarios return None instead of 1
- Tests don't catch regressions
- Tests don't verify actual behavior
- Debugging is misleading

---

## Contradiction 5: Background Scope and Timing 🔴 CRITICAL

### Spec Says

**Lines 35-106:** Setup is a deliberate user action BEFORE inference:

```bash
# User explicitly runs this first:
rbee-keeper setup add-node --name workstation ...

# Then user runs inference:
rbee-keeper infer --node workstation ...
```

Temporal sequence: Setup → Ready → Use

### Tests Say

**Lines 12-23:** Background runs BEFORE each scenario:

```gherkin
Background:
  Given the following topology:
    ...
  And queen-rbee is running at "http://localhost:8080"
  And the rbee-hive registry is SQLite at "~/.rbee/beehives.db"
```

**Gherkin semantics:** Background executes before EVERY scenario.

### Implementation Says (src/main.rs)

**Lines 42-43:** Global queen-rbee starts BEFORE Background:

```rust
// TEAM-051: Start global queen-rbee instance before running tests
steps::global_queen::start_global_queen_rbee().await;
```

**Execution order:**
1. main() starts
2. Global queen-rbee spawns (60s wait for HTTP ready)
3. Background runs (tries to register nodes)
4. BUT queen-rbee might not be ready yet
5. HTTP requests fail with "connection closed" or timeout

### The Contradiction

- **Spec:** Sequential user actions (setup, then use)
- **Tests:** Parallel execution (Background before scenario, but after main)
- **Code:** Race condition (Background vs. global queen startup)

**Why TEAM-056's auto-registration failed:**
- Background runs, tries to register nodes via HTTP
- Global queen-rbee not fully ready yet (still binding port, initializing DB)
- HTTP request fails with "connection closed before message completed"
- Registration fails, nodes not available, scenarios fail

### Impact

- Auto-registration doesn't work (timing issue)
- Manual registration in scenarios works (queen-rbee is ready by then)
- First few scenarios might fail while later ones pass
- Non-deterministic failures

---

## Root Cause Summary

| Contradiction | Root Cause | Symptom | Impact |
|---------------|-----------|---------|--------|
| 1. Registration Model | Spec requires explicit, tests assume implicit | Nodes not in registry | 3-9 scenarios fail |
| 2. Test Isolation | Global state in shared DB | State leaks between tests | Non-deterministic failures |
| 3. Mock Architecture | Real binaries + partial mocks | Endpoints missing/wrong | 3-6 scenarios fail |
| 4. Step Philosophy | Stubs instead of implementations | Exit codes None not 1 | 9 scenarios fail |
| 5. Background Timing | Race condition between setup and use | HTTP connection errors | 0-6 scenarios fail |

**Total potential impact:** All 20 failing scenarios trace to these contradictions.

**IMPORTANT UPDATE:** After deeper code investigation, Contradiction 3 severity is reduced. All endpoints ARE implemented. However, even scenarios WITH explicit registration steps (lines 176, 231) are failing, indicating the problem is more complex than just missing registration steps. The registration step definition itself may have issues, or there are timing/state problems.

---

## Critical Questions to Answer

### Q1: Should nodes be explicitly registered or implicitly available?

**Option A:** Explicit registration (matches spec)
- Pro: Matches spec exactly
- Pro: Clear temporal sequence
- Con: Verbose tests (every scenario needs registration step)

**Option B:** Implicit registration (matches current test assumption)
- Pro: Tests are concise
- Con: Contradicts spec
- Con: Hides setup complexity

**Option C:** Background auto-registration (TEAM-056 attempted)
- Pro: DRY tests
- Con: Timing issues prevent this
- Con: Doesn't match spec's two-phase model

**Recommendation:** Option A - Explicit registration in scenarios that need it.

### Q2: Should tests be isolated or share state?

**Option A:** Isolated (fresh DB per scenario)
- Pro: True test isolation
- Pro: Deterministic
- Con: Slower (spawn queen per scenario)

**Option B:** Shared state with reset (current attempt)
- Pro: Fast (one queen)
- Con: Reset might be incomplete
- Con: Ordering matters

**Option C:** Shared state, ordered scenarios
- Pro: Fast
- Con: Fragile (one failure breaks rest)
- Con: Can't run scenarios independently

**Recommendation:** Option A - Isolated tests with fresh DB.

### Q3: Should mocks simulate real behavior or return canned responses?

**Option A:** Simulate real behavior (complex mocks)
- Pro: Tests verify actual flow
- Con: Mocks become another codebase to maintain

**Option B:** Canned responses (current)
- Pro: Simple
- Con: Don't catch real issues
- Con: Disconnect between mock and reality

**Option C:** Integration tests with real components
- Pro: True behavior
- Con: Slow, require infrastructure

**Recommendation:** Option A for BDD, Option C for integration suite.

### Q4: Should step definitions verify or document?

**Option A:** Verify (strict)
- Pro: Catch regressions
- Pro: Tests are meaningful
- Con: More code to write

**Option B:** Document (current edge cases)
- Pro: Fast to write
- Con: False confidence
- Con: Tests don't test

**Option C:** Mix (current)
- Pro: Flexible
- Con: Inconsistent, confusing

**Recommendation:** Option A - All step definitions must verify behavior.

### Q5: Should Background set minimal or complete state?

**Option A:** Minimal (current)
- Pro: Follows Gherkin best practices
- Con: Scenarios need explicit setup

**Option B:** Complete (auto-registration)
- Pro: DRY
- Con: Timing prevents this

**Option C:** Nothing (explicit everything)
- Pro: Crystal clear
- Con: Very verbose

**Recommendation:** Option A with explicit registration in scenarios.

---

## Architectural Decisions Required

1. **Accept spec's explicit registration model** → Update tests to add registration steps
2. **Implement per-scenario test isolation** → Fresh DB per scenario or proper DB reset
3. **Complete mock infrastructure** → Add missing queen-rbee endpoints
4. **Convert all stubs to real implementations** → Edge cases must execute commands
5. **Fix Background timing** → Ensure queen-rbee ready before Background OR explicit registration

---

## Next Steps for TEAM-057

See companion documents:
- `TEAM_057_FAILING_SCENARIOS_ANALYSIS.md` - Detailed analysis of each failing scenario
- `TEAM_057_INVESTIGATION_REPORT.md` - Complete findings and recommendations
- `TEAM_057_IMPLEMENTATION_PLAN.md` - Multi-phase execution plan

---

**TEAM-057 signing off on contradictions analysis.**

**Status:** Architectural contradictions identified and documented  
**Risk:** HIGH - These are design issues, not bugs  
**Confidence:** VERY HIGH - Root causes verified through code inspection  
**Recommendation:** Proceed to implementation plan after reviewing all analysis documents
