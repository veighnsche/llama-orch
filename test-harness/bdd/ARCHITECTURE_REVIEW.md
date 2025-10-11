# 🚨 CRITICAL: BDD Architecture Review & Contradictions

**Date:** 2025-10-11  
**Reviewer:** TEAM-080  
**Scope:** All 20 feature files, especially NEW files (200-230)

---

## Executive Summary

**🔴 CRITICAL ISSUES FOUND: 5**  
**🟡 MODERATE ISSUES: 8**  
**🟢 MINOR ISSUES: 3**

### Key Finding: Registry Confusion

**The biggest architectural flaw is TWO separate registries with unclear boundaries:**

1. **queen-rbee WorkerRegistry** (050-queen-rbee-worker-registry.feature)
   - Global orchestrator registry
   - In-memory, tracks ALL workers across ALL rbee-hive instances
   - Used for routing and global coordination

2. **rbee-hive WorkerRegistry** (060-rbee-hive-worker-registry.feature)
   - Local pool manager registry
   - Tracks only workers spawned BY THIS rbee-hive instance
   - Used for local lifecycle management

**Problem:** NEW concurrency tests (200-concurrency-scenarios.feature) don't specify WHICH registry they test!

---

## 🔴 CRITICAL ISSUE #1: Registry Ambiguity in Concurrency Tests

### Problem

**200-concurrency-scenarios.feature** line 19:
```gherkin
And queen-rbee is running at "http://localhost:8080"
```

But the scenarios test:
- Line 24: "all 3 instances register worker 'worker-001' simultaneously"
- Line 28: "worker-001 appears exactly once in registry"

**Which registry?** queen-rbee's global registry or rbee-hive's local registry?

### Current Implementation (TEAM-080)

We implemented using `queen-rbee::WorkerRegistry` BUT:
- Background says "queen-rbee is running" (line 19)
- Scenarios mention "rbee-hive instances" (line 23)
- This is architecturally confused!

### Impact

- Tests don't match real architecture
- Step definitions test wrong component
- Concurrency bugs in rbee-hive won't be caught

### Recommendation

**CHOOSE ONE:**

**Option A: Test queen-rbee registry (current implementation)**
```gherkin
Background:
  Given queen-rbee is running at "http://localhost:8080"
  And 3 rbee-hive instances report to queen-rbee

Scenario: Gap-C1 - Concurrent worker registration at queen-rbee
  Given 3 rbee-hive instances are running
  When all 3 instances POST to "http://localhost:8080/v1/workers/register"
  Then queen-rbee's global registry has exactly one worker-001
```

**Option B: Test rbee-hive registry (architecturally correct)**
```gherkin
Background:
  Given rbee-hive is running at "http://localhost:9200"
  And 3 worker processes are starting

Scenario: Gap-C1 - Concurrent worker registration at rbee-hive
  When 3 workers register simultaneously via callback
  Then rbee-hive's local registry has exactly 3 workers
```

---

## 🔴 CRITICAL ISSUE #2: Slot Management Owner Unclear

### Problem

**200-concurrency-scenarios.feature** line 52-57:
```gherkin
Scenario: Gap-C4 - Slot allocation race condition
  Given worker has 4 slots total
  And 3 slots are busy
  When 2 requests arrive simultaneously for last slot
  Then only one request gets the slot
  And the other receives "ALL_SLOTS_BUSY"
```

**Who manages slots?**
- queen-rbee tracks `slots_available` in global registry (050, line 46)
- rbee-hive tracks worker state (060)
- Worker itself manages actual slots

### Current Implementation

TEAM-080 tested slot allocation in `queen-rbee::WorkerRegistry.slots_available` field.

**But this is WRONG!** According to architecture:
- Worker owns slot state
- rbee-hive queries worker for availability
- queen-rbee caches slot info but doesn't allocate

### Impact

- Tests don't reflect real slot allocation logic
- Race conditions in worker won't be caught
- False sense of security

### Recommendation

**Split into 3 separate scenarios:**

1. **Worker-level slot allocation** (where it actually happens)
```gherkin
Feature: Worker Slot Management
  Scenario: Worker handles concurrent slot requests
    Given worker has 1 slot available
    When 2 requests arrive simultaneously
    Then worker allocates slot to one request atomically
    And worker returns 503 to the other
```

2. **rbee-hive slot tracking** (sync with worker)
```gherkin
Feature: rbee-hive Worker Tracking
  Scenario: rbee-hive tracks worker slots
    Given worker reports slots_available=1
    When rbee-hive queries worker status
    Then rbee-hive updates cached slot count
```

3. **queen-rbee slot cache** (eventual consistency)
```gherkin
Feature: queen-rbee Worker Registry
  Scenario: queen-rbee caches slot availability
    Given worker-001 reports slots_available=0
    When heartbeat updates queen-rbee
    Then queen-rbee marks worker as busy
```

---

## 🔴 CRITICAL ISSUE #3: Cancellation Feature Duplication

### Problem

**TWO features cover cancellation:**

**130-inference-execution.feature** lines 189-233:
- `Gap-G12a - Client cancellation with Ctrl+C`
- `Gap-G12b - Client disconnects`
- `Gap-G12c - Explicit cancellation endpoint`

**220-request-cancellation.feature** lines 23-89:
- `Gap-G12a - User cancels with Ctrl+C`
- `Gap-G12b - Client disconnects during streaming`
- `Gap-G12c - Explicit cancellation via DELETE endpoint`
- Plus 4 more scenarios

**EXACT SAME scenario IDs (Gap-G12a, G12b, G12c)!**

### Impact

- Duplicate test effort
- Maintenance nightmare (update both files?)
- Cucumber will have duplicate step definitions
- Confusion about which file owns cancellation

### Recommendation

**CONSOLIDATE:**

**Option A: Keep in 130-inference-execution.feature (better)**
- Cancellation is part of inference flow
- Keep 3 basic scenarios in 130
- Delete 220-request-cancellation.feature

**Option B: Move all to 220-request-cancellation.feature**
- Separate feature for cancellation logic
- Move all 7 scenarios to 220
- Remove from 130-inference-execution.feature

**DO NOT keep both with duplicate IDs!**

---

## 🔴 CRITICAL ISSUE #4: Model Catalog Concurrency Mismatch

### Problem

**200-concurrency-scenarios.feature** line 42-48:
```gherkin
Scenario: Gap-C3 - Concurrent model catalog registration
  Given 3 rbee-hive instances are downloading "tinyllama-q4"
  When all 3 complete download simultaneously
  And all 3 attempt to register in catalog
  Then only one INSERT succeeds
```

**But model-catalog is per-rbee-hive instance, not shared!**

According to architecture:
- Each rbee-hive has its own SQLite catalog at `~/.rbee/models.db`
- No shared catalog across nodes
- No concurrent INSERT conflicts possible

### Current Implementation

TEAM-080 stubbed this but didn't wire it (correctly, because it doesn't make sense architecturally).

### Impact

- Tests an impossible scenario
- Misleading about actual architecture
- Wastes implementation time

### Recommendation

**DELETE or REWRITE:**

**If testing shared catalog (future feature):**
```gherkin
Scenario: Gap-C3 - Concurrent catalog INSERT with shared database
  Given queen-rbee uses shared PostgreSQL catalog
  And 3 rbee-hive instances download same model
  When all 3 attempt INSERT simultaneously
  Then PostgreSQL UNIQUE constraint prevents duplicates
  And only one INSERT succeeds
```

**If testing current architecture:**
```gherkin
Scenario: Concurrent downloads are independent
  Given rbee-hive-1 downloads "tinyllama-q4"
  And rbee-hive-2 downloads "tinyllama-q4"
  Then each rbee-hive maintains separate catalog
  And both catalogs have "tinyllama-q4" entry
  And no conflicts occur (different databases)
```

---

## 🔴 CRITICAL ISSUE #5: Worker Crash Failover Unrealistic

### Problem

**210-failure-recovery.feature** line 22-29:
```gherkin
Scenario: Gap-F1 - Worker crashes during request processing
  Given worker-001 is processing inference request "req-123"
  And worker-002 is available with same model
  When worker-001 crashes unexpectedly
  Then queen-rbee detects crash within 5 seconds
  And request "req-123" is retried on worker-002
  And user receives result without manual intervention
```

**This requires state machine persistence that doesn't exist!**

For this to work:
- queen-rbee must track request "req-123" → worker-001 mapping
- queen-rbee must detect worker crash
- queen-rbee must re-route to worker-002
- Request state must be preserved

**But current architecture:**
- rbee-keeper sends request directly to worker
- No queen-rbee involvement in active requests
- No request tracking
- No state persistence

### Impact

- Tests feature that doesn't exist
- Sets false expectations
- Requires major architectural change

### Recommendation

**REWRITE for current architecture:**

```gherkin
Scenario: Worker crash detection and registry cleanup
  Given worker-001 is registered in queen-rbee
  And heartbeat interval is 30s
  When worker-001 crashes
  Then heartbeat timeout occurs after 120s
  And queen-rbee removes worker-001 from registry
  And subsequent requests are NOT routed to worker-001
  # User must manually retry
```

**OR mark as future feature:**
```gherkin
@future @requires-state-machine
Scenario: Automatic failover with request tracking (v2.0)
  # Requires: Request state machine in queen-rbee
  # Requires: Persistent request log
  # Requires: Request ID tracking
```

---

## 🟡 MODERATE ISSUE #6: Heartbeat Timeout Inconsistency

### Problem

**Multiple heartbeat timeout values across features:**

**050-queen-rbee-worker-registry.feature** line 85:
```gherkin
Then queen-rbee marks worker-002 as stale (no heartbeat for >120s)
```

**210-failure-recovery.feature** line 65:
```gherkin
When heartbeat times out (>120s)
```

**130-inference-execution.feature** line 112:
```gherkin
And no tokens generated for 60 seconds
Then rbee-keeper detects stall timeout
```

**Different timeouts:**
- Heartbeat: 120s
- Token generation stall: 60s
- Connection timeout: 10s (060, line 59)

### Recommendation

**Standardize and document:**

```markdown
# Timeout Policy

- **Heartbeat interval:** 30s
- **Heartbeat timeout:** 120s (4 missed heartbeats)
- **Token stall timeout:** 60s (generation stuck)
- **HTTP connection timeout:** 10s
- **Retry backoff:** 1s, 2s, 4s (exponential)
```

Create `timeout-policy.md` and reference in all features.

---

## 🟡 MODERATE ISSUE #7: State Transition Validation Missing

### Problem

**Multiple features mention state transitions but don't validate:**

**200-concurrency-scenarios.feature** line 78:
```gherkin
Given worker-001 is transitioning from "idle" to "busy"
```

**130-inference-execution.feature** line 49:
```gherkin
And the worker transitions from "idle" to "busy" to "idle"
```

**But no validation of invalid transitions:**
- Can worker go directly from "loading" to "idle"?
- Can worker skip "busy" and go "loading" → "idle"?
- What if heartbeat updates state during transition?

### Recommendation

**Add state machine validation scenarios:**

```gherkin
Feature: Worker State Machine
  @state-machine
  Scenario Outline: Valid state transitions
    Given worker is in state "<from>"
    When worker transitions to "<to>"
    Then transition is <allowed>

    Examples:
      | from    | to      | allowed |
      | idle    | loading | valid   |
      | loading | idle    | valid   |
      | idle    | busy    | valid   |
      | busy    | idle    | valid   |
      | loading | busy    | invalid |
      | busy    | loading | invalid |

  Scenario: Heartbeat during transition is queued
    Given worker is transitioning from "idle" to "busy" (atomic operation)
    When heartbeat arrives mid-transition
    Then heartbeat update is queued
    And processed after transition completes
```

---

## 🟡 MODERATE ISSUE #8: Resource Management Scenarios Missing Context

### Problem

**230-resource-management.feature** tests dynamic resource monitoring BUT:
- No integration with worker lifecycle
- No connection to preflight checks (090-worker-resource-preflight.feature)
- Duplicate concerns with 090

**090-worker-resource-preflight.feature:**
- RAM check (line 21-27)
- VRAM check (line 75-91)
- Disk space check (line 130-145)

**230-resource-management.feature:**
- Multi-GPU selection (line 22-29)
- Dynamic RAM monitoring (line 32-38)
- GPU temperature monitoring (line 41-47)

### Recommendation

**Merge or clearly separate:**

**090 = Pre-flight (before worker starts)**
```gherkin
Feature: Worker Preflight Checks
  Scenario: Static resource validation
    When rbee-hive validates resources BEFORE starting worker
    Then checks are one-time snapshots
```

**230 = Runtime (during worker operation)**
```gherkin
Feature: Runtime Resource Monitoring
  Scenario: Dynamic resource monitoring
    When worker is running
    Then rbee-hive monitors resources continuously
    And adapts to changing conditions
```

Add cross-references:
```gherkin
# See also: 090-worker-resource-preflight.feature for startup checks
```

---

## 🟡 MODERATE ISSUE #9: Background Topology Duplication

### Problem

**EVERY feature has identical Background:**

```gherkin
Background:
  Given the following topology:
    | node        | hostname              | components                      | capabilities         |
    | blep        | blep.home.arpa        | rbee-keeper, queen-rbee         | cpu                  |
    | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee      | cuda:0, cuda:1, cpu  |
  And I am on node "blep"
  And queen-rbee is running at "http://localhost:8080"
```

**Repeated in 20 files!**

### Recommendation

**Extract to shared file:**

```gherkin
# tests/features/support/common-background.feature
Background: Standard Two-Node Topology
  Given the following topology:
    | node        | hostname              | components                      | capabilities         |
    | blep        | blep.home.arpa        | rbee-keeper, queen-rbee         | cpu                  |
    | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee      | cuda:0, cuda:1, cpu  |
  And I am on node "blep"
  And queen-rbee is running at "http://localhost:8080"
```

**OR use tags:**
```gherkin
@standard-topology
Feature: Concurrency Scenarios
  # Background loaded via tag
```

---

## 🟡 MODERATE ISSUE #10: Split-Brain Scenario Impractical

### Problem

**210-failure-recovery.feature** line 42-50:
```gherkin
Scenario: Gap-F3 - Registry split-brain resolution
  Given queen-rbee-1 has workers [A, B]
  And queen-rbee-2 has workers [C, D]
  When network partition heals
```

**But architecture only supports ONE queen-rbee instance!**

No HA setup, no federation, no split-brain possible with current design.

### Recommendation

**Mark as future or delete:**

```gherkin
@future @v2.0 @requires-ha
Scenario: Gap-F3 - Registry split-brain resolution (HA Setup)
  # Requires: High-availability queen-rbee deployment
  # Requires: Raft consensus or similar
  # Status: Not in v1.0 roadmap
```

**OR test simpler scenario:**
```gherkin
Scenario: Single queen-rbee restart preserves workers
  Given queen-rbee has workers [A, B, C, D]
  When queen-rbee restarts
  Then workers re-register via heartbeat
  And registry is rebuilt from heartbeats
```

---

## 🟢 MINOR ISSUE #11-13

### 11. Error Code Standardization

Different error codes across features:
- `ALL_SLOTS_BUSY` (130, line 63)
- `WORKER_ALREADY_REGISTERED` (200, line 26)
- `WORKER_ALREADY_BUSY` (200, line 37)

**Recommendation:** Create error code registry in `docs/error-codes.md`

### 12. Request ID Format Inconsistency

- `req-123` (210, line 23)
- `req-456` (220, line 47)
- `<request_id>` (130, line 195)

**Recommendation:** Standardize as UUIDs: `req-550e8400-e29b-41d4-a716-446655440000`

### 13. Port Number Inconsistency

- queen-rbee: `8080` (most features)
- rbee-hive: `9200` (060, line 23) vs `8081` (some scenarios)
- worker: `8081` (130, line 25)

**Recommendation:** Standardize in `docs/port-assignments.md`

---

## 📋 Action Plan

### IMMEDIATE (Block v1.0)

1. ✅ **Fix Registry Confusion** - Clarify which registry each scenario tests
2. ✅ **Consolidate Cancellation** - Remove duplicates, keep one feature
3. ✅ **Delete Invalid Scenarios** - Remove split-brain, catalog concurrency
4. ✅ **Fix Slot Management** - Move to worker-level tests

### SOON (Block v1.1)

5. **Standardize Timeouts** - Document timeout policy
6. **Add State Machine Tests** - Validate transitions
7. **Extract Common Background** - DRY principle
8. **Merge Resource Features** - 090 vs 230 clarification

### LATER (Nice to have)

9. **Error Code Registry**
10. **Request ID Standard**
11. **Port Assignment Doc**

---

## Recommended File Changes

### DELETE

- **220-request-cancellation.feature** (duplicate of 130)

### MAJOR REWRITE

- **200-concurrency-scenarios.feature** - Specify which registry
- **210-failure-recovery.feature** - Remove HA scenarios

### MODERATE CHANGES

- **230-resource-management.feature** - Clarify vs 090
- **050-queen-rbee-worker-registry.feature** - Add concurrency notes
- **060-rbee-hive-worker-registry.feature** - Clarify scope

### CREATE NEW

- `docs/registry-architecture.md` - Explain two registries
- `docs/timeout-policy.md` - Standard timeouts
- `docs/error-codes.md` - Error code registry
- `tests/features/240-worker-state-machine.feature` - State transitions

---

## Conclusion

**The BDD suite has grown organically without architectural clarity.**

**Two paths forward:**

**Path A: Quick Fix (2-4 hours)**
- Delete 220-request-cancellation.feature
- Mark @future on HA scenarios
- Add comments clarifying registry confusion
- Ship v1.0 with known tech debt

**Path B: Proper Fix (1-2 days)**
- Rewrite 200-concurrency-scenarios.feature per registry
- Split slot management scenarios
- Consolidate resource features
- Document architecture
- Clean, maintainable test suite

**Recommendation:** Path B - Fix it now while codebase is small.

---

**Created by:** TEAM-080  
**Date:** 2025-10-11  
**Status:** CRITICAL - Review Required Before v1.0
