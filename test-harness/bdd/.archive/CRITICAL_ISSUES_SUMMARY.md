# 🚨 CRITICAL ISSUES SUMMARY - REQUIRES DECISION

**5 CRITICAL architectural contradictions found that will cause problems:**

---

## Issue #1: Registry Confusion ⚠️ MOST CRITICAL

**Problem:** Concurrency tests don't specify WHICH registry they test.

**Two registries exist:**
```
queen-rbee (global)          rbee-hive (local)
└─ All workers across        └─ Only workers spawned
   all nodes                    by THIS instance
└─ In-memory                 └─ Per-instance
└─ For routing               └─ For lifecycle
```

**200-concurrency-scenarios.feature says:**
- Background: "queen-rbee is running" 
- Scenario: "3 rbee-hive instances register worker"
- Then: "worker appears in registry"

**Which registry?** 🤔

**Current implementation (TEAM-080):** Tests queen-rbee registry
**Architectural reality:** Should test BOTH separately

**Decision needed:** Rewrite 200-concurrency-scenarios.feature to split:
- `200a-queen-rbee-concurrency.feature` (global registry races)
- `200b-rbee-hive-concurrency.feature` (local registry races)

---

## Issue #2: Duplicate Cancellation Features 🔴

**Problem:** Exact same scenarios in TWO files:

```
130-inference-execution.feature      220-request-cancellation.feature
├─ Gap-G12a (Ctrl+C)                ├─ Gap-G12a (Ctrl+C)
├─ Gap-G12b (disconnect)            ├─ Gap-G12b (disconnect)
└─ Gap-G12c (DELETE)                └─ Gap-G12c (DELETE)
                                    ├─ Plus 4 more scenarios
```

**Same scenario IDs in both files = Cucumber conflict!**

**Decision needed:** 
- Option A: Delete 220-request-cancellation.feature (keep all in 130)
- Option B: Move all to 220-request-cancellation.feature (remove from 130)

**Recommendation:** Option A (cancellation is part of inference flow)

---

## Issue #3: Slot Management Owner Unclear 🔴

**Problem:** Who owns slot allocation logic?

```
Worker         rbee-hive         queen-rbee
├─ Actual      ├─ Tracks         ├─ Caches
│  slots       │  worker          │  slot info
├─ Allocates   │  status          ├─ Routes based
│  atomically  └─ Queries         │  on cache
└─ Returns        worker          └─ Eventually
   503                               consistent
```

**200-concurrency-scenarios.feature tests slots at queen-rbee level.**
**This is WRONG!** Slots are allocated AT THE WORKER.

**Decision needed:** Rewrite Gap-C4 to test at worker level:
```gherkin
Feature: Worker Slot Allocation
  Scenario: Worker handles concurrent slot requests atomically
    Given worker has 1 slot available
    When 2 inference requests arrive simultaneously
    Then worker allocates to ONE request (atomic CAS)
    And returns 503 to the other
```

---

## Issue #4: Catalog Concurrency Impossible 🔴

**Problem:** Tests concurrent catalog INSERT conflicts.

**But:** Each rbee-hive has **separate SQLite file**:
```
workstation-1: ~/.rbee/models.db
workstation-2: ~/.rbee/models.db  (different file!)
```

**No shared database = No concurrent INSERT conflicts possible!**

**Current scenario is testing an impossible situation.**

**Decision needed:**
- Option A: DELETE Gap-C3 (tests non-existent architecture)
- Option B: Mark @future and require PostgreSQL shared catalog

**Recommendation:** Option A - delete it

---

## Issue #5: Failover Requires State Machine 🔴

**Problem:** 210-failure-recovery.feature tests automatic request retry:

```gherkin
When worker-001 crashes
Then request "req-123" is retried on worker-002
And user receives result without manual intervention
```

**This requires:**
- ❌ queen-rbee tracking active requests (doesn't exist)
- ❌ Request state persistence (doesn't exist)
- ❌ Automatic retry logic (doesn't exist)
- ❌ Request ID mapping (doesn't exist)

**Current architecture:** rbee-keeper → worker (direct, no tracking)

**Decision needed:**
- Option A: Rewrite as "Worker crash detection and cleanup" (realistic)
- Option B: Mark @future @v2.0 and implement state machine later

**Recommendation:** Option A - test what exists now

---

## Quick Decision Matrix

| Issue | Action | Effort | Impact if not fixed |
|-------|--------|--------|---------------------|
| #1 Registry confusion | Split into 2 files | 2h | Tests wrong component |
| #2 Duplicate cancellation | Delete 220-*.feature | 5min | Cucumber errors |
| #3 Slot management | Rewrite scenario | 1h | Tests wrong layer |
| #4 Catalog concurrency | Delete scenario | 2min | Tests impossible case |
| #5 Failover unrealistic | Rewrite scenario | 1h | False expectations |

**Total fix time: ~4 hours**

---

## Recommended Immediate Actions

### 1. DELETE (5 minutes)
```bash
rm tests/features/220-request-cancellation.feature
# Keep cancellation in 130-inference-execution.feature
```

### 2. MARK FUTURE (2 minutes)
In 210-failure-recovery.feature:
```gherkin
@future @v2.0 @requires-state-machine
Scenario: Gap-F1 - Automatic failover
  # Requires: Request tracking in queen-rbee
```

In 200-concurrency-scenarios.feature:
```gherkin
@future @requires-shared-database
Scenario: Gap-C3 - Catalog concurrent INSERT
  # Requires: PostgreSQL shared catalog
```

### 3. REWRITE (4 hours)
- Split 200-concurrency into registry-specific files
- Move slot allocation to worker-level test
- Clarify failover as "detection" not "retry"

---

## Path Forward

**Path A: Ship with @future tags (30 minutes)**
- Mark unrealistic scenarios as @future
- Delete 220-request-cancellation.feature
- Document known issues
- Ship v1.0, fix in v1.1

**Path B: Fix architecture now (4 hours)**
- Split concurrency tests per registry
- Rewrite slot/failover scenarios
- Clean test suite
- Ship v1.0 with correct architecture

**Your choice!** 🎯

---

**Created by:** TEAM-080  
**Date:** 2025-10-11  
**Urgency:** HIGH - Decisions needed before continuing
