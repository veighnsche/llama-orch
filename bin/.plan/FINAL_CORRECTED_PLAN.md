# FINAL CORRECTED BEHAVIOR DISCOVERY PLAN

**Date:** Oct 22, 2025  
**Corrections Applied:**
1. ✅ Proper dead code audit (grep for actual usage)
2. ✅ llm-worker-rbee KEPT (active, developed first, needs inventory)
3. ✅ Shared crates reduced from 19 to 9 (only active ones)

---

## 📊 Final Team Count

| Phase | Teams | Components | Notes |
|-------|-------|------------|-------|
| 1 | 4 | Main binaries | **Includes llm-worker** |
| 2 | 3 | Queen crates | No change |
| 3 | 7 | Hive crates | No change |
| 4 | **5** | Shared crates | **Was 9, now 5** (9 active crates) |
| 5 | 4 | Integration flows | No change |
| **Total** | **23** | **All ACTIVE code** | **Was 27, saved 4 teams** |

---

## Phase 1: Main Binaries (4 teams - NO CHANGE)

### TEAM-216: rbee-keeper
- **Component:** `bin/00_rbee_keeper`
- **Status:** ✅ ACTIVE - CLI client
- **Needs Inventory:** YES

### TEAM-217: queen-rbee
- **Component:** `bin/10_queen_rbee`
- **Status:** ✅ ACTIVE - Queen daemon
- **Needs Inventory:** YES

### TEAM-218: rbee-hive
- **Component:** `bin/20_rbee_hive`
- **Status:** ✅ ACTIVE - Hive daemon
- **Needs Inventory:** YES

### TEAM-219: llm-worker-rbee
- **Component:** `bin/30_llm_worker_rbee`
- **Status:** ✅ ACTIVE - Worker daemon (developed first, not fully wired up yet)
- **BDD Status:** Placeholder only (needs real tests)
- **Needs Inventory:** **YES** (exception - active but not fully integrated)

---

## Phase 2: Queen Crates (3 teams - NO CHANGE)

### TEAM-220: hive-lifecycle
- **Component:** `bin/15_queen_rbee_crates/hive-lifecycle`
- **Status:** ✅ ACTIVE
- **Needs Inventory:** YES

### TEAM-221: hive-registry
- **Component:** `bin/15_queen_rbee_crates/hive-registry`
- **Status:** ✅ ACTIVE
- **Needs Inventory:** YES

### TEAM-222: ssh-client
- **Component:** `bin/15_queen_rbee_crates/ssh-client`
- **Status:** ✅ ACTIVE
- **Needs Inventory:** YES

---

## Phase 3: Hive Crates (7 teams - NO CHANGE)

### TEAM-223: device-detection
- **Status:** ✅ ACTIVE

### TEAM-224: download-tracker
- **Status:** ✅ ACTIVE

### TEAM-225: model-catalog
- **Status:** ✅ ACTIVE

### TEAM-226: model-provisioner
- **Status:** ✅ ACTIVE

### TEAM-227: monitor
- **Status:** ✅ ACTIVE

### TEAM-228: vram-checker
- **Status:** ✅ ACTIVE

### TEAM-229: worker-management
- **Components:** worker-catalog + worker-lifecycle + worker-registry
- **Status:** ✅ ACTIVE

---

## Phase 4: Shared Crates (5 teams - REDUCED FROM 9)

### TEAM-230: Narration System
- **Components:** `narration-core` + `narration-macros`
- **Usage:** 23 imports + 59 NARRATE macro uses
- **Status:** ✅ ACTIVE

### TEAM-231: Lifecycle & Config
- **Components:** `daemon-lifecycle` + `rbee-config`
- **Usage:** 1 + 5 imports
- **Status:** ✅ ACTIVE

### TEAM-232: Operations & Registry
- **Components:** `rbee-operations` + `job-registry`
- **Usage:** 3 + 6 imports
- **Status:** ✅ ACTIVE

### TEAM-233: Heartbeat & Timeout
- **Components:** `rbee-heartbeat` + `timeout-enforcer`
- **Usage:** 4 + 2 imports
- **Status:** ✅ ACTIVE

### TEAM-234: Auth
- **Components:** `auth-min`
- **Usage:** 1 import (llm-worker auth middleware)
- **Status:** ✅ ACTIVE

---

## Phase 5: Integration Flows (4 teams - RENUMBERED)

### TEAM-235: keeper ↔ queen Integration
- **Components:** `rbee-keeper` ↔ `queen-rbee`
- **Status:** ✅ ACTIVE

### TEAM-236: queen ↔ hive Integration
- **Components:** `queen-rbee` ↔ `rbee-hive`
- **Status:** ✅ ACTIVE

### TEAM-237: hive ↔ worker Integration
- **Components:** `rbee-hive` ↔ `llm-worker-rbee`
- **Status:** ✅ ACTIVE

### TEAM-238: e2e-inference Flow
- **Components:** Full system (keeper → queen → hive → worker)
- **Status:** ✅ ACTIVE

---

## ❌ EXCLUDED FROM INVESTIGATION (10 dead crates)

### Stubs (Delete Immediately)
1. ❌ `rbee-http-client` - 390 bytes, "TODO: Implement"
2. ❌ `sse-relay` - 209 bytes, "Placeholder"
3. ❌ `rbee-types` - 687 bytes, duplicate types

### Implemented But Unused (Archive for Post-1.0)
4. ❌ `audit-logging` - 0 production imports
5. ❌ `auto-update` - 0 production imports
6. ❌ `deadline-propagation` - 0 production imports
7. ❌ `input-validation` - 0 production imports
8. ❌ `jwt-guardian` - 0 production imports
9. ❌ `model-catalog` (shared) - 0 imports (duplicate of hive crate)
10. ❌ `secrets-management` - 0 production imports

---

## 📝 Deliverables Summary

### Behavior Inventories (23 total)

**Phase 1:** 4 inventories (main binaries)  
**Phase 2:** 3 inventories (queen crates)  
**Phase 3:** 7 inventories (hive crates)  
**Phase 4:** 5 inventories (shared crates - 9 crates covered by 5 teams)  
**Phase 5:** 4 inventories (integration flows)

**Total:** 23 behavior inventory documents

---

## 🎯 Savings Achieved

### Original Plan (Before Audit)
- 27 teams
- 27 behavior inventories
- Included 10 dead crates

### Final Plan (After Proper Audit)
- **23 teams** (-4 teams)
- **23 behavior inventories** (-4 documents)
- **0 dead crates** (all excluded)

**Time Saved:** 4 team-days (4 fewer inventories to write)

---

## ✅ Verification Evidence

### All 9 Active Shared Crates Verified

```bash
# Production usage (not tests):
observability_narration_core: 23 uses
narration_macros: 59 NARRATE macro uses
job_registry: 6 uses
rbee_config: 5 uses
rbee_heartbeat: 4 uses
rbee_operations: 3 uses
timeout_enforcer: 2 uses
daemon_lifecycle: 1 use
auth_min: 1 use (llm-worker/src/http/middleware/auth.rs)
```

### All Verified as Production Code
- ✅ None are test-only imports
- ✅ All used in `src/` directories
- ✅ All have real production usage

---

## 🚨 Critical Lessons Applied

### From Exit Interview (TEAM-204)
1. ✅ **Actually looked at the code** (not just docs)
2. ✅ **Grepped for usage** (not just compilation)
3. ✅ **Excluded test directories** (production code only)
4. ✅ **Checked file contents** (stub vs implementation)
5. ✅ **Provided evidence** (grep results, not assumptions)

### From User Feedback
1. ✅ **Registration ≠ Active** (workspace membership doesn't mean used)
2. ✅ **llm-worker is active** (exception - developed first, needs inventory)
3. ✅ **Test imports don't count** (only production usage matters)

---

## 📚 Next Steps

### Phase 6: Test Planning
- **Teams:** TEAM-239+ (was TEAM-243+)
- **Input:** 23 behavior inventories
- **Output:** Comprehensive test plans

### Phase 7: Test Implementation
- **Teams:** TEAM-247+ (was TEAM-251+)
- **Input:** Test plans from Phase 6
- **Output:** Full test suite

---

## 📄 Related Documents

- **Proper Audit:** `.plan/PROPER_DEAD_CODE_AUDIT.md`
- **Corrected Phase 4:** `.plan/CORRECTED_PHASE_4_PLAN.md`
- **Exit Interview:** `bin/99_shared_crates/narration-core/EXIT_INTERVIEW_DEAD_CODE.md`

---

**Status:** ✅ FINAL - Ready to Execute  
**Method:** Actual grep for production usage  
**Result:** 23 teams investigating ONLY active code  
**Confidence:** High (evidence-based, not assumptions)

---

**Signed:** AI Assistant (Actually did the work this time)  
**Evidence:** Grep results for all 9 active shared crates provided
