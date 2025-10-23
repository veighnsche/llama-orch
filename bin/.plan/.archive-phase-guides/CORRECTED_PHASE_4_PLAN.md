# CORRECTED Phase 4 Plan - After Proper Dead Code Audit

**Date:** Oct 22, 2025  
**Reason:** Proper usage audit revealed 58% of shared crates are dead code  
**Method:** Grepped for actual imports in product code (not just workspace registration)

---

## 🔍 Audit Results

**Total Shared Crates:** 19  
**Active (used in product code):** 9 crates (47%)  
**Dead Code:** 10 crates (53%)

---

## ✅ ACTIVE SHARED CRATES (9 total)

1. ✅ **observability-narration-core** - 23 imports (core narration)
2. ✅ **narration-macros** - NARRATE macro used 59 times
3. ✅ **job-registry** - 6 imports (job tracking)
4. ✅ **rbee-config** - 5 imports (configuration)
5. ✅ **rbee-heartbeat** - 4 imports (heartbeat system)
6. ✅ **rbee-operations** - 3 imports (operation types)
7. ✅ **timeout-enforcer** - 2 imports (timeout with SSE)
8. ✅ **daemon-lifecycle** - 1 import (process lifecycle)
9. ✅ **auth-min** - 1 import (llm-worker auth middleware)

---

## ❌ DEAD CODE (10 crates)

### Stubs (Never Implemented - 3 crates)
1. ❌ **rbee-http-client** - 390 bytes, "TODO: Implement"
2. ❌ **sse-relay** - 209 bytes, "Placeholder"
3. ❌ **rbee-types** - 687 bytes, duplicate types

### Implemented But Unused (7 crates)
4. ❌ **audit-logging** - 0 imports
5. ❌ **auto-update** - 0 imports
6. ❌ **deadline-propagation** - 0 imports
7. ❌ **input-validation** - 0 imports
8. ❌ **jwt-guardian** - 0 imports
9. ❌ **model-catalog** (shared) - 0 imports (duplicate of hive crate)
10. ❌ **secrets-management** - 0 imports

---

## 📋 REVISED Phase 4 Team Assignments

### Original Plan (9 teams, 19 crates)
- TEAM-230 through TEAM-238
- **Problem:** 10 crates are dead code

### Corrected Plan (5 teams, 9 crates)

**TEAM-230: Narration System**
- Components: `narration-core` + `narration-macros`
- Complexity: High
- Status: ✅ ACTIVE (23 + 59 uses)

**TEAM-231: Lifecycle & Config**
- Components: `daemon-lifecycle` + `rbee-config`
- Complexity: Medium
- Status: ✅ ACTIVE (1 + 5 uses)

**TEAM-232: Operations & Registry**
- Components: `rbee-operations` + `job-registry`
- Complexity: Medium
- Status: ✅ ACTIVE (3 + 6 uses)

**TEAM-233: Heartbeat & Timeout**
- Components: `rbee-heartbeat` + `timeout-enforcer`
- Complexity: Medium
- Status: ✅ ACTIVE (4 + 2 uses)

**TEAM-234: Auth**
- Components: `auth-min`
- Complexity: Low
- Status: ✅ ACTIVE (1 use in llm-worker)

---

## 📊 Updated Team Counts

### Discovery Phases

| Phase | Teams | Components | Change |
|-------|-------|------------|--------|
| 1 | 4 | Main binaries | No change |
| 2 | 3 | Queen crates | No change |
| 3 | 7 | Hive crates | No change |
| 4 | **5** | Shared crates | **Was 9, now 5** |
| 5 | 4 | Integration flows | No change |
| **Total** | **23** | **All ACTIVE code** | **Was 27, now 23** |

### Phase 5 Team Numbers (Shifted)

**Before:** TEAM-239 through TEAM-242  
**After:** TEAM-235 through TEAM-238 (shifted back by 4)

**TEAM-235:** keeper ↔ queen integration  
**TEAM-236:** queen ↔ hive integration  
**TEAM-237:** hive ↔ worker integration  
**TEAM-238:** End-to-end inference flows

### Test Planning & Implementation

**Phase 6:** TEAM-239+ (was TEAM-243+)  
**Phase 7:** TEAM-247+ (was TEAM-251+)

---

## 🗑️ Dead Code Cleanup Actions

### Immediate Deletions (Stubs)

```bash
# Remove stub crates
rm -rf bin/99_shared_crates/rbee-http-client
rm -rf bin/99_shared_crates/sse-relay
rm -rf bin/99_shared_crates/rbee-types

# Remove from Cargo.toml workspace members
# (Lines 72, 67, 73-74)
```

### Archive for Post-1.0 (Implemented but Unused)

```bash
# Move to archive
mkdir -p bin/99_shared_crates/.archive-unused
mv bin/99_shared_crates/audit-logging bin/99_shared_crates/.archive-unused/
mv bin/99_shared_crates/auto-update bin/99_shared_crates/.archive-unused/
mv bin/99_shared_crates/deadline-propagation bin/99_shared_crates/.archive-unused/
mv bin/99_shared_crates/input-validation bin/99_shared_crates/.archive-unused/
mv bin/99_shared_crates/jwt-guardian bin/99_shared_crates/.archive-unused/
mv bin/99_shared_crates/model-catalog bin/99_shared_crates/.archive-unused/
mv bin/99_shared_crates/secrets-management bin/99_shared_crates/.archive-unused/
```

---

## 📝 Documents Requiring Updates

1. ✅ `.plan/PHASE_4_GUIDES.md` - Reduce from 9 to 5 teams
2. ✅ `.plan/PHASE_5_GUIDES.md` - Shift team numbers (239-242 → 235-238)
3. ✅ `.plan/BEHAVIOR_DISCOVERY_MASTER_PLAN.md` - Update counts (27 → 23)
4. ✅ `.plan/QUICK_REFERENCE_TESTING_PLAN.md` - Update counts
5. ✅ `.plan/00_INDEX.md` - Update all tables
6. ✅ `.plan/README.md` - Update summary

---

## ✅ Verification Evidence

### Grep Results (Product Code Only)

```bash
# Active crates (9)
observability_narration_core: 23 uses
narration_macros: 59 NARRATE macro uses
job_registry: 6 uses
rbee_config: 5 uses
rbee_heartbeat: 4 uses
rbee_operations: 3 uses
timeout_enforcer: 2 uses
daemon_lifecycle: 1 use
auth_min: 1 use (llm-worker/src/http/middleware/auth.rs)

# Dead crates (10)
audit_logging: 0 uses
auto_update: 0 uses
deadline_propagation: 0 uses
input_validation: 0 uses
jwt_guardian: 0 uses
model_catalog: 0 uses
rbee_http_client: 0 uses (stub)
rbee_types: 0 uses (duplicate)
secrets_management: 0 uses
sse_relay: 0 uses (stub)
```

---

## 🎯 Final Team Assignments

### Phase 4 (5 teams)
- TEAM-230: narration-core + narration-macros
- TEAM-231: daemon-lifecycle + rbee-config
- TEAM-232: rbee-operations + job-registry
- TEAM-233: rbee-heartbeat + timeout-enforcer
- TEAM-234: auth-min

### Phase 5 (4 teams)
- TEAM-235: keeper ↔ queen
- TEAM-236: queen ↔ hive
- TEAM-237: hive ↔ worker
- TEAM-238: e2e inference

### Phase 6 (Test Planning)
- TEAM-239+

### Phase 7 (Test Implementation)
- TEAM-247+

---

## 📚 Related Documents

- **Proper Audit:** `.plan/PROPER_DEAD_CODE_AUDIT.md`
- **Exit Interview:** `bin/99_shared_crates/narration-core/EXIT_INTERVIEW_DEAD_CODE.md`
- **Lesson:** Don't trust workspace registration, verify actual usage

---

**Status:** ✅ CORRECTED  
**Method:** Actual grep for imports in product code  
**Result:** 10 dead crates excluded, 9 active crates remain  
**Savings:** 4 fewer teams, 4 fewer behavior inventories

---

**Signed:** AI Assistant (Actually looked at the code this time)  
**Evidence:** Grep results provided, not assumptions
