# Dead Code Sweep Summary

**Date:** Oct 22, 2025  
**Purpose:** Verify testing plan only covers active codebase (no dead code)

---

## ✅ Verification Complete

Performed comprehensive sweep of `/bin/` to identify dead code and ensure testing plan only covers active workspace members.

---

## 🔍 Dead Code Found

### ❌ bin/99_shared_crates/hive-core
- **Status:** DEPRECATED
- **Evidence:** Has `DEPRECATED.md`, NOT in `Cargo.toml` workspace
- **Replacement:** `bin/99_shared_crates/rbee-types`
- **Action:** Excluded from testing plan

### ❌ bin/99_shared_crates/hive-operations
- **Status:** Empty stub (never implemented)
- **Evidence:** Empty `src/` directory, NO `Cargo.toml`, NOT in workspace
- **Action:** Excluded from testing plan
- **Recommendation:** Delete directory

---

## ✅ Active Crates Verified

### Workspace Members (from root Cargo.toml)

**Main Binaries (4):**
1. ✅ `bin/00_rbee_keeper`
2. ✅ `bin/10_queen_rbee`
3. ✅ `bin/20_rbee_hive`
4. ✅ `bin/30_llm_worker_rbee`

**Queen Crates (3):**
5. ✅ `bin/15_queen_rbee_crates/ssh-client`
6. ✅ `bin/15_queen_rbee_crates/hive-registry`
7. ✅ `bin/15_queen_rbee_crates/hive-lifecycle`

**Hive Crates (7 groups, 9 crates):**
8. ✅ `bin/25_rbee_hive_crates/worker-lifecycle`
9. ✅ `bin/25_rbee_hive_crates/worker-registry`
10. ✅ `bin/25_rbee_hive_crates/model-catalog`
11. ✅ `bin/25_rbee_hive_crates/model-provisioner`
12. ✅ `bin/25_rbee_hive_crates/monitor`
13. ✅ `bin/25_rbee_hive_crates/download-tracker`
14. ✅ `bin/25_rbee_hive_crates/device-detection`
15. ✅ `bin/25_rbee_hive_crates/vram-checker`
16. ✅ `bin/25_rbee_hive_crates/worker-catalog`

**Shared Crates (19):**
17. ✅ `bin/99_shared_crates/audit-logging`
18. ✅ `bin/99_shared_crates/auth-min`
19. ✅ `bin/99_shared_crates/auto-update`
20. ✅ `bin/99_shared_crates/daemon-lifecycle`
21. ✅ `bin/99_shared_crates/deadline-propagation`
22. ✅ `bin/99_shared_crates/heartbeat`
23. ✅ `bin/99_shared_crates/input-validation`
24. ✅ `bin/99_shared_crates/job-registry`
25. ✅ `bin/99_shared_crates/jwt-guardian`
26. ✅ `bin/99_shared_crates/model-catalog` (shared)
27. ✅ `bin/99_shared_crates/narration-core`
28. ✅ `bin/99_shared_crates/narration-macros`
29. ✅ `bin/99_shared_crates/rbee-config`
30. ✅ `bin/99_shared_crates/rbee-http-client`
31. ✅ `bin/99_shared_crates/rbee-operations`
32. ✅ `bin/99_shared_crates/rbee-types`
33. ✅ `bin/99_shared_crates/secrets-management`
34. ✅ `bin/99_shared_crates/sse-relay`
35. ✅ `bin/99_shared_crates/timeout-enforcer`

**Total:** 35 active crates (some with BDD subcrates)

---

## 📝 Plan Updates Made

### Documents Updated (7)

1. **`.plan/PHASE_4_GUIDES.md`**
   - ❌ Removed: `hive-core` from TEAM-237
   - ✅ Added: `timeout-enforcer` to TEAM-237
   - ✅ Added: TEAM-238 for `secrets-management` + `sse-relay` + `model-catalog`
   - Updated team count: 8 → 9

2. **`.plan/PHASE_5_GUIDES.md`**
   - Updated team numbers: TEAM-238→241 became TEAM-239→242
   - (Shifted by 1 due to TEAM-238 addition in Phase 4)

3. **`.plan/BEHAVIOR_DISCOVERY_MASTER_PLAN.md`**
   - Added "Dead Code Exclusions" section
   - Updated Phase 4: 8 teams → 9 teams
   - Updated Phase 5: TEAM-238→241 → TEAM-239→242
   - Updated timeline: 26 teams → 27 teams
   - Updated document index

4. **`.plan/QUICK_REFERENCE_TESTING_PLAN.md`**
   - Updated Phase 4: 8 teams → 9 teams
   - Updated Phase 5: TEAM-238→241 → TEAM-239→242
   - Updated total: 26 inventories → 27 inventories
   - Updated test planning: TEAM-242+ → TEAM-243+

5. **`.plan/00_INDEX.md`**
   - Updated Phase 4 table (added TEAM-238)
   - Updated Phase 5 table (shifted team numbers)
   - Updated team counts throughout
   - Updated test planning team numbers

6. **`.plan/DEAD_CODE_EXCLUSIONS.md`** (NEW)
   - Comprehensive documentation of dead code
   - Verification methodology
   - Active crate listing
   - Cleanup recommendations

7. **`.plan/DEAD_CODE_SWEEP_SUMMARY.md`** (THIS FILE)
   - Summary of verification
   - List of changes made

---

## 📊 Team Assignment Changes

### Phase 4 (Shared Crates)

**Before (8 teams):**
- TEAM-230 through TEAM-237

**After (9 teams):**
- TEAM-230 through TEAM-238

**Changes:**
- **TEAM-237:** `heartbeat` + `auto-update` + ~~`hive-core`~~ → `timeout-enforcer` ✅
- **TEAM-238:** `secrets-management` + `sse-relay` + `model-catalog` (NEW) ✅

### Phase 5 (Integration Flows)

**Before:**
- TEAM-238 through TEAM-241

**After:**
- TEAM-239 through TEAM-242 (shifted by 1)

**No content changes, just renumbering.**

### Phase 6 (Test Planning)

**Before:** TEAM-242+  
**After:** TEAM-243+ (shifted by 1)

### Phase 7 (Test Implementation)

**Before:** TEAM-250+  
**After:** TEAM-251+ (shifted by 1)

---

## 🎯 Coverage Summary

### Total Components to Investigate

| Phase | Teams | Components | Status |
|-------|-------|------------|--------|
| 1 | 4 | Main binaries | ✅ Verified |
| 2 | 3 | Queen crates | ✅ Verified |
| 3 | 7 | Hive crates | ✅ Verified |
| 4 | 9 | Shared crates | ✅ Verified (was 8) |
| 5 | 4 | Integration flows | ✅ Verified |
| **Total** | **27** | **All active code** | **✅ No dead code** |

### Previously Missing Coverage (Now Added)

1. ✅ `secrets-management` - Security-critical crate
2. ✅ `sse-relay` - SSE infrastructure
3. ✅ `model-catalog` (shared) - Model metadata
4. ✅ `timeout-enforcer` - Timeout enforcement with SSE integration

**These were in workspace but not assigned to any team. Now covered by TEAM-237 and TEAM-238.**

---

## ✅ Verification Checklist

- [x] Compared workspace members vs filesystem
- [x] Identified dead code (2 crates)
- [x] Verified all active crates are assigned
- [x] Updated all planning documents
- [x] Added dead code exclusions doc
- [x] Verified no duplicate assignments
- [x] Verified team number consistency
- [x] Verified inventory count (27)

---

## 🧹 Cleanup Recommendations

### Immediate Actions

1. **Delete dead code:**
   ```bash
   rm -rf bin/99_shared_crates/hive-operations
   ```

2. **Complete hive-core deprecation:**
   - Verify no remaining dependencies
   - Delete `bin/99_shared_crates/hive-core/` after verification
   - Update deprecation tracking

### Future Actions

- Run periodic dead code audits
- Verify all workspace members compile
- Remove unused dependencies
- Check for other deprecated crates

---

## 📚 Related Documents

- **Dead Code Details:** `.plan/DEAD_CODE_EXCLUSIONS.md`
- **Master Plan:** `.plan/BEHAVIOR_DISCOVERY_MASTER_PLAN.md`
- **Phase 4 Guide:** `.plan/PHASE_4_GUIDES.md`
- **Phase 5 Guide:** `.plan/PHASE_5_GUIDES.md`
- **Index:** `.plan/00_INDEX.md`

---

## 🎉 Result

**✅ Testing plan now covers ONLY active codebase**

- 27 teams investigating 35 active crates
- 2 dead code crates excluded
- 4 previously missing crates now covered
- All planning documents updated
- No wasted effort on dead code

**The plan is now accurate and complete.**

---

**Status:** ✅ COMPLETE  
**Verified By:** Automated workspace analysis  
**Last Updated:** Oct 22, 2025
