# Dead Code Exclusions from Behavior Discovery

**Date:** Oct 22, 2025  
**Purpose:** Document crates that should NOT be investigated (not in workspace)

---

## Excluded Crates

### ❌ bin/99_shared_crates/hive-core

**Status:** DEPRECATED  
**Reason:** Renamed to `rbee-types`  
**Evidence:**
- Has `DEPRECATED.md` file
- NOT in root `Cargo.toml` workspace members
- Replaced by `bin/99_shared_crates/rbee-types`

**Migration:**
- All functionality moved to `rbee-types`
- See: `bin/99_shared_crates/hive-core/DEPRECATED.md`

**Action:** DO NOT investigate. Use `rbee-types` instead (covered by TEAM-233).

---

### ❌ bin/99_shared_crates/hive-operations

**Status:** DEAD CODE (empty stub)  
**Reason:** Never implemented  
**Evidence:**
- Directory exists: `bin/99_shared_crates/hive-operations/`
- Has empty `src/` directory
- NO `Cargo.toml` file
- NOT in root `Cargo.toml` workspace members

**Action:** DO NOT investigate. Directory should be deleted.

---

## Verification Method

To verify a crate is active (should be investigated):

```bash
# Check if crate is in workspace
grep "bin/99_shared_crates/[crate-name]" Cargo.toml

# If found → investigate
# If not found → skip (dead code)
```

**Active crates only:** Only crates listed in root `Cargo.toml` workspace members should be investigated.

---

## Impact on Testing Plan

### Phase 4 Adjustments

**Original TEAM-237 assignment:**
- heartbeat + auto-update + hive-core

**Corrected TEAM-237 assignment:**
- heartbeat + auto-update + timeout-enforcer

**Rationale:**
- `hive-core` is dead code (deprecated)
- `timeout-enforcer` is active and important (SSE narration integration)
- Better coverage of actual codebase

---

## Active Shared Crates (Phase 4)

Verified from `Cargo.toml` workspace members:

1. ✅ `audit-logging` (+ bdd)
2. ✅ `auth-min`
3. ✅ `auto-update`
4. ✅ `daemon-lifecycle` (+ bdd)
5. ✅ `deadline-propagation`
6. ✅ `heartbeat`
7. ✅ `input-validation` (+ bdd)
8. ✅ `job-registry`
9. ✅ `jwt-guardian`
10. ✅ `model-catalog`
11. ✅ `narration-core` (+ bdd)
12. ✅ `narration-macros`
13. ✅ `rbee-config`
14. ✅ `rbee-http-client` (+ bdd)
15. ✅ `rbee-operations`
16. ✅ `rbee-types` (+ bdd)
17. ✅ `secrets-management` (+ bdd)
18. ✅ `sse-relay`
19. ✅ `timeout-enforcer`

**Total:** 19 active shared crates (some with BDD subcrates)

---

## Team Assignment Impact

### Updated Phase 4 Teams

- **TEAM-230:** narration-core + narration-macros ✅
- **TEAM-231:** daemon-lifecycle ✅
- **TEAM-232:** rbee-http-client ✅
- **TEAM-233:** rbee-config + rbee-operations ✅
- **TEAM-234:** job-registry + deadline-propagation ✅
- **TEAM-235:** auth-min + jwt-guardian ✅
- **TEAM-236:** audit-logging + input-validation ✅
- **TEAM-237:** heartbeat + auto-update + timeout-enforcer ✅ (CORRECTED)

**Note:** Some teams investigate multiple related crates to reduce total team count.

---

## Crates Not Assigned (Low Priority)

The following active crates are NOT assigned to any team (consider for future phases):

1. `secrets-management` - Security-related, may need dedicated investigation
2. `sse-relay` - SSE infrastructure, may need dedicated investigation
3. `model-catalog` (shared) - Duplicate of hive crate, may be consolidated

**Recommendation:** Add TEAM-238A, 238B, 238C if these need investigation.

---

## Cleanup Recommendations

### Immediate Actions

1. **Delete dead code:**
   ```bash
   rm -rf bin/99_shared_crates/hive-operations
   ```

2. **Complete hive-core deprecation:**
   - Verify no dependencies remain
   - Remove from filesystem
   - Update deprecation docs

### Future Actions

- Audit for other dead code directories
- Verify all workspace members compile
- Remove unused dependencies

---

## Verification Checklist

Before investigating ANY crate:

- [ ] Crate is listed in root `Cargo.toml` workspace members
- [ ] Crate has a `Cargo.toml` file
- [ ] Crate has source code in `src/`
- [ ] Crate is NOT marked DEPRECATED
- [ ] Crate compiles with `cargo check -p [crate-name]`

**If any check fails:** DO NOT investigate (dead code).

---

## Related Documents

- **Master Plan:** `.plan/BEHAVIOR_DISCOVERY_MASTER_PLAN.md`
- **Phase 4 Guide:** `.plan/PHASE_4_GUIDES.md`
- **hive-core Deprecation:** `bin/99_shared_crates/hive-core/DEPRECATED.md`

---

**Status:** ✅ VERIFIED  
**Last Updated:** Oct 22, 2025  
**Verified By:** Automated workspace analysis
