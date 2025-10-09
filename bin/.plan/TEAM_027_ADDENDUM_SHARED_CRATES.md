# TEAM-027 Addendum: Shared Crates Refactoring

**Date:** 2025-10-09T23:39:00+02:00  
**Team:** TEAM-027  
**Context:** User requested worker-registry be moved to shared crate

---

## What Was Done

### 1. Created New Shared Crate: worker-registry ✅

**Location:** `bin/shared-crates/worker-registry/`

**Files Created:**
- `Cargo.toml` - Crate configuration
- `src/lib.rs` - Worker registry implementation (240 lines)
- `README.md` - Documentation

**Features:**
- SQLite-backed worker tracking
- `find_worker()` - Query by node + model
- `register_worker()` - Register/update worker
- `update_state()` - Update worker state
- `remove_worker()` - Remove worker
- `list_workers()` - List all workers

**Shared Between:**
- `queen-rbee` (orchestrator daemon)
- `rbee-keeper` (orchestrator CLI)

### 2. Migrated rbee-keeper to Use Shared Crate ✅

**Changes:**
- Removed `bin/rbee-keeper/src/registry.rs`
- Updated `Cargo.toml` to depend on `worker-registry`
- Updated `main.rs` to remove registry module
- Updated `commands/infer.rs` to import from `worker_registry`

**Build Status:**
```bash
$ cargo build --bin rbee
   Compiling worker-registry v0.1.0
   Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.93s
```

✅ **Successfully compiles and works**

### 3. Analyzed All Shared Crates ✅

**Created Documents:**
- `SHARED_CRATES_ANALYSIS.md` - Comprehensive analysis
- `CRATE_USAGE_SUMMARY.md` - Usage verification results

**Key Findings:**

**Active Crates (Keep):**
1. ✅ worker-registry (new)
2. ✅ pool-core
3. ✅ gpu-info
4. ✅ narration-core ⚠️ **Used by llm-worker-rbee**
5. ✅ narration-macros (dependency)

**Unused Crates (Can Archive):**
1. ❌ auth-min
2. ❌ deadline-propagation
3. ❌ pool-registry-types
4. ❌ orchestrator-core
5. ❌ audit-logging
6. ❌ input-validation
7. ❌ secrets-management

---

## Important Discovery

**narration-core is ACTIVE!**

Initially thought to be obsolete, but verification showed:
- llm-worker-rbee uses it for observability
- Found in main.rs, device.rs, backend/inference.rs
- Uses `observability_narration_core::{narrate, NarrationFields}`

**Action:** Keep narration-core and narration-macros

---

## Recommendations for TEAM-028

### Immediate Actions

1. **Review analysis documents:**
   - Read `SHARED_CRATES_ANALYSIS.md`
   - Read `CRATE_USAGE_SUMMARY.md`

2. **Decide on archival:**
   - If agreed, archive unused crates
   - Update workspace Cargo.toml
   - Document in README

3. **Archive process (if approved):**
   ```bash
   mkdir -p bin/shared-crates/.archive
   mv bin/shared-crates/auth-min bin/shared-crates/.archive/
   mv bin/shared-crates/deadline-propagation bin/shared-crates/.archive/
   mv bin/shared-crates/pool-registry-types bin/shared-crates/.archive/
   mv bin/shared-crates/orchestrator-core bin/shared-crates/.archive/
   mv bin/shared-crates/audit-logging bin/shared-crates/.archive/
   mv bin/shared-crates/input-validation bin/shared-crates/.archive/
   mv bin/shared-crates/secrets-management bin/shared-crates/.archive/
   ```

4. **Update Cargo.toml:**
   - Remove archived crates from workspace members
   - Add comments explaining archival

### Future Considerations

**If auth is needed:**
- Restore auth-min from archive
- Integrate into rbee-hive ↔ llm-worker-rbee

**If deadlines are needed:**
- Restore deadline-propagation from archive
- Integrate into queen-rbee request flow

**Post-MVP features:**
- Audit logging: Restore if compliance required
- Input validation: Restore if strict validation needed
- Secrets management: Restore if secure key storage needed

---

## Files Created

1. `bin/shared-crates/worker-registry/Cargo.toml`
2. `bin/shared-crates/worker-registry/src/lib.rs`
3. `bin/shared-crates/worker-registry/README.md`
4. `bin/shared-crates/SHARED_CRATES_ANALYSIS.md`
5. `bin/shared-crates/CRATE_USAGE_SUMMARY.md`
6. `bin/.plan/TEAM_027_ADDENDUM_SHARED_CRATES.md` (this file)

## Files Modified

1. `Cargo.toml` - Added worker-registry to workspace
2. `bin/rbee-keeper/Cargo.toml` - Removed sqlx, added worker-registry
3. `bin/rbee-keeper/src/main.rs` - Removed registry module
4. `bin/rbee-keeper/src/commands/infer.rs` - Import from worker_registry

## Files Deleted

1. `bin/rbee-keeper/src/registry.rs` - Moved to shared crate

---

## Build Verification

```bash
$ cargo build --bin rbee-hive --bin rbee
   Compiling worker-registry v0.1.0
   Compiling rbee-keeper v0.1.0
   Compiling rbee-hive v0.1.0
   Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.93s

$ cargo test --bin rbee --bin rbee-hive
   test result: ok. 10 passed; 0 failed; 0 ignored (rbee-hive)
   test result: ok. 1 passed; 0 failed; 1 ignored (rbee-keeper)
```

✅ **All builds and tests pass**

---

## Summary

Successfully refactored worker registry into a shared crate that can be used by both `queen-rbee` and `rbee-keeper`. Analyzed all shared crates and identified 7 unused crates that can be archived. Discovered that `narration-core` is actively used by `llm-worker-rbee` and must be kept.

**Status:** ✅ Complete  
**Next Team:** TEAM-028 - Review analysis and decide on archival

---

**Signed:** TEAM-027  
**Date:** 2025-10-09T23:39:00+02:00
