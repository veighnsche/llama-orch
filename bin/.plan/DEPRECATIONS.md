# Deprecations and Removals

**Last Updated:** 2025-10-19

This document tracks deprecated crates and their replacements.

---

## Active Deprecations

### 1. shared-crates/gpu-info → rbee-hive-crates/device-detection

**Status:** DEPRECATED (2025-10-19)  
**Reason:** GPU detection is hive-specific, not shared

**Details:**
- **Deprecated:** `bin/shared-crates/gpu-info/`
- **Replacement:** `bin/rbee-hive-crates/device-detection/`
- **Migration Guide:** `bin/rbee-hive-crates/device-detection/MIGRATION_FROM_GPU_INFO.md`
- **Deprecation Notice:** `bin/shared-crates/gpu-info/DEPRECATED.md`

**Why:**
- ONLY rbee-hive needs GPU/device detection
- Workers don't detect GPUs (told which GPU to use)
- Queen doesn't detect GPUs (receives GPU state from hives)
- Keeper doesn't detect GPUs (CLI tool)

**Action Items:**
- [x] Migrate implementation to `device-detection`
- [ ] Update rbee-hive imports
- [ ] Remove from root `Cargo.toml`
- [ ] Delete `shared-crates/gpu-info/` directory

---

### 2. shared-crates/hive-core → shared-crates/rbee-types

**Status:** DEPRECATED (2025-10-19)  
**Reason:** Renamed to better reflect purpose as system-wide shared types

**Details:**
- **Deprecated:** `bin/shared-crates/hive-core/`
- **Replacement:** `bin/shared-crates/rbee-types/`
- **Deprecation Notice:** `bin/shared-crates/hive-core/DEPRECATED.md`

**Why:**
- `hive-core` implies hive-specific types
- Actually contains system-wide shared types (WorkerInfo, Backend, ModelCatalog)
- `rbee-types` better reflects its purpose

**What Changed:**
- Crate name: `hive-core` → `rbee-types`
- Import path: `hive_core::` → `rbee_types::`
- No API changes (all types remain the same)

**Action Items:**
- [x] Merge implementation into `rbee-types`
- [x] Mark `hive-core` as deprecated
- [ ] Find all uses of `hive-core` and update to `rbee-types`
- [ ] Remove from root `Cargo.toml`
- [ ] Delete `shared-crates/hive-core/` directory

---

## Completed Moves

### 1. worker-rbee-crates/heartbeat → shared-crates/heartbeat

**Status:** COMPLETED (2025-10-19)  
**Reason:** Both workers AND hives send heartbeats

**Details:**
- **Old Location:** `bin/worker-rbee-crates/heartbeat/`
- **New Location:** `bin/shared-crates/heartbeat/`
- **Migration Notes:** `bin/shared-crates/heartbeat/MIGRATION_NOTES.md`

**Why:**
- Workers send heartbeats to hives (30s interval)
- Hives send heartbeats to queen (15s interval)
- Same mechanism, different payloads
- ~200 LOC reused instead of duplicated

**Actions Completed:**
- ✅ Moved to `shared-crates/`
- ✅ Renamed to `rbee-heartbeat`
- ✅ Updated root `Cargo.toml`
- ✅ Deleted old directory

---

## Planned Removals

### 1. shared-crates/hive-core

**Status:** UNUSED  
**Reason:** Never integrated, duplicate of rbee-types

**Details:**
- Has duplicate `WorkerInfo` struct
- 100 LOC, 0 uses
- Should be deleted or merged into `rbee-types`

**Reference:** TEAM-130E_CONSOLIDATION_SUMMARY.md

---

### 2. shared-crates/secrets-management

**Status:** UNUSED  
**Reason:** Declared but never used

**Details:**
- Declared in llm-worker but 0 uses
- Should be removed from dependencies

**Reference:** TEAM-130E_CONSOLIDATION_SUMMARY.md

---

## Deprecation Policy

### When to Deprecate

A crate should be deprecated when:
1. **Single-binary usage:** Only 1 binary uses it (move to binary-specific crates)
2. **Never used:** Declared but never imported (remove entirely)
3. **Duplicate functionality:** Another crate provides same functionality
4. **Architectural violation:** Violates separation of concerns

### Deprecation Process

1. **Mark as deprecated:**
   - Create `DEPRECATED.md` in the crate
   - Document reason and replacement
   - Add deprecation warnings

2. **Create migration guide:**
   - Document API changes
   - Provide code examples
   - List action items

3. **Update documentation:**
   - Update architecture docs
   - Update this file
   - Update related READMEs

4. **Remove from workspace:**
   - Remove from root `Cargo.toml`
   - Verify no dependencies remain
   - Delete directory

---

## Related Documentation

- **Architecture:** `bin/.plan/TEAM_130G_FINAL_ARCHITECTURE.md`
- **Consolidation:** `bin/.plan/TEAM_130E_CONSOLIDATION_SUMMARY.md`
- **Scaffolding:** `bin/.plan/TEAM_135_SCAFFOLDING_ASSIGNMENT.md`
