# Shared Crates Cleanup Tasks

**Date:** 2025-10-09T23:46:00+02:00  
**Based on:** User feedback on CRATE_USAGE_SUMMARY.md

---

## Summary of Decisions

### ✅ Keep (10 crates)
1. worker-registry (new)
2. hive-core (rename from pool-core)
3. gpu-info
4. narration-core
5. narration-macros
6. audit-logging
7. secrets-management
8. input-validation
9. deadline-propagation

### ❌ Delete (2 crates)
1. pool-registry-types
2. orchestrator-core

### ⚠️ Uncertain (1 crate)
1. auth-min (needs decision)

---

## Task Checklist

### Task 1: Delete Obsolete Crates ❌

```bash
# Delete pool-registry-types
rm -rf bin/shared-crates/pool-registry-types

# Delete orchestrator-core
rm -rf bin/shared-crates/orchestrator-core
```

- [ ] Delete pool-registry-types directory
- [ ] Delete orchestrator-core directory
- [ ] Remove from workspace Cargo.toml
- [ ] Verify no imports exist

### Task 2: Rename pool-core to hive-core 🔄

**Step 1: Rename directory**
```bash
mv bin/shared-crates/pool-core bin/shared-crates/hive-core
```

**Step 2: Update hive-core/Cargo.toml**
```toml
[package]
name = "hive-core"  # Changed from pool-core
```

**Step 3: Update workspace Cargo.toml**
```toml
# Change:
"bin/shared-crates/pool-core",
# To:
"bin/shared-crates/hive-core",
```

**Step 4: Update rbee-hive/Cargo.toml**
```toml
[dependencies]
# Change:
pool-core = { path = "../shared-crates/pool-core" }
# To:
hive-core = { path = "../shared-crates/hive-core" }
```

**Step 5: Update imports in rbee-hive**
```bash
# Find all uses of pool-core
rg "use.*pool.core|pool_core" bin/rbee-hive/

# Replace with hive-core
# (Do this manually or with sed)
```

- [ ] Rename directory
- [ ] Update hive-core/Cargo.toml name
- [ ] Update workspace Cargo.toml
- [ ] Update rbee-hive/Cargo.toml
- [ ] Update all imports in rbee-hive
- [ ] Test: `cargo build --bin rbee-hive`

### Task 3: Decide on auth-min ⚠️

**Questions to answer:**
1. Do we need auth between rbee-hive ↔ llm-worker-rbee?
2. Do we need auth between queen-rbee ↔ rbee-hive?
3. Do we need auth for rbee-keeper CLI?

**If YES to any:**
- [ ] Keep auth-min
- [ ] Document usage plan

**If NO to all:**
- [ ] Delete auth-min
- [ ] Remove from workspace

### Task 4: Update Documentation 📝

- [ ] Update README.md in bin/shared-crates/
- [ ] List all active crates with purpose
- [ ] Document the rename (pool-core → hive-core)
- [ ] Note deleted crates (pool-registry-types, orchestrator-core)

---

## Verification Commands

**After Task 1 (Delete):**
```bash
# Verify deleted
ls bin/shared-crates/ | grep -E "pool-registry-types|orchestrator-core"
# Should return nothing

# Verify no imports
rg "pool.registry.types|pool_registry_types" --type rust bin/
rg "orchestrator.core|orchestrator_core" --type rust bin/
# Should return nothing
```

**After Task 2 (Rename):**
```bash
# Verify renamed
ls bin/shared-crates/ | grep hive-core
# Should show hive-core

# Verify no old imports
rg "pool.core|pool_core" --type rust bin/rbee-hive/
# Should return nothing

# Build test
cargo build --bin rbee-hive
cargo test --bin rbee-hive
```

**After Task 3 (auth-min decision):**
```bash
# If kept, verify it's in workspace
grep "auth-min" Cargo.toml

# If deleted, verify it's gone
ls bin/shared-crates/ | grep auth-min
# Should return nothing
```

---

## Final State

**Active shared crates (9 or 10):**
1. ✅ worker-registry
2. ✅ hive-core (renamed from pool-core)
3. ✅ gpu-info
4. ✅ narration-core
5. ✅ narration-macros
6. ✅ audit-logging
7. ✅ secrets-management
8. ✅ input-validation
9. ✅ deadline-propagation
10. ⚠️ auth-min (if decision is to keep)

**Deleted:**
- ❌ pool-registry-types
- ❌ orchestrator-core
- ❌ auth-min (if decision is to delete)

---

## Estimated Time

- Task 1 (Delete): 15 minutes
- Task 2 (Rename): 30-45 minutes
- Task 3 (Decision): 15 minutes discussion
- Task 4 (Docs): 15 minutes

**Total:** ~1.5-2 hours

---

**Created by:** TEAM-027  
**For:** TEAM-028 or next session
