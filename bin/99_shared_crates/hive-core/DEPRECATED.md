# ⚠️ DEPRECATED: hive-core

**Status:** DEPRECATED  
**Date:** 2025-10-19  
**Reason:** Renamed to rbee-types

---

## Deprecation Notice

This crate is **DEPRECATED** and has been renamed.

### Replacement

**Use instead:** `bin/shared-crates/rbee-types`

### Why Renamed?

**Original name (WRONG):**
- `hive-core` - Implies hive-specific types
- Located in `bin/shared-crates/hive-core/`

**New name (CORRECT):**
- `rbee-types` - System-wide shared types
- Located in `bin/shared-crates/rbee-types/`
- Better reflects its purpose as shared types for ALL binaries

### What Changed?

**Crate rename:**
```
FROM: hive-core (hive_core)
TO:   rbee-types (rbee_types)
```

**No API changes:**
- All types remain the same
- All functions remain the same
- Just the crate name changed

---

## Migration Guide

### Update Cargo.toml

**Before:**
```toml
[dependencies]
hive-core = { path = "../shared-crates/hive-core" }
```

**After:**
```toml
[dependencies]
rbee-types = { path = "../shared-crates/rbee-types" }
```

### Update Imports

**Before:**
```rust
use hive_core::{WorkerInfo, Backend, ModelCatalog};
use hive_core::error::{PoolError, Result};
```

**After:**
```rust
use rbee_types::{WorkerInfo, Backend, ModelCatalog};
use rbee_types::error::{PoolError, Result};
```

---

## Removal Timeline

### Phase 1: Mark as Deprecated (DONE)
- ✅ Create `DEPRECATED.md`
- ✅ Merge implementation into `rbee-types`
- ✅ Update documentation

### Phase 2: Update Dependencies (TODO)
- [ ] Find all uses of `hive-core`
- [ ] Update to `rbee-types`
- [ ] Verify compilation

### Phase 3: Remove from Workspace (TODO)
- [ ] Remove from root `Cargo.toml`
- [ ] Delete `bin/shared-crates/hive-core/` directory

---

## Related Documentation

- **New location:** `bin/shared-crates/rbee-types/`
- **Deprecation tracking:** `bin/.plan/DEPRECATIONS.md`
- **Architecture:** `bin/.plan/TEAM_130G_FINAL_ARCHITECTURE.md`

---

## DO NOT USE THIS CRATE

Use `rbee-types` instead.
