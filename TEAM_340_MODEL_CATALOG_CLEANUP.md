# TEAM-340: Model Catalog SQLite Removal

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025

## Problem

Found duplicate `model-catalog` crate at `bin/99_shared_crates/model-catalog/` that was:
- ❌ Using SQLite (434 LOC)
- ❌ Not used anywhere in the codebase
- ❌ Outdated architecture (stateful database)
- ❌ Conflicting with the correct implementation

## Root Cause

The project evolved from SQLite-based catalogs to **filesystem-based catalogs** using the `artifact-catalog` abstraction. The old SQLite implementation was never removed.

## Solution

### 1. Deleted Unused SQLite Crate

**Removed:** `bin/99_shared_crates/model-catalog/` (434 LOC)

This crate was:
```rust
// OLD (SQLite-based)
pub struct ModelCatalog {
    db_path: String,  // ❌ SQLite database
}

impl ModelCatalog {
    pub async fn init(&self) -> Result<()> {
        // Creates SQLite tables
    }
    
    pub async fn find_model(&self, reference: &str, provider: &str) -> Result<Option<ModelInfo>> {
        // SQL queries
    }
}
```

### 2. Verified Correct Implementation

**Kept:** `bin/25_rbee_hive_crates/model-catalog/` (124 LOC)

This is the CORRECT implementation:
```rust
// NEW (Filesystem-based)
pub struct ModelCatalog {
    inner: FilesystemCatalog<ModelEntry>,  // ✅ Filesystem catalog
}

impl ModelCatalog {
    pub fn new() -> Result<Self> {
        let catalog_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("rbee")
            .join("models");
        
        let inner = FilesystemCatalog::new(catalog_dir)?;
        Ok(Self { inner })
    }
}

// Delegates to FilesystemCatalog
impl ArtifactCatalog<ModelEntry> for ModelCatalog {
    fn add(&self, model: ModelEntry) -> Result<()> {
        self.inner.add(model)
    }
    // ... other methods
}
```

## Architecture

### Current (Correct) Design

```
rbee-hive-model-catalog
├── Uses: artifact-catalog (filesystem abstraction)
├── Storage: ~/.cache/rbee/models/
├── Pattern: Stateless, folder-based
└── Consistent with: worker-catalog, artifact-catalog
```

### Benefits of Filesystem Approach

1. ✅ **Stateless** - No database to initialize/migrate
2. ✅ **Simple** - Just read the folder
3. ✅ **Consistent** - Same pattern as worker-catalog
4. ✅ **Portable** - Works across platforms
5. ✅ **Debuggable** - Just look at the filesystem

## Verification

```bash
# Compilation: PASS
cargo check --package rbee-hive-model-catalog
cargo check --bin rbee-hive

# Tests: PASS
cargo test --package rbee-hive-model-catalog
# running 1 test
# test tests::test_model_catalog_crud ... ok

# Usage in rbee-hive
bin/20_rbee_hive/src/main.rs:29:
use rbee_hive_model_catalog::ModelCatalog;

bin/20_rbee_hive/Cargo.toml:49:
rbee-hive-model-catalog = { path = "../25_rbee_hive_crates/model-catalog" }
```

## Code Structure

The correct implementation is already well-structured:

```
bin/25_rbee_hive_crates/model-catalog/
├── src/
│   ├── lib.rs (42 LOC) - ModelCatalog struct + ArtifactCatalog impl
│   └── types.rs (82 LOC) - ModelEntry + ModelStatus types
├── Cargo.toml
└── README.md
```

### File Breakdown

**src/lib.rs** (42 LOC):
- `ModelCatalog` struct
- `new()` - Creates catalog in `~/.cache/rbee/models/`
- `with_dir()` - Custom directory (for testing)
- `model_path()` - Get path for a model
- `ArtifactCatalog` trait delegation to `FilesystemCatalog`

**src/types.rs** (82 LOC):
- `ModelEntry` struct (id, name, path, size, status, added_at)
- `ModelStatus` type alias
- `Artifact` trait implementation
- Serde serialization

## Impact

- ✅ Removed 434 LOC of dead code
- ✅ Eliminated SQLite dependency confusion
- ✅ Clarified architecture (filesystem-based, stateless)
- ✅ No breaking changes (unused crate removed)

## Related Work

- **TEAM-267:** Created filesystem-based model-catalog
- **TEAM-273:** Created artifact-catalog abstraction
- **TEAM-274:** Created worker-catalog (same pattern)

## Lessons Learned

**Pattern:** When migrating from stateful (SQLite) to stateless (filesystem), DELETE the old implementation immediately. Don't leave it around "just in case."

**Why this happened:** The SQLite implementation was created early (TEAM-029), then replaced by the filesystem implementation (TEAM-267), but the old code was never removed.

**Prevention:** Regular dead code audits, especially after architectural changes.

---

## Summary

**TEAM-340 completed:**
1. ✅ Deleted unused SQLite-based model-catalog (434 LOC)
2. ✅ Fixed test import in filesystem-based model-catalog
3. ✅ Verified compilation of rbee-hive binary
4. ✅ Verified all tests pass

**Files Changed:**
- DELETED: `bin/99_shared_crates/model-catalog/` (entire directory, 434 LOC)
- MODIFIED: `bin/25_rbee_hive_crates/model-catalog/src/lib.rs` (1 line - fixed test import)

**Compilation:** ✅ PASS  
**Tests:** ✅ PASS  
**Binary:** ✅ rbee-hive compiles successfully
