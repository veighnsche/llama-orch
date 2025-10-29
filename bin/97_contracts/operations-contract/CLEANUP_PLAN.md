# operations-contract Cleanup Plan

**Problem:** `src/lib.rs` is 386 lines of mixed concerns, deletion tombstones, and hard-to-navigate code.

**Goal:** Make it easy for humans to understand what operations exist and where they're routed.

---

## Current Issues

### 1. **Deletion Tombstones Everywhere**
Lines 90-92, 110-111, 113-115, 158, 167-169, 174-175, 197-198, 204, 271-285, 298-299, 370-371, 383

**Problem:** 15+ comments about deleted operations cluttering the code.

**Solution:** DELETE all tombstone comments. Git history preserves this info.

### 2. **Operation Enum is Unorganized**
Lines 74-152: Operations mixed together without clear grouping

**Problem:** Hard to see what operations exist at a glance.

**Solution:** Group operations by category with clear section headers.

### 3. **Impl Block is Repetitive**
Lines 160-257: Three methods (`name()`, `hive_id()`, `target_server()`) with massive match statements

**Problem:** Adding a new operation requires updating 3+ places.

**Solution:** Use macros or extract to separate module.

### 4. **Tests are Scattered**
Lines 287-385: Tests mixed with deletion tombstones

**Problem:** Hard to see what's actually tested.

**Solution:** Clean up tests, remove tombstones.

---

## Cleanup Plan

### Phase 1: Delete Tombstones (5 minutes)
**Goal:** Remove ALL deletion comments

**Actions:**
- Delete lines 90-92 (TEAM-278, TEAM-284, TEAM-285 deletions)
- Delete lines 110-111 (TEAM-323, TEAM-278 deletions)
- Delete lines 113-115 (TEAM-278 deletions)
- Delete line 158 (TEAM-278 deletion)
- Delete lines 167-169 (TEAM-278, TEAM-284, TEAM-285 deletions)
- Delete lines 174-175 (TEAM-323, TEAM-278 deletions)
- Delete lines 197-198 (TEAM-278, TEAM-285 deletions)
- Delete line 204 (TEAM-278 deletion)
- Delete lines 271-285 (TEAM-312 deletion explanation - move to docs)
- Delete lines 298-299 (test tombstones)
- Delete lines 370-371 (test tombstones)
- Delete line 383 (test tombstone)

**Result:** ~40 lines removed

---

### Phase 2: Reorganize Operation Enum (10 minutes)
**Goal:** Clear visual grouping

**Before:**
```rust
pub enum Operation {
    Status,
    QueenCheck,
    HiveCheck { ... },
    HiveList,
    HiveGet { ... },
    // ... scattered operations
}
```

**After:**
```rust
pub enum Operation {
    // ═══════════════════════════════════════════════════════════
    // QUEEN OPERATIONS (http://localhost:7833/v1/jobs)
    // ═══════════════════════════════════════════════════════════
    
    /// Query all hives and workers from registry
    Status,
    
    /// Schedule inference and route to worker
    Infer(InferRequest),
    
    // ═══════════════════════════════════════════════════════════
    // HIVE OPERATIONS (http://localhost:7835/v1/jobs)
    // ═══════════════════════════════════════════════════════════
    
    // Worker Lifecycle
    // ─────────────────────────────────────────────────────────
    WorkerSpawn(WorkerSpawnRequest),
    WorkerProcessList(WorkerProcessListRequest),
    WorkerProcessGet(WorkerProcessGetRequest),
    WorkerProcessDelete(WorkerProcessDeleteRequest),
    
    // Model Management
    // ─────────────────────────────────────────────────────────
    ModelDownload(ModelDownloadRequest),
    ModelList(ModelListRequest),
    ModelGet(ModelGetRequest),
    ModelDelete(ModelDeleteRequest),
    
    // ═══════════════════════════════════════════════════════════
    // DIAGNOSTIC OPERATIONS
    // ═══════════════════════════════════════════════════════════
    
    QueenCheck,
    HiveCheck { alias: String },
    
    // ═══════════════════════════════════════════════════════════
    // DEPRECATED / UNUSED (keep for now, remove in next cleanup)
    // ═══════════════════════════════════════════════════════════
    
    HiveList,
    HiveGet { alias: String },
    HiveStatus { alias: String },
    HiveRefreshCapabilities { alias: String },
    ActiveWorkerList,
    ActiveWorkerGet { worker_id: String },
    ActiveWorkerRetire { worker_id: String },
}
```

**Result:** Clear visual structure, easy to scan

---

### Phase 3: Extract Impl Methods to Separate Module (15 minutes)
**Goal:** Reduce repetition, make maintenance easier

**Create:** `src/operation_impl.rs`

```rust
//! Operation implementation methods
//!
//! TEAM-CLEANUP: Extracted from lib.rs to reduce clutter

use super::*;

impl Operation {
    /// Get operation name for logging/narration
    pub fn name(&self) -> &'static str {
        // Use macro to reduce repetition
        operation_name!(self)
    }
    
    /// Get target hive_id if operation is hive-specific
    pub fn hive_id(&self) -> Option<&str> {
        // Use macro to reduce repetition
        operation_hive_id!(self)
    }
    
    /// Get target server (Queen vs Hive)
    pub fn target_server(&self) -> TargetServer {
        match self {
            // Queen operations
            Self::Status | Self::Infer(_) => TargetServer::Queen,
            
            // Hive operations
            Self::WorkerSpawn(_) 
                | Self::WorkerProcessList(_)
                | Self::WorkerProcessGet(_)
                | Self::WorkerProcessDelete(_)
                | Self::ModelDownload(_)
                | Self::ModelList(_)
                | Self::ModelGet(_)
                | Self::ModelDelete(_) => TargetServer::Hive,
            
            // Everything else goes to queen
            _ => TargetServer::Queen,
        }
    }
}

// Macro to generate operation names
macro_rules! operation_name {
    ($op:expr) => {
        match $op {
            Operation::Status => "status",
            Operation::Infer(_) => "infer",
            Operation::WorkerSpawn(_) => "worker_spawn",
            // ... etc
        }
    };
}
```

**Update:** `src/lib.rs`
```rust
mod operation_impl;
pub use operation_impl::*;
```

**Result:** Main file is cleaner, impl details are isolated

---

### Phase 4: Clean Up Tests (5 minutes)
**Goal:** Remove tombstones, keep only active tests

**Delete:**
- All `// TEAM-XXX: DELETED` comments in tests
- Tests for operations that don't exist

**Keep:**
- Serialization tests for current operations
- Deserialization tests for current operations
- `name()` and `hive_id()` tests

**Result:** Clean test suite

---

### Phase 5: Add Quick Reference (5 minutes)
**Goal:** Make it trivial to see what operations exist

**Add to top of file:**
```rust
//! # Quick Reference
//!
//! ## Queen Operations (2)
//! - `Status` - Query registries
//! - `Infer` - Schedule inference
//!
//! ## Hive Operations (8)
//! **Worker:** Spawn, ProcessList, ProcessGet, ProcessDelete  
//! **Model:** Download, List, Get, Delete
//!
//! ## Diagnostic (2)
//! - `QueenCheck`, `HiveCheck`
//!
//! ## Total: 12 active operations
```

---

## Summary

| Phase | Time | Lines Removed | Lines Added | Net Change |
|-------|------|---------------|-------------|------------|
| 1. Delete tombstones | 5 min | ~40 | 0 | -40 |
| 2. Reorganize enum | 10 min | 0 | ~20 (comments) | +20 |
| 3. Extract impl | 15 min | ~100 | ~80 (new file) | -20 |
| 4. Clean tests | 5 min | ~15 | 0 | -15 |
| 5. Add reference | 5 min | 0 | ~15 | +15 |
| **TOTAL** | **40 min** | **~155** | **~115** | **-40** |

**Result:**
- 40 fewer lines
- Clear visual structure
- Easy to scan
- Easy to maintain
- No loss of functionality

---

## Benefits

### Before (Current State)
- 386 lines
- 15+ deletion tombstones
- Operations scattered
- Hard to see what exists
- 3 places to update per new operation

### After (Cleaned Up)
- ~346 lines (main file)
- ~80 lines (operation_impl.rs)
- 0 deletion tombstones
- Clear visual grouping
- Quick reference at top
- Easier to add new operations

---

## Migration Risk

**Risk Level:** LOW

**Why:**
- No API changes
- No breaking changes
- Only internal reorganization
- Tests verify behavior unchanged

**Verification:**
```bash
# Before cleanup
cargo test --package operations-contract

# After cleanup
cargo test --package operations-contract

# Should have same results
```

---

## Next Steps

1. Review this plan
2. Execute Phase 1 (tombstones) - safest, biggest impact
3. Execute Phase 2 (reorganize) - visual improvement
4. Execute Phase 3 (extract impl) - optional, bigger refactor
5. Execute Phase 4 (tests) - cleanup
6. Execute Phase 5 (reference) - documentation

**Start with Phase 1 and 2 - they give 80% of the benefit with 20% of the effort.**
