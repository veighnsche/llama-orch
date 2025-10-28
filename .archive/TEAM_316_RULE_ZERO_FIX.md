# TEAM-316: RULE ZERO Violation Fix

**Date:** 2025-10-27  
**Status:** ✅ COMPLETE

## Problem

**RULE ZERO VIOLATION:** Type aliases for backwards compatibility

### Violations Found

1. **queen-lifecycle/src/types.rs**
   ```rust
   pub use daemon_contract::DaemonHandle as QueenHandle;
   ```

2. **hive-lifecycle/src/lib.rs**
   ```rust
   pub type HiveHandle = daemon_contract::DaemonHandle;
   ```

**Why This Violates RULE ZERO:**

From `.windsurf/rules/engineering-rules.md`:

> ❌ **BANNED - Entropy Patterns:**
> - Creating wrapper functions that just call new implementations
> - "Let's keep both APIs for compatibility"

**Result:** 3 names for the same thing (`DaemonHandle`, `QueenHandle`, `HiveHandle`)

**Impact:** Permanent confusion for all future developers: "Which one should I use?"

## Solution

### ✅ JUST USE `DaemonHandle` EVERYWHERE

Deleted type aliases, use `daemon_contract::DaemonHandle` directly.

## Changes Made

### 1. queen-lifecycle/src/types.rs
- **BEFORE:** `pub use daemon_contract::DaemonHandle as QueenHandle;`
- **AFTER:** File converted to tombstone explaining the deletion
- **Reason:** Type alias is entropy

### 2. queen-lifecycle/src/ensure.rs
- **BEFORE:** `use crate::types::QueenHandle;`
- **AFTER:** `use daemon_contract::DaemonHandle;`
- **BEFORE:** `pub async fn ensure_queen_running(base_url: &str) -> Result<QueenHandle>`
- **AFTER:** `pub async fn ensure_queen_running(base_url: &str) -> Result<DaemonHandle>`
- **BEFORE:** `QueenHandle::already_running()`, `QueenHandle::started_by_us()`
- **AFTER:** `DaemonHandle::already_running()`, `DaemonHandle::started_by_us()`

### 3. queen-lifecycle/src/lib.rs
- Removed `types` module from module list
- Removed `pub use types::QueenHandle;`
- Added `pub use daemon_contract::DaemonHandle;`
- Updated documentation to remove references to `QueenHandle`

### 4. hive-lifecycle/src/lib.rs
- **BEFORE:** `pub type HiveHandle = daemon_contract::DaemonHandle;`
- **AFTER:** `pub use daemon_contract::DaemonHandle;`
- Added tombstone comment explaining the deletion

## Verification

### Compilation Status

✅ **All crates compile successfully:**

```bash
cargo check -p queen-lifecycle  # ✅ PASS
cargo check -p hive-lifecycle   # ✅ PASS
cargo check -p rbee-keeper      # ✅ PASS
```

### Breaking Changes

**Files changed:** 4 files  
**Lines changed:** ~20 lines  
**Compilation errors:** 0 (no external usage found)

**Impact:** Zero - no code outside these crates used `QueenHandle` or `HiveHandle`

## Why This Matters

### Entropy is Permanent
- Every new developer asks: "QueenHandle vs HiveHandle vs DaemonHandle?"
- Documentation must explain all 3 names
- Grep searches return 3x results
- Refactoring tools must handle all 3 names
- Must maintain 3 names forever

### Breaking Changes are Temporary
- Compiler finds all `QueenHandle` → `DaemonHandle` in 30 seconds
- Fix them all at once
- Done forever

## Decision Matrix

| Current (Entropy) | Correct (Breaking) |
|-------------------|-------------------|
| 3 types for same thing | 1 type |
| Permanent confusion | Temporary compiler errors |
| "Which one do I use?" | "Use DaemonHandle" |
| Must maintain 3 names forever | Fix once, done |

## ROI

**Time to fix:** 15 minutes  
**Benefit:** Eliminates permanent confusion for all future developers  
**Break-even:** Immediate (pre-1.0 software)

## Engineering Rules Compliance

✅ **RULE ZERO:** Breaking changes > backwards compatibility  
✅ **Pre-1.0 Policy:** Destructive actions encouraged  
✅ **No entropy:** Single name for single concept  
✅ **Compiler verification:** All call sites found automatically

## Files Modified

1. `/bin/05_rbee_keeper_crates/queen-lifecycle/src/types.rs` - Converted to tombstone
2. `/bin/05_rbee_keeper_crates/queen-lifecycle/src/ensure.rs` - Use DaemonHandle directly
3. `/bin/05_rbee_keeper_crates/queen-lifecycle/src/lib.rs` - Export DaemonHandle, remove types module
4. `/bin/05_rbee_keeper_crates/hive-lifecycle/src/lib.rs` - Use DaemonHandle directly

## Next Steps

None - violation fixed, all code compiles, no external breakage.

---

**Lesson:** Type aliases for "backwards compatibility" in pre-1.0 software are ALWAYS wrong. Just break the API and let the compiler find all call sites.
