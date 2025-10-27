# TEAM-329: Minimal Cleanup (No Behavior Changes)

**Date:** Oct 27, 2025  
**Status:** ✅ COMPLETE

## Problem

After type alignment, found additional inconsistencies that could be fixed without changing behavior.

## Fixes Applied

### 1. ✅ Removed Unused Imports (3 files)

**rebuild.rs:15**
```rust
- use std::process::Command; // ❌ UNUSED
```

**start.rs:8**
```rust
- use std::path::PathBuf; // ❌ UNUSED
```

**stop.rs:7**
```rust
- use std::path::PathBuf; // ❌ UNUSED
```

**Result:** Eliminated all 3 unused import warnings from `cargo check`

## Remaining Inconsistencies (Documented, Not Fixed)

### 2. Function Naming Pattern
- ✅ `shutdown_daemon_force()` - Good
- ✅ `shutdown_daemon_graceful()` - Good
- ✅ `start_daemon()` - Good
- ✅ `stop_daemon()` - Good
- ⚠️ `update_daemon()` - Should be `rebuild_daemon()` for consistency

**Why not fixed:** Would require updating all call sites. Breaking change.

### 3. Narration Pattern Inconsistency
- ✅ Most files use `#[with_job_id]` macro (modern)
- ✅ `build.rs` uses `n!()` directly (simple, no job_id needed)
- ⚠️ `shutdown.rs` uses `with_narration_context()` wrapper (old pattern)

**Why not fixed:** `shutdown.rs` has complex dual-path logic (HTTP vs signal-based). Refactoring would require careful testing.

### 4. Empty Type Files
- `types/build.rs` - Just comments (no types needed)
- `types/stop.rs` - Just comments (reuses HttpDaemonConfig)

**Why not fixed:** Intentional for structural parity. Each operation has a types file, even if empty.

### 5. Doc Comment Inconsistency
- Some functions have `# Example` sections
- Others don't
- No consistent pattern

**Why not fixed:** Low priority. Examples are helpful but not required.

### 6. Import Ordering
- Some files: `use crate::` before `use std::`
- Others: reversed

**Why not fixed:** Rustfmt doesn't enforce this. Low priority cosmetic issue.

## Verification

```bash
# Before: 4 warnings (3 unused imports + 1 deprecated)
cargo check -p daemon-lifecycle

# After: 1 warning (1 deprecated - intentional)
cargo check -p daemon-lifecycle
# ✅ PASS - Only expected deprecation warning remains
```

## Summary

**Fixed:**
- ✅ 3 unused imports removed
- ✅ Compilation warnings reduced from 4 → 1

**Not Fixed (Documented):**
- ⚠️ Function naming (`update_daemon` → would be breaking change)
- ⚠️ Narration pattern in `shutdown.rs` (complex refactor)
- ⚠️ Empty type files (intentional for parity)
- ⚠️ Doc comment consistency (low priority)
- ⚠️ Import ordering (cosmetic)

**Result:** Minimal, safe cleanup with zero behavior changes. All remaining inconsistencies are documented with rationale.

---

**Files Modified:** 3 files (-3 lines)  
**Behavior Changes:** 0  
**Breaking Changes:** 0  
**Compilation:** ✅ PASS (1 expected warning)
