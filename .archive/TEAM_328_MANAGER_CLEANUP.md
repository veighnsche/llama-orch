# TEAM-328: DaemonManager RULE ZERO Cleanup

**Status:** ✅ COMPLETE

**Mission:** Remove backwards compatibility entropy from `daemon-lifecycle/src/manager.rs`

## Problem

`manager.rs` violated RULE ZERO by providing multiple ways to do the same thing:
1. **Two binary resolution functions** - `find_binary()` and `find_in_target()`
2. **Unused wrapper function** - `spawn_daemon()` (nobody used it)

This created entropy:
- Confusion about which function to use
- Duplicate code paths
- Unnecessary API surface

## Solution

**Deleted 2 functions:**

### 1. Deleted `find_in_target()` (47 lines)

**Before:**
```rust
pub fn find_binary(name: &str) -> Result<PathBuf> {
    // Check installed → debug → release
}

pub fn find_in_target(name: &str) -> Result<PathBuf> {
    // Check debug → release only
}
```

**After:**
```rust
pub fn find_binary(name: &str) -> Result<PathBuf> {
    // Check installed → debug → release
    // (Inlined the target search logic)
}
```

**Why:** `find_in_target()` was redundant - `find_binary()` already does everything it did, plus checks installed location.

### 2. Deleted `spawn_daemon()` (4 lines)

**Before:**
```rust
pub async fn spawn_daemon<P: AsRef<Path>>(binary_path: P, args: Vec<String>) -> Result<Child> {
    let manager = DaemonManager::new(binary_path.as_ref().to_path_buf(), args);
    manager.spawn().await
}
```

**After:**
```rust
// Deleted - use DaemonManager::new().spawn() directly
```

**Why:** Nobody used it! All callers use `DaemonManager::new().spawn()` directly.

## Files Changed

**Modified:**
- `bin/99_shared_crates/daemon-lifecycle/src/manager.rs` (-51 LOC)
  - Deleted `find_in_target()` function
  - Inlined target search logic into `find_binary()`
  - Deleted `spawn_daemon()` wrapper
  
- `bin/99_shared_crates/daemon-lifecycle/src/lib.rs` (-1 LOC, updated docs)
  - Removed `spawn_daemon` from exports
  - Updated documentation to use `find_binary()`
  
- `bin/99_shared_crates/daemon-lifecycle/tests/stdio_null_tests.rs` (+4 LOC, -4 LOC)
  - Updated `test_find_in_target_*` → `test_find_binary_*`
  - Tests now use `find_binary()` instead of deleted function

## Impact

**Breaking changes:**
- `DaemonManager::find_in_target()` - DELETED
  - Only used in 1 test file (updated)
  - No external callers
  
- `spawn_daemon()` - DELETED
  - Zero callers in entire codebase
  - No impact

**Benefits:**
- ✅ Simpler API (2 fewer public functions)
- ✅ Less confusion (one way to find binaries)
- ✅ Follows RULE ZERO (no backwards compatibility entropy)
- ✅ Cleaner codebase (-51 LOC)

## Verification

**Compilation:**
```bash
cargo check -p daemon-lifecycle
# ✅ PASS
```

**Tests:**
```bash
cargo test -p daemon-lifecycle --test stdio_null_tests
# ✅ PASS (9 tests)
# - test_find_binary_debug_binary ... ok
# - test_find_binary_missing_binary_error ... ok
```

## What Remains

**Core DaemonManager functionality (kept):**
- ✅ `new()` - Create daemon manager
- ✅ `spawn()` - Spawn daemon process
- ✅ `find_binary()` - Find binary (installed → debug → release)
- ✅ `enable_auto_update()` - Enable auto-rebuild (used by worker-lifecycle)

**All essential functionality preserved, entropy eliminated.**

## Code Signatures

All changes marked with `// TEAM-328:`

---

**Result:** DaemonManager now follows RULE ZERO - one way to do each thing, no backwards compatibility entropy.
