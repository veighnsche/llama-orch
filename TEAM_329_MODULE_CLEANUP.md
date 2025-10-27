# TEAM-329: Module Cleanup - Accurate Naming

**Date:** 2025-10-27  
**Rule:** RULE ZERO - Single responsibility, accurate naming

## Changes Made

### 1. Renamed `manager.rs` → `find.rs`

**Why:** The module only contains `find_binary()` function. "manager" implied it managed daemons, but after deleting `spawn()`, it only finds binaries.

**Accurate naming:** `find.rs` clearly states its purpose.

### 2. Extracted `build_daemon()` from `install.rs` → `build.rs`

**Why:** Single responsibility principle. Building and installing are separate concerns.

**Before:**
- `install.rs` - 183 LOC (build + install logic mixed)

**After:**
- `build.rs` - 59 LOC (build only)
- `install.rs` - 124 LOC (install only)

### 3. Identified `list.rs` as UNUSED

**Status:** ⚠️ DEAD CODE

**Evidence:**
- Exported in `lib.rs` but never implemented
- No implementations of `ListableConfig` trait exist
- Generic trait pattern but zero usage

**Purpose (intended):**
Generic daemon listing functionality for hive-lifecycle and worker-lifecycle to avoid duplication.

**Reality:**
- hive-lifecycle never implemented it
- worker-lifecycle never implemented it
- Both crates implement their own listing logic

**Recommendation:** DELETE `list.rs` in next cleanup pass.

## Files Changed

### Modified
- `bin/99_shared_crates/daemon-lifecycle/src/find.rs` (renamed from manager.rs)
  - Updated module docs
  - TEAM-329 attribution

- `bin/99_shared_crates/daemon-lifecycle/src/build.rs` (NEW)
  - Extracted `build_daemon()` from install.rs
  - 59 LOC

- `bin/99_shared_crates/daemon-lifecycle/src/install.rs`
  - Removed `build_daemon()` (moved to build.rs)
  - Import from `crate::build`
  - Updated docs

- `bin/99_shared_crates/daemon-lifecycle/src/start.rs`
  - Import from `crate::find` instead of `crate::manager`

- `bin/99_shared_crates/daemon-lifecycle/src/lib.rs`
  - Module declarations updated
  - Exports updated
  - Marked `list` module as UNUSED

## Compilation Status

✅ **daemon-lifecycle:** PASS (2 warnings, 0 errors)

## Module Structure (After Cleanup)

```
daemon-lifecycle/
├── build.rs         ✅ Build binaries from source
├── find.rs          ✅ Find installed/development binaries
├── health.rs        ✅ Health checking
├── install.rs       ✅ Install binaries to ~/.local/bin
├── list.rs          ⚠️  UNUSED - candidate for deletion
├── paths.rs         ✅ Centralized path constants
├── rebuild.rs       ✅ Hot reload (rebuild + restart)
├── shutdown.rs      ✅ Graceful/force shutdown
├── start.rs         ✅ Start daemons (spawn + health poll)
├── stop.rs          ✅ Stop daemons (PID-based)
├── timeout.rs       ✅ Timeout enforcement
└── uninstall.rs     ✅ Uninstall binaries
```

## Why list.rs is UNUSED

### Intended Design
```rust
// Generic trait for listing daemons
pub trait ListableConfig {
    type Info: Serialize;
    fn list_all(&self) -> Vec<Self::Info>;
    fn daemon_type_name(&self) -> &'static str;
}

// Generic list function
pub async fn list_daemons<T: ListableConfig>(
    config: &T,
    job_id: Option<&str>,
) -> Result<Vec<T::Info>>
```

### Reality Check
```bash
# Search for implementations
$ rg "impl ListableConfig" --type rust
# NO RESULTS

# Search for usage
$ rg "list_daemons" --type rust
# Only in lib.rs export, never called
```

### Why It Failed
1. **Over-engineered:** Generic trait for 2 use cases (hive, worker)
2. **Not needed:** Each crate has different listing requirements
3. **Never adopted:** Both hive-lifecycle and worker-lifecycle implemented their own

### Correct Approach
- hive-lifecycle: List from config file
- worker-lifecycle: List from registry/state
- Different data sources = different implementations
- No need for shared trait

## Next Steps

1. ✅ **Compilation verified** - daemon-lifecycle builds
2. ⚠️ **Fix broken tests** - stdio_null_tests.rs uses deleted `spawn()`
3. ⚠️ **Fix worker-lifecycle** - uses deleted `DaemonManager`
4. 🗑️ **Delete list.rs** - unused code (future cleanup)

---

**Key Insight:** Accurate naming prevents confusion. `manager.rs` implied daemon management, but it only finds binaries. `find.rs` is honest.

**RULE ZERO:** If a module name doesn't match its contents, rename it. Don't keep misleading names for "backwards compatibility."
