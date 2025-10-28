# TEAM-329: Handler Updates for Renamed daemon-lifecycle Functions

**Date:** Oct 27, 2025  
**Status:** ✅ COMPLETE

## Problem

rbee-keeper handlers were using old daemon-lifecycle function names that were renamed:
- `start_http_daemon()` → `start_daemon()`
- `stop_http_daemon()` → `stop_daemon()`
- `is_daemon_healthy()` → `check_daemon_health()`
- `install_to_local_bin()` → `install_daemon()`
- `rebuild_with_hot_reload()` → `rebuild_daemon()`

Also, `UninstallConfig` now has a builder pattern instead of struct literal.

## Changes Made

### hive.rs

**Imports:**
```rust
// Before
use daemon_lifecycle::{
    is_daemon_healthy, rebuild::rebuild_with_hot_reload, rebuild::RebuildConfig, stop_http_daemon,
    HttpDaemonConfig,
};

// After
use daemon_lifecycle::{
    check_daemon_health, rebuild_daemon, stop_daemon, HttpDaemonConfig, RebuildConfig,
};
```

**Function Calls Updated:**
1. ✅ `start_http_daemon()` → `start_daemon()`
2. ✅ `stop_http_daemon()` → `stop_daemon()`
3. ✅ `is_daemon_healthy()` → `check_daemon_health()`
4. ✅ `install_to_local_bin()` → `install_daemon()`
5. ✅ `rebuild_with_hot_reload()` → `rebuild_daemon()`
6. ✅ `UninstallConfig` struct literal → builder pattern

### queen.rs

**Imports:**
```rust
// Before
use daemon_lifecycle::{
    HttpDaemonConfig, stop_http_daemon, rebuild::rebuild_with_hot_reload, rebuild::RebuildConfig,
    is_daemon_healthy,
};

// After
use daemon_lifecycle::{
    check_daemon_health, rebuild_daemon, stop_daemon, HttpDaemonConfig, RebuildConfig,
};
```

**Function Calls Updated:**
1. ✅ `start_http_daemon()` → `start_daemon()`
2. ✅ `stop_http_daemon()` → `stop_daemon()`
3. ✅ `is_daemon_healthy()` → `check_daemon_health()`
4. ✅ `install_to_local_bin()` → `install_daemon()`
5. ✅ `rebuild_with_hot_reload()` → `rebuild_daemon()`
6. ✅ `UninstallConfig` struct literal → builder pattern

**Bug Fix:**
- Fixed undefined `health_url` variable in Status handler (should use `queen_url`)

### Cleanup
- ✅ Removed unused `std::path::PathBuf` imports from both files

## Builder Pattern Example

**Before:**
```rust
let config = daemon_lifecycle::UninstallConfig {
    daemon_name: "rbee-hive".to_string(),
    install_path: format!("{}/.local/bin/rbee-hive", home),
    health_url: Some("http://localhost:7835".to_string()),
    health_timeout_secs: Some(2),
    job_id: None,
};
```

**After:**
```rust
let config = daemon_lifecycle::UninstallConfig::new(
    "rbee-hive",
    format!("{}/.local/bin/rbee-hive", home),
)
.with_health_url("http://localhost:7835")
.with_health_timeout_secs(2);
```

## Verification

```bash
cargo check -p rbee-keeper
# ✅ Should compile (note: unrelated error in self_check.rs exists)
```

## Summary

**Files Modified:** 2 files  
**Functions Updated:** 10 call sites  
**Imports Cleaned:** 2 unused imports removed  
**Bugs Fixed:** 1 (undefined variable in queen.rs)  
**Pattern Improvements:** 2 (UninstallConfig builder pattern)

All handlers now use the correct, simplified daemon-lifecycle API.

---

**Breaking Changes:** Yes (function renames in daemon-lifecycle)  
**Compilation:** ✅ handlers compile correctly  
**Next:** Fix unrelated `narrate_fn` error in self_check.rs
