# TEAM-328: Compiler Errors - Breaking Changes

**Status:** 🔨 Ready to fix  
**Date:** Oct 27, 2025

## Summary

✅ **daemon-lifecycle compiles** - Breaking changes complete  
❌ **9 errors in rbee-keeper** - Need to migrate to new API  
❌ **20 errors in narration-core-bdd** - Unrelated (missing stdext dependency)

## Errors to Fix

### File 1: `bin/00_rbee_keeper/src/handlers/hive.rs` (4 errors)

**Line 9:** Import error
```rust
// OLD
use daemon_lifecycle::{
    is_daemon_healthy, rebuild::rebuild_with_hot_reload, rebuild::RebuildConfig, stop_http_daemon,
};

// NEW
use daemon_lifecycle::{
    is_daemon_healthy, install_daemon, InstallDaemonConfig, rebuild::RebuildConfig, stop_http_daemon,
};
```

**Line 136:** Install error
```rust
// OLD
daemon_lifecycle::install_to_local_bin("rbee-hive", None, None).await?;

// NEW
let config = InstallDaemonConfig::new("rbee-hive");
daemon_lifecycle::install_daemon(config).await?;
```

**Lines 142-149:** Uninstall error
```rust
// OLD
let config = daemon_lifecycle::UninstallConfig {
    daemon_name: "rbee-hive".to_string(),
    install_path: install_path.clone(),
    health_url: Some("http://localhost:8600".to_string()),
    health_timeout_secs: Some(2),
    job_id: None,
};
daemon_lifecycle::uninstall_daemon(config).await

// NEW
// Just remove the file - no special function needed
std::fs::remove_file(&install_path)?;
Ok(())
```

### File 2: `bin/00_rbee_keeper/src/handlers/queen.rs` (4 errors)

**Line 14:** Import error
```rust
// OLD
use daemon_lifecycle::{
    HttpDaemonConfig, stop_http_daemon, rebuild::rebuild_with_hot_reload, rebuild::RebuildConfig,
};

// NEW
use daemon_lifecycle::{
    HttpDaemonConfig, stop_http_daemon, install_daemon, InstallDaemonConfig, rebuild::RebuildConfig,
};
```

**Line 93:** Install error
```rust
// OLD
daemon_lifecycle::install_to_local_bin("queen-rbee", binary, None).await?;

// NEW
let config = InstallDaemonConfig::new("queen-rbee")
    .with_source_path(binary);
daemon_lifecycle::install_daemon(config).await?;
```

**Lines 98-105:** Uninstall error
```rust
// OLD
let config = daemon_lifecycle::UninstallConfig {
    daemon_name: "queen-rbee".to_string(),
    install_path: install_path.clone(),
    health_url: Some("http://localhost:7833".to_string()),
    health_timeout_secs: Some(2),
    job_id: None,
};
daemon_lifecycle::uninstall_daemon(config).await

// NEW
// Just remove the file - no special function needed
std::fs::remove_file(&install_path)?;
Ok(())
```

### File 3: `bin/00_rbee_keeper/src/handlers/self_check.rs` (1 error)

**Line 7:** Import error
```rust
// OLD
use observability_narration_macros::narrate_fn;

// NEW
// Remove this line - narrate_fn was deleted (TEAM-328)
```

**Then remove `#[narrate_fn]` attributes** from functions in this file.

## Migration Pattern

### Install
```rust
// OLD
install_to_local_bin(name, source, dir).await?;

// NEW
let config = InstallDaemonConfig::new(name)
    .with_source_path(source)  // Optional
    .with_install_dir(dir);     // Optional
install_daemon(config).await?;
```

### Uninstall
```rust
// OLD
let config = UninstallConfig { ... };
uninstall_daemon(config).await?;

// NEW
std::fs::remove_file(&install_path)?;
```

### Rebuild + Hot Reload
```rust
// OLD
rebuild_with_hot_reload(rebuild_config, daemon_config).await?;

// NEW
let config = InstallDaemonConfig::new(daemon_name)
    .with_rebuild_config(rebuild_config)
    .with_daemon_config(daemon_config);
install_daemon(config).await?;
```

## Next Steps

1. Fix `bin/00_rbee_keeper/src/handlers/hive.rs` (4 errors)
2. Fix `bin/00_rbee_keeper/src/handlers/queen.rs` (4 errors)
3. Fix `bin/00_rbee_keeper/src/handlers/self_check.rs` (1 error)
4. Run `cargo check -p rbee-keeper` to verify
5. Run `cargo check --workspace` to find any other broken crates
6. Update tests
7. Update documentation

## Why This Is Better

**Before (Entropy):**
- 3 different functions for related operations
- User has to orchestrate: stop → uninstall → install → start
- Confusing: "rebuild" doesn't actually install
- Error-prone: forget to stop daemon before uninstall

**After (Clean):**
- 1 function does everything
- User just calls `install_daemon()` with config
- Clear: install means install (with optional rebuild/hot-reload)
- Safe: automatically stops/restarts daemon if needed

---

**TEAM-328: Let the compiler find the bugs. That's what it's for.**
