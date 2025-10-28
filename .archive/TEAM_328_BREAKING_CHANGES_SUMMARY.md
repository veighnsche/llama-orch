# TEAM-328: BREAKING CHANGES - Install/Uninstall/Rebuild Consolidation

**Status:** 🔨 IN PROGRESS - Compiler finding all call sites  
**Date:** Oct 27, 2025

## RULE ZERO Applied

**Breaking changes > Entropy**

Pre-1.0 software is ALLOWED to break. The compiler will catch breaking changes.  
Entropy from "backwards compatibility" functions is PERMANENT TECHNICAL DEBT.

## What We Deleted (Entropy)

### ❌ Deleted Files:
1. `daemon-lifecycle/src/install.rs` - 154 LOC
2. `daemon-lifecycle/src/uninstall.rs` - 90 LOC  
3. `daemon-lifecycle/src/rebuild.rs` - 266 LOC (old version)

**Total deleted:** 510 LOC of entropy

### ❌ Deleted Functions:
1. `install_to_local_bin(binary_name, source_path, install_dir)` - 3 parameters, refused overwrites
2. `uninstall_daemon(config)` - Separate boilerplate
3. `rebuild_with_hot_reload(rebuild_config, daemon_config)` - Misleading name, didn't actually install

### ❌ Deleted Types:
1. `InstallConfig` - Old config type
2. `InstallResult` - Old result type
3. `UninstallConfig` - Separate config type

## What We Created (Clean)

### ✅ New File:
- `daemon-lifecycle/src/install_daemon.rs` - 300 LOC

### ✅ New Function (ONE way to do things):
```rust
pub async fn install_daemon(config: InstallDaemonConfig) -> Result<String>
```

### ✅ New Type:
```rust
pub struct InstallDaemonConfig {
    pub daemon_name: String,
    pub source_path: Option<PathBuf>,        // If None: builds from source
    pub install_dir: Option<PathBuf>,        // If None: ~/.local/bin
    pub rebuild_config: Option<RebuildConfig>, // If Some: builds from source
    pub daemon_config: Option<HttpDaemonConfig>, // If Some: hot reload
    pub job_id: Option<String>,
}
```

### ✅ Simplified rebuild.rs:
- Kept `build_daemon_local()` - Used by install_daemon
- Deleted `rebuild_with_hot_reload()` - Entropy (use install_daemon instead)
- **New size:** 145 LOC (vs 266 LOC before)

## Migration Guide

### Before (3 separate functions):
```rust
// Fresh install
install_to_local_bin("queen-rbee", Some(path), None).await?;

// Uninstall
let config = UninstallConfig { ... };
uninstall_daemon(config).await?;

// Rebuild + hot reload
rebuild_with_hot_reload(rebuild_config, daemon_config).await?;
```

### After (ONE function):
```rust
// Fresh install
let config = InstallDaemonConfig::new("queen-rbee")
    .with_source_path(path);
install_daemon(config).await?;

// Uninstall (just remove the file - no special function needed)
std::fs::remove_file(install_path)?;

// Rebuild + install + hot reload
let config = InstallDaemonConfig::new("queen-rbee")
    .with_rebuild_config(rebuild_config)
    .with_daemon_config(daemon_config);
install_daemon(config).await?;
```

## Compiler Errors (Expected)

### rbee-keeper (9 errors):
```
error[E0432]: unresolved import `daemon_lifecycle::rebuild::rebuild_with_hot_reload`
error[E0425]: cannot find function `install_to_local_bin`
error[E0422]: cannot find struct `UninstallConfig`
error[E0425]: cannot find function `uninstall_daemon`
```

**Fix:** Update to use `install_daemon()` with `InstallDaemonConfig`

### narration-core-bdd (20 errors):
```
error[E0433]: failed to resolve: use of unresolved module or unlinked crate `stdext`
```

**Fix:** Unrelated to this change (missing stdext dependency)

## Benefits

1. ✅ **One function** - Not 3 separate ones
2. ✅ **Overwrites by default** - No "uninstall first" error
3. ✅ **Builds if needed** - Or uses provided binary
4. ✅ **Hot reload built-in** - Stop → install → restart
5. ✅ **No boilerplate duplication** - DRY
6. ✅ **510 LOC deleted** - Less code to maintain

## Code Reduction

**Before:**
- install.rs: 154 LOC
- uninstall.rs: 90 LOC
- rebuild.rs: 266 LOC
- **Total: 510 LOC**

**After:**
- install_daemon.rs: 300 LOC
- rebuild.rs: 145 LOC (simplified)
- **Total: 445 LOC**

**Net savings: 65 LOC (13% reduction)**

Plus:
- **One API** instead of 3
- **No confusion** about which function to use
- **No entropy** from backwards compatibility

## Next Steps

1. ✅ Delete old files
2. ✅ Create new install_daemon.rs
3. ✅ Update lib.rs exports
4. ✅ Let compiler find all call sites
5. 🔨 Fix rbee-keeper (9 errors)
6. 🔨 Fix any other broken crates
7. 🔨 Update tests
8. 🔨 Update documentation

## Files Modified

### Deleted:
- `bin/99_shared_crates/daemon-lifecycle/src/install.rs`
- `bin/99_shared_crates/daemon-lifecycle/src/uninstall.rs`
- `bin/99_shared_crates/daemon-lifecycle/src/rebuild.rs` (old version)

### Created:
- `bin/99_shared_crates/daemon-lifecycle/src/install_daemon.rs`
- `bin/99_shared_crates/daemon-lifecycle/src/rebuild.rs` (simplified)

### Modified:
- `bin/99_shared_crates/daemon-lifecycle/src/lib.rs` (exports)

### Needs Fixing:
- `bin/00_rbee_keeper/src/*.rs` (9 compilation errors)
- Any other crates using old API

---

**TEAM-328: Breaking changes are TEMPORARY. Entropy is FOREVER.**
