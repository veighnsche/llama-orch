# TEAM-373: Rebuild Bug Fix - Binary Not Copied to Remote

**Date:** 2025-11-01  
**Status:** ✅ FIXED

## Problem

`rebuild_daemon()` was not copying the newly built binary to the remote system. The daemon would stop, but the old binary remained installed.

### Root Cause

The `install_daemon()` function had a guard that checked if the binary was already installed and **bailed out** with an error:

```rust
// install.rs line 147-158
if check_binary_installed(daemon_name, ssh_config).await {
    anyhow::bail!("{} is already installed. Use rebuild to update.", daemon_name);
}
```

When `rebuild_daemon()` called `install_daemon()` after building the new binary, this check would fail because the binary existed on the remote system, preventing the new binary from being copied.

### User Impact

Users running `rebuild` would see:
1. ✅ Build complete locally
2. ✅ Daemon stopped on remote
3. ⚠️  "rbee-hive is already installed" (no copy happened)
4. ❌ Old binary still running after restart

## Solution

Added `force_reinstall: bool` field to `InstallConfig` struct:

```rust
pub struct InstallConfig {
    pub daemon_name: String,
    pub ssh_config: SshConfig,
    pub local_binary_path: Option<PathBuf>,
    pub job_id: Option<String>,
    pub force_reinstall: bool,  // TEAM-373: New field
}
```

### Behavior

- **Normal install** (`force_reinstall: false`): Check if binary exists, bail if already installed
- **Rebuild** (`force_reinstall: true`): Skip existence check, overwrite existing binary

### Implementation

**File 1: `lifecycle-ssh/src/install.rs`**
- Added `force_reinstall: bool` field to `InstallConfig`
- Modified existence check: `if !install_config.force_reinstall && check_binary_installed(...)`

**File 2: `lifecycle-ssh/src/rebuild.rs`**
- Set `force_reinstall: true` when calling `install_daemon()`

**File 3: `rbee-keeper/src/handlers/hive_lifecycle.rs`**
- Set `force_reinstall: false` for normal install operations

## Files Changed

### lifecycle-ssh (Remote Operations)
1. `bin/96_lifecycle/lifecycle-ssh/src/install.rs` (+4 lines, modified 1 line)
   - Added `force_reinstall: bool` field to `InstallConfig`
   - Modified existence check to respect `force_reinstall` flag

2. `bin/96_lifecycle/lifecycle-ssh/src/rebuild.rs` (+1 line)
   - Set `force_reinstall: true` when calling `install_daemon()`

### lifecycle-local (Local Operations)
3. `bin/96_lifecycle/lifecycle-local/src/install.rs` (+4 lines, modified 1 line)
   - Added `force_reinstall: bool` field to `InstallConfig`
   - Modified existence check to respect `force_reinstall` flag

4. `bin/96_lifecycle/lifecycle-local/src/rebuild.rs` (+1 line)
   - Set `force_reinstall: true` when calling `install_daemon()`

### rbee-keeper (CLI Tool)
5. `bin/00_rbee_keeper/src/handlers/hive_lifecycle.rs` (+2 lines)
   - Set `force_reinstall: false` for normal install (both local and remote)

6. `bin/00_rbee_keeper/src/handlers/queen.rs` (+1 line)
   - Set `force_reinstall: false` for queen install

## Testing

```bash
# Verify compilation
cargo check --bin rbee-keeper

# Test rebuild flow
# 1. Build locally ✅
# 2. Stop remote daemon ✅
# 3. Copy new binary (now works!) ✅
# 4. Start daemon ✅
```

## Code Signatures

All changes marked with `// TEAM-373: [description]`

---

**TEAM-373: Fixed rebuild not copying new binary to remote system**
