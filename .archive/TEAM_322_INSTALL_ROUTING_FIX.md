# TEAM-322: Fixed HiveAction::Install Routing + Removed build_remote

**Status:** ✅ COMPLETE

## Problem

The `HiveAction::Install` handler was incorrectly calling `install_hive()` for all hosts (localhost and remote). However, `install_hive` is just an alias for `daemon_lifecycle::install_to_local_bin`, which only handles **local** installation.

```rust
// WRONG - install_hive only works for localhost
HiveAction::Install { host, binary, install_dir } => {
    install_hive(&host, binary, install_dir).await
}
```

## Root Cause

- `install_hive` (line 71 in hive-lifecycle/src/lib.rs) is an alias for `install_to_local_bin`
- `install_to_local_bin` only copies binaries to `~/.local/bin` (local filesystem)
- `install_hive_remote` existed in `install.rs` but was private and not exported
- Handler didn't route based on host type

## Solution

### 1. Made `install_hive_remote` Public
**File:** `bin/05_rbee_keeper_crates/hive-lifecycle/src/install.rs`
- Changed `async fn install_hive_remote` → `pub async fn install_hive_remote`
- Added TEAM-322 documentation
- Removed all `build_remote` logic (only supports build locally + upload)

### 2. Exported `install_hive_remote`
**File:** `bin/05_rbee_keeper_crates/hive-lifecycle/src/lib.rs`
```rust
pub use install::install_hive_remote; // TEAM-322: Remote installation via SSH
```

### 3. Removed `build_remote` Flag
**Files:** CLI, handlers, install.rs, rebuild.rs
- Removed `build_remote: bool` parameter from all functions
- Removed `--build-remote` CLI flag from Install and Rebuild commands
- Deleted `rebuild_hive_remote_onsite` function (~100 LOC)
- Simplified to single installation mode: **Build locally + upload binary**

### 4. Fixed Handler Routing
**File:** `bin/00_rbee_keeper/src/handlers/hive.rs`
```rust
HiveAction::Install { host, binary, install_dir } => {
    if host == "localhost" || host == "127.0.0.1" {
        // TEAM-322: Local installation - use daemon-lifecycle
        daemon_lifecycle::install_to_local_bin("rbee-hive", binary, install_dir).await?;
        Ok(())
    } else {
        // TEAM-322: Remote installation - use hive-lifecycle
        install_hive_remote(&host, binary, install_dir).await
    }
}
```

## Implementation Details

### Local Installation (localhost)
- Uses `daemon_lifecycle::install_to_local_bin`
- Copies binary to `~/.local/bin`
- No SSH required

### Remote Installation
1. Find local binary (provided path → target/debug → target/release)
2. Connect via SSH
3. Create install directory on remote
4. Upload binary via SCP
5. Make executable (`chmod +x`)
6. Verify installation (`--version`)

## Files Changed

1. **bin/00_rbee_keeper/src/cli/hive.rs**
   - Removed `build_remote` flag from Install command
   - Removed `build_remote` flag from Rebuild command

2. **bin/00_rbee_keeper/src/handlers/hive.rs**
   - Added routing logic (localhost vs remote)
   - Imported `install_hive_remote`
   - Removed `build_remote` parameter

3. **bin/00_rbee_keeper/src/tauri_commands.rs**
   - Removed `build_remote: false` from tauri command

4. **bin/05_rbee_keeper_crates/hive-lifecycle/src/install.rs**
   - Made `install_hive_remote` public
   - Removed `build_remote` parameter
   - Removed remote build logic (~70 LOC deleted)

5. **bin/05_rbee_keeper_crates/hive-lifecycle/src/rebuild.rs**
   - Removed `build_remote` parameter from `rebuild_hive`
   - Removed `build_remote` parameter from `rebuild_hive_remote`
   - Deleted `rebuild_hive_remote_onsite` function (~100 LOC deleted)

6. **bin/05_rbee_keeper_crates/hive-lifecycle/src/lib.rs**
   - Exported `install_hive_remote`

## Verification

```bash
# Both packages compile successfully
cargo check -p hive-lifecycle  # ✅ PASS
cargo check -p rbee-keeper     # ✅ PASS
```

## Usage

```bash
# Local installation (no SSH)
rbee-keeper hive install localhost

# Remote installation (build locally + upload)
rbee-keeper hive install gpu-server

# Rebuild (local)
rbee-keeper hive rebuild localhost

# Rebuild (remote: build locally + upload)
rbee-keeper hive rebuild gpu-server
```

## Benefits

- ✅ Correct routing based on host type
- ✅ Local installation uses fast local copy
- ✅ Remote installation: build locally + upload (fast, no git needed on remote)
- ✅ Simplified CLI (removed unnecessary `--build-remote` flag)
- ✅ Clear separation of concerns
- ✅ ~170 LOC removed (build_remote logic)

## Code Signatures

All changes marked with `// TEAM-322:` comments for traceability.
