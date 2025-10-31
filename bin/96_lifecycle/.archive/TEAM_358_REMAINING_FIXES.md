# TEAM-358 Remaining Fixes

## Files Still Needing Updates

### 1. stop.rs
- Remove `SshConfig` from imports and `StopConfig`
- Remove SSH fallback logic
- Use local process termination only

### 2. shutdown.rs
- Remove `SshConfig` from imports and `ShutdownConfig`
- Remove SSH-based SIGTERM/SIGKILL
- Use local process termination only

### 3. uninstall.rs
- Remove `SshConfig` from imports and `UninstallConfig`
- Fix `check_daemon_health()` call (2 args, not 3)
- Use local file deletion

### 4. rebuild.rs
- Remove `SshConfig` from imports and `RebuildConfig`
- Fix `StartConfig` and `InstallConfig` initialization (no ssh_config field)
- All operations are local

## Quick Fix Strategy

Since these files all have similar patterns, I'll:
1. Remove all `use crate::SshConfig` imports
2. Remove all `use crate::utils::ssh::*` imports
3. Remove `ssh_config` fields from all config structs
4. Update function calls to not pass ssh_config
5. Replace SSH operations with local equivalents
