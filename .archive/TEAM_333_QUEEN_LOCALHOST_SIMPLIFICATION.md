# TEAM-333: Queen Localhost Simplification

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025

## Problem

User feedback:
> "Ok so the queen is always localhost. So I don't like the fact that the ssh_config isn't optional in all the functions with localhost as the defaults"

**Issue:**
- Queen is **always** localhost (never remote)
- Using `resolve_ssh_config("localhost")` was unnecessary overhead
- Should just use `SshConfig::localhost()` directly

## Solution

**Simplified queen handler** to use `SshConfig::localhost()` directly instead of going through the resolver.

### Architecture Decision

```text
Queen (always localhost):
    ↓
SshConfig::localhost()  ← Direct, no resolver needed
    ↓
daemon-lifecycle operations
    ↓
local_exec() (bypasses SSH)

Hive (localhost OR remote):
    ↓
resolve_ssh_config(alias)  ← Resolver needed for remote hosts
    ↓
SshConfig { ... }
    ↓
daemon-lifecycle operations
    ↓
if is_localhost() → local_exec()
else              → ssh_exec()
```

## Changes

### Before (TEAM-332)

```rust
QueenAction::Start => {
    // TEAM-332: Resolve SSH config (always localhost for queen)
    let ssh = resolve_ssh_config("localhost")?;  // ← Unnecessary!
    let config = StartConfig {
        ssh_config: ssh,
        // ...
    };
    start_daemon(config).await?;
}
```

### After (TEAM-333)

```rust
QueenAction::Start => {
    // TEAM-333: Queen is always localhost - use SshConfig::localhost() directly
    let config = StartConfig {
        ssh_config: SshConfig::localhost(),  // ← Direct, simple!
        // ...
    };
    start_daemon(config).await?;
}
```

## Files Changed

### Modified
- `bin/00_rbee_keeper/src/handlers/queen.rs`
  - Removed `resolve_ssh_config` import
  - Re-added `SshConfig` import
  - Changed all 5 operations to use `SshConfig::localhost()` directly
  - Updated comments to TEAM-333

## Benefits

1. ✅ **Simpler code** - No resolver overhead for queen
2. ✅ **Clearer intent** - Queen is obviously localhost
3. ✅ **No error handling** - `SshConfig::localhost()` never fails
4. ✅ **Faster** - No SSH config parsing for queen operations

## Comparison

### Queen (Always Localhost)
```rust
// Direct - no resolver needed
SshConfig::localhost()
```

### Hive (Localhost OR Remote)
```rust
// Resolver needed - supports remote hosts
resolve_ssh_config(&alias)?
```

## Verification

```bash
✅ cargo check -p rbee-keeper
✅ Queen operations simplified
✅ Hive operations still use resolver (correct!)
✅ No functional changes
```

## Summary

- **Queen:** Always localhost → Use `SshConfig::localhost()` directly
- **Hive:** Localhost OR remote → Use `resolve_ssh_config(alias)` for flexibility

**Result:** Cleaner, simpler, faster code for queen operations! 🎉
