# Port 7833 Fix - TEAM-294

**Date:** Oct 25, 2025  
**Issue:** Queen was starting on port 8500 instead of 7833  
**Status:** ✅ FIXED

## Problem

When clicking "Start Queen" in the rbee-keeper UI, queen-rbee was starting on port **8500** instead of the configured port **7833** as specified in `PORT_CONFIGURATION.md`.

## Root Cause

Hardcoded port in `bin/05_rbee_keeper_crates/queen-lifecycle/src/ensure.rs` line 142:

```rust
// OLD (WRONG):
let args = vec!["--port".to_string(), "8500".to_string()];
```

This overrode the correct default port (7833) configured in:
- `bin/00_rbee_keeper/src/config.rs` - default_queen_port() = 7833
- `bin/10_queen_rbee/src/main.rs` - default_value = "7833"

## Solution

Extract the port from the `base_url` parameter instead of hardcoding:

```rust
// NEW (CORRECT):
// base_url format: "http://localhost:7833"
let port = base_url
    .split(':')
    .last()
    .context("Failed to extract port from base_url")?
    .to_string();

let args = vec!["--port".to_string(), port];
```

## Files Modified

### Core Fix
- `bin/05_rbee_keeper_crates/queen-lifecycle/src/ensure.rs` - Extract port from base_url

### Documentation Updates
- `bin/10_queen_rbee/src/main.rs` - Updated comment (8500 → 7833)
- `bin/10_queen_rbee/src/hive_forwarder.rs` - Updated example (8500 → 7833)
- `bin/10_queen_rbee/src/http/info.rs` - Updated example (8500 → 7833)
- `bin/10_queen_rbee/src/http/build_info.rs` - Updated example (8500 → 7833)

## Verification

After this fix:
1. rbee-keeper loads config with queen_port = 7833
2. Calls `ensure_queen_running("http://localhost:7833")`
3. ensure.rs extracts "7833" from the URL
4. Spawns queen with `--port 7833`
5. Queen starts on correct port ✅

## Port Configuration Reference

According to `PORT_CONFIGURATION.md`:

| Service | Port | Purpose |
|---------|------|---------|
| Keeper GUI | 5173 | Tauri frontend (Vite) |
| queen-rbee | **7833** | Backend HTTP API |
| queen-rbee UI | 7834 | Frontend dev server |
| rbee-hive | 7835 | Backend HTTP API |
| rbee-hive UI | 7836 | Frontend dev server |

## Impact

- ✅ Queen now starts on correct port (7833)
- ✅ Consistent with PORT_CONFIGURATION.md
- ✅ No breaking changes (config already had 7833 as default)
- ✅ Works with both CLI and GUI
