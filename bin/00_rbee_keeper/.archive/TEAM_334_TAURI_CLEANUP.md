# TEAM-334: Tauri Commands Cleanup

**Date:** Oct 28, 2025  
**Status:** ✅ COMPLETE

## Mission

Clean up `tauri_commands.rs` - remove 90% of unused commands, keep only `ssh_list` for now. Rest will be re-implemented later when architecture stabilizes.

## Changes Made

### 1. tauri_commands.rs (466 → 137 lines, 71% reduction)

**Removed:**
- All Queen commands (start, stop, status, rebuild, install, uninstall)
- All Hive commands (install, uninstall, start, stop, status, refresh_capabilities)
- All Worker commands (spawn, process_list, process_get, process_delete)
- All Model commands (download, list, get, delete)
- All Inference commands (infer)
- CommandResponse helper types
- to_response_unit helper function

**Kept:**
- `ssh_list` command (TEAM-333)
- `SshTarget` and `SshTargetStatus` types
- TypeScript binding test

### 2. main.rs

**Updated `launch_gui()` function:**
```rust
// Before: 28 commands registered
tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![
        get_status,
        queen_start, queen_stop, queen_status, queen_rebuild,
        queen_install, queen_uninstall,
        hive_install, hive_uninstall, hive_start, hive_stop,
        hive_status, hive_refresh_capabilities,
        ssh_list,
        worker_spawn, worker_process_list, worker_process_get, worker_process_delete,
        model_download, model_list, model_get, model_delete,
        infer,
    ])

// After: 1 command registered
tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![
        ssh_list,
    ])
```

**Fixed unused imports:**
- Removed `config` import (unused after cleanup)
- Prefixed `_hive_id` parameter in `job_client.rs` (unused but kept for future)

### 3. TypeScript Bindings

**Generated bindings now only expose:**
```typescript
export const commands = {
  async sshList() : Promise<Result<SshTarget[], string>> {
    return { status: "ok", data: await TAURI_INVOKE("ssh_list") };
  }
}
```

## Verification

✅ **Compilation:** Clean build, no errors  
✅ **Tests:** TypeScript binding test passes  
✅ **Warnings:** Fixed all unused import/variable warnings  
✅ **TypeScript:** Generated bindings only expose `ssh_list`

## Code Reduction

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| tauri_commands.rs | 466 lines | 137 lines | 329 lines (71%) |
| main.rs (launch_gui) | 28 commands | 1 command | 27 commands removed |

## Why This Cleanup?

1. **Architecture in flux:** Too many changes needed across the board
2. **90% unused:** Most commands not being used by UI yet
3. **Better to rebuild:** When architecture stabilizes, re-implement with correct patterns
4. **SSH works:** Keep what's working (ssh_list for hive discovery)

## Next Steps

When ready to re-implement Tauri commands:
1. Start with Queen lifecycle (start/stop/status)
2. Add Hive management (start/stop/status)
3. Add Worker operations (spawn/list/delete)
4. Add Model operations (download/list/delete)
5. Add Inference (last, most complex)

Each command should be added incrementally with proper testing and UI integration.

## Team Signatures

- TEAM-334: Tauri commands cleanup (this document)
- TEAM-333: SSH list command (preserved)
- TEAM-297: Specta v2 TypeScript bindings (preserved)
- TEAM-293: Original Tauri command wrappers (mostly removed)
