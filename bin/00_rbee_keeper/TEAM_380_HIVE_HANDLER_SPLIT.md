# TEAM-380: Hive Handler Split

**Date:** Oct 31, 2025  
**Team:** TEAM-380  
**Status:** ✅ COMPLETE

## Mission

Split the hive handler into two separate files:
1. **hive_lifecycle.rs** - Lifecycle operations (start, stop, status, install, uninstall, rebuild)
2. **hive_jobs.rs** - Job operations using job-client (worker/model management)

## Rationale

The keeper performs two distinct types of hive operations:
1. **Lifecycle management** - Uses lifecycle-local/lifecycle-ssh crates
2. **Job operations** - Uses job-client to submit operations to hive's job server

Separating these concerns improves code organization and makes it clear which operations use which pattern.

## Changes Made

### 1. ✅ Created `handlers/hive_lifecycle.rs`

**Purpose:** Handles all lifecycle operations for rbee-hive daemon

**Operations:**
- `Start` - Start rbee-hive daemon (local or remote via SSH)
- `Stop` - Stop rbee-hive daemon
- `Status` - Check if hive is running
- `Install` - Install rbee-hive binary
- `Uninstall` - Remove rbee-hive binary
- `Rebuild` - Rebuild rbee-hive from source

**Key features:**
- Conditional dispatch (localhost uses lifecycle-local, remote uses lifecycle-ssh)
- Network-aware queen URL for remote hives
- SSH config resolution via middleware
- All narration for visibility

**Lines of code:** 253 lines

### 2. ✅ Created `handlers/hive_jobs.rs`

**Purpose:** Submit job operations to hive's job server using job-client

**Functions:**
- `submit_hive_job(operation, hive_url)` - Generic handler for all hive job operations
- `get_hive_url(alias)` - Helper to resolve hive URL from alias

**Usage pattern:**
```rust
use operations_contract::Operation;
use handlers::hive_jobs::submit_hive_job;

let operation = Operation::WorkerSpawn(WorkerSpawnRequest { ... });
submit_hive_job(operation, "http://localhost:7835").await?;
```

**Key features:**
- Uses shared job-client crate
- Streams narration events to stdout
- Handles [DONE], [ERROR], [CANCELLED] markers automatically
- Generic - works for all hive job operations

**Lines of code:** 45 lines

### 3. ✅ Updated `handlers/mod.rs`

**Changes:**
- Removed `pub mod hive;`
- Added `pub mod hive_lifecycle;`
- Added `pub mod hive_jobs;`
- Exported `handle_hive_lifecycle` function
- Exported `submit_hive_job` and `get_hive_url` helpers

### 4. ✅ Updated `cli/mod.rs`

**Changes:**
- Changed `pub use crate::handlers::hive::HiveAction;`
- To `pub use crate::handlers::hive_lifecycle::HiveLifecycleAction;`

### 5. ✅ Updated `cli/commands.rs`

**Changes:**
- Changed `use super::HiveAction;`
- To `use super::HiveLifecycleAction;`
- Updated `Commands::Hive { action: HiveLifecycleAction }`

### 6. ✅ Updated `main.rs`

**Changes:**
- Changed `use handlers::handle_hive;`
- To `use handlers::handle_hive_lifecycle;`
- Updated match arm: `Commands::Hive { action } => handle_hive_lifecycle(action, &queen_url).await`

### 7. ✅ Updated `tauri_commands.rs`

**Changes (5 functions):**
- `hive_start` - Updated imports and function call
- `hive_stop` - Updated imports and function call
- `hive_install` - Updated imports and function call
- `hive_uninstall` - Updated imports and function call
- `hive_rebuild` - Updated imports and function call

All changed from:
```rust
use crate::cli::HiveAction;
use crate::handlers::handle_hive;
handle_hive(HiveAction::Start { ... }, &queen_url)
```

To:
```rust
use crate::cli::HiveLifecycleAction;
use crate::handlers::handle_hive_lifecycle;
handle_hive_lifecycle(HiveLifecycleAction::Start { ... }, &queen_url)
```

### 8. ✅ Updated `lib.rs`

**Changes:**
- Changed `pub use handlers::handle_hive;`
- To `pub use handlers::handle_hive_lifecycle;`

### 9. ✅ Deleted `handlers/hive.rs`

**Reason:** Functionality split into hive_lifecycle.rs and hive_jobs.rs

## Architecture

### Before (Single File)

```
handlers/hive.rs (253 lines)
├─ HiveAction enum
├─ handle_hive() function
│  ├─ Start (lifecycle)
│  ├─ Stop (lifecycle)
│  ├─ Status (lifecycle)
│  ├─ Install (lifecycle)
│  ├─ Uninstall (lifecycle)
│  └─ Rebuild (lifecycle)
└─ (No job operations yet)
```

### After (Two Files)

```
handlers/hive_lifecycle.rs (253 lines)
├─ HiveLifecycleAction enum
└─ handle_hive_lifecycle() function
   ├─ Start (uses lifecycle-local/lifecycle-ssh)
   ├─ Stop (uses lifecycle-local/lifecycle-ssh)
   ├─ Status (uses lifecycle-local/lifecycle-ssh)
   ├─ Install (uses lifecycle-local/lifecycle-ssh)
   ├─ Uninstall (uses lifecycle-local/lifecycle-ssh)
   └─ Rebuild (uses lifecycle-local/lifecycle-ssh)

handlers/hive_jobs.rs (45 lines)
├─ submit_hive_job() - Generic handler using job-client
└─ get_hive_url() - Helper for URL resolution
   (Future: Worker/Model operations will use this)
```

## Usage Examples

### Lifecycle Operations (CLI)

```bash
# Start hive
./rbee hive start

# Stop hive
./rbee hive stop

# Check status
./rbee hive status

# Install binary
./rbee hive install

# Uninstall binary
./rbee hive uninstall

# Rebuild from source
./rbee hive rebuild
```

### Job Operations (Future - Not Yet Wired to CLI)

```rust
// In worker.rs or model.rs handlers
use handlers::hive_jobs::{submit_hive_job, get_hive_url};
use operations_contract::Operation;

// Submit worker spawn operation
let hive_url = get_hive_url(&hive_id);
let operation = Operation::WorkerSpawn(WorkerSpawnRequest {
    hive_id: hive_id.clone(),
    model: "llama-3.2-1b".to_string(),
    worker: "cpu".to_string(),
    device: 0,
});
submit_hive_job(operation, &hive_url).await?;
```

## Benefits

### 1. **Clear Separation of Concerns**
- Lifecycle operations in one file
- Job operations in another file
- Each file has a single, clear purpose

### 2. **Easier to Extend**
- Adding new lifecycle operations → modify hive_lifecycle.rs
- Adding new job operations → use hive_jobs.rs helpers
- No mixing of patterns

### 3. **Better Code Organization**
- 253 lines (lifecycle) + 45 lines (jobs) = 298 total
- Original: 253 lines (lifecycle only, no jobs yet)
- Future job operations won't bloat lifecycle file

### 4. **Consistent with Contract Architecture**
- hive_lifecycle.rs → Uses lifecycle crates (daemon management)
- hive_jobs.rs → Uses job-client (aligns with contracts)
- Clear mapping to architecture layers

## Files Modified

1. **NEW:** `bin/00_rbee_keeper/src/handlers/hive_lifecycle.rs` (253 lines)
2. **NEW:** `bin/00_rbee_keeper/src/handlers/hive_jobs.rs` (45 lines)
3. **MODIFIED:** `bin/00_rbee_keeper/src/handlers/mod.rs` (exports updated)
4. **MODIFIED:** `bin/00_rbee_keeper/src/cli/mod.rs` (re-export updated)
5. **MODIFIED:** `bin/00_rbee_keeper/src/cli/commands.rs` (enum name updated)
6. **MODIFIED:** `bin/00_rbee_keeper/src/main.rs` (import and call updated)
7. **MODIFIED:** `bin/00_rbee_keeper/src/tauri_commands.rs` (5 functions updated)
8. **MODIFIED:** `bin/00_rbee_keeper/src/lib.rs` (re-export updated)
9. **DELETED:** `bin/00_rbee_keeper/src/handlers/hive.rs` (split into 2 files)

## Verification

✅ **Compilation:** `cargo check --bin rbee-keeper` passes

## Next Steps

The hive_jobs.rs module is ready to be used by:
- `handlers/worker.rs` - For worker process operations
- `handlers/model.rs` - For model management operations

These handlers can now use `submit_hive_job()` instead of implementing their own HTTP client logic.

## Summary

Successfully split hive handler into two focused modules:
- ✅ hive_lifecycle.rs - Daemon lifecycle management
- ✅ hive_jobs.rs - Job operations using job-client
- ✅ All references updated across codebase
- ✅ Compilation successful
- ✅ Clear separation of concerns achieved
