# TEAM-329: PID Operations Centralized

**Date:** 2025-10-27  
**Rule:** Centralize related operations - No duplication

## The Problem

**PID operations scattered across start.rs and stop.rs:**

### Before (Duplicated)

**start.rs:**
```rust
fn get_pid_file_path(daemon_name: &str) -> Result<PathBuf> { ... }
// Write PID inline: std::fs::write(&pid_file, pid.to_string())?
```

**stop.rs:**
```rust
fn get_pid_file_path(daemon_name: &str) -> Result<PathBuf> { ... }
fn read_pid_file(daemon_name: &str) -> Result<u32> { ... }
fn remove_pid_file(daemon_name: &str) -> Result<()> { ... }
```

**Result:** Duplication, inconsistency, maintenance burden.

## The Solution

**Centralized in utils/pid.rs:**

```rust
utils/pid.rs:
├── get_pid_file_path()   - Get path to PID file
├── write_pid_file()      - Write PID to file
├── read_pid_file()       - Read PID from file
└── remove_pid_file()     - Remove PID file
```

## Changes Made

### 1. Created utils/pid.rs
- ✅ `get_pid_file_path()` - Get PID file path (with directory creation)
- ✅ `write_pid_file()` - Write PID to file
- ✅ `read_pid_file()` - Read PID from file
- ✅ `remove_pid_file()` - Remove PID file

### 2. Updated start.rs
- ✅ Removed local `get_pid_file_path()` function
- ✅ Import `write_pid_file` from `utils/pid`
- ✅ Use `write_pid_file(&config.daemon_name, pid)?`

### 3. Updated stop.rs
- ✅ Removed all 3 local PID functions
- ✅ Import `read_pid_file`, `remove_pid_file` from `utils/pid`

### 4. Updated utils/paths.rs
- ✅ Removed `get_pid_file_path()` (moved to utils/pid.rs)
- ✅ Kept `get_pid_dir()` (used by utils/pid.rs)

### 5. Updated exports
- ✅ Added `read_pid_file`, `write_pid_file`, `remove_pid_file` to public API

## Before vs After

### Before (Scattered)
```
start.rs:
  - get_pid_file_path() (local)
  - std::fs::write() (inline)

stop.rs:
  - get_pid_file_path() (local, duplicate!)
  - read_pid_file() (local)
  - remove_pid_file() (local)

utils/paths.rs:
  - get_pid_file_path() (exported, but different!)
```

### After (Centralized)
```
utils/pid.rs:
  - get_pid_file_path() ✅
  - write_pid_file() ✅
  - read_pid_file() ✅
  - remove_pid_file() ✅

start.rs:
  - uses write_pid_file()

stop.rs:
  - uses read_pid_file()
  - uses remove_pid_file()

utils/paths.rs:
  - get_pid_dir() (helper for pid.rs)
```

## Benefits

### 1. No Duplication
**Before:** 3 different implementations of PID operations  
**After:** 1 centralized implementation

### 2. Consistency
All PID operations use the same logic:
- Same error messages
- Same path resolution
- Same directory creation

### 3. Easier Testing
Test PID operations once in `utils/pid.rs`, not in multiple places.

### 4. Better Error Handling
Centralized error handling with proper context:
```rust
.context(format!("Failed to write PID file: {}", pid_file.display()))?
```

## Public API

```rust
use daemon_lifecycle::{
    // PID operations (now exported)
    get_pid_file_path,
    write_pid_file,
    read_pid_file,
    remove_pid_file,
    
    // Operations still work the same
    start_daemon,
    stop_daemon,
};
```

**New exports:** PID operations are now part of public API.

## Compilation

✅ **daemon-lifecycle:** PASS (4 warnings, 0 errors)

## Key Insight

**Centralize related operations.**

PID operations are tightly related:
- All work with PID files
- All use same path logic
- All need same error handling

**Solution:** One module (`utils/pid.rs`) for all PID operations.

**No duplication. No inconsistency. One source of truth.**

---

**CENTRALIZATION COMPLETE.**

**4 PID operations → 1 module**

**No more scattered PID logic.**
