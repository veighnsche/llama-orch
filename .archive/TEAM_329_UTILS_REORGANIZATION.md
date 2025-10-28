# TEAM-329: Utils Reorganization - Final Structure

**Date:** 2025-10-27  
**Rule:** Operations vs Utilities - Clear separation

## Changes Made

### 1. Moved to utils/
- ✅ `paths.rs` → `utils/paths.rs` (path utilities)
- ✅ `find.rs` → `utils/find.rs` (already moved)
- ✅ `timeout.rs` → `utils/timeout.rs` (already moved)

### 2. Split health.rs
- ✅ `health.rs` → `status.rs` (checking if daemon is running)
- ✅ `poll_daemon_health()` → `utils/poll.rs` (polling utility)

### 3. Reasoning

**health.rs was confusing:**
- `check_daemon_health()` - Just checks if daemon responds (status check)
- `poll_daemon_health()` - Polls with backoff until running (utility function)

**Solution:**
- **status.rs** - Simple status checking (is it running?)
- **utils/poll.rs** - Polling utility (wait until running)

## Final Structure

```
daemon-lifecycle/src/
├── operations (8 files)
│   ├── build.rs
│   ├── install.rs
│   ├── rebuild.rs
│   ├── shutdown.rs
│   ├── start.rs
│   ├── status.rs          ← Renamed from health.rs
│   ├── stop.rs
│   └── uninstall.rs
│
├── types/ (7 files) - ALL config types
│   ├── handle.rs
│   ├── health.rs          (StatusRequest, StatusResponse, HealthPollConfig)
│   ├── install.rs
│   ├── rebuild.rs
│   ├── shutdown.rs
│   ├── start.rs
│   └── timeout.rs
│
└── utils/ (4 files) - Utilities
    ├── find.rs            (find_binary)
    ├── paths.rs           (get_install_dir, get_pid_file_path, etc.)
    ├── poll.rs            (poll_daemon_health)
    └── timeout.rs         (timeout_after, with_timeout)
```

## What's What

### Operations (src/)
**What you DO with daemons:**
- build - Build from source
- install - Install to ~/.local/bin
- rebuild - Rebuild (with hot reload)
- shutdown - Graceful/force shutdown
- start - Start daemon (spawn + health poll)
- status - Check if running
- stop - Stop daemon (PID-based)
- uninstall - Remove from ~/.local/bin

### Types (types/)
**Config for operations:**
- Each operation has its config types
- types/{operation}.rs matches src/{operation}.rs

### Utils (utils/)
**Helper functions that don't fit operations:**
- find - Find binaries in target/ or ~/.local/bin
- paths - Path utilities (install dir, PID file paths)
- poll - Polling with exponential backoff
- timeout - Timeout enforcement

## Why status.rs?

**Before (health.rs):**
```rust
// Confusing - "health" implies medical check
check_daemon_health()  // Just checks HTTP 200
poll_daemon_health()   // Polls until HTTP 200
```

**After (status.rs + utils/poll.rs):**
```rust
// Clear - "status" = is it running?
status::check_daemon_health()  // Check status (is it running?)
utils::poll::poll_daemon_health()  // Poll until running (utility)
```

**Reasoning:**
- "Health" implies complex checks (CPU, memory, etc.)
- We're just checking "is it responding?" = status check
- Polling is a utility function, not a status operation

## Why paths in utils/?

**paths.rs doesn't DO anything with daemons:**
- It's just helper functions for paths
- `get_install_dir()` - Where to install?
- `get_pid_file_path()` - Where's the PID file?
- Pure utilities, no daemon operations

## Public API (Unchanged)

```rust
use daemon_lifecycle::{
    // Operations
    build_daemon,
    check_daemon_health,    // From status.rs
    install_daemon,
    poll_daemon_health,     // From utils/poll.rs
    rebuild_daemon,
    shutdown_daemon_force,
    shutdown_daemon_graceful,
    start_daemon,
    stop_daemon,
    uninstall_daemon,
    
    // Utils
    find_binary,            // From utils/find.rs
    get_install_dir,        // From utils/paths.rs
    get_install_path,       // From utils/paths.rs
    get_pid_file_path,      // From utils/paths.rs
    timeout_after,          // From utils/timeout.rs
    with_timeout,           // From utils/timeout.rs
    
    // Types (all from types/)
    DaemonHandle,
    HealthPollConfig,
    HttpDaemonConfig,
    InstallConfig,
    InstallResult,
    RebuildConfig,
    ShutdownConfig,
    StatusRequest,
    StatusResponse,
    TimeoutConfig,
    UninstallConfig,
};
```

**All re-exports preserved** - Zero breaking changes.

## Compilation

✅ **daemon-lifecycle:** PASS (2 warnings, 0 errors)

## Key Insights

### 1. Operations vs Utilities
**Operations** = Things you DO (start, stop, install)  
**Utilities** = Helper functions (find, poll, timeout)

### 2. Status vs Health
**Status** = Is it running? (simple HTTP check)  
**Health** = Complex checks (CPU, memory, etc.)

We're doing status checks, not health checks.

### 3. Polling is a Utility
Polling with exponential backoff is a utility function, not a daemon operation.

---

**Final structure:**
- **8 operations** - What you DO
- **7 types** - Config for operations
- **4 utils** - Helper functions

**Clear, organized, no confusion.**
