# TEAM-329: Final Structure - Complete Reorganization

**Date:** 2025-10-27  
**Rule:** RULE ZERO - Consistent structure, no scattered types

## Final Structure

```
daemon-lifecycle/src/
├── operations/ (11 files)
│   ├── build.rs
│   ├── health.rs
│   ├── install.rs
│   ├── paths.rs
│   ├── rebuild.rs
│   ├── shutdown.rs
│   ├── start.rs
│   ├── stop.rs
│   └── uninstall.rs
│
├── types/ (7 files) - ALL config types
│   ├── handle.rs         (DaemonHandle)
│   ├── health.rs          (StatusRequest, StatusResponse, HealthPollConfig)
│   ├── install.rs         (InstallConfig, InstallResult, UninstallConfig)
│   ├── rebuild.rs         (RebuildConfig)
│   ├── shutdown.rs        (ShutdownConfig)
│   ├── start.rs           (HttpDaemonConfig)
│   └── timeout.rs         (TimeoutConfig)
│
└── utils/ (2 files) - Utilities
    ├── find.rs            (find_binary)
    └── timeout.rs         (timeout_after, with_timeout)
```

## Mapping: Operations → Types

| Operation | Config Type | Location |
|-----------|-------------|----------|
| build.rs | (none) | - |
| health.rs | HealthPollConfig | types/health.rs |
| install.rs | InstallConfig, InstallResult | types/install.rs |
| paths.rs | (none) | - |
| rebuild.rs | RebuildConfig | types/rebuild.rs |
| shutdown.rs | ShutdownConfig | types/shutdown.rs |
| start.rs | HttpDaemonConfig | types/start.rs |
| stop.rs | (uses HttpDaemonConfig) | types/start.rs |
| uninstall.rs | UninstallConfig | types/install.rs |
| utils/timeout.rs | TimeoutConfig | types/timeout.rs |

## What Changed

### 1. Deleted
- ✅ `src/list.rs` - UNUSED (zero consumers)
- ✅ `bin/97_contracts/daemon-contract/` - Inlined into types/

### 2. Moved to utils/
- ✅ `src/find.rs` → `src/utils/find.rs`
- ✅ `src/timeout.rs` → `src/utils/timeout.rs`

### 3. Extracted to types/
- ✅ `HealthPollConfig` from `src/health.rs` → `types/health.rs`
- ✅ `RebuildConfig` from `src/rebuild.rs` → `types/rebuild.rs`
- ✅ `TimeoutConfig` from `src/utils/timeout.rs` → `types/timeout.rs`

### 4. Renamed in types/
- ✅ `types/lifecycle.rs` → `types/start.rs` (matches src/start.rs)
- ✅ `types/status.rs` → `types/health.rs` (matches src/health.rs)

## Public API (Unchanged)

```rust
use daemon_lifecycle::{
    // Operations
    build_daemon,
    check_daemon_health,
    install_daemon,
    poll_daemon_health,
    rebuild_daemon,
    shutdown_daemon_force,
    shutdown_daemon_graceful,
    start_daemon,
    stop_daemon,
    uninstall_daemon,
    
    // Utils
    find_binary,
    timeout_after,
    with_timeout,
    
    // Types
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

**All re-exports preserved** - Users see no breaking changes.

## Principles Applied

### 1. No Types in Root
**Before:** Config types scattered in operation modules  
**After:** ALL config types in `types/` folder

### 2. File Naming Parity
**Rule:** `types/{operation}.rs` matches `src/{operation}.rs`

**Examples:**
- `src/start.rs` ←→ `types/start.rs`
- `src/health.rs` ←→ `types/health.rs`
- `src/rebuild.rs` ←→ `types/rebuild.rs`

### 3. Utils Folder
**Before:** `find.rs` and `timeout.rs` mixed with operations  
**After:** Moved to `utils/` folder (not operations)

### 4. Complete Extraction
**Before:** Some types in daemon-contract, some in root modules  
**After:** ALL types in `types/` folder

## Compilation Status

✅ **daemon-lifecycle:** PASS (2 warnings, 0 errors)  
✅ **All imports updated**  
✅ **Public API unchanged**

## Benefits

### 1. Clear Organization
```
operations/  - What you can DO
types/       - Config for operations
utils/       - Helper functions
```

### 2. Easy Navigation
Want config for `start.rs`? → Look in `types/start.rs`  
Want config for `health.rs`? → Look in `types/health.rs`

### 3. No Confusion
- Operations are verbs (start, stop, build)
- Types are nouns (HttpDaemonConfig, RebuildConfig)
- Utils are helpers (find_binary, timeout_after)

### 4. Scalability
Adding new operation? Create:
1. `src/{operation}.rs` - Implementation
2. `types/{operation}.rs` - Config types

**Convention enforced by structure.**

---

**Key Insight:** Structure should make the right thing obvious. When you see `src/start.rs`, you know config is in `types/start.rs`. No guessing.

**RULE ZERO:** Consistent structure > scattered files. Fix it now, not later.
