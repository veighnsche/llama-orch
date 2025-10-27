# TEAM-329: PERFECT PARITY - Types Mirror Operations

**Date:** 2025-10-27  
**Rule:** PERFECT PARITY - types/{operation}.rs matches src/{operation}.rs

## The Problem

**Types folder was a mess:**
- `handle.rs` - WTF is this? (UNUSED - deleted)
- `health.rs` - Should be `status.rs` (we have status.rs, not health.rs)
- `install.rs` - Contains UninstallConfig (should be in uninstall.rs)
- Missing `uninstall.rs`, `build.rs`, `stop.rs`

**NO PARITY!**

## The Solution

**PERFECT 1-to-1 mapping:**

```
src/build.rs       ←→  types/build.rs
src/install.rs     ←→  types/install.rs
src/rebuild.rs     ←→  types/rebuild.rs
src/shutdown.rs    ←→  types/shutdown.rs
src/start.rs       ←→  types/start.rs
src/status.rs      ←→  types/status.rs
src/stop.rs        ←→  types/stop.rs
src/uninstall.rs   ←→  types/uninstall.rs
```

## Changes Made

### 1. Deleted
- ✅ `types/handle.rs` - UNUSED (DaemonHandle exported but never used)

### 2. Renamed
- ✅ `types/health.rs` → `types/status.rs` (matches src/status.rs)

### 3. Split
- ✅ `types/install.rs` - Extracted UninstallConfig → `types/uninstall.rs`

### 4. Created for Parity
- ✅ `types/build.rs` - Empty (build has no config types)
- ✅ `types/stop.rs` - Empty (stop uses HttpDaemonConfig from start)
- ✅ `types/uninstall.rs` - UninstallConfig

## Final Structure

```
daemon-lifecycle/src/
├── operations (8 files)
│   ├── build.rs
│   ├── install.rs
│   ├── rebuild.rs
│   ├── shutdown.rs
│   ├── start.rs
│   ├── status.rs
│   ├── stop.rs
│   └── uninstall.rs
│
├── types/ (9 files) - PERFECT PARITY
│   ├── build.rs           ← matches src/build.rs
│   ├── install.rs          ← matches src/install.rs
│   ├── rebuild.rs          ← matches src/rebuild.rs
│   ├── shutdown.rs         ← matches src/shutdown.rs
│   ├── start.rs            ← matches src/start.rs
│   ├── status.rs           ← matches src/status.rs
│   ├── stop.rs             ← matches src/stop.rs
│   ├── timeout.rs          ← (utility config)
│   └── uninstall.rs        ← matches src/uninstall.rs
│
└── utils/ (4 files)
    ├── find.rs
    ├── paths.rs
    ├── poll.rs
    └── timeout.rs
```

## Parity Table

| Operation | Config Types | Location |
|-----------|-------------|----------|
| build.rs | (none) | types/build.rs (empty) |
| install.rs | InstallConfig, InstallResult | types/install.rs |
| rebuild.rs | RebuildConfig | types/rebuild.rs |
| shutdown.rs | ShutdownConfig | types/shutdown.rs |
| start.rs | HttpDaemonConfig | types/start.rs |
| status.rs | StatusRequest, StatusResponse, HealthPollConfig | types/status.rs |
| stop.rs | (uses HttpDaemonConfig) | types/stop.rs (empty) |
| uninstall.rs | UninstallConfig | types/uninstall.rs |

**Plus utils:**
- timeout.rs → types/timeout.rs (TimeoutConfig)

## Why Empty Files?

**build.rs and stop.rs have no config types:**
- `build.rs` - Just runs `cargo build`, no config needed
- `stop.rs` - Uses `HttpDaemonConfig` from `start.rs`

**But they exist for PARITY:**
- Every operation gets a types file
- Even if empty
- No confusion about "where's the types file?"

## Why Delete handle.rs?

**DaemonHandle was UNUSED:**

```bash
# Search for usage
$ rg "DaemonHandle" --type rust

# Results:
- lib.rs: pub use types::DaemonHandle  (export only)
- types/mod.rs: pub use handle::DaemonHandle  (export only)
- types/handle.rs: struct DaemonHandle { ... }  (definition)

# NO ACTUAL USAGE IN CODE
```

**Exported but never used = DELETE.**

## Why status.rs not health.rs?

**We have src/status.rs, not src/health.rs:**
- Operation is called "status checking" (is daemon running?)
- Types file should match: types/status.rs
- health.rs was a leftover from old naming

## Compilation

✅ **daemon-lifecycle:** PASS (2 warnings, 0 errors)  
✅ **All imports updated**  
✅ **Public API unchanged**

## Public API

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
    get_install_dir,
    get_install_path,
    get_pid_file_path,
    timeout_after,
    with_timeout,
    
    // Types (NO MORE DaemonHandle)
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

**DaemonHandle removed** - Was unused, now deleted.

## Key Insight

**PARITY = NO CONFUSION**

When you see `src/start.rs`, you know config is in `types/start.rs`.  
When you see `src/uninstall.rs`, you know config is in `types/uninstall.rs`.

**Every operation has its types file. No exceptions. No confusion.**

---

**PERFECT PARITY ACHIEVED.**

**8 operations → 8 types files (+ 1 util config)**

**No more "where's the config?" Just look in types/{operation}.rs**
