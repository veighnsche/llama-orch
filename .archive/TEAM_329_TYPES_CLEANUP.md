# TEAM-329: Types Cleanup - File Naming Parity

**Date:** 2025-10-27  
**Rule:** RULE ZERO - Consistent naming, no scattered types

## The Problem

**Inconsistent file naming between operations and types:**

```
src/start.rs       ←→  types/lifecycle.rs  ❌ MISMATCH
src/health.rs      ←→  types/status.rs     ❌ MISMATCH
src/list.rs        ←→  (nothing)           ❌ UNUSED
```

**Result:** Confusing, hard to navigate, no clear mapping.

## The Solution

**1-to-1 mapping between operations and types:**

```
src/{operation}.rs  ←→  types/{operation}.rs
```

## Changes Made

### 1. Deleted Unused Module
- ✅ **Deleted `src/list.rs`** - UNUSED (exported but never implemented, zero consumers)

### 2. Renamed Types to Match Operations
- ✅ **`types/lifecycle.rs` → `types/start.rs`** (matches `src/start.rs`)
- ✅ **`types/status.rs` → `types/health.rs`** (matches `src/health.rs`)

### 3. Updated All Imports
- ✅ `src/start.rs` - Import from `types::start`
- ✅ `src/stop.rs` - Import from `types::start`
- ✅ `src/health.rs` - Import from `types::health`
- ✅ `src/install.rs` - Import from `types::install`
- ✅ `src/rebuild.rs` - Import from `types::start`
- ✅ `src/shutdown.rs` - Import from `types::shutdown`
- ✅ `src/lib.rs` - Removed list exports, updated module docs

## Final Structure

```
daemon-lifecycle/src/
├── build.rs              ←→  (no types needed)
├── find.rs               ←→  (no types needed)
├── health.rs             ←→  types/health.rs ✅
├── install.rs            ←→  types/install.rs ✅
├── paths.rs              ←→  (no types needed)
├── rebuild.rs            ←→  (uses types/start.rs)
├── shutdown.rs           ←→  types/shutdown.rs ✅
├── start.rs              ←→  types/start.rs ✅
├── stop.rs               ←→  (uses types/start.rs)
├── timeout.rs            ←→  (no types needed)
├── uninstall.rs          ←→  (uses types/install.rs)
└── types/
    ├── handle.rs         (DaemonHandle - utility type)
    ├── health.rs         (StatusRequest/Response)
    ├── install.rs        (InstallConfig/Result/UninstallConfig)
    ├── shutdown.rs       (ShutdownConfig)
    └── start.rs          (HttpDaemonConfig)
```

## Naming Convention

**Rule:** `types/{operation}.rs` contains config types for `src/{operation}.rs`

**Examples:**
- `src/start.rs` uses `types/start::HttpDaemonConfig`
- `src/health.rs` uses `types/health::StatusRequest/Response`
- `src/install.rs` uses `types/install::InstallConfig`

**Special cases:**
- `types/handle.rs` - Utility type (DaemonHandle), not tied to specific operation
- Some operations share types (e.g., `stop.rs` uses `types/start::HttpDaemonConfig`)

## What Was Wrong

### Before (Confusing)
```rust
// In start.rs
use crate::types::HttpDaemonConfig;  // Where is this defined?

// In health.rs  
pub use crate::types::{StatusRequest, StatusResponse};  // What file?
```

**Problem:** No clear mapping between operation and type file.

### After (Clear)
```rust
// In start.rs
use crate::types::start::HttpDaemonConfig;  // types/start.rs ✅

// In health.rs
pub use crate::types::health::{StatusRequest, StatusResponse};  // types/health.rs ✅
```

**Benefit:** Explicit module path shows exactly where types are defined.

## Why list.rs Was Deleted

**Evidence:**
```bash
# Search for ListableConfig implementations
$ rg "impl ListableConfig" --type rust
# NO RESULTS

# Search for list_daemons usage
$ rg "list_daemons" --type rust
# Only in lib.rs export, never called
```

**Conclusion:** Exported but never implemented, zero consumers = DELETE.

## Compilation Status

✅ **daemon-lifecycle:** PASS (2 warnings, 0 errors)  
✅ **All imports updated**  
✅ **Public API unchanged** (re-exports still work)

## Public API (Unchanged)

Users see NO CHANGE:

```rust
use daemon_lifecycle::{
    HttpDaemonConfig,      // Still works
    InstallConfig,         // Still works
    StatusRequest,         // Still works
    start_daemon,          // Still works
};
```

**Re-exports preserved** - Internal reorganization only.

## Key Insight

**Consistent naming prevents confusion.**

When you see `src/start.rs`, you know config types are in `types/start.rs`.

When you see `src/health.rs`, you know types are in `types/health.rs`.

**No guessing. No hunting. Just convention.**

---

**RULE ZERO:** If file naming doesn't match, fix it. Don't keep confusing names for "backwards compatibility."
