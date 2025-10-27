# TEAM-316: Lifecycle Split & Contract Migration

**Date:** 2025-10-27  
**Status:** ✅ COMPLETE

## Problem

**Architecture Violation:** `lifecycle.rs` had multiple responsibilities and `HttpDaemonConfig` was in the wrong place.

### Issues Found

1. **`lifecycle.rs` violated single responsibility**
   - Contained both start and stop operations in one file
   - 239 lines doing two different things

2. **`HttpDaemonConfig` was in wrong crate**
   - Located in `daemon-lifecycle` (implementation crate)
   - Should be in `daemon-contract` (contract crate)
   - Had duplicate/extended version in `daemon-lifecycle`

## Solution

### ✅ Split `lifecycle.rs` into `start.rs` and `stop.rs`

**Single Responsibility Principle:** Each file does ONE thing.

### ✅ Move `HttpDaemonConfig` to `daemon-contract`

**Contracts belong in contracts/:** Configuration types are contracts between crates.

## Changes Made

### 1. daemon-contract/src/lifecycle.rs
**BEFORE:** Minimal config (4 fields)
```rust
pub struct HttpDaemonConfig {
    pub daemon_name: String,
    pub health_url: String,
    pub shutdown_endpoint: Option<String>,
    pub job_id: Option<String>,
}
```

**AFTER:** Complete config with lifecycle fields (8 fields + builder methods)
```rust
pub struct HttpDaemonConfig {
    // Contract fields
    pub daemon_name: String,
    pub health_url: String,
    pub shutdown_endpoint: Option<String>,
    pub job_id: Option<String>,
    
    // Lifecycle fields (moved from daemon-lifecycle)
    pub binary_path: PathBuf,
    pub args: Vec<String>,
    pub max_health_attempts: Option<usize>,
    pub health_initial_delay_ms: Option<u64>,
}

impl HttpDaemonConfig {
    pub fn new(...) -> Self { ... }
    pub fn with_args(self, ...) -> Self { ... }
    pub fn with_job_id(self, ...) -> Self { ... }
    pub fn with_shutdown_endpoint(self, ...) -> Self { ... }
    pub fn with_max_health_attempts(self, ...) -> Self { ... }
    pub fn with_health_initial_delay_ms(self, ...) -> Self { ... }
}
```

### 2. daemon-lifecycle/src/start.rs (NEW)
**Created:** 67 lines
- `start_http_daemon(config)` - Spawn + health polling
- Single responsibility: START operations only

### 3. daemon-lifecycle/src/stop.rs (NEW)
**Created:** 58 lines
- `stop_http_daemon(config)` - Graceful shutdown
- Single responsibility: STOP operations only

### 4. daemon-lifecycle/src/lifecycle.rs (DELETED)
**Removed:** 239 lines
- Split into start.rs (67 lines) and stop.rs (58 lines)
- Eliminated duplicate HttpDaemonConfig (moved to contract)

### 5. daemon-lifecycle/src/lib.rs
**Updated:**
- Removed `pub mod lifecycle;`
- Added `pub mod start;` and `pub mod stop;`
- Changed exports:
  - `pub use lifecycle::{start_http_daemon, stop_http_daemon, HttpDaemonConfig};`
  - → `pub use start::start_http_daemon;`
  - → `pub use stop::stop_http_daemon;`
  - → `pub use daemon_contract::HttpDaemonConfig;`

## Architecture Improvements

### Before
```
daemon-lifecycle/
├── lifecycle.rs (239 lines)
│   ├── HttpDaemonConfig (duplicate, extended)
│   ├── start_http_daemon()
│   └── stop_http_daemon()
│
daemon-contract/
└── lifecycle.rs
    └── HttpDaemonConfig (minimal, 4 fields)
```

### After
```
daemon-contract/
└── lifecycle.rs
    └── HttpDaemonConfig (complete, 8 fields + builders)
        ↑
        │ (contract - single source of truth)
        │
daemon-lifecycle/
├── start.rs (67 lines)
│   └── start_http_daemon()  ← uses HttpDaemonConfig from contract
│
└── stop.rs (58 lines)
    └── stop_http_daemon()   ← uses HttpDaemonConfig from contract
```

## Benefits

### 1. Single Responsibility
- ✅ `start.rs` - Only start operations
- ✅ `stop.rs` - Only stop operations
- ✅ Easier to understand, test, and maintain

### 2. Contracts in Correct Location
- ✅ `HttpDaemonConfig` is in `daemon-contract` (where it belongs)
- ✅ No duplication between crates
- ✅ Single source of truth

### 3. Cleaner Dependencies
- ✅ Implementation depends on contract (correct direction)
- ✅ No circular dependencies
- ✅ Clear separation of concerns

## Verification

### Compilation Status

✅ **All crates compile successfully:**

```bash
cargo check -p daemon-contract    # ✅ PASS
cargo check -p daemon-lifecycle   # ✅ PASS
cargo check -p queen-lifecycle    # ✅ PASS
cargo check -p hive-lifecycle     # ✅ PASS
cargo check -p rbee-keeper        # ✅ PASS
```

### Breaking Changes

**Impact:** Zero - all exports maintained at crate level

Users still write:
```rust
use daemon_lifecycle::{start_http_daemon, stop_http_daemon, HttpDaemonConfig};
```

The re-exports in `lib.rs` make the split transparent to consumers.

## Code Reduction

**Before:** 239 lines in lifecycle.rs  
**After:** 67 lines (start.rs) + 58 lines (stop.rs) = 125 lines  
**Savings:** 114 lines removed (duplicate config + tests moved to contract)

## Engineering Rules Compliance

✅ **Single Responsibility:** Each file does ONE thing  
✅ **Contracts in contracts/:** HttpDaemonConfig moved to correct location  
✅ **No duplication:** Single source of truth for config  
✅ **Clean architecture:** Implementation depends on contract (not vice versa)

## Files Modified

1. `/bin/97_contracts/daemon-contract/src/lifecycle.rs` - Added complete HttpDaemonConfig
2. `/bin/99_shared_crates/daemon-lifecycle/src/start.rs` - NEW (extracted from lifecycle.rs)
3. `/bin/99_shared_crates/daemon-lifecycle/src/stop.rs` - NEW (extracted from lifecycle.rs)
4. `/bin/99_shared_crates/daemon-lifecycle/src/lifecycle.rs` - DELETED
5. `/bin/99_shared_crates/daemon-lifecycle/src/lib.rs` - Updated module declarations

## Next Steps

None - refactoring complete, all code compiles, no external breakage.

---

**Lesson:** 
1. **Contracts belong in contracts/** - Configuration types are contracts, not implementation
2. **One file, one responsibility** - Don't mix start and stop in the same file
3. **Re-exports hide internal structure** - Users don't see the split, they just use the API
