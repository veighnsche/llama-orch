# TEAM-328: Health Module Consolidation

**Status:** ✅ COMPLETE

**Mission:** Consolidate get.rs, status.rs into health.rs (RULE ZERO cleanup)

## Problem

Three separate files for health/status checking:
- `get.rs` (116 LOC) - Unused trait-based abstraction, zero callers
- `status.rs` (97 LOC) - Duplicated HTTP health check logic
- `health.rs` (200 LOC) - Core health checking

**RULE ZERO violations:**
1. ❌ `get.rs` - Dead code (zero external callers)
2. ❌ `status.rs` - Duplicated HTTP client logic from `health.rs`
3. ❌ Multiple ways to check daemon health

## Solution

**Consolidated everything into `health.rs`:**

### Deleted Files
1. ✅ `src/get.rs` (116 LOC deleted)
   - `get_daemon()` function
   - `GettableConfig` trait
   - Zero external callers - pure dead code

2. ✅ `src/status.rs` (97 LOC deleted)
   - `check_daemon_status()` function moved to `health.rs`
   - Refactored to use `is_daemon_healthy()` internally
   - Eliminated duplicate HTTP client logic

### Updated health.rs

**Added:**
```rust
// TEAM-328: Re-export status types from daemon-contract
pub use daemon_contract::{StatusRequest, StatusResponse};

/// Check daemon status via HTTP health check
///
/// TEAM-328: Moved from status.rs, refactored to use is_daemon_healthy()
pub async fn check_daemon_status(
    id: &str,
    health_url: &str,
    daemon_type: Option<&str>,
    job_id: Option<&str>,
) -> Result<StatusResponse> {
    // ...
    // TEAM-328: Use is_daemon_healthy() instead of duplicating HTTP logic
    let is_running = is_daemon_healthy(&health_url, None, Some(Duration::from_secs(5))).await;
    // ...
}
```

**Key improvement:** `check_daemon_status()` now uses `is_daemon_healthy()` internally instead of duplicating HTTP client logic.

### Updated lib.rs

**Before:**
```rust
pub mod get;
pub mod health;
pub mod status;

pub use get::{get_daemon, GettableConfig};
pub use health::{is_daemon_healthy, poll_until_healthy, HealthPollConfig};
pub use status::{check_daemon_status, StatusRequest, StatusResponse};
```

**After:**
```rust
// TEAM-328: Deleted get.rs and status.rs - consolidated into health.rs
pub mod health;

// TEAM-328: Consolidated get.rs and status.rs into health.rs (RULE ZERO)
pub use health::{
    check_daemon_status, is_daemon_healthy, poll_until_healthy, HealthPollConfig, StatusRequest,
    StatusResponse,
};
```

## Files Changed

**Deleted:**
- `bin/99_shared_crates/daemon-lifecycle/src/get.rs` (-116 LOC)
- `bin/99_shared_crates/daemon-lifecycle/src/status.rs` (-97 LOC)

**Modified:**
- `bin/99_shared_crates/daemon-lifecycle/src/health.rs` (+76 LOC)
  - Added `check_daemon_status()` function
  - Re-exports `StatusRequest`, `StatusResponse`
  - Uses `is_daemon_healthy()` internally (no duplication)
  
- `bin/99_shared_crates/daemon-lifecycle/src/lib.rs` (-5 LOC)
  - Removed `get` and `status` module declarations
  - Updated exports to use `health` module
  - Added TEAM-328 comments

## Benefits

### Code Reduction
- **Total deleted:** -213 LOC (get.rs + status.rs)
- **Total added:** +76 LOC (health.rs additions)
- **Net reduction:** -137 LOC

### Eliminated Duplication
**Before (status.rs):**
```rust
// Duplicate HTTP client logic
let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build()?;
let running = match client.get(&health_url).send().await { ... }
```

**After (health.rs):**
```rust
// Reuses existing is_daemon_healthy()
let is_running = is_daemon_healthy(&health_url, None, Some(Duration::from_secs(5))).await;
```

### Simplified API
**Before:** 3 modules for health/status
- `health::is_daemon_healthy()`
- `health::poll_until_healthy()`
- `status::check_daemon_status()`
- `get::get_daemon()` (unused)

**After:** 1 module for all health checking
- `health::is_daemon_healthy()`
- `health::poll_until_healthy()`
- `health::check_daemon_status()`

### Better Organization
All health-related functionality in one place:
- Simple bool check: `is_daemon_healthy()`
- Retry with backoff: `poll_until_healthy()`
- Status with narration: `check_daemon_status()`

## Impact

### External Callers (Unchanged)
**rbee-keeper still works:**
```rust
// bin/00_rbee_keeper/src/handlers/queen.rs
use daemon_lifecycle::check_daemon_status;

check_daemon_status("localhost", &format!("{}/health", queen_url), Some("queen"), None).await?;
```

**No breaking changes** - all exports preserved, just from different module.

### Internal Improvements
- Single source of truth for HTTP health checks
- No duplicate HTTP client logic
- Easier to maintain and test
- Clear hierarchy: simple → complex

## Compilation

✅ `cargo check -p daemon-lifecycle` - PASS  
✅ `cargo build --bin rbee-keeper` - PASS

## Code Signatures

All changes marked with `// TEAM-328:`

## Summary

**Deleted:**
- `get.rs` (116 LOC) - Dead code, zero callers
- `status.rs` (97 LOC) - Duplicated HTTP logic

**Consolidated into:**
- `health.rs` - All health/status checking in one place

**Results:**
- -137 LOC net reduction
- Eliminated HTTP client duplication
- Simpler API (1 module instead of 3)
- Better code organization
- Zero breaking changes

---

**TEAM-328 Result:** health.rs is now the single source of truth for all daemon health checking
