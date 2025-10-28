# TEAM-328: Final Health Module Cleanup

**Status:** ✅ COMPLETE

**Mission:** Remove `check_daemon_status()` and use `is_daemon_healthy()` directly everywhere

## Problem

After consolidating `get.rs` and `status.rs` into `health.rs`, we still had `check_daemon_status()` which was just a wrapper around `is_daemon_healthy()` with extra narration.

**RULE ZERO violation:** Two ways to check daemon health
- `is_daemon_healthy()` - Simple bool check
- `check_daemon_status()` - Wrapper with narration + StatusResponse

## Solution

**Deleted `check_daemon_status()` entirely** and updated all callers to use `is_daemon_healthy()` directly with their own narration.

### Deleted from health.rs

```rust
// DELETED (70 LOC)
pub async fn check_daemon_status(
    id: &str,
    health_url: &str,
    daemon_type: Option<&str>,
    job_id: Option<&str>,
) -> Result<StatusResponse> {
    // Wrapper around is_daemon_healthy()
    // ...
}
```

### Updated Callers

**Before (queen.rs):**
```rust
use daemon_lifecycle::check_daemon_status;

check_daemon_status("localhost", &format!("{}/health", queen_url), Some("queen"), None).await?;
```

**After (queen.rs):**
```rust
use daemon_lifecycle::is_daemon_healthy;
use observability_narration_core::n;

let health_url = format!("{}/health", queen_url);
let is_running = is_daemon_healthy(&health_url, None, None).await;

if is_running {
    n!("queen_status", "✅ queen 'localhost' is running on {}", health_url);
} else {
    n!("queen_status", "❌ queen 'localhost' is not running on {}", health_url);
}
```

**Before (hive.rs):**
```rust
use daemon_lifecycle::check_daemon_status;

check_daemon_status("localhost", &format!("{}/health", hive_url), Some("hive"), None).await?;
```

**After (hive.rs):**
```rust
use daemon_lifecycle::is_daemon_healthy;
use observability_narration_core::n;

let health_url = "http://localhost:7835/health";
let is_running = is_daemon_healthy(health_url, None, None).await;

if is_running {
    n!("hive_status", "✅ hive 'localhost' is running on {}", health_url);
} else {
    n!("hive_status", "❌ hive 'localhost' is not running on {}", health_url);
}
```

## Files Changed

**Modified:**
- `bin/99_shared_crates/daemon-lifecycle/src/health.rs` (-70 LOC)
  - Deleted `check_daemon_status()` function
  - Removed `StatusRequest`, `StatusResponse` re-exports
  
- `bin/99_shared_crates/daemon-lifecycle/src/lib.rs` (-1 LOC)
  - Removed `check_daemon_status` from exports
  
- `bin/00_rbee_keeper/src/handlers/queen.rs` (+7 LOC, -1 LOC)
  - Use `is_daemon_healthy()` directly
  - Add own narration
  
- `bin/00_rbee_keeper/src/handlers/hive.rs` (+7 LOC, -1 LOC)
  - Use `is_daemon_healthy()` directly
  - Add own narration

## Benefits

### Simpler API
**Before:** 3 health functions
- `is_daemon_healthy()` - Simple bool
- `poll_until_healthy()` - Retry with backoff
- `check_daemon_status()` - Wrapper with narration

**After:** 2 health functions
- `is_daemon_healthy()` - Simple bool
- `poll_until_healthy()` - Retry with backoff

### No Duplication
**Before:** `check_daemon_status()` wrapped `is_daemon_healthy()`
```rust
// Inside check_daemon_status()
let is_running = is_daemon_healthy(&health_url, None, Some(Duration::from_secs(5))).await;
```

**After:** Callers use `is_daemon_healthy()` directly
```rust
let is_running = is_daemon_healthy(&health_url, None, None).await;
```

### Better Separation of Concerns
- **health.rs** - Pure health checking (no narration)
- **Callers** - Add their own narration as needed

### Code Reduction
- **health.rs:** -70 LOC
- **lib.rs:** -1 LOC
- **queen.rs:** +6 LOC (net)
- **hive.rs:** +6 LOC (net)
- **Total:** -59 LOC net

## Compilation & Testing

✅ `cargo build --bin rbee-keeper` - PASS  
✅ `./rbee queen status` - Works correctly

**Output:**
```
❌ queen 'localhost' is not running on http://localhost:7833/health
```

## Summary

**Deleted:**
- `check_daemon_status()` function (70 LOC)
- Wrapper abstraction that added no value

**Result:**
- Simpler API (2 functions instead of 3)
- No duplication (callers use `is_daemon_healthy()` directly)
- Better separation (health checking vs narration)
- -59 LOC net reduction

**health.rs now has exactly 2 functions:**
1. `is_daemon_healthy()` - Simple bool check
2. `poll_until_healthy()` - Retry with exponential backoff

---

**TEAM-328 Final Result:** health.rs is now minimal and focused - just health checking, no wrappers
