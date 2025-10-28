# TEAM-338: Extended Status Check

**Date:** Oct 28, 2025  
**Status:** ✅ COMPLETE

## Summary

Extended `daemon-lifecycle` crate with `check_daemon_status()` function that returns both running and installed status in a single call. Optimizes by only checking installation when daemon is not running.

## Changes

### `/bin/99_shared_crates/daemon-lifecycle/src/status.rs`

**Added:**

1. **`DaemonStatus` struct** (lines 43-52)
   ```rust
   #[derive(Debug, Clone)]
   pub struct DaemonStatus {
       pub is_running: bool,
       pub is_installed: bool,
   }
   ```

2. **`check_daemon_status()` function** (lines 73-92)
   ```rust
   pub async fn check_daemon_status(
       health_url: &str,
       binary_path: &std::path::Path
   ) -> DaemonStatus
   ```

**Kept:**
- `check_daemon_health()` - Still used by polling utilities

### `/bin/99_shared_crates/daemon-lifecycle/src/lib.rs`

**Updated exports:**
```rust
pub use status::{
    check_daemon_health,      // Existing
    check_daemon_status,       // NEW
    DaemonStatus               // NEW
};
```

### `/bin/00_rbee_keeper/src/tauri_commands.rs`

**Simplified `queen_status()` command:**

**Before:**
```rust
let is_running = check_daemon_health(&health_url).await;
let is_installed = binary_path.exists();

Ok(QueenStatus { is_running, is_installed })
```

**After:**
```rust
let status = check_daemon_status(&health_url, &binary_path).await;

Ok(QueenStatus {
    is_running: status.is_running,
    is_installed: status.is_installed,
})
```

## Implementation Details

### Optimization Logic

```rust
let is_running = check_daemon_health(health_url).await;

let is_installed = if is_running {
    true  // If running, it must be installed
} else {
    binary_path.exists()  // Only check filesystem if not running
};
```

**Why this optimization?**
- If daemon is running, we know it's installed (can't run without binary)
- Avoids unnecessary filesystem check in the common case (daemon running)
- Only checks installation when daemon is offline

### Function Signatures

```rust
// Simple health check (HTTP only)
pub async fn check_daemon_health(health_url: &str) -> bool

// Extended status check (HTTP + filesystem)
pub async fn check_daemon_status(
    health_url: &str,
    binary_path: &std::path::Path
) -> DaemonStatus
```

## Use Cases

### When to use `check_daemon_health()`
- Polling loops (utils/poll.rs)
- Quick health checks
- When you only care if daemon is running

### When to use `check_daemon_status()`
- UI status displays
- Deciding which buttons to show
- When you need both running and installed info

## Benefits

### 1. Single Network Call
**Before:**
```rust
// Two separate checks
let is_running = check_daemon_health(url).await;
let is_installed = binary_path.exists();
```

**After:**
```rust
// One function call (internally optimized)
let status = check_daemon_status(url, path).await;
```

### 2. Optimization
- Running → Skip filesystem check (we know it's installed)
- Not running → Check filesystem

### 3. Type Safety
```rust
// Returns structured data, not loose booleans
DaemonStatus {
    is_running: bool,
    is_installed: bool,
}
```

### 4. Reusability
Can be used for any daemon (queen, hive, worker):
```rust
// Queen
let queen_status = check_daemon_status(
    "http://localhost:7833/health",
    Path::new("~/.local/bin/queen-rbee")
).await;

// Hive
let hive_status = check_daemon_status(
    "http://localhost:7835/health",
    Path::new("~/.local/bin/rbee-hive")
).await;
```

## Performance

### Network Calls
- **Before:** 1 HTTP request (health check)
- **After:** 1 HTTP request (health check)
- **No change** - Still single network call

### Filesystem Checks
- **Before:** Always checked (even if running)
- **After:** Only checked if not running
- **Improvement:** Avoids unnecessary I/O in common case

### Typical Case (Daemon Running)
```
check_daemon_status()
  ↓
check_daemon_health() → HTTP GET /health → 200 OK
  ↓
is_running = true
  ↓
is_installed = true (inferred, no filesystem check)
  ↓
return DaemonStatus { is_running: true, is_installed: true }
```

### Edge Case (Daemon Not Running)
```
check_daemon_status()
  ↓
check_daemon_health() → HTTP GET /health → Connection refused
  ↓
is_running = false
  ↓
is_installed = binary_path.exists() (filesystem check)
  ↓
return DaemonStatus { is_running: false, is_installed: true/false }
```

## Error Handling

### Network Errors
```rust
// check_daemon_health() returns false on any error
match client.get(health_url).send().await {
    Ok(response) => response.status().is_success(),
    Err(_) => false,  // Connection refused, timeout, etc.
}
```

### Filesystem Errors
```rust
// Path::exists() returns false if path doesn't exist or can't be accessed
binary_path.exists()  // false on permission errors, missing file, etc.
```

**No panics** - All errors handled gracefully with boolean returns.

## Future Enhancements

Consider adding:

1. **Version info**
   ```rust
   pub struct DaemonStatus {
       pub is_running: bool,
       pub is_installed: bool,
       pub version: Option<String>,  // NEW
   }
   ```

2. **Uptime**
   ```rust
   pub struct DaemonStatus {
       pub is_running: bool,
       pub is_installed: bool,
       pub uptime_secs: Option<u64>,  // NEW
   }
   ```

3. **Health details**
   ```rust
   pub struct DaemonStatus {
       pub is_running: bool,
       pub is_installed: bool,
       pub health: Option<HealthDetails>,  // NEW
   }
   
   pub struct HealthDetails {
       pub memory_mb: u64,
       pub cpu_percent: f32,
       pub active_connections: u32,
   }
   ```

These would require extending the `/health` endpoint to return JSON instead of just 200 OK.

## Testing

### Manual Test
```bash
# Install queen
rbee-keeper queen install

# Check status (should show installed but not running)
# Frontend badge should show "Stopped" (red)

# Start queen
rbee-keeper queen start

# Check status (should show running and installed)
# Frontend badge should show "Running" (green)

# Uninstall queen
rbee-keeper queen uninstall

# Check status (should show not running and not installed)
# Frontend badge should show "Unknown" (gray)
```

### Unit Test (Future)
```rust
#[tokio::test]
async fn test_status_running_implies_installed() {
    // Mock health endpoint returning 200 OK
    let status = check_daemon_status("http://mock/health", Path::new("/fake")).await;
    
    assert!(status.is_running);
    assert!(status.is_installed);  // Should be true even if path doesn't exist
}

#[tokio::test]
async fn test_status_not_running_checks_filesystem() {
    // Mock health endpoint returning connection refused
    let temp_file = create_temp_binary();
    let status = check_daemon_status("http://mock/health", &temp_file).await;
    
    assert!(!status.is_running);
    assert!(status.is_installed);  // Should check filesystem
}
```

## Related Files

**Backend:**
- `/bin/99_shared_crates/daemon-lifecycle/src/status.rs` (new function)
- `/bin/99_shared_crates/daemon-lifecycle/src/lib.rs` (exports)
- `/bin/00_rbee_keeper/src/tauri_commands.rs` (usage)

**Frontend:**
- `/bin/00_rbee_keeper/ui/src/store/queenStore.ts` (calls command)
- `/bin/00_rbee_keeper/ui/src/components/QueenCard.tsx` (displays status)
- `/bin/00_rbee_keeper/ui/src/components/StatusBadge.tsx` (visual indicator)

## Architecture

```
Frontend (QueenCard)
    ↓
Zustand Store (fetchStatus)
    ↓
Tauri Command (queen_status)
    ↓
daemon-lifecycle (check_daemon_status)
    ↓
    ├─ HTTP GET /health (check_daemon_health)
    └─ filesystem check (binary_path.exists) [only if not running]
```

---

**Pattern:** Extended status checks combine multiple signals (network + filesystem) with optimization to avoid unnecessary I/O.
