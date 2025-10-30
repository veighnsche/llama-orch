# Health Check Fix - Empty Daemon Name Bug

**TEAM-341** | **Date:** 2025-10-29  
**Status:** ✅ FIXED

## Problem

Health polling was failing with infinite retries because `check_binary_installed()` was being called with an **empty daemon name**:

```
daemon_lifecycle::utils::binary::check_binary_installed check_binary        
🔍 Checking if  is installed
                    ^^^ Empty string!
```

The daemon was actually **running and healthy**, but the health check was stuck in a loop checking if an empty binary name was installed.

## Root Cause

In `start.rs` line 271, the `HealthPollConfig` was created using the builder pattern but didn't set the required fields:

```rust
// OLD CODE (BROKEN)
let poll_config = HealthPollConfig::new(&daemon_config.health_url)
    .with_max_attempts(30);

poll_daemon_health(poll_config).await?;
```

This left `daemon_binary_name` as an empty string (from `Default::default()`), causing `check_daemon_health()` to call `check_binary_installed("", ssh_config)`.

## The Fix

**File:** `src/start.rs` (TEAM-341)

Changed from builder pattern to struct initialization to ensure all fields are set:

```rust
// NEW CODE (FIXED)
let poll_config = HealthPollConfig {
    base_url: daemon_config.health_url.clone(),
    health_endpoint: None,
    max_attempts: 30,
    initial_delay_ms: 200,
    backoff_multiplier: 1.5,
    job_id: daemon_config.job_id.clone(),
    daemon_name: Some(daemon_name.to_string()),
    daemon_binary_name: daemon_name.to_string(),  // ← CRITICAL FIX
    ssh_config: ssh_config.clone(),               // ← CRITICAL FIX
};

poll_daemon_health(poll_config).await?;
```

## Why This Happened

The `HealthPollConfig` struct has required fields (`daemon_binary_name`, `ssh_config`) but the builder pattern methods didn't enforce setting them:

```rust
impl HealthPollConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self { base_url: base_url.into(), ..Default::default() }
        // ↑ Uses Default, which sets daemon_binary_name to empty string
    }
}
```

The builder pattern is convenient but can lead to missing required fields.

## Expected Behavior Now

When starting a daemon, health polling will:

1. ✅ Check HTTP health endpoint (daemon is running)
2. ✅ Check if binary is installed with **correct daemon name** (e.g., "queen-rbee")
3. ✅ Return success when daemon is healthy

**Logs should show:**
```
daemon_lifecycle::utils::binary::check_binary_installed check_binary        
🔍 Checking if queen-rbee is installed
                    ^^^^^^^^^^^ Correct name!
```

## Testing

```bash
# Rebuild rbee-keeper with the fix
cargo build -p rbee-keeper

# Start queen-rbee from Tauri GUI
# Should now succeed without infinite retries
```

## Related Files

- **`src/start.rs`** - Fixed health polling config (line 271-281)
- **`src/utils/poll.rs`** - Health polling implementation
- **`src/utils/binary.rs`** - Binary installation check
- **`src/status.rs`** - Daemon health check logic

## Lesson Learned

**Builder patterns are convenient but dangerous for structs with required fields.**

Consider:
1. Making required fields non-optional in the struct
2. Using a `new()` method that takes all required parameters
3. Or using `#[must_use]` on builder methods

---

**Compilation:** ✅ PASS  
**Impact:** Fixes infinite retry loop during daemon startup  
**Severity:** HIGH (blocked daemon startup)
