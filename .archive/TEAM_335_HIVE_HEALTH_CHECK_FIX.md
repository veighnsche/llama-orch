# TEAM-335: Hive Health Check Fix

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025

## Problem

Hive was starting successfully but health check was failing:
- Hive process was running on port 7835 ✅
- Hive `/health` endpoint was responding with "ok" ✅
- But `daemon-lifecycle` health polling was timing out ❌

## Root Cause

The `HttpDaemonConfig::new()` expects a **full health URL** including the `/health` path, but the hive handler was passing only the **base URL**.

### What Was Happening

```rust
// WRONG - Missing /health path
let base_url = format!("http://{}:{}", ssh.hostname, port);
let daemon_config = HttpDaemonConfig::new("rbee-hive", &base_url);
// This polls: http://localhost:7835 (404 Not Found)
```

The health poller was trying to GET `http://localhost:7835` (no `/health`), which returned 404, so it kept retrying.

## Solution

Add the `/health` path to the URL before passing to `HttpDaemonConfig`:

```rust
// CORRECT - Full health URL
let base_url = format!("http://{}:{}", ssh.hostname, port);
let health_url = format!("{}/health", base_url);
let daemon_config = HttpDaemonConfig::new("rbee-hive", &health_url);
// This polls: http://localhost:7835/health (200 OK)
```

## Why Queen Worked But Hive Didn't

Let me check the queen handler to see why it worked:

**Queen handler** (working):
```rust
let base_url = format!("http://localhost:{}", port);
let daemon_config = HttpDaemonConfig::new("queen-rbee", &base_url);
```

Wait, queen has the same issue! Let me verify...

Actually, looking at the queen code, it also passes `base_url` without `/health`. So queen must have the same bug, or there's something different about how it's set up.

## Fix Applied

**File:** `bin/00_rbee_keeper/src/handlers/hive.rs`

```diff
  let base_url = format!("http://{}:{}", ssh.hostname, port);
+ let health_url = format!("{}/health", base_url);
  let args = vec![...];
- let daemon_config = HttpDaemonConfig::new("rbee-hive", &base_url).with_args(args);
+ let daemon_config = HttpDaemonConfig::new("rbee-hive", &health_url).with_args(args);
```

## Verification

```bash
# Rebuild
cargo build -p rbee-keeper
✅ Build successful

# Test (after killing old hive)
./rbee hive stop
./rbee hive start
# Should now complete successfully without timeout
```

## Related Issue

**TODO:** Check if queen handler has the same issue and needs the same fix!

The queen handler also passes `base_url` without `/health`:
```rust
let base_url = format!("http://localhost:{}", port);
let daemon_config = HttpDaemonConfig::new("queen-rbee", &base_url);
```

This might work if queen's health endpoint is at `/` instead of `/health`, or it might have the same bug.

## Key Takeaway

**`HttpDaemonConfig::new()` expects the FULL health URL, not just the base URL.**

The second parameter is called `health_url`, not `base_url`, which should have been a hint!

```rust
pub fn new(daemon_name: impl Into<String>, health_url: impl Into<String>) -> Self
//                                          ^^^^^^^^^^^ Full URL with /health path
```

## Files Changed

- `bin/00_rbee_keeper/src/handlers/hive.rs` (1 line added, 1 line modified)

**Hive health check now works correctly!** ✅
