# TEAM-317: Start Parity (Part 2)

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025  
**Priority:** CRITICAL - Massive Code Duplication

## Problem

**hive-lifecycle/src/start.rs** manually reimplemented everything that `daemon-lifecycle` already provides:

**Manual implementation (150+ LOC):**
1. Binary resolution
2. Process spawning with `tokio::process::Command`
3. Stderr capture to temp file
4. Crash detection (check if process exited)
5. Manual health check with `reqwest::Client`
6. Manual error handling and diagnostics

**All of this already exists in `daemon-lifecycle::start_http_daemon()`!**

## Comparison

### queen-lifecycle (CORRECT)

```rust
// Uses daemon-lifecycle properly
let config = HttpDaemonConfig::new("queen-rbee", binary_path, base_url)
    .with_args(args);
let child = start_http_daemon(config).await?;
std::mem::forget(child);
```

**Total:** ~20 LOC

### hive-lifecycle (BEFORE - WRONG)

```rust
// Manual spawning
let stderr_file = std::fs::File::create(&stderr_path)?;
let mut child = tokio::process::Command::new(&binary_path)
    .arg("--port").arg(port.to_string())
    .arg("--queen-url").arg(queen_url)
    .arg("--hive-id").arg("localhost")
    .stdout(std::process::Stdio::null())
    .stderr(stderr_file)
    .spawn()?;

// Manual crash detection
tokio::time::sleep(Duration::from_secs(2)).await;
match child.try_wait() {
    Ok(Some(status)) => {
        let stderr_content = std::fs::read_to_string(&stderr_path)?;
        anyhow::bail!("Crashed: {}\n\nStderr:\n{}", status, stderr_content);
    }
    // ... more manual error handling
}

// Manual health check
let client = reqwest::Client::new();
match client.get(&health_url).timeout(Duration::from_secs(3)).send().await {
    Ok(response) if response.status().is_success() => { /* ok */ }
    // ... more manual error handling
}
```

**Total:** ~150 LOC

### hive-lifecycle (AFTER - CORRECT)

```rust
// Uses daemon-lifecycle (same as queen)
let config = HttpDaemonConfig::new("rbee-hive", binary_path, &base_url)
    .with_args(args);
let child = start_http_daemon(config).await?;
std::mem::forget(child);
```

**Total:** ~20 LOC

## Solution

**Migrated hive-lifecycle to use `daemon-lifecycle::start_http_daemon()`**

**Before (185 LOC):**
- Manual process spawning
- Manual stderr capture
- Manual crash detection
- Manual health polling
- Manual error diagnostics

**After (88 LOC):**
- Uses `daemon_lifecycle::start_http_daemon()`
- All crash detection handled by daemon-lifecycle
- All health polling handled by daemon-lifecycle
- All error handling handled by daemon-lifecycle

**Savings:** 97 LOC removed from `start_hive_local()`

## Files Changed

1. **hive-lifecycle/src/start.rs** (185 → 88 LOC)
   - Removed manual spawning logic
   - Removed stderr capture and crash detection
   - Removed manual health checking
   - Now uses `daemon_lifecycle::start_http_daemon()`
   - Added TEAM-317 signatures

## Benefits

### ✅ RULE ZERO Compliance
- **Single startup pattern:** Both queen and hive use `daemon-lifecycle`
- **No duplication:** Crash detection, health polling in one place
- **Contract compliance:** Both use `HttpDaemonConfig`

### ✅ Code Reduction
- **hive-lifecycle start:** 97 LOC removed (185 → 88)
- **Combined with stop:** 245 LOC total eliminated

### ✅ Maintainability
- Bugs fixed in one place (`daemon-lifecycle`)
- Consistent error handling across all daemons
- Single source of truth for daemon startup

### ✅ Feature Parity
- Both queen and hive get same crash detection
- Both get same health polling
- Both get same error diagnostics
- No more "queen has X but hive doesn't"

## What daemon-lifecycle Provides

**All of this is now shared:**
1. ✅ Process spawning with proper stdio handling
2. ✅ Health polling with exponential backoff
3. ✅ Crash detection during startup
4. ✅ Timeout enforcement
5. ✅ Structured error messages
6. ✅ Narration with job_id routing

## Verification

```bash
# Compilation
cargo check -p hive-lifecycle
# ✅ PASS

# Test hive start
./rbee hive start -a localhost
# ✅ Uses daemon-lifecycle (same as queen)
```

## Key Insight

**If two daemons need the same startup logic, they should use the same code.**

The fact that hive had 150 LOC of manual implementation while queen used daemon-lifecycle was a clear sign of duplication.

---

**RULE ZERO:** Breaking changes > backwards compatibility  
**Result:** 97 LOC removed, feature parity restored, single startup pattern

**Total TEAM-317 savings:** 245 LOC eliminated (148 from stop + 97 from start)
