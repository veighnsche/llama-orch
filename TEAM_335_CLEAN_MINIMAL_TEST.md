# TEAM-335: Clean Minimal Test Setup

**Date:** 2025-10-28  
**Status:** ✅ READY TO TEST

## What You Asked For

> "Can you please do a complete git reset and just comment out the with timeout in the start queen flow"

## What I Did

1. ✅ **Git reset** - Restored all daemon-lifecycle files to clean state
2. ✅ **Minimal change** - Only commented out `#[with_timeout]` in `start.rs`
3. ✅ **Kept with_job_id** - SSE routing still works
4. ✅ **Building** - Release binary compiling now

## The Change

**File:** `bin/99_shared_crates/daemon-lifecycle/src/start.rs`

```rust
// Line 213-216 (before):
#[with_job_id(config_param = "start_config")]
#[with_timeout(secs = 120, label = "Start daemon")]
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {

// Line 213-216 (after):
#[with_job_id(config_param = "start_config")]
// #[with_timeout(secs = 120, label = "Start daemon")]  ← COMMENTED OUT
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
```

**That's it.** Only 1 line changed in 1 file.

## How to Test

### Option 1: CLI (Recommended)
```bash
# Wait for build to finish
# Kill existing queen
pkill queen-rbee

# Test queen start
./target/release/rbee-keeper queen start
```

### Option 2: Tauri GUI
1. Wait for build to finish
2. Kill existing queen: `pkill queen-rbee`
3. Click "Start Queen" button in UI
4. Check browser console (F12) for result

## What to Look For

### ✅ Success (timeout was the problem)
```
🚀 Starting queen-rbee on vince@localhost
🔍 Locating queen-rbee binary on remote...
✅ Found binary at /home/vince/.local/bin/queen-rbee
🚀 Starting daemon...
⏳ Waiting for queen-rbee to become healthy...
✅ queen-rbee is healthy!
```

**Means:** `#[with_timeout]` macro causes stack overflow, `#[with_job_id]` is fine

### ❌ Stack Overflow (with_job_id is the problem)
```
thread 'tokio-runtime-worker' has overflowed its stack
fatal runtime error: stack overflow
```

**Means:** `#[with_job_id]` macro causes stack overflow, need to investigate both macros

## Current Build Status

🔨 **Building** `cargo build --bin rbee-keeper --release`

Progress: ~94% complete (620/658 crates)

Should finish in ~1-2 minutes.

## Why This Approach Works

**Previous attempts:** Changed too many things at once
- Commented out all macros in all files
- Hard to isolate which macro was the problem

**This approach:** Surgical precision
- Only 1 macro commented out
- Only in the queen start flow
- Easy to test, easy to understand results
- If it works, we know exactly which macro to fix

## Next Steps After Testing

### If timeout is the culprit:
1. Investigate `timeout-enforcer` crate implementation
2. Look for deep async nesting in macro expansion
3. Consider alternative timeout approach
4. Keep `#[with_job_id]` (it works)

### If with_job_id is the culprit:
1. Comment out `#[with_job_id]` too
2. Test with neither macro
3. Investigate both macro implementations
4. Consider alternative approaches for both

---

**Build finishing soon. Then test and report what happens!** 🚀
