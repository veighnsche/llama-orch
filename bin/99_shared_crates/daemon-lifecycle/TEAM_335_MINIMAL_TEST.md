# TEAM-335: Minimal Test - Only Timeout Commented Out

**Date:** 2025-10-28  
**Status:** 🧪 READY TO TEST  
**Configuration:** `#[with_job_id]` ✅ ON | `#[with_timeout]` ❌ OFF (only in start.rs)

## What Changed

**Minimal change approach:**
1. ✅ Git reset all daemon-lifecycle files to clean state
2. ✅ Only commented out `#[with_timeout]` in `start.rs` (line 215)
3. ✅ Kept `#[with_job_id]` **enabled** everywhere
4. ✅ All other files unchanged (stop, shutdown, rebuild, etc. still have both macros)

## Files Modified

**Only 1 file changed:**
- `bin/99_shared_crates/daemon-lifecycle/src/start.rs`
  - Line 215: `// #[with_timeout(secs = 120, label = "Start daemon")]`
  - Line 65: Commented out unused import

**All other files at git HEAD:**
- `stop.rs` - Both macros enabled
- `shutdown.rs` - Both macros enabled  
- `uninstall.rs` - Both macros enabled
- `rebuild.rs` - Both macros enabled
- `utils/poll.rs` - `#[with_job_id]` enabled

## Queen Start Flow

```
CLI: rbee queen start
  ↓
handlers/queen.rs → handle_queen(QueenAction::Start)
  ↓
daemon-lifecycle/start.rs → start_daemon(config)
  ↓
#[with_job_id] ✅ ENABLED
#[with_timeout] ❌ DISABLED (commented out)
```

## Compilation Status

✅ **PASS** (with warning about unused import - fixed)
```bash
cargo check -p daemon-lifecycle  # 0.71s - PASS
cargo build --bin rbee-keeper --release  # Building...
```

## Test Instructions

### From CLI (Recommended)
```bash
# Kill existing queen
pkill queen-rbee

# Test queen start
./target/release/rbee-keeper queen start

# Expected: Should work without stack overflow
# Watch for narration events in terminal
```

### From Tauri GUI (Alternative)
1. Kill existing queen: `pkill queen-rbee`
2. Open browser console (F12)
3. Click "Start Queen" button
4. Check console for success/error

## Expected Outcomes

### Scenario A: Works Fine ✅
```
🚀 Starting queen-rbee on vince@localhost
🔍 Locating queen-rbee binary on remote...
✅ Found binary at /home/vince/.local/bin/queen-rbee
🚀 Starting daemon...
⏳ Waiting for queen-rbee to become healthy at http://localhost:7833
✅ queen-rbee is healthy!
```

**Conclusion:** The `#[with_timeout]` macro was the culprit!

**Next Steps:**
1. Document that timeout macro causes stack overflow
2. Investigate timeout macro implementation
3. Fix or replace timeout mechanism
4. Keep `#[with_job_id]` (it works fine)

### Scenario B: Stack Overflow Returns ❌
```
thread 'tokio-runtime-worker' has overflowed its stack
fatal runtime error: stack overflow
```

**Conclusion:** The `#[with_job_id]` macro is the problem (or both macros together)

**Next Steps:**
1. Comment out `#[with_job_id]` too
2. Test with neither macro
3. Investigate both macro implementations

### Scenario C: Different Error ⚠️
Some other error appears

**Next Steps:** Document and investigate separately

## Why This Approach?

**Previous approach:** Commented out all macros in all files (8 files)
- ✅ Proved macros were the problem
- ❌ Too broad - couldn't isolate which macro

**Current approach:** Minimal change - only timeout in start.rs
- ✅ Isolates the specific macro
- ✅ Only affects queen start flow
- ✅ Easy to test from CLI
- ✅ Easy to revert if needed

## Verification

```bash
# Check what's modified
git diff bin/99_shared_crates/daemon-lifecycle/src/start.rs

# Should show only 2 changes:
# - Line 65: Commented out import
# - Line 215: Commented out #[with_timeout]
```

## Benefits if This Works

If queen start works with this minimal change:
- ✅ SSE routing works (with_job_id enabled)
- ✅ Real-time narration in CLI
- ✅ job_id propagation works
- ✅ Confirmed timeout macro is the culprit
- ❌ No timeout enforcement (can hang, but acceptable for now)

---

**Ready to test!** Run `./target/release/rbee-keeper queen start` and see what happens. 🚀
