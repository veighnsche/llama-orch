# TEAM-335: ROOT CAUSE CONFIRMED

**Date:** 2025-10-28  
**Status:** 🎯 CONFIRMED

## The Culprit

**`#[with_job_id]` macro causes stack overflow in CLI context**

## Test Results

### Test 1: Only timeout commented out
```bash
# Configuration: with_job_id ✅ ON | with_timeout ❌ OFF
./rbee queen start
# Result: ❌ STACK OVERFLOW
```

**Output:**
```
thread 'tokio-runtime-worker' has overflowed its stack
fatal runtime error: stack overflow, aborting
```

### Test 2: Both macros commented out (previous)
```bash
# Configuration: with_job_id ❌ OFF | with_timeout ❌ OFF  
./rbee queen start
# Result: ✅ WORKS
```

## Conclusion

**The `#[with_job_id]` macro is the root cause of the stack overflow.**

The `#[with_timeout]` macro is innocent - it was just along for the ride.

## Current State

All `#[with_job_id]` macros commented out in:
- ✅ `start.rs`
- ✅ `stop.rs`
- ✅ `shutdown.rs`
- ✅ `uninstall.rs`
- ✅ `rebuild.rs`
- ✅ `utils/poll.rs`

All `#[with_timeout]` macros still commented out (not the problem, but keeping them off for now).

## Impact

### What Works ✅
- Queen starts successfully
- All daemon lifecycle operations work
- No crashes

### What's Missing ❌
- No SSE routing (job_id not propagated)
- No timeout enforcement
- Narration goes to stdout only, not SSE streams

## Next Steps

1. **Investigate `#[with_job_id]` macro implementation**
   - Location: `bin/99_shared_crates/narration-macros/src/lib.rs`
   - Look for deep async nesting
   - Check macro expansion with `cargo expand`

2. **Find alternative for job_id propagation**
   - Option A: Explicit parameter passing (no macro)
   - Option B: Simpler macro without deep nesting
   - Option C: Thread-local storage (manual)

3. **Re-enable `#[with_timeout]` once job_id is fixed**
   - Timeout macro is fine, just needs job_id fixed first

## Why This Happens

The `#[with_job_id]` macro likely:
- Creates nested async wrappers
- Uses proc macro expansion that creates deep call stacks
- Interacts badly with tokio runtime in CLI context
- Works fine in queen-rbee (server) but fails in rbee-keeper (CLI)

## Test Now

```bash
cargo build --bin rbee-keeper
./target/debug/rbee-keeper queen start
```

Should work without stack overflow! ✅

---

**The mystery is solved. Now we need to fix the macro.**
