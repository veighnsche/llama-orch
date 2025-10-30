# TEAM-335: All Timeouts & with_job_id Commented Out for Debugging

**Date:** 2025-10-28  
**Status:** 🐛 DEBUGGING (NOT FOR PRODUCTION)  
**Issue:** Stack overflow in Tauri queen start flow

## What Was Done

Commented out ALL `#[with_timeout]` and `#[with_job_id]` macro usage throughout the entire daemon-lifecycle crate for debugging purposes.

## Files Modified

### 1. src/start.rs
- ❌ Commented out: `use observability_narration_macros::with_job_id;`
- ❌ Commented out: `use timeout_enforcer::with_timeout;`
- ❌ Commented out: `#[with_job_id(config_param = "start_config")]`
- ❌ Commented out: `#[with_timeout(secs = 120, label = "Start daemon")]`

### 2. src/stop.rs
- ❌ Commented out: `use observability_narration_macros::with_job_id;`
- ❌ Commented out: `use timeout_enforcer::with_timeout;`
- ❌ Commented out: `#[with_job_id(config_param = "stop_config")]`
- ❌ Commented out: `#[with_timeout(secs = 20, label = "Stop daemon")]`

### 3. src/shutdown.rs
- ❌ Commented out: `use observability_narration_macros::with_job_id;`
- ❌ Commented out: `use timeout_enforcer::with_timeout;`
- ❌ Commented out: `#[with_job_id(config_param = "shutdown_config")]`
- ❌ Commented out: `#[with_timeout(secs = 15, label = "SSH shutdown")]`

### 4. src/uninstall.rs
- ❌ Commented out: `use observability_narration_macros::with_job_id;`
- ❌ Commented out: `use timeout_enforcer::with_timeout;`
- ❌ Commented out: `#[with_job_id(config_param = "uninstall_config")]`
- ❌ Commented out: `#[with_timeout(secs = 60, label = "Uninstall daemon")]`

### 5. src/rebuild.rs
- ❌ Commented out: `use observability_narration_macros::with_job_id;`
- ❌ Commented out: `use timeout_enforcer::with_timeout;`
- ❌ Commented out: `#[with_job_id(config_param = "rebuild_config")]`
- ❌ Commented out: `#[with_timeout(secs = 600, label = "Rebuild daemon")]`

### 6. src/utils/poll.rs
- ❌ Commented out: `use observability_narration_macros::with_job_id;`
- ❌ Commented out: `#[with_job_id]` (on `poll_daemon_health`)

### 7. src/build.rs (Already done by previous TEAM-335)
- ✅ Already commented out: `use observability_narration_macros::with_job_id;`
- ✅ Already commented out: `#[with_job_id(config_param = "build_config")]`

### 8. src/install.rs (Already done by previous TEAM-335)
- ✅ Already commented out: `use observability_narration_macros::with_job_id;`
- ✅ Already commented out: `use timeout_enforcer::with_timeout;`
- ✅ Already commented out: `#[with_job_id(config_param = "install_config")]`
- ✅ Already commented out: `#[with_timeout(secs = 300, label = "Install daemon")]`

## Impact

### What Still Works
- ✅ All async functions still execute normally
- ✅ SSH/SCP operations unaffected
- ✅ Health polling still works
- ✅ Build/install/start/stop logic intact
- ✅ Error handling preserved

### What's Temporarily Disabled
- ❌ No timeout enforcement (operations can hang indefinitely)
- ❌ No SSE routing for narration events (job_id not propagated)
- ❌ Narration events go to stdout only (not SSE streams)
- ❌ No timeout countdown in UI

## Testing Focus

With all async wrapper macros removed, we can isolate:
1. **Is the stack overflow in the daemon-lifecycle functions themselves?**
2. **Is it caused by the macro-generated async wrappers?**
3. **Is it in the Tauri command layer calling these functions?**

## Expected Behavior After This Change

### If Stack Overflow is FIXED ✅
→ The macros were the root cause (too many nested async wrappers)

### If Stack Overflow PERSISTS ❌
→ The problem is in:
- The daemon-lifecycle function implementations themselves
- The Tauri command layer
- Something else entirely

## Restoration Plan

Once debugging is complete:

### Option A: Fix Found in Macros
1. Redesign timeout/job_id propagation without nested async wrappers
2. Consider alternative approaches (explicit parameters vs macros)
3. Test thoroughly before re-enabling

### Option B: Fix Found Elsewhere
1. Uncomment all macros (git checkout or manual restore)
2. Document the real root cause
3. Apply fix in correct location

## Verification Commands

```bash
# Build daemon-lifecycle
cargo check -p daemon-lifecycle

# Build rbee-keeper (uses daemon-lifecycle)
cargo check -p rbee-keeper

# Test queen start flow
cargo build --bin rbee-keeper
./target/debug/rbee-keeper queen start
```

## Verification Results ✅

**Compilation Status:** PASS

```bash
# daemon-lifecycle
✅ cargo check -p daemon-lifecycle
   Finished in 9.78s (no errors)

# rbee-keeper  
✅ cargo check -p rbee-keeper
   Finished in 48.91s (no errors)
```

All code compiles successfully with timeouts and with_job_id commented out.

## Related Files
- `/home/vince/Projects/llama-orch/bin/00_rbee_keeper/TEAM_335_STACK_OVERFLOW_ROOT_CAUSE.md`
- `/home/vince/Projects/llama-orch/bin/99_shared_crates/daemon-lifecycle/TEAM_335_TIMEOUT_REMOVAL.md`

---

**⚠️ WARNING:** These changes are TEMPORARY for debugging only. Do NOT merge to main without proper fix!
