# TEAM-335: Queen Start Works But No User Feedback

**Date:** 2025-10-28  
**Status:** ✅ FUNCTIONAL but ❌ NO UI FEEDBACK

## Summary

After commenting out `#[with_timeout]` and `#[with_job_id]` macros in daemon-lifecycle:
- ✅ **Stack overflow is FIXED**
- ✅ **Queen starts successfully**
- ✅ **Health check passes**
- ❌ **No user feedback in UI**

## Evidence

### Queen is Running ✅
```bash
$ pgrep -f queen-rbee
135941

$ ss -tlnp | grep 7833
LISTEN 0  1024  127.0.0.1:7833  0.0.0.0:*  users:(("queen-rbee",pid=135941,fd=9))

$ curl http://localhost:7833/health
HTTP/1.1 200 OK
```

### Tauri Command Returns Success ✅
```rust
// tauri_commands.rs:157
Ok(format!("Queen started successfully (PID: {})", pid))
```

### UI Ignores Result ❌
```typescript
// ServicesPage.tsx:27-28
case "queen-start":
  await invoke("queen_start");  // ← Result thrown away!
  break;
```

## Root Cause

The frontend code:
1. Calls `invoke("queen_start")`
2. Gets back `Ok("Queen started successfully (PID: 135941)")`
3. **Ignores the result completely**
4. No toast, no alert, no console.log

User experience:
- Clicks "Start Queen" button
- Buttons disable for 500ms
- Buttons re-enable
- **Nothing else happens** (appears broken, but queen IS running)

## Quick Fix Options

### Option 1: Console log (minimal)
```typescript
case "queen-start":
  const result = await invoke("queen_start");
  console.log(result);  // ← At least show in console
  break;
```

### Option 2: Toast notification (better UX)
```typescript
case "queen-start":
  try {
    const result = await invoke("queen_start");
    showToast("success", result);  // ← Need toast system
  } catch (error) {
    showToast("error", `Failed: ${error}`);
  }
  break;
```

### Option 3: Status update (best)
```typescript
case "queen-start":
  const result = await invoke("queen_start");
  // Update ServiceCard status to "running"
  updateQueenStatus("running");
  break;
```

## What About SSE?

The frontend might be expecting SSE narration events (like the CLI):
- CLI: Streams progress through SSE (build steps, health checks, etc.)
- Tauri: Direct implementation, no SSE

With `#[with_job_id]` commented out:
- ❌ No SSE channel setup
- ❌ No narration events
- ❌ No progress updates

This is fine for debugging, but production needs SSE back for real-time feedback.

## Next Steps

### Immediate (debugging continues):
1. ✅ Add console.log to show result
2. ✅ Verify other commands work too (stop, install, etc.)
3. Document which macro caused stack overflow (timeout or with_job_id)

### After debugging:
1. Re-enable macros one at a time to isolate culprit
2. Fix macro implementation (reduce nesting)
3. Restore SSE communication for rich feedback
4. Add toast system for user notifications

## Related Files
- `/home/vince/Projects/llama-orch/bin/00_rbee_keeper/ui/src/pages/ServicesPage.tsx` (line 27-28)
- `/home/vince/Projects/llama-orch/bin/00_rbee_keeper/src/tauri_commands.rs` (line 157)
- `/home/vince/Projects/llama-orch/bin/99_shared_crates/daemon-lifecycle/TEAM_335_DEBUG_TIMEOUTS_COMMENTED.md`

---

**Bottom line:** Queen works! Just need to show the user what happened.
