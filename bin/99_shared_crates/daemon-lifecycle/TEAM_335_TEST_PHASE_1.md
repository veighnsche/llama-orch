# TEAM-335: TEST PHASE 1 - with_job_id Enabled

**Date:** 2025-10-28  
**Status:** 🧪 READY TO TEST  
**Configuration:** `#[with_job_id]` ✅ ON | `#[with_timeout]` ❌ OFF

## What Changed

Re-enabled `#[with_job_id]` macro in all daemon-lifecycle functions:
- ✅ `start.rs` - `#[with_job_id(config_param = "start_config")]`
- ✅ `stop.rs` - `#[with_job_id(config_param = "stop_config")]`
- ✅ `shutdown.rs` - `#[with_job_id(config_param = "shutdown_config")]`
- ✅ `uninstall.rs` - `#[with_job_id(config_param = "uninstall_config")]`
- ✅ `rebuild.rs` - `#[with_job_id(config_param = "rebuild_config")]`
- ✅ `utils/poll.rs` - `#[with_job_id]`

Kept `#[with_timeout]` **commented out** in all files.

## Compilation Status

✅ **PASS**
```bash
cargo check -p daemon-lifecycle  # 0.64s - PASS
cargo check -p rbee-keeper       # 1.59s - PASS
```

## Test Instructions

### 1. Kill Existing Queen (if running)
```bash
pkill queen-rbee
# Verify it's dead
pgrep -f queen-rbee  # Should return nothing
```

### 2. Test Queen Start in Tauri GUI

**Action:** Click "Start Queen" button in rbee-keeper UI

**Expected Outcomes:**

#### Scenario A: Stack Overflow Returns ❌
```
thread 'tokio-runtime-worker' has overflowed its stack
fatal runtime error: stack overflow
```

**Conclusion:** The `#[with_job_id]` macro is the culprit (or part of it)

**Next Step:** Comment out `#[with_job_id]`, enable `#[with_timeout]` to test timeout macro alone

#### Scenario B: Works Fine ✅
```
✅ QUEEN START: Queen started successfully (PID: XXXXX)
```

**Conclusion:** The `#[with_job_id]` macro is **NOT** the problem. The `#[with_timeout]` macro is the culprit.

**Next Step:** 
1. Keep `#[with_job_id]` enabled
2. Enable `#[with_timeout]` to confirm it causes the issue
3. Investigate timeout macro implementation

#### Scenario C: Different Error ⚠️
Some other error appears (not stack overflow)

**Next Step:** Document the error, investigate separately

### 3. Check Console Output

Open browser console (F12 → Console) to see:
- ✅ Success message with PID
- ❌ Error messages
- 🔍 Any JavaScript errors

### 4. Verify Queen is Running (if successful)

```bash
pgrep -f queen-rbee  # Should show PID
ss -tlnp | grep 7833  # Should show listener
curl http://localhost:7833/health  # Should return 200 OK
```

## What's Different from Previous Test?

### Before (All Macros Off)
- ❌ No `#[with_job_id]` - No SSE routing
- ❌ No `#[with_timeout]` - No timeout enforcement
- ✅ **Result:** Stack overflow FIXED, queen works

### Now (with_job_id On)
- ✅ `#[with_job_id]` **enabled** - SSE routing active
- ❌ No `#[with_timeout]` - Still no timeout enforcement
- ❓ **Result:** TO BE DETERMINED

### If This Works, Next Test (timeout On)
- ✅ `#[with_job_id]` enabled - SSE routing active
- ✅ `#[with_timeout]` **enabled** - Timeout enforcement active
- ❓ **Result:** Should stack overflow again if timeout is the culprit

## Benefits of with_job_id Being Enabled

If this test passes, we get SSE routing back:
- ✅ Narration events flow to frontend
- ✅ Real-time progress updates in UI
- ✅ job_id propagation works
- ✅ Multi-user scenarios work (events route to correct client)

## What's Still Missing

Even if this works:
- ❌ No timeout enforcement (can hang forever)
- ❌ No timeout countdown UI feedback

## Results Section (Fill After Testing)

### Test Result:
**Status:** _[PENDING - Run test and fill this in]_

**Behavior:** _[Describe what happened]_

**Console Output:** _[Paste relevant console messages]_

**Queen Status:** _[Running/Crashed/Not Started]_

**Conclusion:** _[Which macro is the culprit?]_

---

## Test Checklist

Before testing:
- [ ] Kill existing queen (`pkill queen-rbee`)
- [ ] Open browser console (F12)
- [ ] Note the time (for log correlation)

During testing:
- [ ] Click "Start Queen" button
- [ ] Watch for crash or success
- [ ] Check browser console for messages
- [ ] Note any unusual behavior

After testing:
- [ ] Check if queen is running (`pgrep queen-rbee`)
- [ ] Test health endpoint (`curl http://localhost:7833/health`)
- [ ] Document results in "Results Section" above

---

**Ready to test!** Click the Start Queen button and see what happens. 🚀
