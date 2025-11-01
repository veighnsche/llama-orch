# TEAM-350: Enhanced Error Logging + Reduced Heartbeat Noise

**Status:** ✅ COMPLETE

## Problem

1. RHAI test failing with "Unknown error" - no useful error information
2. Heartbeat logs polluting console with excessive noise

## Solution

### 1. Enhanced Error Logging in RHAI Test

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useRhaiScripts.ts`

Added comprehensive logging at every step:

```typescript
console.log('[RHAI Test] Starting test...')
console.log('[RHAI Test] Client created, baseUrl:', baseUrl)
console.log('[RHAI Test] Operation:', operation)
console.log('[RHAI Test] Submitting and streaming...')
console.log('[RHAI Test] SSE line:', line)
console.log('[RHAI Test] Stream complete, receivedDone:', receivedDone)

// Enhanced error logging
console.error('[RHAI Test] Error caught:', err)
console.error('[RHAI Test] Error type:', typeof err)
console.error('[RHAI Test] Error details:', {
  message: (err as Error).message,
  stack: (err as Error).stack,
  name: (err as Error).name,
})
```

**Now you'll see:**
- Exactly where the error occurs
- Full error details including stack trace
- Step-by-step progress through the test flow
- Whether [DONE] marker was received

### 2. Reduced Heartbeat Logging

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/heartbeat.rs`

**Before (NOISY):**
```rust
web_sys::console::log_1("🐝 [SDK] Connecting to SSE: http://...");
web_sys::console::log_1("🐝 [SDK] EventSource created, ready_state: 1");
web_sys::console::log_1("🐝 [SDK] SSE connection OPENED");
web_sys::console::log_1("🐝 [SDK] Received 'heartbeat' event");
web_sys::console::log_1("🐝 [SDK] Event data: {...}");
web_sys::console::log_1("🐝 [SDK] JSON parsed successfully");
// ... repeated every 2 seconds!
```

**After (SILENT):**
```rust
// TEAM-350: Reduced logging - only log on errors

// Connection opened - no log needed

// Parse the event data (no logging)

// TEAM-350: Silent - call callback without logging
```

**Now only logs:**
- ⚠️ Warnings on connection errors (auto-retry)
- ❌ Errors on JSON parse failures
- ❌ Warnings if event has no data

### 3. Rebuilt WASM SDK

```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-sdk
pnpm build
# ✅ Success - new WASM bundle with reduced logging
```

## Expected Behavior

### When Test Button is Pressed

Console will show clear progress:

```
[RHAI Test] Starting test...
[RHAI Test] Client created, baseUrl: http://localhost:7833
[RHAI Test] Operation: { RhaiScriptTest: { content: "..." } }
[RHAI Test] Submitting and streaming...
[RHAI Test] SSE line: data: {...}
[RHAI Test] Narration event: {...}
[RHAI Test] SSE line: [DONE]
[RHAI Test] Stream complete, receivedDone: true
[RHAI Test] Finished
```

### If Error Occurs

Console will show detailed error:

```
[RHAI Test] Error caught: Error: ...
[RHAI Test] Error type: object
[RHAI Test] Error details: {
  message: "Script content cannot be empty",
  stack: "Error: Script content cannot be empty\n  at ...",
  name: "Error"
}
```

### Heartbeat (Background)

Console will be **silent** unless there's an actual error:

```
// Normal operation: NOTHING logged ✅

// Only on error:
⚠️ 🐝 [SDK] SSE connection error (state: CONNECTING [0]). Browser will retry automatically.
```

## Files Changed

1. **useRhaiScripts.ts** - Added comprehensive error logging
2. **heartbeat.rs** - Removed verbose success logging, kept error logging
3. **Rebuilt WASM SDK** - New bundle with reduced logging

## Next Steps

**With this enhanced logging, you should now see exactly where and why the test is failing.**

Possible issues to look for:
1. Job submission failing (wrong endpoint?)
2. SSE stream not connecting
3. Backend returning error (check server logs too)
4. [DONE] marker not being sent
5. Empty content causing bail in backend

---

**TEAM-350 Signature:** Enhanced error logging + reduced heartbeat noise
