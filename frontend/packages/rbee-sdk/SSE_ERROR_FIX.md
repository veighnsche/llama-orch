# SSE Error Handling Fix

**Date:** Oct 24, 2025  
**Issue:** Console showing unhelpful error: `🐝 [SDK] SSE ERROR: Event { obj: Object { obj: JsValue(Event) } }`

## Root Cause

The SSE error event handler in `heartbeat.rs` was logging the raw JavaScript `Event` object using Rust's `{:?}` debug formatting. This produces an unhelpful message because:

1. JavaScript `Event` objects don't contain error details in their properties
2. SSE errors are typically connection failures (network issues, server down, etc.)
3. The browser's EventSource API doesn't expose detailed error information in the error event
4. Rust's debug formatting of WASM objects just shows the wrapper structure, not useful data

## The Fix

**File:** `frontend/packages/rbee-sdk/src/heartbeat.rs` (lines 75-96)

**Before:**
```rust
let error_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
    web_sys::console::error_1(&JsValue::from_str(&format!("🐝 [SDK] SSE ERROR: {:?}", event)));
}) as Box<dyn FnMut(web_sys::Event)>);
```

**After:**
```rust
let es_for_error = event_source.clone();
let error_closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
    let ready_state = es_for_error.ready_state();
    let state_str = match ready_state {
        0 => "CONNECTING",
        1 => "OPEN",
        2 => "CLOSED",
        _ => "UNKNOWN",
    };
    
    // SSE errors are typically connection failures
    // The browser will automatically retry, so this is informational
    web_sys::console::warn_1(&JsValue::from_str(&format!(
        "🐝 [SDK] SSE connection error (state: {} [{}]). Browser will retry automatically.",
        state_str, ready_state
    )));
}) as Box<dyn FnMut(web_sys::Event)>);
```

## Key Improvements

1. **Useful Information:** Now logs the EventSource's `readyState` with human-readable labels:
   - `0` = CONNECTING
   - `1` = OPEN
   - `2` = CLOSED

2. **Appropriate Log Level:** Changed from `console.error` to `console.warn` because:
   - SSE errors are expected during normal operation (network hiccups, server restarts)
   - The browser automatically retries SSE connections
   - This is informational, not a critical error

3. **Better Context:** Message explains that the browser will retry automatically

## Example Output

**Old (unhelpful):**
```
🐝 [SDK] SSE ERROR: Event { obj: Object { obj: JsValue(Event) } }
```

**New (helpful):**
```
🐝 [SDK] SSE connection error (state: CONNECTING [0]). Browser will retry automatically.
```

## Testing

1. Rebuilt WASM package: `pnpm build` in `frontend/packages/rbee-sdk`
2. Restart Next.js dev server to pick up new WASM binary
3. Monitor console when:
   - Server is down (should see CONNECTING state)
   - Network is interrupted (should see state transitions)
   - Server restarts (should see reconnection)

## Related Files

- `frontend/packages/rbee-sdk/src/heartbeat.rs` - Fixed error handler
- `frontend/packages/rbee-sdk/pkg/bundler/rbee_sdk_bg.js` - Generated bindings (now uses `__wbg_warn` instead of `__wbg_error`)

## Notes

- The Event object in SSE error handlers is intentionally minimal by design
- For detailed error information, you need to check EventSource properties (readyState, url)
- SSE connections automatically retry with exponential backoff - this is browser behavior
- If you need custom retry logic, you must implement it at the application level
