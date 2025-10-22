# TEAM-205: Timeout Enforcer Added to SSE Streaming

**Date:** 2025-10-22 11:27 AM  
**Status:** ✅ IMPLEMENTED - Timeout protection in place

---

## What We Did

Added **30-second hard timeout** to SSE streaming in `bin/00_rbee_keeper/src/job_client.rs`.

### Before (HANGING FOREVER)

```rust
while let Some(chunk) = stream.next().await {
    // Process chunks... FOREVER if queen stops responding
}
```

**Problem:** If Queen stops sending events, keeper hangs forever waiting for more chunks.

### After (TIMEOUT AT 30s)

```rust
let stream_result = TimeoutEnforcer::new(Duration::from_secs(30))
    .with_label("Streaming job results")
    .silent()
    .enforce(async move {
        // SSE streaming logic here
        while let Some(chunk) = stream.next().await {
            // ... process chunks
        }
        Ok(())
    })
    .await;
```

**Fixed:** Operation WILL fail after 30 seconds with clear error message.

---

## What This Fixes

### Symptom
```
[keeper    ] job_submit     : 📋 Job job-xxx submitted
[keeper    ] job_stream     : 📡 Streaming results...

<HANGS FOREVER - NO OUTPUT>
```

### After Fix
```
[keeper    ] job_submit     : 📋 Job job-xxx submitted
[keeper    ] job_stream     : 📡 Streaming results...

<After 30 seconds>
Error: Streaming job results timed out after 30 seconds
```

---

## Why 30 Seconds?

Same timeout used for queen startup (`ensure_queen_running`):

```rust
// bin/00_rbee_keeper/src/queen_lifecycle.rs
TimeoutEnforcer::new(Duration::from_secs(30))
    .with_label("Starting queen-rbee")
    .with_countdown()  // Shows progress bar
    .enforce(ensure_queen_running_inner(base_url))
    .await
```

**Rationale:**
- Operations should complete quickly (seconds, not minutes)
- 30 seconds is generous for local operations
- Fail-fast is better than hanging forever
- User gets clear error message instead of infinite wait

---

## Implementation Details

### Changed File
- `bin/00_rbee_keeper/src/job_client.rs`

### Added Imports
```rust
use std::time::Duration;
use timeout_enforcer::TimeoutEnforcer;
```

### Wrapped Operation
Entire SSE streaming loop is wrapped in timeout enforcer:
- Connect to SSE endpoint
- Stream chunks
- Process events
- Return on [DONE] marker

All of this MUST complete within 30 seconds or fail.

---

## Current State

### The REAL Bug

The timeout doesn't fix the root cause - it just prevents hanging forever.

**Root cause:** SSE channel closes immediately with "Closed" error because:
1. Broadcast channel has no initial receiver
2. When client subscribes, it gets "Closed" immediately
3. No events are ever received

**See:** `/home/vince/Projects/llama-orch/bin/TEAM_205_SSE_CHANNEL_CLOSES_IMMEDIATELY.md`

### What Happens Now

```
[keeper    ] job_stream     : 📡 Streaming results...

<30 seconds pass>

Error: Streaming job results timed out after 30 seconds - operation was hanging
```

**Better than before:** At least it fails instead of hanging forever!

---

## Next Steps

1. ✅ Timeout added (prevents infinite hang)
2. ⚠️  Still need to fix SSE channel "Closed" issue
3. ⚠️  Events are being emitted but not received
4. ⚠️  Need to keep initial receiver alive in broadcast channel

---

## Testing

```bash
# Should timeout after 30 seconds instead of hanging forever
./rbee hive start

# Expected after 30s:
# Error: Streaming job results timed out after 30 seconds
```

---

**TEAM-205 Status:** Timeout protection ✅ | SSE fix still needed ⚠️
