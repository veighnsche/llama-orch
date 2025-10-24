# SSH Test Streaming Fix - TEAM-189

**Problem:** SSH test operation doesn't stream any results to the client.

## Root Cause Analysis

### The Issue

The SSE stream handler (`handle_stream_job`) was only reading from the **job registry token receiver**, which is NEVER populated for non-inference operations like SSH test.

```rust
// OLD CODE (BROKEN)
pub async fn handle_stream_job(...) -> Sse<...> {
    let token_stream = crate::job_router::execute_job(job_id, state.into()).await;
    let event_stream = token_stream.map(|data| Ok(Event::default().data(data)));
    Sse::new(event_stream)
}
```

**What happens:**
1. ✅ SSH test executes and emits narrations via `Narration::new(...).emit()`
2. ✅ Narrations are sent to SSE broadcaster (line 362-364 in `narration-core/src/lib.rs`)
3. ❌ **BUT** the HTTP handler is waiting for tokens from `registry.take_token_receiver(&job_id)`
4. ❌ This receiver is `None` because no channel was ever created for SSH test
5. ❌ Stream immediately ends with no output

### The Flow

```
SSH Test Execution:
  execute_ssh_test()
    ↓
  Narration::new(...).emit()
    ↓
  narrate_at_level()
    ↓
  sse_sink::send(&fields)  ← Narrations ARE sent here!
    ↓
  SSE_BROADCASTER.send(event)  ← Global broadcast channel

HTTP Handler (OLD):
  handle_stream_job()
    ↓
  execute_job() returns stream::unfold(receiver, ...)
    ↓
  receiver = registry.take_token_receiver(&job_id)  ← Returns None!
    ↓
  Stream ends immediately  ← No output!
```

## Solution

Subscribe to the **SSE broadcaster** in the HTTP handler to receive narration events.

### Changes Made

**File:** `bin/10_queen_rbee/src/http/jobs.rs`

1. **Import SSE sink:**
   ```rust
   use observability_narration_core::sse_sink;
   ```

2. **Subscribe to broadcaster:**
   ```rust
   let mut sse_rx = sse_sink::subscribe().expect("SSE sink not initialized");
   ```

3. **Merge streams:**
   ```rust
   let combined_stream = async_stream::stream! {
       tokio::pin!(token_stream);
       
       let mut token_stream_ended = false;
       let mut last_event_time = std::time::Instant::now();
       let completion_timeout = std::time::Duration::from_millis(500);

       loop {
           tokio::select! {
               // Narration events from SSE broadcaster
               result = sse_rx.recv() => {
                   match result {
                       Ok(event) => {
                           last_event_time = std::time::Instant::now();
                           let formatted = format!("[{}] {}", event.actor, event.human);
                           yield Ok(Event::default().data(formatted));
                       }
                       Err(_) => break,
                   }
               }
               // Tokens from job registry (for inference)
               token_opt = token_stream.next() => {
                   match token_opt {
                       Some(token) => {
                           last_event_time = std::time::Instant::now();
                           yield Ok(Event::default().data(token));
                       }
                       None => {
                           token_stream_ended = true;
                       }
                   }
               }
               // Timeout: if no events for 500ms, job is complete
               _ = tokio::time::sleep(completion_timeout), if token_stream_ended => {
                   if last_event_time.elapsed() >= completion_timeout {
                       yield Ok(Event::default().data("[DONE]"));
                       break;
                   }
               }
           }
       }
   };
   ```

### How It Works Now

```
SSH Test Execution:
  execute_ssh_test()
    ↓
  Narration::new(...).emit()
    ↓
  sse_sink::send(&fields)
    ↓
  SSE_BROADCASTER.send(event)

HTTP Handler (NEW):
  handle_stream_job()
    ↓
  sse_rx = sse_sink::subscribe()  ← Subscribe to broadcaster!
    ↓
  execute_job() (spawns in background)
    ↓
  combined_stream polls both:
    - sse_rx.recv() ← Receives narration events!
    - token_stream.next() ← Empty for SSH test
    ↓
  Streams narrations to client  ← Works!
```

## Completion Detection

For operations without token streams (like SSH test), we use **timeout-based completion**:

1. Token stream ends immediately (returns `None`)
2. Set `token_stream_ended = true`
3. Track `last_event_time` for each narration
4. If no events for 500ms after token stream ends → send `[DONE]`

This ensures the stream doesn't hang waiting for more events.

## Known Limitations

### Global Broadcaster Issue

The SSE broadcaster is **global** - all narration events from ALL jobs are broadcast to ALL subscribers.

**Current behavior:**
- Client A connects to `/v1/jobs/job-123/stream`
- Client B connects to `/v1/jobs/job-456/stream`
- Both clients receive narrations from BOTH jobs!

**Why this works for now:**
- Only one job runs at a time in testing
- Narrations don't have `job_id` set during execution

**Future fix needed:**
- Propagate `job_id` through execution context
- Filter narration events by `job_id` in the stream handler
- Or use per-job channels instead of global broadcaster

## Testing

```bash
./rbee hive ssh-test --ssh-host workstation.arpa.hoe --ssh-user vince
```

**Expected output:**
```
✅ rbee-keeper is up-to-date
⏱️  Starting queen-rbee (timeout: 30s)
[🧑‍🌾 rbee-keeper / ⚙️ queen-lifecycle]
  Queen is already running and healthy
[🧑‍🌾 rbee-keeper]
  📋 Job job-xxx submitted
[🧑‍🌾 rbee-keeper]
  📡 Streaming results...
[🐝 hive-lifecycle] 🔐 Testing SSH connection to vince@workstation.arpa.hoe:22
[🐝 hive-lifecycle] ✅ SSH test successful
[👑 queen-router] ✅ SSH test successful: test
[DONE]
```

## Files Modified

- `bin/10_queen_rbee/src/http/jobs.rs` - Subscribe to SSE broadcaster, merge streams

## Verification

```bash
cargo check --bin queen-rbee  # ✅ Compiles successfully
```

---

**Status:** ✅ Fix implemented, ready for testing
**Team:** TEAM-189
**Date:** 2025-01-21
