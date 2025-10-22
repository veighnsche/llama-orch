# TEAM-205: Critical Bug - SSE Channel Closes Immediately

**Date:** 2025-10-22 11:23 AM  
**Status:** 🔴 CRITICAL BUG - System is completely broken

---

## The Problem

`./rbee hive start` HANGS because SSE channel closes before any narrations are sent.

### Debug Output

```
[TEAM-205] SSE receiver subscribed to job: job-0b355bb8-27d4-44e7-8535-aa639ca955cc
[TEAM-205] Starting SSE event loop for job: job-0b355bb8-27d4-44e7-8535-aa639ca955cc
[TEAM-205] SSE receiver error: Closed
                              ^^^^^^
                              IMMEDIATELY CLOSED - NO EVENTS RECEIVED!

[keeper    ] job_submit     : 📋 Job job-0b355bb8-27d4-44e7-8535-aa639ca955cc submitted
[keeper    ] job_stream     : 📡 Streaming results...
                              
<HANGS FOREVER>
```

### What This Means

1. ✅ Job channel is created: `observability_narration_core::sse_sink::create_job_channel()`
2. ✅ SSE receiver subscribes: `sse_sink::subscribe_to_job()` returns `Some(rx)`
3. ❌ **Receiver immediately gets "Closed" error**
4. ❌ **NO narrations are ever received**
5. ❌ Keeper hangs waiting for events that never come

---

## Root Cause

The `tokio::sync::broadcast` channel closes when **ALL senders are dropped**.

### The Bug

```rust
// In sse_sink.rs
pub fn create_job_channel(job_id: String, capacity: usize) {
    let (tx, _) = broadcast::channel(capacity);
    //           ^ This receiver is dropped immediately!
    
    self.jobs.lock().unwrap().insert(job_id, tx);
    //                                       ^^ ONLY sender is stored
}

// Later when we subscribe...
pub fn subscribe_to_job(&self, job_id: &str) -> Option<broadcast::Receiver<NarrationEvent>> {
    self.jobs.lock().unwrap()
        .get(job_id)
        .map(|tx| tx.subscribe())  // Creates NEW receiver
}

// But if tx.subscribe() is called and there are NO OTHER RECEIVERS...
// The channel is considered "closed" because no one is listening!
```

### Why It Fails

**Tokio broadcast channels have this behavior:**
- When `tx.subscribe()` is called, it creates a new receiver
- But if the channel has no messages yet AND no receivers, it's considered "empty/closed"
- The receiver immediately returns `Err(RecvError::Closed)`

---

## The Flow (Broken)

```
1. create_job() is called
   └─> create_job_channel(job_id, 1000)
       └─> Creates (tx, _rx) but drops _rx immediately
       └─> Stores only tx in HashMap

2. Client connects to SSE: GET /v1/jobs/{id}/stream
   └─> handle_stream_job() is called
       └─> sse_rx = subscribe_to_job(job_id) 
           └─> tx.subscribe() creates NEW receiver
           └─> But channel is "empty" (no messages yet)
           └─> Receiver gets Closed error immediately
       
3. execute_job() spawns background task
   └─> Narrations are emitted: NARRATE.action("...").job_id(&job_id).emit()
       └─> sse_sink::send(&fields) is called
           └─> tx.send(event) is called
           └─> BUT RECEIVER ALREADY CLOSED!
           └─> Events are dropped silently

4. Keeper waits forever for events that never arrive
```

---

## Why This Happens

### The Race Condition

There's a timing issue in `handle_stream_job()`:

```rust
pub async fn handle_stream_job(...) -> Sse<...> {
    // 1. Subscribe to job BEFORE execution starts
    let sse_rx_opt = sse_sink::subscribe_to_job(&job_id);
    
    // 2. Trigger execution (spawns in background)
    let _token_stream = crate::job_router::execute_job(job_id.clone(), state.into()).await;
    
    // 3. Start listening to SSE events
    let combined_stream = async_stream::stream! {
        let Some(mut sse_rx) = sse_rx_opt else { ... };
        
        // 4. Receiver gets "Closed" because:
        //    - No messages sent yet
        //    - execute_job() is still starting up
        //    - broadcast channel considers it "empty/closed"
        
        result = sse_rx.recv() => {
            match result {
                Err(_) => {  // Closed!
                    break;   // Exit immediately
                }
            }
        }
    };
}
```

---

## The Solution

### Option 1: Keep Initial Receiver Alive (RECOMMENDED)

Don't drop the initial receiver when creating the channel:

```rust
// In sse_sink.rs
pub struct SseBroadcaster {
    /// Per-job channels with BOTH sender AND initial receiver
    jobs: Arc<Mutex<HashMap<String, (broadcast::Sender<NarrationEvent>, broadcast::Receiver<NarrationEvent>)>>>,
}

pub fn create_job_channel(&self, job_id: String, capacity: usize) {
    let (tx, rx) = broadcast::channel(capacity);
    // Keep BOTH sender and receiver!
    self.jobs.lock().unwrap().insert(job_id, (tx, rx));
}

pub fn subscribe_to_job(&self, job_id: &str) -> Option<broadcast::Receiver<NarrationEvent>> {
    self.jobs.lock().unwrap()
        .get(job_id)
        .map(|(tx, _rx)| tx.subscribe())  // Still use tx.subscribe()
}
```

This keeps the channel "alive" because there's always at least one receiver (even if unused).

### Option 2: Use Different Channel Type

Use `tokio::sync::mpsc` instead of `broadcast`:
- mpsc doesn't have this "closed" issue
- Single receiver model (fits our use case)
- More predictable behavior

### Option 3: Send Initial "Connected" Event

Send a dummy event immediately after creating the channel:

```rust
pub fn create_job_channel(job_id: String, capacity: usize) {
    let (tx, _) = broadcast::channel(capacity);
    
    // Send initial event to keep channel alive
    let _ = tx.send(NarrationEvent {
        formatted: "[CONNECTED]".to_string(),
        ...
    });
    
    self.jobs.lock().unwrap().insert(job_id, tx);
}
```

---

## Fix Implementation

**Recommended: Option 1 (Keep Initial Receiver)**

This is the cleanest fix that preserves the broadcast pattern and fixes the root cause.

---

## Testing After Fix

```bash
# Should see narrations flowing:
./rbee hive start

# Expected output (NO LONGER HANGS):
[qn-router ] job_create     : Job job-xxx created
[qn-router ] route_job      : Executing: hive_start
[qn-router ] hive_start     : 🚀 Starting hive 'localhost'
[qn-router ] hive_check     : 📋 Checking if already running
[DONE]
```

---

## Impact

**This bug breaks EVERYTHING:**
- ✅ Queen emits narrations (visible in stderr)
- ❌ SSE channel closes immediately
- ❌ Keeper never receives narrations
- ❌ Commands hang forever
- ❌ System is completely unusable

**Priority: CRITICAL - FIX IMMEDIATELY**

---

## Related Files

- `bin/99_shared_crates/narration-core/src/sse_sink.rs` - SSE broadcaster (BUG HERE)
- `bin/10_queen_rbee/src/http/jobs.rs` - SSE stream handler
- `bin/10_queen_rbee/src/job_router.rs` - Job execution and narration

---

## Next Steps

1. ✅ Implement Option 1 (keep initial receiver alive)
2. ✅ Test `./rbee hive start` - should complete immediately
3. ✅ Remove debug logging (TEAM-205 markers)
4. ✅ Verify all operations work

---

**End of TEAM-205 Bug Report**
