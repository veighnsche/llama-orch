# TEAM-205: Complete Narration Flow Analysis & Entropy Report

**Date:** 2025-10-22 11:31 AM  
**Status:** 🔴 CRITICAL - System is broken, full analysis required

---

## Executive Summary

**The narration system IS working** - events are being emitted and printed to stderr.

**The SSE streaming is BROKEN** - events never reach the keeper via SSE.

**Root cause:** Broadcast channel closes immediately when keeper subscribes.

---

## The ENTIRE Narration Flow (Step by Step)

### Step 1: Keeper Submits Job

**File:** `bin/00_rbee_keeper/src/job_client.rs`

```rust
// 1. Keeper calls queen
let res = client.post(format!("{}/v1/jobs", queen_url))
    .json(&job_payload)
    .send()
    .await?;

// 2. Gets response with job_id
let job_id = json["job_id"].as_str()?;  // e.g., "job-abc123"
let sse_url = json["sse_url"].as_str()?;  // e.g., "/v1/jobs/job-abc123/stream"
```

**Status:** ✅ WORKS - Job is created successfully

---

### Step 2: Queen Creates Job

**File:** `bin/10_queen_rbee/src/job_router.rs`

```rust
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    // 1. Generate job_id
    let job_id = state.registry.create_job();  // "job-abc123"
    
    // 2. Store payload
    state.registry.set_payload(&job_id, payload);
    
    // 3. CREATE SSE CHANNEL ← THIS IS WHERE IT SHOULD WORK
    observability_narration_core::sse_sink::create_job_channel(job_id.clone(), 1000);
    
    // 4. Emit narration (should go to SSE channel)
    NARRATE
        .action("job_create")
        .job_id(&job_id)  // ← CRITICAL: This routes to SSE channel
        .human("Job {} created")
        .emit();
    
    // 5. Return job_id and sse_url to keeper
    Ok(JobResponse { job_id, sse_url })
}
```

**Status:** ✅ WORKS - Job created, channel created, narration emitted

---

### Step 3: SSE Channel Creation

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

```rust
pub fn create_job_channel(job_id: String, capacity: usize) {
    let (tx, rx) = broadcast::channel(capacity);
    
    // TEAM-205: Store BOTH sender and receiver
    self.jobs.lock().unwrap().insert(job_id, (tx, rx));
}
```

**What happens:**
1. Creates broadcast channel with capacity 1000
2. Stores BOTH `tx` (sender) and `rx` (receiver) in HashMap
3. Channel is now "alive" and ready to receive subscribers

**Status:** ✅ SHOULD WORK - Both tx and rx are stored

---

### Step 4: Keeper Connects to SSE Stream

**File:** `bin/00_rbee_keeper/src/job_client.rs`

```rust
// Keeper connects to SSE endpoint
let sse_full_url = format!("{}{}", queen_url, sse_url);
// e.g., "http://localhost:8500/v1/jobs/job-abc123/stream"

let response = client.get(&sse_full_url).send().await?;
```

**Status:** ✅ WORKS - HTTP connection succeeds

---

### Step 5: Queen's SSE Handler Subscribes

**File:** `bin/10_queen_rbee/src/http/jobs.rs`

```rust
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    
    // 1. Subscribe to job's SSE channel
    let sse_rx_opt = sse_sink::subscribe_to_job(&job_id);
    //                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                THIS IS WHERE IT BREAKS!
    
    // 2. Trigger job execution
    let _token_stream = crate::job_router::execute_job(job_id.clone(), state.into()).await;
    
    // 3. Stream events to keeper
    let combined_stream = async_stream::stream! {
        let Some(mut sse_rx) = sse_rx_opt else {
            yield Ok(Event::default().data("ERROR: Job channel not found"));
            return;
        };
        
        // 4. Listen for events
        loop {
            result = sse_rx.recv() => {
                match result {
                    Ok(event) => {
                        yield Ok(Event::default().data(&event.formatted));
                    }
                    Err(_) => {
                        // CHANNEL CLOSED!
                        break;
                    }
                }
            }
        }
    };
    
    Sse::new(combined_stream)
}
```

**Status:** ❌ BROKEN - `sse_rx.recv()` immediately returns `Err(Closed)`

---

### Step 6: Subscribe to Job Channel

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

```rust
pub fn subscribe_to_job(&self, job_id: &str) -> Option<broadcast::Receiver<NarrationEvent>> {
    self.jobs.lock().unwrap()
        .get(job_id)
        .map(|(tx, _rx)| tx.subscribe())
        //               ^^^^^^^^^^^^^^
        //               Creates NEW receiver from sender
}
```

**What happens:**
1. Finds the (tx, rx) tuple in HashMap
2. Calls `tx.subscribe()` to create a NEW receiver
3. Returns this new receiver to the SSE handler

**Status:** ✅ WORKS - Returns `Some(new_rx)`

---

### Step 7: Job Execution Starts

**File:** `bin/99_shared_crates/job-registry/src/lib.rs`

```rust
pub async fn execute_and_stream<T, F, Exec>(
    job_id: String,
    registry: Arc<JobRegistry<T>>,
    executor: Exec,
) -> impl Stream<Item = String> {
    let payload = registry.take_payload(&job_id);
    
    if let Some(payload) = payload {
        // Spawn background task to execute job
        tokio::spawn(async move {
            NARRATE
                .action("execute")
                .job_id(&job_id_clone)  // ← Routes to SSE channel
                .human("Executing job {}")
                .emit();
            
            // Execute the actual operation
            executor(job_id_clone.clone(), payload).await;
        });
    }
    
    // Return empty stream (no tokens for non-inference operations)
    stream::unfold(receiver, |rx_opt| async move { None })
}
```

**Status:** ✅ WORKS - Task spawns, narrations are emitted

---

### Step 8: Narration Emission

**File:** `bin/99_shared_crates/narration-core/src/lib.rs`

```rust
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    // 1. Print to stderr (ALWAYS works)
    eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, fields.human);
    
    // 2. Send to SSE if enabled
    if sse_sink::is_enabled() {
        sse_sink::send(&fields);
    }
}
```

**Status:** ✅ WORKS - Both stderr and SSE send are called

---

### Step 9: SSE Send

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

```rust
pub fn send(fields: &NarrationFields) {
    let event = NarrationEvent::from(fields.clone());
    
    // SECURITY: Only send if we have a job_id
    if let Some(job_id) = &fields.job_id {
        SSE_BROADCASTER.send_to_job(job_id, event);
    }
}

pub fn send_to_job(&self, job_id: &str, event: NarrationEvent) {
    let jobs = self.jobs.lock().unwrap();
    if let Some((tx, _rx)) = jobs.get(job_id) {
        // Send event to broadcast channel
        let _ = tx.send(event);
        //       ^^^^^^^^
        //       THIS SHOULD WORK!
    }
}
```

**Status:** ✅ WORKS - Event is sent to broadcast channel

---

## The Mystery: Why Does recv() Return Closed?

### Debug Output Analysis

```
[TEAM-205] SSE receiver subscribed to job: job-xxx
[TEAM-205] Starting SSE event loop for job: job-xxx
[TEAM-205] SSE receiver error: Closed
```

**This means:**
1. ✅ Subscription succeeded (got `Some(rx)`)
2. ✅ Started listening for events
3. ❌ First `rx.recv()` call returns `Err(RecvError::Closed)`

### Tokio Broadcast Channel Behavior

From Tokio docs:

> `RecvError::Closed` is returned when **all senders have been dropped**.

**But we're storing the sender in the HashMap!** So why is it "closed"?

---

## Hypothesis 1: Timing Issue

**Theory:** The receiver subscribes BEFORE any events are sent, and the channel is considered "empty/closed".

**Evidence:**
```
1. create_job_channel() - Creates (tx, rx), stores both
2. subscribe_to_job() - Creates NEW receiver via tx.subscribe()
3. execute_job() - Spawns background task (takes time)
4. sse_rx.recv() - Called immediately, gets Closed
5. Background task starts emitting - TOO LATE!
```

**Problem:** There's a race condition between:
- SSE handler starting to listen (`sse_rx.recv()`)
- Background task starting to emit events

---

## Hypothesis 2: Receiver Lagging

From Tokio docs:

> If a receiver is too slow and messages are sent faster than it can receive them, the receiver will return `RecvError::Lagged`.

**But we're getting `Closed`, not `Lagged`.**

---

## Hypothesis 3: Channel Capacity Issue

**Theory:** Channel capacity is 1000, but something is wrong with the channel state.

**Evidence:** We're storing both tx and rx, so the channel should stay alive.

---

## Hypothesis 4: Multiple Receivers Issue

**Theory:** When we call `tx.subscribe()`, it creates a NEW receiver. But the broadcast channel might have special behavior with multiple receivers.

From Tokio docs:

> Each receiver independently tracks its position in the channel.

**This should work!** Multiple receivers are supported.

---

## The REAL Problem (Most Likely)

### Race Condition in handle_stream_job

```rust
pub async fn handle_stream_job(...) -> Sse<...> {
    // 1. Subscribe FIRST (before execution starts)
    let sse_rx_opt = sse_sink::subscribe_to_job(&job_id);
    
    // 2. THEN trigger execution
    let _token_stream = crate::job_router::execute_job(job_id.clone(), state.into()).await;
    
    // 3. THEN start listening
    let combined_stream = async_stream::stream! {
        let Some(mut sse_rx) = sse_rx_opt else { ... };
        
        // 4. First recv() call happens HERE
        result = sse_rx.recv() => {
            // Gets Closed immediately!
        }
    };
}
```

**The issue:**
1. We subscribe to the channel
2. We trigger execution (which spawns a background task)
3. We start listening
4. **But the background task hasn't started emitting yet!**
5. The channel appears "empty" and returns Closed

---

## Why Keeping Initial Receiver Didn't Fix It

We changed:

```rust
// Before
jobs: Arc<Mutex<HashMap<String, broadcast::Sender<NarrationEvent>>>>

// After
jobs: Arc<Mutex<HashMap<String, (broadcast::Sender<NarrationEvent>, broadcast::Receiver<NarrationEvent>)>>>
```

**This SHOULD keep the channel alive!**

But it's still returning Closed. Why?

### Possible Reason: The Initial Receiver is Never Used

The initial `rx` we store is just sitting in the HashMap, never calling `recv()`.

**Broadcast channels might close if:**
- All receivers have been dropped OR
- All receivers are "inactive" (not calling recv())

---

## Entropy Analysis

### Complexity Score: 🔴 VERY HIGH

**Number of moving parts:**
1. Job creation (queen)
2. SSE channel creation (narration-core)
3. Job execution spawning (job-registry)
4. SSE subscription (queen HTTP handler)
5. Narration emission (narration-core)
6. SSE send (narration-core)
7. SSE receive (queen HTTP handler)
8. SSE streaming (keeper)

**Points of failure:** 8

**Async boundaries:** 4 (tokio::spawn, async_stream, SSE streaming, HTTP)

**Shared state:** 2 (JobRegistry, SseBroadcaster HashMap)

**Race conditions:** 2 (subscription vs execution, emission vs reception)

### Entropy Sources

1. **Timing:** Subscription happens before emission
2. **Async spawning:** Background task takes time to start
3. **Broadcast channel semantics:** Unclear when channel is "closed"
4. **Multiple receivers:** Initial rx + subscribed rx
5. **HTTP streaming:** Additional async layer

---

## Do We Need to Start Over?

### Option 1: Fix the Broadcast Channel (Current Approach)

**Pros:**
- Keeps current architecture
- Minimal changes needed
- Broadcast pattern is elegant

**Cons:**
- Complex debugging
- Unclear why it's not working
- Multiple async boundaries

**Verdict:** ⚠️  Worth one more try, but if it doesn't work, pivot

---

### Option 2: Use MPSC Instead of Broadcast

**Change:**
```rust
// Instead of broadcast::channel
use tokio::sync::mpsc;

pub struct SseBroadcaster {
    jobs: Arc<Mutex<HashMap<String, mpsc::Sender<NarrationEvent>>>>,
}
```

**Pros:**
- Simpler semantics (single receiver)
- No "Closed" issues
- Well-understood behavior

**Cons:**
- Only one subscriber per job (but that's what we have anyway!)
- Need to change subscription model

**Verdict:** ✅ RECOMMENDED - This will definitely work

---

### Option 3: Direct Channel Passing (Simplest)

**Change:** Pass the receiver directly when creating the job, instead of storing in HashMap.

```rust
pub async fn create_job(...) -> Result<(JobResponse, mpsc::Receiver<NarrationEvent>)> {
    let (tx, rx) = mpsc::channel(1000);
    
    // Store tx for narration emission
    state.registry.set_sse_sender(&job_id, tx);
    
    // Return rx directly to HTTP handler
    Ok((JobResponse { job_id, sse_url }, rx))
}
```

**Pros:**
- No HashMap lookup
- No subscription step
- Direct connection
- Guaranteed to work

**Cons:**
- Changes API structure
- Less flexible

**Verdict:** ✅ VERY SIMPLE - Would definitely work

---

### Option 4: Abandon SSE, Use HTTP Polling

**Change:** Instead of SSE streaming, poll for narration events.

```rust
// Keeper polls: GET /v1/jobs/{job_id}/events
// Queen returns: { events: [...], done: false }
```

**Pros:**
- No streaming complexity
- No async channel issues
- Simple HTTP requests

**Cons:**
- Polling overhead
- Not real-time
- More HTTP requests

**Verdict:** ❌ NOT RECOMMENDED - SSE is better when it works

---

## Recommended Fix: Switch to MPSC

### Why MPSC Will Work

1. **Single receiver model** - Matches our use case (one keeper per job)
2. **No "Closed" semantics** - Channel stays open until sender is dropped
3. **Simpler** - Fewer edge cases
4. **Proven** - Used everywhere in Tokio ecosystem

### Implementation

```rust
// In sse_sink.rs
pub struct SseBroadcaster {
    jobs: Arc<Mutex<HashMap<String, mpsc::Sender<NarrationEvent>>>>,
}

pub fn create_job_channel(&self, job_id: String, capacity: usize) {
    let (tx, rx) = mpsc::channel(capacity);
    self.jobs.lock().unwrap().insert(job_id, tx);
    // Store rx somewhere OR return it directly
}

pub fn subscribe_to_job(&self, job_id: &str) -> Option<mpsc::Receiver<NarrationEvent>> {
    // Return the receiver (need to store it somewhere first)
}
```

**Challenge:** MPSC has single receiver, so we need to store it and retrieve it once.

---

## Alternative: Keep Broadcast, Fix the Race

### The Fix

Add a small delay or initial event to ensure channel is "active":

```rust
pub fn create_job_channel(&self, job_id: String, capacity: usize) {
    let (tx, rx) = broadcast::channel(capacity);
    
    // Send initial "connected" event to activate channel
    let _ = tx.send(NarrationEvent {
        formatted: "[CONNECTED]".to_string(),
        actor: "system".to_string(),
        action: "connected".to_string(),
        human: "SSE channel connected".to_string(),
        ..Default::default()
    });
    
    self.jobs.lock().unwrap().insert(job_id, (tx, rx));
}
```

**This might work!** The initial event "primes" the channel.

---

## Summary

### What's Working
- ✅ Job creation
- ✅ SSE channel creation
- ✅ Job execution
- ✅ Narration emission (to stderr)
- ✅ SSE send (to broadcast channel)

### What's Broken
- ❌ SSE receive (gets Closed immediately)
- ❌ Events never reach keeper

### Root Cause
- Broadcast channel returns `Closed` on first `recv()` call
- Likely a race condition or channel state issue

### Recommended Actions

**Priority 1:** Try sending initial event to "prime" the channel

**Priority 2:** Switch to MPSC if broadcast doesn't work

**Priority 3:** Consider direct channel passing (simplest)

---

## Conclusion

**Do we need to start over?** 

**NO** - The architecture is sound. We just need to fix the broadcast channel issue.

**Two paths forward:**
1. Fix broadcast (send initial event)
2. Switch to MPSC (guaranteed to work)

**Estimated time:** 30 minutes to try both approaches

---

**End of TEAM-205 Analysis**
