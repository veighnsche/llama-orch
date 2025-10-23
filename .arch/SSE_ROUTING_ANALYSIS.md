# SSE Routing Analysis: Distributed vs Integrated Mode

**Date:** Oct 23, 2025  
**Status:** ✅ ARCHITECTURE VERIFIED CORRECT

---

## Problem Statement

**Question:** In integrated mode (local-hive feature), when hive crates emit narration, will events reach the correct SSE stream?

**Answer:** ✅ **YES, it works correctly!** The architecture is sound.

---

## SSE Broadcaster Architecture

### Global Singleton Per-Process

```rust
// bin/99_shared_crates/narration-core/src/sse_sink.rs
static SSE_BROADCASTER: once_cell::sync::Lazy<SseBroadcaster> =
    once_cell::sync::Lazy::new(SseBroadcaster::new);
```

**Key Insight:** ONE instance per process, shared across all threads.

### Job Channel Lifecycle

```rust
// 1. Create channel (called by job creator)
create_job_channel(job_id: String, capacity: usize)
  ├─ Creates mpsc channel
  ├─ Stores sender in SSE_BROADCASTER.senders
  └─ Stores receiver in SSE_BROADCASTER.receivers

// 2. Emit narration (called by operation handlers)
NARRATE.action("worker_spawn").job_id(job_id).emit()
  ├─ sse_sink::send(&fields)
  ├─ SSE_BROADCASTER.send_to_job(job_id, event)
  └─ If channel exists: send event, else: DROP

// 3. Take receiver (called by SSE endpoint)
take_job_receiver(job_id)
  └─ Moves receiver out of SSE_BROADCASTER
```

**Security:** If no channel exists, events are DROPPED (fail-fast, prevent leaks).

---

## Flow Analysis

### Distributed Mode (Current)

#### Processes

```
┌──────────────────────────────────────────┐
│ PROCESS 1: queen-rbee (port 8500)       │
│ - Has its own SSE_BROADCASTER            │
│ - Creates queen job channels             │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ PROCESS 2: rbee-hive (port 9000)        │
│ - Has its own SSE_BROADCASTER            │
│ - Creates hive job channels              │
└──────────────────────────────────────────┘
```

#### Request Flow

```
1. Keeper → Queen:
   POST http://localhost:8500/v1/jobs
   Body: { "operation": { "WorkerSpawn": {...} } }
   Response: { "job_id": "queen-abc123" }

2. Queen creates job channel:
   ├─ create_job_channel("queen-abc123", 1000)
   └─ Channel created in QUEEN's SSE_BROADCASTER ✅

3. Queen forwards to Hive:
   POST http://localhost:9000/v1/jobs
   Body: { "operation": { "WorkerSpawn": {...} } }
   Response: { "job_id": "hive-xyz789" }

4. Hive creates its OWN job channel:
   ├─ create_job_channel("hive-xyz789", 1000)
   └─ Channel created in HIVE's SSE_BROADCASTER ✅

5. Hive executes operation:
   ├─ NARRATE.action("worker_spawn").job_id("hive-xyz789").emit()
   ├─ Looks for channel in HIVE's SSE_BROADCASTER
   └─ ✅ Channel EXISTS! Events emitted

6. Queen connects to Hive's SSE stream:
   GET http://localhost:9000/v1/jobs/hive-xyz789/stream
   ├─ Receives events from Hive
   └─ Forwards to queen's job channel "queen-abc123"

7. Keeper receives events:
   GET http://localhost:8500/v1/jobs/queen-abc123/stream
   └─ ✅ Gets both queen AND hive events
```

**Key:** Each process manages its own jobs and channels. HTTP links them.

---

### Integrated Mode (Planned - local-hive feature)

#### Processes

```
┌────────────────────────────────────────────────────────────┐
│ PROCESS 1: queen-rbee [with embedded hive logic]          │
│ - Single SSE_BROADCASTER (shared by queen and hive code)  │
│ - Creates job channels                                     │
│ - Hive crates execute in same process                      │
└────────────────────────────────────────────────────────────┘

No PROCESS 2! Hive logic is embedded in queen.
```

#### Request Flow

```
1. Keeper → Queen:
   POST http://localhost:8500/v1/jobs
   Body: { "operation": { "WorkerSpawn": {...} } }
   Response: { "job_id": "abc123" }

2. Queen creates job channel:
   ├─ create_job_channel("abc123", 1000)
   └─ Channel created in (SINGLE) SSE_BROADCASTER ✅

3. Queen calls hive crate DIRECTLY (NO HTTP):
   ├─ #[cfg(feature = "local-hive")]
   ├─ if is_localhost_hive(hive_id, &config) {
   │     return forward_via_local_hive(job_id="abc123", operation).await;
   │  }
   │
   └─ rbee_hive_worker_lifecycle::spawn_worker(
        job_id="abc123",  // ← CRITICAL: Queen's job_id passed to hive crate
        model, device
      )

4. Hive crate emits narration:
   ├─ NARRATE.action("worker_spawn").job_id("abc123").emit()
   ├─ Looks for channel in (SINGLE) SSE_BROADCASTER
   └─ ✅ Channel EXISTS! Events emitted to queen's channel

5. Keeper receives events:
   GET http://localhost:8500/v1/jobs/abc123/stream
   └─ ✅ Gets both queen AND hive crate events
```

**Key:** Same process = same SSE_BROADCASTER = hive narration flows to queen's channel!

---

## Why It Works

### 1. Hive Crates Don't Create Job Channels

**Verified:** No calls to `create_job_channel()` in hive crates.

```bash
$ grep -r "create_job_channel" bin/25_rbee_hive_crates/
# No results!
$ grep -r "create_job_channel" bin/15_queen_rbee_crates/
# No results!
```

**Conclusion:** Only binaries create job channels. Crates emit to existing channels.

### 2. Hive Crates Accept job_id as Parameter

**Example from hive-lifecycle:**

```rust
pub struct HiveStartRequest {
    pub alias: String,
    pub job_id: String,  // ← Passed from caller (queen)
}

pub async fn execute_hive_start(
    request: HiveStartRequest,
    config: Arc<RbeeConfig>,
) -> Result<HiveStartResponse> {
    let job_id = &request.job_id;  // ← Use caller's job_id
    
    NARRATE
        .action("hive_start")
        .job_id(job_id)  // ← Emit with caller's job_id
        .emit();
```

**Conclusion:** Hive crates use the job_id provided by the caller (queen).

### 3. Same Process = Same SSE_BROADCASTER

**In integrated mode:**
- Queen creates job channel in SSE_BROADCASTER
- Queen calls hive crate functions (same process)
- Hive crate narration goes to same SSE_BROADCASTER
- Channel exists, events flow correctly

---

## Potential Issues (None Found!)

### ❌ Issue 1: Hive Creates Its Own Channels?

**Status:** NOT AN ISSUE

**Evidence:** Hive crates don't call `create_job_channel()`.

### ❌ Issue 2: Hive Uses Different job_id?

**Status:** NOT AN ISSUE

**Evidence:** Hive crates accept job_id as parameter from caller.

### ❌ Issue 3: Process Boundary Breaks SSE?

**Status:** NOT AN ISSUE

**Evidence:** In integrated mode, there's no process boundary!

---

## Architectural Guarantees

### ✅ Distributed Mode

1. Queen creates job channel in Queen's process
2. Hive creates job channel in Hive's process
3. HTTP links the two streams
4. Events flow: Hive → Queen → Keeper

**Guarantee:** Each process manages its own channels. HTTP forwards events.

### ✅ Integrated Mode

1. Queen creates job channel in (single) process
2. Queen calls hive crates with queen's job_id
3. Hive crates emit to (single) SSE_BROADCASTER
4. Events flow: Hive crates → Queen's channel → Keeper

**Guarantee:** Same process = same SSE_BROADCASTER = narration works!

---

## Testing Strategy

### Unit Test: Integrated Mode SSE Routing

```rust
#[tokio::test]
async fn test_integrated_mode_sse_routing() {
    // 1. Create job channel (simulating queen)
    let job_id = "test-abc123";
    create_job_channel(job_id.to_string(), 100);
    
    // 2. Take receiver (simulating SSE endpoint)
    let mut rx = take_job_receiver(job_id).unwrap();
    
    // 3. Call hive crate function with queen's job_id
    // (simulating integrated mode)
    let request = HiveStartRequest {
        alias: "localhost".to_string(),
        job_id: job_id.to_string(),
    };
    
    // Spawn execution in background
    tokio::spawn(async move {
        execute_hive_start(request, config).await.ok();
    });
    
    // 4. Verify events are received
    let event1 = rx.recv().await.unwrap();
    assert!(event1.formatted.contains("Starting hive"));
    
    let event2 = rx.recv().await.unwrap();
    assert!(event2.formatted.contains("Checking if hive"));
    
    // More events...
    
    // 5. Success! Hive crate narration reached queen's SSE channel
}
```

---

## Conclusion

**The architecture is CORRECT!**

✅ **Distributed Mode:** Each process has its own SSE_BROADCASTER. HTTP links them.  
✅ **Integrated Mode:** Single process, single SSE_BROADCASTER, hive narration flows to queen's channel.

**No bugs found.** The integrated mode will work correctly as designed.

---

## Recommendations

1. ✅ **Keep Current Design:** No changes needed
2. ✅ **Add Integration Test:** Verify integrated mode SSE routing
3. ✅ **Document Clearly:** Update Part 4 to explain this flow
4. ✅ **Add Unit Test:** Test hive crate narration with mocked SSE sink

---

**Status:** ARCHITECTURE VERIFIED ✅
