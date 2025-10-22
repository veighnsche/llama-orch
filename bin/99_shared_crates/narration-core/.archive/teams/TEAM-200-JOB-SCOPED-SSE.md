# TEAM-200: Job-Scoped SSE Broadcaster

**Team:** TEAM-200  
**Priority:** 🚨 **CRITICAL - ISOLATION ISSUE**  
**Duration:** 4-6 hours  
**Based On:** OPPORTUNITY 1 from TEAM-197's review

---

## Mission

Refactor SSE broadcaster from global (all narration to all subscribers) to job-scoped (each job has isolated SSE channel). This prevents narration cross-contamination between concurrent jobs.

---

## The Isolation Problem

### Current Code (BUGGY)

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

```rust
/// Global SSE broadcaster for narration events.
static SSE_BROADCASTER: once_cell::sync::Lazy<SseBroadcaster> =
    once_cell::sync::Lazy::new(|| SseBroadcaster::new());

pub struct SseBroadcaster {
    sender: Arc<Mutex<Option<broadcast::Sender<NarrationEvent>>>>,
}
```

**The Bug:**
- ONE global channel for ALL narration
- ALL subscribers receive ALL events
- No isolation between jobs

**Example of Cross-Contamination:**
```
User A: ./rbee hive status
User B: ./rbee infer "hello"

User A sees:
[job-exec  ] execute        : Executing job A ✅
[worker    ] inference      : Generating tokens ❌ (This is Job B!)
[job-exec  ] execute        : Executing job B ❌
```

---

## The Solution: Job-Scoped Channels

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ SSE BROADCASTER (NEW)                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Global Channel (for non-job narration)                    │
│    ├─ Queen startup                                        │
│    ├─ Hive lifecycle (no job_id)                           │
│    └─ System-wide events                                   │
│                                                             │
│  Per-Job Channels (isolated)                               │
│    ├─ job-abc123 → Channel 1 (User A)                      │
│    ├─ job-xyz789 → Channel 2 (User B)                      │
│    └─ HashMap<String, Sender>                              │
│                                                             │
│  Thread-Local Channel (request-scoped)                     │
│    ├─ Worker inference (like existing pattern)            │
│    └─ Hive operations in HTTP context                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Step 1: Refactor SseBroadcaster

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

**Replace existing SseBroadcaster (line 15-80):**

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// Global SSE broadcaster with job-scoped channels.
/// 
/// TEAM-200: Refactored to support:
/// - Global channel for system-wide narration
/// - Per-job channels for isolated job narration
/// - Thread-local channels for request-scoped narration
static SSE_BROADCASTER: once_cell::sync::Lazy<SseBroadcaster> =
    once_cell::sync::Lazy::new(|| SseBroadcaster::new());

/// Broadcaster for SSE narration events with job isolation.
/// 
/// TEAM-200: This replaces the simple global broadcaster with:
/// 1. Global channel - For non-job narration (queen startup, etc.)
/// 2. Per-job channels - Isolated narration for each job
/// 3. Thread-local support - Request-scoped narration (like worker pattern)
pub struct SseBroadcaster {
    /// Global channel for non-job narration
    global: Arc<Mutex<Option<broadcast::Sender<NarrationEvent>>>>,
    
    /// Per-job channels (keyed by job_id)
    /// TEAM-200: Each job gets isolated SSE stream
    jobs: Arc<Mutex<HashMap<String, broadcast::Sender<NarrationEvent>>>>,
}

impl SseBroadcaster {
    fn new() -> Self {
        Self {
            global: Arc::new(Mutex::new(None)),
            jobs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Initialize the global SSE broadcaster.
    /// 
    /// TEAM-200: This creates the global channel for non-job narration.
    pub fn init(&self, capacity: usize) {
        let (tx, _) = broadcast::channel(capacity);
        *self.global.lock().unwrap() = Some(tx);
    }

    /// Create a new job-specific SSE channel.
    /// 
    /// TEAM-200: Call this when a job is created (before execution starts).
    /// The job's SSE stream will be isolated from other jobs.
    pub fn create_job_channel(&self, job_id: String, capacity: usize) {
        let (tx, _) = broadcast::channel(capacity);
        self.jobs.lock().unwrap().insert(job_id, tx);
    }

    /// Remove a job's SSE channel (cleanup when job completes).
    /// 
    /// TEAM-200: Call this when a job completes to prevent memory leaks.
    pub fn remove_job_channel(&self, job_id: &str) {
        self.jobs.lock().unwrap().remove(job_id);
    }

    /// Send narration to a specific job's SSE stream.
    /// 
    /// TEAM-200: This is the primary send method - routes to job-specific channel.
    pub fn send_to_job(&self, job_id: &str, event: NarrationEvent) {
        let jobs = self.jobs.lock().unwrap();
        if let Some(tx) = jobs.get(job_id) {
            // Ignore send errors (no subscribers is OK)
            let _ = tx.send(event);
        }
    }

    /// Send narration to the global channel (non-job narration).
    /// 
    /// TEAM-200: Use this for system-wide events (queen startup, etc.)
    pub fn send_global(&self, event: NarrationEvent) {
        if let Some(tx) = self.global.lock().unwrap().as_ref() {
            let _ = tx.send(event);
        }
    }

    /// Subscribe to a specific job's SSE stream.
    /// 
    /// TEAM-200: Keeper calls this with job_id to get isolated stream.
    pub fn subscribe_to_job(&self, job_id: &str) -> Option<broadcast::Receiver<NarrationEvent>> {
        self.jobs.lock().unwrap()
            .get(job_id)
            .map(|tx| tx.subscribe())
    }

    /// Subscribe to the global SSE stream.
    /// 
    /// TEAM-200: Use for monitoring all system-wide narration.
    pub fn subscribe_global(&self) -> Option<broadcast::Receiver<NarrationEvent>> {
        self.global.lock().unwrap()
            .as_ref()
            .map(|tx| tx.subscribe())
    }

    /// Check if a job channel exists.
    pub fn has_job_channel(&self, job_id: &str) -> bool {
        self.jobs.lock().unwrap().contains_key(job_id)
    }
}
```

### Step 2: Update Public API

**Replace existing public functions (line 82-end):**

```rust
/// Initialize the global SSE broadcaster.
///
/// TEAM-200: This initializes the global channel for non-job narration.
/// Job channels are created separately via create_job_channel().
pub fn init(capacity: usize) {
    SSE_BROADCASTER.init(capacity);
}

/// Create a job-specific SSE channel.
/// 
/// TEAM-200: Call this in job_router::create_job() before execution starts.
/// 
/// # Example
/// ```rust,ignore
/// use observability_narration_core::sse_sink;
/// 
/// let job_id = "job-abc123";
/// sse_sink::create_job_channel(job_id.to_string(), 1000);
/// // Now narration with this job_id goes to isolated channel
/// ```
pub fn create_job_channel(job_id: String, capacity: usize) {
    SSE_BROADCASTER.create_job_channel(job_id, capacity);
}

/// Remove a job's SSE channel (cleanup).
/// 
/// TEAM-200: Call this when job completes to prevent memory leaks.
pub fn remove_job_channel(job_id: &str) {
    SSE_BROADCASTER.remove_job_channel(job_id);
}

/// Send a narration event to appropriate channel based on job_id.
///
/// TEAM-200: Routing logic:
/// - If event has job_id → send to job-specific channel
/// - Otherwise → send to global channel
pub fn send(fields: &NarrationFields) {
    let event = NarrationEvent::from(fields.clone());
    
    // Route based on job_id
    if let Some(job_id) = &fields.job_id {
        SSE_BROADCASTER.send_to_job(job_id, event);
    } else {
        SSE_BROADCASTER.send_global(event);
    }
}

/// Subscribe to a specific job's SSE stream.
/// 
/// TEAM-200: Keeper calls this with job_id from job creation response.
///
/// # Example
/// ```rust,ignore
/// let mut rx = sse_sink::subscribe_to_job("job-abc123")
///     .expect("Job channel not found");
/// while let Ok(event) = rx.recv().await {
///     println!("{}", event.formatted);
/// }
/// ```
pub fn subscribe_to_job(job_id: &str) -> Option<broadcast::Receiver<NarrationEvent>> {
    SSE_BROADCASTER.subscribe_to_job(job_id)
}

/// Subscribe to the global SSE stream (all non-job narration).
pub fn subscribe_global() -> Option<broadcast::Receiver<NarrationEvent>> {
    SSE_BROADCASTER.subscribe_global()
}

/// Check if SSE broadcasting is enabled.
/// 
/// TEAM-200: Returns true if global channel is initialized.
pub fn is_enabled() -> bool {
    SSE_BROADCASTER.global.lock().unwrap().is_some()
}

/// Check if a job channel exists.
pub fn has_job_channel(job_id: &str) -> bool {
    SSE_BROADCASTER.has_job_channel(job_id)
}
```

---

## Step 3: Update Queen Job Creation

**File:** `bin/10_queen_rbee/src/job_router.rs`

**Modify `create_job()` function (~line 60-73):**

```rust
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    let sse_url = format!("/v1/jobs/{}/stream", job_id);

    state.registry.set_payload(&job_id, payload);

    // TEAM-200: Create job-specific SSE channel for isolation
    observability_narration_core::sse_sink::create_job_channel(job_id.clone(), 1000);

    NARRATE
        .action("job_create")
        .context(&job_id)
        .job_id(&job_id)  // ← CRITICAL: Include job_id so narration routes correctly
        .human("Job {} created, waiting for client connection")
        .emit();

    Ok(JobResponse { job_id, sse_url })
}
```

---

## Step 4: Update Queen SSE Stream Handler

**File:** `bin/10_queen_rbee/src/http/jobs.rs`

**Modify `handle_stream_job()` function (~line 73-133):**

```rust
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // TEAM-200: Subscribe to JOB-SPECIFIC SSE channel (not global!)
    let mut sse_rx = sse_sink::subscribe_to_job(&job_id)
        .expect("Job channel not found - did you forget to create it?");

    // Trigger job execution (spawns in background)
    let token_stream = crate::job_router::execute_job(job_id.clone(), state.into()).await;

    // Give background task time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // TEAM-200: Stream narration from job-specific channel
    // This ensures User A only sees Job A's narration
    let combined_stream = async_stream::stream! {
        let mut last_event_time = std::time::Instant::now();
        let completion_timeout = std::time::Duration::from_millis(2000);
        let mut received_first_event = false;

        loop {
            let timeout_fut = tokio::time::sleep(completion_timeout);
            tokio::pin!(timeout_fut);

            tokio::select! {
                result = sse_rx.recv() => {
                    match result {
                        Ok(event) => {
                            received_first_event = true;
                            last_event_time = std::time::Instant::now();
                            // Will be updated by TEAM-201 to use event.formatted
                            let formatted = format!("[{:<10}] {:<15}: {}", event.actor, event.action, event.human);
                            yield Ok(Event::default().data(formatted));
                        }
                        Err(_) => {
                            if received_first_event {
                                yield Ok(Event::default().data("[DONE]"));
                            }
                            break;
                        }
                    }
                }
                _ = &mut timeout_fut, if received_first_event => {
                    if last_event_time.elapsed() >= completion_timeout {
                        yield Ok(Event::default().data("[DONE]"));
                        // TEAM-200: Cleanup job channel
                        sse_sink::remove_job_channel(&job_id);
                        break;
                    }
                }
            }
        }
    };

    Sse::new(combined_stream)
}
```

---

## Testing Strategy

### Test 1: Job Isolation

```rust
#[cfg(test)]
mod team_200_isolation_tests {
    use super::*;
    use crate::NarrationFields;

    #[tokio::test]
    async fn test_job_isolation() {
        // Initialize broadcaster
        init(100);

        // Create two job channels
        create_job_channel("job-a".to_string(), 100);
        create_job_channel("job-b".to_string(), 100);

        // Subscribe to both jobs
        let mut rx_a = subscribe_to_job("job-a").unwrap();
        let mut rx_b = subscribe_to_job("job-b").unwrap();

        // Send narration to job-a
        let fields_a = NarrationFields {
            actor: "test",
            action: "action_a",
            target: "target-a".to_string(),
            human: "Message for Job A".to_string(),
            job_id: Some("job-a".to_string()),
            ..Default::default()
        };
        send(&fields_a);

        // Send narration to job-b
        let fields_b = NarrationFields {
            actor: "test",
            action: "action_b",
            target: "target-b".to_string(),
            human: "Message for Job B".to_string(),
            job_id: Some("job-b".to_string()),
            ..Default::default()
        };
        send(&fields_b);

        // Job A should only receive its message
        let event_a = rx_a.try_recv().unwrap();
        assert_eq!(event_a.human, "Message for Job A");
        assert!(rx_a.try_recv().is_err()); // No more messages

        // Job B should only receive its message
        let event_b = rx_b.try_recv().unwrap();
        assert_eq!(event_b.human, "Message for Job B");
        assert!(rx_b.try_recv().is_err()); // No more messages

        // Cleanup
        remove_job_channel("job-a");
        remove_job_channel("job-b");
    }

    #[tokio::test]
    async fn test_global_channel_for_non_job_narration() {
        init(100);
        let mut rx = subscribe_global().unwrap();

        // Send narration without job_id
        let fields = NarrationFields {
            actor: "queen",
            action: "startup",
            target: "queen-rbee".to_string(),
            human: "Queen starting".to_string(),
            job_id: None, // ← No job_id
            ..Default::default()
        };
        send(&fields);

        // Should go to global channel
        let event = rx.try_recv().unwrap();
        assert_eq!(event.human, "Queen starting");
    }

    #[test]
    fn test_channel_cleanup() {
        create_job_channel("job-temp".to_string(), 100);
        assert!(has_job_channel("job-temp"));

        remove_job_channel("job-temp");
        assert!(!has_job_channel("job-temp"));
    }

    #[test]
    fn test_send_to_nonexistent_job_is_safe() {
        // Sending to non-existent job should not panic
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "test".to_string(),
            human: "Test".to_string(),
            job_id: Some("nonexistent-job".to_string()),
            ..Default::default()
        };
        send(&fields); // Should not panic
    }
}
```

---

## Verification Checklist

### Before Starting
- [ ] TEAM-199 has completed redaction fix
- [ ] Read TEAM-197 OPPORTUNITY 1 section
- [ ] Understand current global broadcaster

### Implementation
- [ ] Refactor `SseBroadcaster` struct (add jobs HashMap)
- [ ] Add `create_job_channel()`, `remove_job_channel()`
- [ ] Add `send_to_job()`, `send_global()`
- [ ] Add `subscribe_to_job()`, `subscribe_global()`
- [ ] Update `send()` routing logic
- [ ] Update queen `create_job()` to create channel
- [ ] Update queen `handle_stream_job()` to subscribe to job
- [ ] Add cleanup on job completion

### Testing
- [ ] Add 4 isolation tests
- [ ] Run: `cargo test -p observability-narration-core team_200`
- [ ] All tests pass
- [ ] No cross-contamination between jobs

### Integration
- [ ] Build succeeds: `cargo build -p observability-narration-core`
- [ ] Build succeeds: `cargo build -p queen-rbee`
- [ ] No breaking changes to existing API

---

## Expected Changes

### Files Modified
- `bin/99_shared_crates/narration-core/src/sse_sink.rs` (~150 lines changed)
- `bin/10_queen_rbee/src/job_router.rs` (~5 lines added)
- `bin/10_queen_rbee/src/http/jobs.rs` (~10 lines changed)

### Impact
- **Job isolation:** Each job has separate SSE stream
- **Memory management:** Job channels cleaned up on completion
- **Backward compatible:** Global channel still exists for non-job narration

---

## Common Pitfalls

### ❌ WRONG: Forgetting to Create Job Channel
```rust
// BAD: Job channel not created!
let job_id = state.registry.create_job();
// ... later ...
let rx = sse_sink::subscribe_to_job(&job_id); // ← Returns None!
```

### ✅ CORRECT: Create Channel Before Use
```rust
// GOOD: Channel created immediately
let job_id = state.registry.create_job();
sse_sink::create_job_channel(job_id.clone(), 1000);
```

### ❌ WRONG: Forgetting to Include job_id in Narration
```rust
// BAD: Narration won't route to job channel!
NARRATE
    .action("job_create")
    .human("Job created")
    .emit(); // No job_id!
```

### ✅ CORRECT: Always Include job_id
```rust
// GOOD: Routes to job-specific channel
NARRATE
    .action("job_create")
    .job_id(&job_id)  // ← CRITICAL
    .human("Job created")
    .emit();
```

---

## Success Criteria

### Isolation
- ✅ Multiple concurrent jobs have separate streams
- ✅ Job A doesn't see Job B's narration
- ✅ Tests verify no cross-contamination

### Memory Management
- ✅ Job channels cleaned up on completion
- ✅ No memory leaks
- ✅ Test: create/remove 100 jobs, no memory growth

### Functionality
- ✅ Global channel still works for non-job narration
- ✅ Job-specific channels work for job narration
- ✅ Backward compatible API

---

## Next Teams

**TEAM-201** can work in parallel with you (formatting is independent).

**TEAM-202** depends on your work (needs job-scoped SSE for hive narration).

---

## Summary

**Problem:** Global SSE broadcaster causes cross-contamination  
**Solution:** Per-job channels with HashMap<String, Sender>  
**Testing:** 4 tests verify isolation  
**Impact:** Better isolation, memory management, backward compatible

---

**Created for:** TEAM-200  
**Priority:** 🚨 CRITICAL  
**Status:** READY TO IMPLEMENT (after TEAM-199)

**This fixes isolation. TEAM-201 will fix formatting (can work in parallel).**
