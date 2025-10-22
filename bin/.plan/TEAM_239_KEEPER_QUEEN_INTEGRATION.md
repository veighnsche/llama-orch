# TEAM-239: Keeper ↔ Queen Integration Flow Inventory

**Date:** Oct 22, 2025  
**Components:** `rbee-keeper` ↔ `queen-rbee`  
**Complexity:** High  
**Status:** ✅ COMPLETE

// TEAM-239: Investigated

---

## 1. Happy Path Flows

### 1.1 HiveList Flow

**Complete Flow:**
```text
1. User: ./rbee hive list
2. Keeper: Parse CLI → Operation::HiveList
3. Keeper: ensure_queen_running() → Check /health
4. Keeper: POST /v1/jobs with {"operation": "hive_list"}
5. Queen: Create job_id, create SSE channel
6. Queen: Return {job_id, sse_url}
7. Keeper: GET /v1/jobs/{job_id}/stream
8. Queen: Execute hive_list operation
9. Queen: Emit narration events → SSE channel
10. Keeper: Stream SSE events to stdout
11. Queen: Send [DONE] marker
12. Keeper: Display "✅ Complete"
```

**Key Files:**
- Keeper: `src/main.rs` (lines 385-402), `src/job_client.rs` (lines 37-171)
- Queen: `src/http/jobs.rs` (lines 47-146), `src/job_router.rs` (lines 60-374)

### 1.2 HiveStart Flow

**Complete Flow:**
```text
1. User: ./rbee hive start --alias localhost
2. Keeper: Parse CLI → Operation::HiveStart { alias: "localhost" }
3. Keeper: ensure_queen_running()
   - Check GET /health
   - If not running: spawn queen-rbee daemon
   - Poll health until ready (30s timeout with countdown)
4. Keeper: POST /v1/jobs with {"operation": "hive_start", "alias": "localhost"}
5. Queen: Create job_id, create SSE channel
6. Queen: Return {job_id, sse_url}
7. Keeper: GET /v1/jobs/{job_id}/stream (with 30s timeout)
8. Queen: Execute hive_start operation
   - Load config from ~/.config/rbee/
   - Validate hive exists in hives.conf
   - Find rbee-hive binary (debug → release)
   - Spawn rbee-hive daemon (Stdio::null())
   - Poll health until ready
   - Fetch capabilities from hive
   - Update capabilities cache
9. Queen: Emit narration events → SSE channel
10. Keeper: Stream SSE events to stdout
11. Queen: Send [DONE] marker
12. Keeper: Display "✅ Complete"
```

**Narration Events (Example):**
```
[qn-router ] route_job      : Executing operation: hive_start
[hive-life ] start          : Starting hive: localhost
[hive-life ] spawn          : Spawning daemon: target/debug/rbee-hive with args: ["--port", "8081", ...]
[hive-life ] spawned        : Daemon spawned with PID: 12345
[hive-life ] health_poll    : Polling health: http://localhost:8081/health
[hive-life ] health_ok      : Hive is healthy
[hive-life ] caps_fetch     : Fetching capabilities from http://localhost:8081/capabilities
[hive-life ] caps_save      : Capabilities cached
[DONE]
```

### 1.3 HiveStop Flow

**Complete Flow:**
```text
1. User: ./rbee hive stop --alias localhost
2. Keeper: Parse CLI → Operation::HiveStop { alias: "localhost" }
3. Keeper: ensure_queen_running()
4. Keeper: POST /v1/jobs with {"operation": "hive_stop", "alias": "localhost"}
5. Queen: Create job_id, create SSE channel
6. Queen: Return {job_id, sse_url}
7. Keeper: GET /v1/jobs/{job_id}/stream
8. Queen: Execute hive_stop operation
   - Validate hive exists
   - Send SIGTERM to hive process
   - Wait 5s for graceful shutdown
   - Send SIGKILL if still running
9. Queen: Emit narration events → SSE channel
10. Keeper: Stream SSE events to stdout
11. Queen: Send [DONE] marker
12. Keeper: Display "✅ Complete"
```

---

## 2. Data Transformations

### 2.1 CLI → Operation Enum

**Keeper (main.rs:385-402):**
```rust
Commands::Hive { action } => {
    let operation = match action {
        HiveAction::Start { alias } => Operation::HiveStart { alias },
        HiveAction::Stop { alias } => Operation::HiveStop { alias },
        HiveAction::List => Operation::HiveList,
        // ... etc
    };
    submit_and_stream_job(&client, &queen_url, operation).await
}
```

**Transformation:** CLI args → Operation enum (type-safe)

### 2.2 Operation → JSON

**Keeper (job_client.rs:42-43):**
```rust
let job_payload = serde_json::to_value(&operation)
    .expect("Failed to serialize operation");
```

**JSON Format:**
```json
{
  "operation": "hive_start",
  "alias": "localhost"
}
```

### 2.3 JSON → Operation

**Queen (job_router.rs:116-117):**
```rust
let operation: Operation = serde_json::from_value(payload)
    .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;
```

**Transformation:** JSON → Operation enum (type-safe)

### 2.4 Narration → SSE Events

**Queen (narration-core/src/sse_sink.rs):**
```rust
pub fn send(fields: &NarrationFields) {
    if let Some(job_id) = &fields.job_id {
        if let Some(tx) = CHANNELS.lock().unwrap().get(job_id) {
            let event = NarrationEvent {
                formatted: format_narration(fields),
                // ... other fields
            };
            let _ = tx.try_send(event); // Drop if full
        }
    }
}
```

**SSE Format:**
```
data: [actor     ] action         : message
```

---

## 3. State Synchronization

### 3.1 Job State (Queen)

**Lifecycle:**
```text
1. POST /v1/jobs → JobState::Queued
2. GET /v1/jobs/{job_id}/stream → JobState::Running
3. Operation completes → JobState::Completed
4. [DONE] sent → Job removed from registry
```

**Storage:** In-memory HashMap in JobRegistry

### 3.2 SSE Channel State

**Lifecycle:**
```text
1. create_job() → create_job_channel(job_id, 1000)
2. Narration emitted → send() → channel.try_send()
3. GET /v1/jobs/{job_id}/stream → take_job_receiver()
4. Stream closes → remove_job_channel(job_id)
```

**Critical:** Channel created BEFORE execution starts (prevents race condition)

### 3.3 Queen Lifecycle State (Keeper)

**QueenHandle:**
```rust
pub struct QueenHandle {
    base_url: String,
    started_by_us: bool,
}
```

**Lifecycle:**
```text
1. ensure_queen_running() → Check /health
2. If not running → spawn daemon → QueenHandle { started_by_us: true }
3. If already running → QueenHandle { started_by_us: false }
4. After job → std::mem::forget(handle) (keep alive)
```

**Policy:** Queen stays alive for future tasks (no auto-shutdown)

---

## 4. Error Propagation

### 4.1 HTTP Errors

**Keeper → Queen:**
```rust
// Connection refused
let res = client.post(...).send().await?;
// → Err: "connection refused"
// → Keeper: "Failed to connect to queen"
```

**Queen → Keeper:**
```rust
// Operation failed
return Err(anyhow::anyhow!("Hive not found"));
// → HTTP 500
// → Keeper: Stream shows error narration
```

### 4.2 SSE Errors

**Channel Not Found:**
```rust
// Queen (jobs.rs:97-99)
let Some(mut sse_rx) = sse_rx_opt else {
    yield Ok(Event::default().data("ERROR: Job channel not found..."));
    return;
};
```

**Stream Closes Early:**
```rust
// Keeper (job_client.rs:121)
None => {
    // Sender dropped (job completed)
    if received_first_event {
        yield Ok(Event::default().data("[DONE]"));
    }
    break;
}
```

### 4.3 Timeout Errors

**Keeper Timeouts:**
```rust
// Job submission timeout (10s)
let submit_client = reqwest::Client::builder()
    .timeout(Duration::from_secs(10))
    .build()?;

// SSE connection timeout (10s)
let sse_client = reqwest::Client::builder()
    .timeout(Duration::from_secs(10))
    .build()?;

// SSE streaming timeout (30s)
TimeoutEnforcer::new(Duration::from_secs(30))
    .with_label("Streaming job results")
    .enforce(stream_future).await?;
```

**Queen Timeouts:**
```rust
// Hive start timeout (15s)
TimeoutEnforcer::new(Duration::from_secs(15))
    .with_job_id(&job_id)  // ← CRITICAL for SSE routing
    .enforce(hive_start_future).await?;
```

**Error Flow:**
```text
1. Operation times out in queen
2. TimeoutEnforcer emits error narration with job_id
3. Narration routes to SSE channel
4. Keeper receives timeout event
5. Keeper displays: "❌ Operation TIMED OUT after 15s"
```

### 4.4 Error Display

**Keeper (job_client.rs:116-132):**
```rust
// Track job failures
if data.contains("failed:") || (data.contains("Job") && data.contains("failed")) {
    job_failed = true;
}

// Show ❌ Failed for failures, ✅ Complete for successes
let narration = if job_failed {
    NARRATE.action("job_complete").human("❌ Failed")
} else {
    NARRATE.action("job_complete").human("✅ Complete")
};
```

---

## 5. Timeout Handling

### 5.1 Layered Timeouts

**Layer 1: HTTP Client (10s)**
- Prevents hanging on connection
- Applies to: POST /v1/jobs, GET /v1/jobs/{job_id}/stream

**Layer 2: SSE Streaming (30s)**
- Prevents hanging on slow operations
- Applies to: Entire SSE stream consumption

**Layer 3: Operation Execution (15s)**
- Prevents hanging operations in queen
- Applies to: Hive start, capabilities fetch, etc.

**Layer 4: Queen Startup (30s)**
- Prevents hanging on queen spawn
- Applies to: ensure_queen_running()

### 5.2 Timeout Propagation

**Example: Hive Start Timeout**
```text
1. Keeper: 30s timeout on SSE streaming
2. Queen: 15s timeout on hive start operation
3. Hive start hangs (e.g., health poll never succeeds)
4. After 15s: Queen timeout fires
5. Queen: Emit timeout narration with job_id
6. Narration → SSE channel → Keeper
7. Keeper: Display timeout error
8. After 30s: Keeper timeout fires (if queen never sent [DONE])
9. Keeper: Return error to user
```

---

## 6. Resource Cleanup

### 6.1 Normal Completion

**Flow:**
```text
1. Operation completes successfully
2. Queen: Emit [DONE] marker
3. Keeper: Receive [DONE], close stream
4. Queen: Sender dropped → receiver sees None
5. Queen: remove_job_channel(job_id)
6. Queen: Job removed from registry (optional)
```

### 6.2 Error Completion

**Flow:**
```text
1. Operation fails with error
2. Queen: Emit error narration
3. Queen: Emit [DONE] marker (still sent!)
4. Keeper: Receive error + [DONE], close stream
5. Queen: Cleanup (same as normal)
```

### 6.3 Client Disconnect

**Flow:**
```text
1. User presses Ctrl+C
2. Keeper: HTTP connection closed
3. Queen: Stream future dropped
4. Queen: Receiver dropped → sender fails
5. Queen: remove_job_channel(job_id) (eventually)
```

**Issue:** No explicit cleanup on disconnect (potential memory leak)

### 6.4 Timeout Expiration

**Flow:**
```text
1. Keeper: 30s timeout fires
2. Keeper: Close HTTP connection
3. Queen: Stream future dropped
4. Queen: Cleanup (same as disconnect)
```

---

## 7. Edge Cases

### 7.1 Queen Unreachable

**Scenario:** Queen not running, auto-start fails

**Flow:**
```text
1. Keeper: Check /health → connection refused
2. Keeper: Spawn queen-rbee daemon
3. Keeper: Poll /health for 30s
4. If timeout: Return error "Failed to start queen"
5. User sees: "❌ Failed to start queen after 30s"
```

**Handled:** Yes (with timeout)

### 7.2 SSE Stream Closes Early

**Scenario:** Queen crashes mid-operation

**Flow:**
```text
1. Keeper: Streaming SSE events
2. Queen: Crashes (process killed)
3. Keeper: HTTP connection closed
4. Keeper: Stream returns None
5. Keeper: No [DONE] received
6. Keeper: Return error "Stream closed unexpectedly"
```

**Handled:** Partially (no [DONE] detection)

### 7.3 Multiple Clients Same job_id

**Scenario:** Two clients try to stream same job

**Flow:**
```text
1. Client A: GET /v1/jobs/{job_id}/stream
2. Queen: take_job_receiver() → Some(rx)
3. Client B: GET /v1/jobs/{job_id}/stream
4. Queen: take_job_receiver() → None (already taken)
5. Client B: Receives "ERROR: Job channel not found"
```

**Handled:** Yes (take semantics prevent double-consumption)

### 7.4 Network Failures

**Scenario:** Network partition during operation

**Flow:**
```text
1. Keeper: POST /v1/jobs → Success
2. Keeper: GET /v1/jobs/{job_id}/stream → Success
3. Network partition occurs
4. Keeper: HTTP timeout (10s)
5. Keeper: Return error "Connection timeout"
```

**Handled:** Yes (HTTP client timeout)

### 7.5 Timeout Scenarios

**Scenario 1: Queen startup timeout**
- Handled: Yes (30s with countdown)

**Scenario 2: Job submission timeout**
- Handled: Yes (10s HTTP timeout)

**Scenario 3: SSE connection timeout**
- Handled: Yes (10s HTTP timeout)

**Scenario 4: SSE streaming timeout**
- Handled: Yes (30s TimeoutEnforcer)

**Scenario 5: Operation execution timeout**
- Handled: Yes (15s TimeoutEnforcer with job_id)

---

## 8. Critical Invariants

### 8.1 SSE Channel Lifecycle

**Invariant:** Channel created BEFORE execution starts

**Why:** Prevents race condition where narration emitted before channel exists

**Enforcement:** `create_job()` creates channel, `execute_job()` uses it

### 8.2 job_id Propagation

**Invariant:** All narration MUST include job_id for SSE routing

**Why:** Without job_id, events are dropped (fail-fast security)

**Enforcement:** All operation handlers receive job_id parameter

### 8.3 [DONE] Marker

**Invariant:** Every SSE stream MUST end with [DONE] marker

**Why:** Keeper uses [DONE] to detect completion

**Enforcement:** Queen sends [DONE] in all code paths (success, error, timeout)

### 8.4 Take Semantics

**Invariant:** SSE receiver can only be taken once per job

**Why:** Prevents multiple clients streaming same job

**Enforcement:** `take_job_receiver()` moves receiver out of registry

### 8.5 Queen Stays Alive

**Invariant:** Queen NEVER auto-shuts down after task

**Why:** Reduces startup latency for future tasks

**Enforcement:** `std::mem::forget(queen_handle)` prevents cleanup

---

## 9. Existing Test Coverage

### 9.1 Integration Tests

**E2E Tests:**
- `bin/test_happy_flow.sh` - Full happy path
- `bin/test_keeper_queen_sse.sh` - SSE streaming

**Coverage:**
- ✅ HiveList operation
- ✅ HiveStart operation
- ✅ SSE streaming
- ✅ Narration flow

### 9.2 Test Gaps

**Missing Tests:**
- ❌ Queen unreachable (auto-start)
- ❌ SSE stream closes early
- ❌ Multiple clients same job_id
- ❌ Network failures
- ❌ Timeout scenarios (all layers)
- ❌ Error propagation (HTTP → SSE → CLI)
- ❌ Resource cleanup (disconnect, timeout)
- ❌ Concurrent operations
- ❌ Queen crash mid-operation

---

## 10. Flow Checklist

- [x] All happy paths documented
- [x] All error paths documented
- [x] All state transitions documented
- [x] All cleanup flows documented
- [x] All edge cases documented
- [x] Test coverage gaps identified

---

**Handoff:** Ready for Phase 6 (test planning)  
**Next:** TEAM-240 (queen-hive integration)
