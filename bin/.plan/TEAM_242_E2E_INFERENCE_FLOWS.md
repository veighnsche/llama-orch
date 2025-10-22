# TEAM-242: End-to-End Inference Flow Inventory

**Date:** Oct 22, 2025  
**Components:** Full system (keeper → queen → hive → worker)  
**Complexity:** Very High  
**Status:** ✅ COMPLETE (DOCUMENTED - NOT YET IMPLEMENTED)

// TEAM-242: Investigated

---

## CRITICAL NOTE

**Inference flow is NOT YET IMPLEMENTED.**

This document describes the INTENDED end-to-end architecture based on:
- Existing patterns (hive operations flow)
- TODO markers in codebase
- Dual-call pattern design
- SSE streaming infrastructure

**Status:**
- ✅ Infrastructure exists (job-registry, SSE, narration)
- ✅ Operation enum includes Infer
- ❌ Worker spawn NOT implemented
- ❌ Model loading NOT implemented
- ❌ Inference execution NOT implemented
- ❌ Token streaming NOT implemented

---

## 1. Happy Path Flow

### 1.1 Complete Inference Flow (INTENDED)

**Step-by-Step:**
```text
1. USER: ./rbee infer --hive-id localhost --model llama2 --prompt "Hello world"

2. KEEPER (rbee-keeper):
   - Parse CLI arguments
   - Create Operation::Infer {
       hive_id: "localhost",
       model: "llama2",
       prompt: "Hello world",
       max_tokens: 100,
       temperature: 0.7,
       stream: true,
     }
   - ensure_queen_running() (30s timeout)
   - POST /v1/jobs to queen with operation payload
   - Receive {job_id: "job-abc123", sse_url: "/v1/jobs/job-abc123/stream"}

3. KEEPER:
   - GET /v1/jobs/job-abc123/stream (30s timeout)
   - Wait for SSE events

4. QUEEN (queen-rbee):
   - Receive POST /v1/jobs
   - Create job_id, create SSE channel
   - Store payload in job registry
   - Return {job_id, sse_url}

5. QUEEN:
   - Receive GET /v1/jobs/job-abc123/stream
   - Take SSE receiver from registry
   - Trigger job execution in background
   - Parse Operation::Infer from payload

6. QUEEN:
   - Validate hive exists in config
   - Get hive URL: http://localhost:8081
   - Forward operation to hive: POST http://localhost:8081/v1/jobs
   - Receive {job_id: "job-xyz789", sse_url: "/v1/jobs/job-xyz789/stream"}

7. QUEEN:
   - GET http://localhost:8081/v1/jobs/job-xyz789/stream
   - Proxy SSE events from hive to keeper

8. HIVE (rbee-hive):
   - Receive POST /v1/jobs
   - Create job_id, create SSE channel
   - Store payload in job registry
   - Return {job_id, sse_url}

9. HIVE:
   - Receive GET /v1/jobs/job-xyz789/stream
   - Parse Operation::Infer from payload
   - Lookup workers with model "llama2"
   - Select worker (round-robin / least loaded)
   - Forward to worker: POST http://localhost:9001/v1/inference

10. WORKER (llm-worker-rbee):
    - Receive POST /v1/inference
    - Validate model loaded (if not, load it)
    - Initialize inference context
    - Tokenize prompt: "Hello world"
    - Start token generation loop

11. WORKER:
    - Generate token: "Hello"
    - Send via SSE: data: {"token": "Hello"}
    - Generate token: " there"
    - Send via SSE: data: {"token": " there"}
    - Generate token: "!"
    - Send via SSE: data: {"token": "!"}
    - Generation complete
    - Send: data: [DONE]

12. HIVE:
    - Receive tokens from worker
    - Emit narration for each token (optional)
    - Forward tokens to queen via SSE

13. QUEEN:
    - Receive tokens from hive
    - Forward tokens to keeper via SSE

14. KEEPER:
    - Receive tokens from queen
    - Display streaming output:
      "Hello there!"
    - Receive [DONE]
    - Display "✅ Complete"
    - Close stream

15. CLEANUP:
    - Worker: Close inference context
    - Hive: Remove job from registry, remove SSE channel
    - Queen: Remove job from registry, remove SSE channel
    - Keeper: Exit
```

**Total Latency:**
- Queen startup: 0-5s (if not running)
- Hive startup: 0-3s (if not running)
- Worker startup: 0-10s (if not spawned)
- Model loading: 5-60s (if not loaded)
- First token: 0.5-2s
- Subsequent tokens: 20-100ms each

---

## 2. Narration Flow

### 2.1 Narration Events Through System

**Component-by-Component:**

**KEEPER (Client-Side):**
- No job_id (client-side narration)
- Goes to stderr only
- Example: "📋 Job job-abc123 submitted"

**QUEEN (Server-Side):**
- Has job_id (server-side narration)
- Goes to SSE channel + stderr
- Example: "[qn-router ] route_job      : Executing operation: infer"

**HIVE (Server-Side):**
- Has job_id (forwarded from queen)
- Goes to SSE channel + stderr
- Example: "[hive      ] infer_start    : Starting inference on worker-123"

**WORKER (Server-Side):**
- Has job_id (forwarded from hive)
- Goes to SSE channel + stderr
- Example: "[worker    ] token_gen      : Generated token: 'Hello'"

### 2.2 Job ID Enables SSE Routing

**Critical Flow:**
```text
1. Queen: create_job() → job_id = "job-abc123"
2. Queen: create_job_channel(job_id, 1000)
3. Queen: Execute operation with job_id parameter
4. Queen: All narration includes .job_id(&job_id)
5. Narration: Check job_id, route to SSE channel
6. SSE channel: Buffer events
7. Keeper: GET /v1/jobs/job-abc123/stream
8. Queen: take_job_receiver(job_id)
9. Queen: Stream events to keeper
```

**Without job_id:**
- Events go to stderr only
- Never reach SSE stream
- Keeper sees nothing

### 2.3 Event Flow to Client

**Path:**
```text
Worker narration
  ↓ (with job_id)
Worker SSE sink
  ↓
Hive SSE stream
  ↓ (proxy)
Queen SSE stream
  ↓ (proxy)
Keeper HTTP client
  ↓
Stdout
```

**Proxying:**
- Hive proxies worker SSE to queen
- Queen proxies hive SSE to keeper
- Each hop maintains job_id

### 2.4 Stdout vs SSE

**Stdout (Client-Side):**
- Keeper narration (no job_id)
- Direct to terminal
- Example: "✅ Complete"

**SSE (Server-Side):**
- Queen/Hive/Worker narration (with job_id)
- Routed through SSE channels
- Example: "[qn-router ] route_job : Executing operation: infer"

**Both:**
- All narration also goes to stderr (for logging)
- SSE is additional routing, not replacement

---

## 3. Error Scenarios

### 3.1 Hive Not Running

**Flow:**
```text
1. Keeper: POST /v1/jobs to queen
2. Queen: Forward to hive
3. Queen: HTTP connection refused
4. Queen: Emit error narration with job_id
5. Narration: "❌ Hive not reachable: localhost"
6. Narration → SSE → Keeper
7. Keeper: Display error
8. Queen: Send [DONE]
9. Keeper: Display "❌ Failed"
```

**Handled:** Yes (with narration)

### 3.2 Worker Not Available

**Flow:**
```text
1. Hive: Lookup workers with model "llama2"
2. Hive: No workers found
3. Hive: Emit error narration with job_id
4. Narration: "❌ No workers available for model: llama2"
5. Narration → SSE → Queen → Keeper
6. Keeper: Display error
7. Hive: Send [DONE]
```

**Handled:** Yes (with narration)

### 3.3 Model Not Found

**Flow:**
```text
1. Hive: Forward to worker
2. Worker: Model "llama2" not found
3. Worker: Return error
4. Hive: Emit error narration with job_id
5. Narration: "❌ Model not found: llama2"
6. Narration → SSE → Queen → Keeper
7. Keeper: Display error
```

**Handled:** Yes (with narration)

### 3.4 Model Load Failure

**Flow:**
```text
1. Worker: Attempt to load model
2. Worker: Load fails (corrupted file, insufficient VRAM)
3. Worker: Return error
4. Hive: Emit error narration with job_id
5. Narration: "❌ Model load failed: <error>"
6. Narration → SSE → Queen → Keeper
7. Keeper: Display error
```

**Handled:** Yes (with narration)

### 3.5 VRAM Exhaustion

**Flow:**
```text
1. Worker: Allocate VRAM for model
2. Worker: VRAM exhausted
3. Worker: Return error
4. Hive: Emit error narration with job_id
5. Narration: "❌ VRAM exhausted"
6. Narration → SSE → Queen → Keeper
7. Keeper: Display error
```

**Handled:** Yes (with narration)

### 3.6 Network Timeout

**Flow:**
```text
1. Keeper: GET /v1/jobs/job-abc123/stream (30s timeout)
2. Queen: Forward to hive (no timeout on queen side)
3. Hive: Forward to worker (no timeout on hive side)
4. Worker: Hangs (infinite loop, deadlock)
5. After 30s: Keeper timeout fires
6. Keeper: Close connection
7. Keeper: Display "❌ Operation timed out after 30s"
```

**Issue:** No timeout on queen/hive side (only keeper)

### 3.7 Client Disconnect

**Flow:**
```text
1. User: Press Ctrl+C
2. Keeper: Close HTTP connection
3. Queen: Stream future dropped
4. Queen: Receiver dropped
5. Hive: Sender fails (no receiver)
6. Hive: Stop forwarding tokens
7. Worker: Continue generating (orphaned)
```

**Issue:** Worker doesn't know client disconnected

---

## 4. State Management

### 4.1 Job State (Queen)

**Lifecycle:**
```text
1. POST /v1/jobs → JobState::Queued
2. GET /v1/jobs/{job_id}/stream → JobState::Running
3. Forward to hive → JobState::Running (still)
4. Hive completes → JobState::Completed
5. [DONE] sent → Job removed from registry
```

**Storage:** In-memory HashMap

### 4.2 Hive State (Queen + Hive)

**Queen Side:**
```rust
pub struct HiveState {
    pub hive_id: String,
    pub last_seen: i64,
    pub workers: Vec<WorkerState>,
}
```

**Hive Side:**
- No state about queen
- Only knows worker states

### 4.3 Worker State (Hive + Worker)

**Hive Side:**
```rust
pub struct WorkerState {
    pub worker_id: String,
    pub model_id: String,
    pub health_status: HealthStatus,
    pub available_slots: usize,
}
```

**Worker Side:**
- No state about hive
- Only knows own inference state

### 4.4 Model State (Worker)

**State:**
```rust
pub struct ModelState {
    pub model_id: String,
    pub loaded: bool,
    pub vram_allocated: u64,
    pub context: Option<LlamaContext>,
}
```

**Lifecycle:**
```text
1. Worker starts → Model not loaded
2. First inference → Load model
3. Model loaded → Ready for inference
4. Worker shutdown → Unload model
```

---

## 5. Timeout Handling

### 5.1 Client Timeout (Keeper)

**Timeout:** 30 seconds

**Applies To:**
- SSE streaming (entire operation)

**Behavior:**
- Close HTTP connection
- Display "❌ Operation timed out"
- No cleanup on server side

### 5.2 Queen Timeout

**No Timeout:** Queen doesn't timeout operations

**Why:**
- Queen is just a proxy
- Timeout handled by keeper

**Issue:** If keeper disconnects, queen keeps running

### 5.3 Hive Timeout

**No Timeout:** Hive doesn't timeout operations

**Why:**
- Hive is just a proxy
- Timeout handled by keeper

**Issue:** If queen disconnects, hive keeps running

### 5.4 Worker Timeout

**No Timeout:** Worker doesn't timeout inference

**Why:**
- Inference can take arbitrary time
- Timeout handled by keeper

**Issue:** If hive disconnects, worker keeps generating

### 5.5 Timeout Propagation

**Current:**
```text
Keeper (30s) → Queen (none) → Hive (none) → Worker (none)
```

**Issue:** Only keeper has timeout, server keeps running

**Better:**
```text
Keeper (30s) → Queen (25s) → Hive (20s) → Worker (15s)
```

**Benefit:** Each layer times out before parent, clean error propagation

---

## 6. Resource Cleanup

### 6.1 Normal Completion

**Flow:**
```text
1. Worker: Generation complete
2. Worker: Send [DONE]
3. Hive: Forward [DONE]
4. Hive: Remove job from registry
5. Hive: Remove SSE channel
6. Queen: Forward [DONE]
7. Queen: Remove job from registry
8. Queen: Remove SSE channel
9. Keeper: Close stream
10. Worker: Free inference context (if one-shot)
```

### 6.2 Error Completion

**Flow:**
```text
1. Worker: Inference fails
2. Worker: Send error token
3. Worker: Send [DONE]
4. Hive: Forward error + [DONE]
5. Hive: Cleanup (same as normal)
6. Queen: Forward error + [DONE]
7. Queen: Cleanup (same as normal)
8. Keeper: Display error
9. Keeper: Close stream
```

### 6.3 Client Disconnect

**Flow:**
```text
1. Keeper: Close HTTP connection
2. Queen: Stream future dropped
3. Queen: Receiver dropped
4. Queen: Sender fails
5. Hive: Sender fails (no receiver)
6. Hive: Stop forwarding
7. Worker: Continue generating (orphaned)
```

**Issue:** No cleanup signal to worker

**Better:**
```text
1. Keeper: Close connection
2. Queen: Detect disconnect
3. Queen: Send cancel signal to hive
4. Hive: Send cancel signal to worker
5. Worker: Stop generation
6. Worker: Free resources
```

### 6.4 Timeout Expiration

**Flow:**
```text
1. Keeper: 30s timeout fires
2. Keeper: Close connection
3. Queen/Hive/Worker: Continue running (orphaned)
```

**Issue:** Same as client disconnect

---

## 7. Edge Cases

### 7.1 Multiple Concurrent Requests

**Scenario:** User sends 10 inference requests simultaneously

**Flow:**
```text
1. Keeper: Send 10 POST /v1/jobs
2. Queen: Create 10 jobs
3. Queen: Forward to hive
4. Hive: Create 10 jobs
5. Hive: Route to workers (round-robin)
6. Workers: Process in parallel (if slots available)
7. Workers: Stream tokens back
8. Keeper: Display 10 streams in parallel
```

**Handled:** Yes (job-registry supports concurrent jobs)

### 7.2 Worker Crash Mid-Inference

**Scenario:** Worker crashes while generating tokens

**Flow:**
```text
1. Worker: Generating tokens
2. Worker: Crashes (segfault, OOM)
3. Hive: HTTP connection closed
4. Hive: Emit error narration
5. Narration: "❌ Worker crashed"
6. Narration → SSE → Queen → Keeper
7. Keeper: Display error
8. Hive: Send [DONE]
```

**Handled:** Partially (error detection, but no retry)

### 7.3 Hive Crash Mid-Operation

**Scenario:** Hive crashes while proxying tokens

**Flow:**
```text
1. Hive: Proxying tokens
2. Hive: Crashes (killed, OOM)
3. Queen: HTTP connection closed
4. Queen: Emit error narration
5. Narration: "❌ Hive unreachable"
6. Narration → SSE → Keeper
7. Keeper: Display error
8. Queen: Send [DONE]
```

**Handled:** Partially (error detection, but no retry)

### 7.4 Queen Restart

**Scenario:** Queen restarts while operations in progress

**Flow:**
```text
1. Queen: Crashes or restarts
2. Keeper: HTTP connection closed
3. Keeper: Display "❌ Connection lost"
4. Hive: Continue running (orphaned)
5. Worker: Continue running (orphaned)
```

**Issue:** No state persistence, all jobs lost

**Better:**
- Persist job state to disk
- Reconnect to hives on restart
- Resume operations

### 7.5 Network Partitions

**Scenario:** Network partition between queen and hive

**Flow:**
```text
1. Queen: Forward to hive
2. Network: Partition occurs
3. Queen: HTTP timeout (if configured)
4. Queen: Emit error narration
5. Narration: "❌ Hive unreachable"
6. Keeper: Display error
```

**Handled:** Partially (depends on HTTP timeout)

---

## 8. Critical Invariants

### 8.1 job_id Propagation

**Invariant:** job_id MUST propagate through all layers

**Why:** Without job_id, narration doesn't reach SSE stream

**Enforcement:** All operation handlers receive job_id parameter

### 8.2 [DONE] Marker

**Invariant:** Every SSE stream MUST end with [DONE]

**Why:** Keeper uses [DONE] to detect completion

**Enforcement:** All components send [DONE] in all code paths

### 8.3 Token Ordering

**Invariant:** Tokens MUST arrive in generation order

**Why:** Out-of-order tokens produce gibberish

**Enforcement:** Single-threaded generation, sequential streaming

### 8.4 Model Consistency

**Invariant:** Worker MUST use requested model

**Why:** Wrong model produces wrong output

**Enforcement:** Validate model_id on inference start

---

## 9. Existing Test Coverage

### 9.1 Integration Tests

**Coverage:**
- ✅ Dual-call pattern (job creation + streaming)
- ✅ SSE streaming infrastructure
- ✅ Narration flow (keeper ↔ queen)
- ❌ Inference flow (NOT IMPLEMENTED)

### 9.2 Test Gaps

**Missing Tests (ALL - NOT IMPLEMENTED):**
- ❌ End-to-end inference (keeper → queen → hive → worker)
- ❌ Token streaming
- ❌ Model loading
- ❌ Worker selection
- ❌ Concurrent inference requests
- ❌ Worker crash mid-inference
- ❌ Hive crash mid-operation
- ❌ Queen restart
- ❌ Network partitions
- ❌ Timeout propagation
- ❌ Resource cleanup (all scenarios)
- ❌ VRAM tracking
- ❌ Slot management

---

## 10. Flow Checklist

- [x] All happy paths documented (INTENDED)
- [x] All error paths documented (INTENDED)
- [x] All state transitions documented (INTENDED)
- [x] All cleanup flows documented (INTENDED)
- [x] All edge cases documented (INTENDED)
- [x] Test coverage gaps identified (ALL MISSING)

---

**Handoff:** Ready for Phase 6 (test planning)  
**Phase 5 Complete:** All 4 integration flows documented
