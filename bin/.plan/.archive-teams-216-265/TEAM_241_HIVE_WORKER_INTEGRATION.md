# TEAM-241: Hive ↔ Worker Integration Flow Inventory

**Date:** Oct 22, 2025  
**Components:** `rbee-hive` ↔ `llm-worker-rbee`  
**Complexity:** High  
**Status:** ✅ COMPLETE (DOCUMENTED - NOT YET IMPLEMENTED)

// TEAM-241: Investigated

---

## CRITICAL NOTE

**Most hive-worker integration is NOT YET IMPLEMENTED.**

This document describes the INTENDED architecture based on:
- Existing patterns (keeper ↔ queen, queen ↔ hive)
- TODO markers in codebase
- Shared crate designs (heartbeat, job-registry)

**Status:**
- ✅ Heartbeat infrastructure exists (shared crate)
- ✅ Worker state provider pattern exists
- ❌ Worker spawn NOT implemented
- ❌ Worker registry NOT implemented
- ❌ Model provisioning NOT implemented
- ❌ Inference coordination NOT implemented

---

## 1. Worker Lifecycle (INTENDED)

### 1.1 Worker Spawn Flow

**Complete Flow (NOT YET IMPLEMENTED):**
```text
1. Keeper: ./rbee worker spawn --hive-id localhost --model llama2 --worker cpu --device 0
2. Keeper: POST /v1/jobs to queen
3. Queen: Parse Operation::WorkerSpawn
4. Queen: Forward to hive: POST http://localhost:8081/v1/jobs
5. Hive: Create job, create SSE channel
6. Hive: Parse Operation::WorkerSpawn
7. Hive: Validate model exists (or trigger download)
8. Hive: Find llm-worker-rbee binary
9. Hive: Spawn worker process:
   - Args: ["--model", "llama2", "--device", "cpu:0", "--port", "9001"]
10. Worker: Start, load model
11. Worker: Register with hive: POST /v1/workers/register
12. Hive: Add worker to registry
13. Hive: Emit narration → SSE → Queen → Keeper
14. Worker: Start heartbeat task (30s interval)
15. Hive: Return success
```

**Key Files (EXPECTED):**
- Hive: `src/worker_manager.rs` (NOT YET CREATED)
- Worker: `src/main.rs` (EXISTS but minimal)

### 1.2 Worker Registration

**Flow (NOT YET IMPLEMENTED):**
```text
1. Worker: POST http://localhost:8081/v1/workers/register
2. Payload: {
     worker_id: "worker-123",
     model_id: "llama2",
     device: "cpu:0",
     port: 9001,
     capabilities: { max_batch_size: 1, ... }
   }
3. Hive: Validate worker_id unique
4. Hive: Add to worker registry
5. Hive: Return acknowledgement
```

**Why Registration:**
- Hive needs to know worker URL
- Hive needs to track worker state
- Enables health monitoring

### 1.3 Worker Heartbeat to Hive

**Flow (INFRASTRUCTURE EXISTS):**
```text
1. Worker: Heartbeat task wakes up (every 30s)
2. Worker: Build WorkerHeartbeatPayload {
     worker_id: "worker-123",
     timestamp: now(),
     health_status: Healthy,
   }
3. Worker: POST http://localhost:8081/v1/heartbeat
4. Hive: Receive at /v1/heartbeat endpoint
5. Hive: handle_worker_heartbeat() called
6. Hive: Update worker registry with timestamp
7. Hive: Return HeartbeatResponse
8. Worker: Sleep 30s, repeat
```

**Shared Crate:** `rbee-heartbeat` (worker module)

**Key Files:**
- Worker: Uses `start_worker_heartbeat_task()`
- Hive: Uses `handle_worker_heartbeat()` (trait-based)

### 1.4 Worker Shutdown

**Flow (NOT YET IMPLEMENTED):**
```text
1. Keeper: ./rbee worker delete --hive-id localhost --id worker-123
2. Keeper: POST /v1/jobs to queen
3. Queen: Forward to hive
4. Hive: Lookup worker in registry
5. Hive: Send SIGTERM to worker process
6. Worker: Graceful shutdown (save state, close connections)
7. Worker: Exit
8. Hive: Wait 5s
9. Hive: Send SIGKILL if still running
10. Hive: Remove from registry
11. Hive: Emit narration → SSE
```

**Graceful Shutdown:**
- Save model state (if needed)
- Close HTTP connections
- Flush logs

---

## 2. Model Provisioning (INTENDED)

### 2.1 Model Discovery

**Flow (NOT YET IMPLEMENTED):**
```text
1. Keeper: ./rbee model list --hive-id localhost
2. Keeper: POST /v1/jobs to queen
3. Queen: Forward to hive
4. Hive: Scan model directory (~/.cache/rbee/models/)
5. Hive: List all .gguf files
6. Hive: Return model list
```

**Model Directory Structure:**
```
~/.cache/rbee/models/
  llama2-7b-q4_0.gguf
  llama2-13b-q4_0.gguf
  mistral-7b-q4_0.gguf
```

### 2.2 Model Download Coordination

**Flow (NOT YET IMPLEMENTED):**
```text
1. Keeper: ./rbee model download --hive-id localhost --model llama2-7b-q4_0
2. Keeper: POST /v1/jobs to queen
3. Queen: Forward to hive
4. Hive: Check if model exists
5. If exists: Return "already downloaded"
6. If not exists:
   - Hive: Start download from Hugging Face
   - Hive: Stream progress via SSE
   - Hive: Save to ~/.cache/rbee/models/
   - Hive: Verify checksum
   - Hive: Return success
```

**Progress Streaming:**
```
[hive      ] download_start  : Downloading llama2-7b-q4_0.gguf
[hive      ] download_progress: 10% (500MB / 5GB)
[hive      ] download_progress: 20% (1GB / 5GB)
...
[hive      ] download_complete: Download complete
[hive      ] verify_checksum  : Verifying checksum
[hive      ] verify_ok        : Checksum verified
```

### 2.3 Model Validation

**Flow (NOT YET IMPLEMENTED):**
```text
1. Hive: Check file exists
2. Hive: Check file size > 0
3. Hive: Check file extension (.gguf)
4. Hive: Verify checksum (if available)
5. Hive: Try to load metadata (gguf header)
6. If valid: Return success
7. If invalid: Return error
```

**Validation Errors:**
- File not found
- File corrupted (checksum mismatch)
- Invalid format (not a gguf file)

### 2.4 Model Loading in Worker

**Flow (NOT YET IMPLEMENTED):**
```text
1. Worker: Receive spawn command with model path
2. Worker: Load model using llama.cpp
3. Worker: Allocate VRAM (if GPU)
4. Worker: Load weights into memory
5. Worker: Initialize context
6. Worker: Register with hive (model loaded)
7. Worker: Ready for inference
```

**Loading Time:**
- Small models (7B): 5-10s
- Large models (70B): 30-60s

---

## 3. Inference Coordination (INTENDED)

### 3.1 Inference Request Routing

**Flow (NOT YET IMPLEMENTED):**
```text
1. Keeper: ./rbee infer --hive-id localhost --model llama2 --prompt "Hello"
2. Keeper: POST /v1/jobs to queen
3. Queen: Forward to hive
4. Hive: Lookup workers with model "llama2"
5. Hive: Select worker (round-robin / least loaded)
6. Hive: Forward to worker: POST http://localhost:9001/v1/inference
7. Worker: Generate tokens
8. Worker: Stream tokens back to hive
9. Hive: Proxy tokens to queen via SSE
10. Queen: Proxy tokens to keeper via SSE
11. Keeper: Display streaming output
```

**Worker Selection Strategies:**
- Round-robin (simple)
- Least loaded (by active requests)
- Device preference (GPU > CPU)

### 3.2 Response Streaming

**Flow (NOT YET IMPLEMENTED):**
```text
1. Worker: Generate token
2. Worker: Send via SSE: data: {"token": "Hello"}
3. Hive: Receive token
4. Hive: Forward via SSE to queen
5. Queen: Forward via SSE to keeper
6. Keeper: Display token
7. Repeat until generation complete
8. Worker: Send: data: [DONE]
9. Hive: Forward [DONE]
10. Queen: Forward [DONE]
11. Keeper: Close stream
```

**Token Format:**
```json
{"token": "Hello", "logprob": -0.5}
{"token": " world", "logprob": -0.3}
{"token": "!", "logprob": -0.7}
```

### 3.3 Worker Slot Management

**Concept (NOT YET IMPLEMENTED):**
- Each worker has N slots (e.g., 4 for batch inference)
- Slot = concurrent inference request
- Hive tracks available slots per worker
- Requests queued if all slots full

**Slot Tracking:**
```rust
pub struct WorkerState {
    pub worker_id: String,
    pub total_slots: usize,
    pub available_slots: usize,
}
```

**Slot Allocation:**
```text
1. Hive: Receive inference request
2. Hive: Find worker with available slot
3. Hive: Decrement available_slots
4. Hive: Forward request to worker
5. Worker: Complete inference
6. Hive: Increment available_slots
```

### 3.4 VRAM Allocation

**Concept (NOT YET IMPLEMENTED):**
- Track VRAM usage per GPU
- Prevent over-allocation
- Reject requests if VRAM exhausted

**VRAM Tracking:**
```rust
pub struct GpuState {
    pub device_id: String,
    pub total_vram_gb: u32,
    pub used_vram_gb: u32,
}
```

**VRAM Allocation:**
```text
1. Hive: Receive worker spawn request
2. Hive: Check GPU VRAM available
3. Hive: Estimate model VRAM usage (from metadata)
4. If available: Spawn worker
5. If exhausted: Return error "VRAM exhausted"
```

---

## 4. Resource Management (INTENDED)

### 4.1 GPU Assignment

**Strategy (NOT YET IMPLEMENTED):**
- Explicit device assignment: `--device gpu:0`
- Auto-assignment: Select GPU with most free VRAM
- CPU fallback: If no GPU available

**Assignment Logic:**
```rust
fn assign_device(requested: Option<String>) -> Result<Device> {
    match requested {
        Some(device) => validate_device(device),
        None => auto_select_device(),
    }
}
```

### 4.2 VRAM Tracking

**Tracking (NOT YET IMPLEMENTED):**
- Query NVML for current VRAM usage
- Track per-worker VRAM allocation
- Update on worker spawn/shutdown

**NVML Query:**
```rust
let device = nvml.device_by_index(0)?;
let memory = device.memory_info()?;
let used_gb = memory.used / (1024 * 1024 * 1024);
```

### 4.3 Worker Capacity Reporting

**Reporting (NOT YET IMPLEMENTED):**
- Worker reports capacity on registration
- Includes: max_batch_size, max_context_length
- Hive uses for request routing

**Capacity Payload:**
```json
{
  "worker_id": "worker-123",
  "max_batch_size": 4,
  "max_context_length": 4096,
  "max_tokens_per_second": 50
}
```

### 4.4 Resource Cleanup

**Cleanup (NOT YET IMPLEMENTED):**
- On worker shutdown: Free VRAM
- On worker crash: Detect via heartbeat, cleanup
- On hive shutdown: Kill all workers

---

## 5. Heartbeat System (INFRASTRUCTURE EXISTS)

### 5.1 Worker → Hive Heartbeat

**Frequency:** 30 seconds

**Payload:**
```rust
pub struct WorkerHeartbeatPayload {
    pub worker_id: String,
    pub timestamp: chrono::DateTime<Utc>,
    pub health_status: HealthStatus,
}
```

**Shared Crate:** `rbee-heartbeat/src/worker.rs`

### 5.2 Worker Status Reporting

**Status Types:**
```rust
pub enum HealthStatus {
    Healthy,    // Normal operation
    Degraded,   // High load but functional
    Unhealthy,  // Errors or failures
}
```

**Status Determination:**
- Healthy: No errors, normal load
- Degraded: High memory usage, slow responses
- Unhealthy: Repeated errors, model load failure

### 5.3 Worker Failure Detection

**Detection (INFRASTRUCTURE EXISTS):**
```text
1. Worker: Stops sending heartbeats
2. Hive: No heartbeat for 60s (2 missed)
3. Hive: Mark worker as unhealthy
4. Hive: Stop routing requests to worker
5. Hive: Attempt to restart worker (optional)
```

**Staleness Threshold:** 60 seconds (2 missed heartbeats)

---

## 6. Error Propagation (INTENDED)

### 6.1 Worker Spawn Failures

**Scenarios:**
- Binary not found
- Port already in use
- Model not found
- VRAM exhausted

**Error Flow:**
```text
1. Hive: Spawn worker fails
2. Hive: Emit error narration with job_id
3. Narration → SSE → Queen → Keeper
4. Keeper: Display "❌ Worker spawn failed: <error>"
```

### 6.2 Model Load Failures

**Scenarios:**
- Model file corrupted
- Insufficient VRAM
- Invalid model format

**Error Flow:**
```text
1. Worker: Model load fails
2. Worker: Send error to hive
3. Hive: Emit error narration
4. Hive: Remove worker from registry
5. Narration → SSE → Queen → Keeper
```

### 6.3 Inference Failures

**Scenarios:**
- Context length exceeded
- VRAM exhausted mid-inference
- Worker crash

**Error Flow:**
```text
1. Worker: Inference fails
2. Worker: Send error token via SSE
3. Hive: Forward error to queen
4. Queen: Forward error to keeper
5. Keeper: Display "❌ Inference failed: <error>"
```

### 6.4 VRAM Exhaustion

**Scenario:** All GPU VRAM allocated

**Error Flow:**
```text
1. Keeper: Request worker spawn
2. Hive: Check VRAM available
3. Hive: VRAM exhausted
4. Hive: Return error "VRAM exhausted"
5. Hive: Emit error narration
6. Keeper: Display "❌ VRAM exhausted, cannot spawn worker"
```

---

## 7. State Synchronization (INTENDED)

### 7.1 Worker Registry State (Hive)

**Structure (NOT YET IMPLEMENTED):**
```rust
pub struct WorkerRegistry {
    workers: Arc<Mutex<HashMap<String, WorkerState>>>,
}

pub struct WorkerState {
    pub worker_id: String,
    pub model_id: String,
    pub device: String,
    pub port: u16,
    pub last_seen: i64,
    pub health_status: HealthStatus,
    pub available_slots: usize,
}
```

**Updates:**
- On registration: Add worker
- On heartbeat: Update last_seen + health_status
- On shutdown: Remove worker

### 7.2 Worker Status Tracking

**Status Derivation:**
```rust
// Active: Heartbeat within last 60s
if last_seen > (now - 60_000) {
    WorkerStatus::Active
} else {
    WorkerStatus::Inactive
}
```

### 7.3 Slot Availability

**Tracking (NOT YET IMPLEMENTED):**
```rust
// On inference start
worker.available_slots -= 1;

// On inference complete
worker.available_slots += 1;
```

**Concurrency:** Protected by Mutex

---

## 8. Critical Invariants (INTENDED)

### 8.1 Worker ID Uniqueness

**Invariant:** Each worker MUST have unique ID

**Enforcement:** Registry checks on registration

### 8.2 Heartbeat Interval < Staleness Threshold

**Invariant:** Heartbeat interval (30s) < Staleness threshold (60s)

**Why:** Ensures at least 2 heartbeats before marked inactive

### 8.3 Slot Count Consistency

**Invariant:** available_slots <= total_slots

**Enforcement:** Atomic increment/decrement

### 8.4 VRAM Accounting

**Invariant:** Sum of worker VRAM <= Total GPU VRAM

**Enforcement:** Check before worker spawn

---

## 9. Existing Test Coverage

### 9.1 Integration Tests

**Coverage:**
- ✅ Heartbeat infrastructure (shared crate)
- ❌ Worker spawn (NOT IMPLEMENTED)
- ❌ Model provisioning (NOT IMPLEMENTED)
- ❌ Inference coordination (NOT IMPLEMENTED)

### 9.2 Test Gaps

**Missing Tests (ALL - NOT IMPLEMENTED):**
- ❌ Worker spawn success/failure
- ❌ Worker registration
- ❌ Worker heartbeat flow
- ❌ Worker failure detection
- ❌ Model download
- ❌ Model validation
- ❌ Inference request routing
- ❌ Token streaming
- ❌ Slot management
- ❌ VRAM tracking
- ❌ GPU assignment
- ❌ Resource cleanup

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
**Next:** TEAM-242 (e2e-inference flow)
