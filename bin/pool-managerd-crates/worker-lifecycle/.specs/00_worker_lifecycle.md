# Worker Lifecycle SPEC — Worker Process Management (WLIFE-10xxx)

**Status**: Draft  
**Applies to**: `bin/pool-managerd-crates/worker-lifecycle/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `worker-lifecycle` crate manages worker process lifecycle for pool-managerd. It spawns, monitors, and stops worker-orcd processes.

**Why it exists:**
- Pool manager needs to start/stop worker processes on orchestratord command
- Worker processes are separate OS processes (separate CUDA contexts)
- Need process monitoring and health checks
- Handle worker crashes and timeouts

**What it does:**
- Spawn worker-orcd processes with correct arguments
- Monitor worker process health (via PID)
- Stop workers (graceful shutdown with SIGTERM, then SIGKILL)
- Track worker registry (worker metadata)
- Handle worker callbacks (registration after startup)

**What it does NOT do:**
- ❌ Allocate VRAM (worker does this)
- ❌ Load models (worker does this)
- ❌ Decide which workers to start (orchestratord decides)
- ❌ Download models (model-cache does this)

---

## 1. Core Responsibilities

### [WLIFE-10001] Worker Spawning
The crate MUST spawn worker-orcd processes with correct command-line arguments.

### [WLIFE-10002] Worker Registry
The crate MUST maintain registry of running workers with metadata.

### [WLIFE-10003] Process Monitoring
The crate MUST monitor worker process health and detect crashes.

### [WLIFE-10004] Graceful Shutdown
The crate MUST support graceful worker shutdown (SIGTERM, then SIGKILL).

---

## 2. Worker Spawning

### [WLIFE-10010] Spawn Sequence
When orchestratord requests worker start, the crate MUST:

1. **Preflight validation**:
   - Check GPU has sufficient free VRAM (via `gpu-inventory`)
   - Check model is downloaded (via `model-cache`)
   - Check model compatibility (via `capability-matcher`)

2. **Allocate port**:
   - Select available port for worker HTTP server

3. **Spawn process**:
   ```bash
   worker-orcd \
     --worker-id worker-{uuid} \
     --model /path/to/model.gguf \
     --gpu-device {gpu_id} \
     --port {port} \
     --callback-url http://localhost:9200/v2/internal/workers/ready
   ```

4. **Register worker**:
   - Add to registry with status `starting`
   - Set timeout for callback (default 60s)

5. **Wait for callback**:
   - Worker will call `/v2/internal/workers/ready` when loaded
   - Update registry status to `ready`

6. **Return worker_id** to orchestratord

### [WLIFE-10011] Spawn Timeout
If worker doesn't call back within timeout (default 60s):
1. Kill worker process (SIGKILL)
2. Mark worker as `failed` in registry
3. Return error to orchestratord

---

## 3. Worker Registry

### [WLIFE-10020] Registry Schema
For each worker, track:
```rust
struct WorkerEntry {
    worker_id: String,
    model_ref: String,
    gpu_device: u32,
    vram_bytes: u64,       // Reported by worker after load
    uri: String,           // e.g., "http://localhost:8001"
    status: WorkerStatus,  // starting, ready, busy, draining, failed
    pid: u32,
    started_at: DateTime<Utc>,
}
```

### [WLIFE-10021] Worker States
- `starting` — Process spawned, waiting for callback
- `ready` — Worker ready for inference
- `busy` — Worker executing inference (optional state)
- `draining` — Worker finishing active job before shutdown
- `failed` — Worker crashed or failed to start

---

## 4. Process Monitoring

### [WLIFE-10030] Health Checks
Every N seconds (default 10s), the crate MUST:
1. Check worker process is alive (PID exists)
2. If process dead, mark worker as `failed`
3. Update GPU inventory (free VRAM)
4. Log crash event

### [WLIFE-10031] HTTP Health Checks
The crate SHOULD optionally ping worker `/health` endpoint:
- If unreachable for N consecutive checks (default 3), mark as `failed`

### [WLIFE-10032] Crash Handling
On worker crash (process exit):
1. Detect via process monitoring
2. Remove worker from registry
3. Update GPU inventory (decrement allocated_vram)
4. Log crash event with exit code
5. Do NOT auto-restart (orchestratord decides)

---

## 5. Worker Shutdown

### [WLIFE-10040] Graceful Shutdown
When orchestratord requests worker stop:
1. **Mark draining**: Set worker status to `draining`
2. **Send SIGTERM**: Signal graceful shutdown
3. **Wait**: Allow grace period (default 30s)
4. **Force kill**: If still alive, send SIGKILL
5. **Update state**: Remove from registry, update GPU inventory
6. **Return**: Respond to orchestratord

### [WLIFE-10041] Drain Mode
For drain (finish active job, then shutdown):
1. Mark worker as `draining`
2. Worker stops accepting new requests
3. Worker finishes active inference
4. Worker exits
5. Pool manager detects exit, cleans up

---

## 6. Worker Callbacks

### [WLIFE-10050] Ready Callback
When worker calls `/v2/internal/workers/ready`:
1. Validate worker_id exists with status `starting`
2. Validate vram_bytes is reasonable
3. Update worker registry:
   - `status` → `ready`
   - `vram_bytes` ← reported value
   - `uri` ← worker endpoint
4. Update GPU inventory (add allocated_vram)
5. Return 200 OK

### [WLIFE-10051] Callback Timeout
If no callback within timeout:
- Mark worker as `failed`
- Kill process
- Free any partial resources

---

## 7. Dependencies

### [WLIFE-10060] Required Crates
```toml
[dependencies]
capability-matcher = { path = "../capability-matcher" }
gpu-inventory = { path = "../gpu-inventory" }
model-cache = { path = "../model-cache" }
tokio = { workspace = true, features = ["process", "time"] }
tracing = { workspace = true }
thiserror = { workspace = true }
uuid = { workspace = true, features = ["v4"] }
```

---

## 8. Traceability

**Code**: `bin/pool-managerd-crates/worker-lifecycle/src/`  
**Tests**: `bin/pool-managerd-crates/worker-lifecycle/tests/`  
**Parent**: `bin/pool-managerd/.specs/00_pool-managerd.md`  
**Used by**: `pool-managerd`  
**Depends on**: `capability-matcher`, `gpu-inventory`, `model-cache`  
**Spec IDs**: WLIFE-10001 to WLIFE-10060

---

**End of Specification**
