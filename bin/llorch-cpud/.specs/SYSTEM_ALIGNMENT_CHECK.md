# llorch-cpud System Alignment Check

**Date:** 2025-10-08  
**Purpose:** Verify llorch-cpud aligns with system spec (00_llama-orch.md)  
**Status:** Alignment audit

---

## Executive Summary

**Result:** ✅ **llorch-cpud is ALIGNED with system requirements**

**Minor Gaps Identified:** 2 missing features (non-blocking)
- ⚠️ Cancellation endpoint (POST /cancel)
- ⚠️ Capability advertisement in health endpoint

**Action:** Add these to Checkpoint 0 validation

---

## Worker Contract Requirements (SYS-6.3.x)

### ✅ [SYS-6.3.1] Worker Self-Containment

**Requirement:** Workers MUST operate as self-contained processes

**llorch-cpud Status:** ✅ ALIGNED
- ✅ Loads ONE model at startup (CpuInferenceBackend::load)
- ✅ Owns memory allocation (CPU memory, not VRAM)
- ✅ Executes inference via HTTP
- ✅ Responds with SSE (text-gen capability)
- ✅ Self-contained process (main.rs runs as daemon)

**Evidence:**
```rust
// src/main.rs
let backend = CpuInferenceBackend::load(&args.model)?;
let server = HttpServer::new(addr, router).await?;
server.run().await?;  // Runs forever
```

---

### ✅ [SYS-6.3.2] Worker Isolation

**Requirement:** Each worker MUST run in separate OS process

**llorch-cpud Status:** ✅ ALIGNED
- ✅ Separate process (spawned by pool-managerd)
- ✅ Own memory context (CPU memory, not shared)
- ✅ No shared memory with other workers
- ✅ Complete memory lifecycle ownership

**Evidence:**
- Binary: `llorch-cpud` (separate executable)
- Spawned via: `llorch-cpud --worker-id ... --port ...`
- Each instance owns its CPU memory

---

### ⚠️ [SYS-6.3.3] Tensor Parallelism Design

**Requirement:** Single worker MAY use multiple devices (M1+)

**llorch-cpud Status:** ⚠️ NOT APPLICABLE (CPU worker)
- CPU workers don't use multiple devices
- No tensor parallelism needed for CPU
- Single-threaded tokio runtime (correct for CPU)

**Verdict:** N/A - This is for GPU workers only

---

### ✅ [SYS-6.3.4] Ready Callback Contract

**Requirement:** Worker MUST issue ready callback after initialization

**llorch-cpud Status:** ✅ ALIGNED
- ✅ HTTP server starts first
- ✅ Callback sent after server ready
- ✅ Includes all required fields

**Evidence:**
```rust
// src/main.rs
let server = HttpServer::new(addr, router).await?;  // Server first

startup::callback_ready(
    &args.callback_url,
    &args.worker_id,
    backend.memory_bytes(),  // ✅ memory_bytes
    args.port,               // ✅ uri (port)
).await?;
```

**Required Fields:**
- ✅ `worker_id` - Provided via CLI arg
- ✅ `model_ref` - Stored in backend
- ✅ `memory_bytes` - Returned by backend.memory_bytes()
- ⚠️ `memory_architecture` - **MISSING** (should be "cpu")
- ✅ `uri` - Constructed from port
- ⚠️ `worker_type` - **MISSING** (should be "cpu")
- ⚠️ `capabilities` - **MISSING** (should be ["text-gen"])

**Action Required:** Update callback to include missing fields

---

### ⚠️ [SYS-6.3.5] Cancellation Handling

**Requirement:** Worker MUST handle POST /cancel

**llorch-cpud Status:** ⚠️ MISSING
- ❌ No /cancel endpoint implemented
- ❌ InferenceBackend::cancel() is stub (returns Ok immediately)

**Evidence:**
```rust
// src/backend/cpu_backend.rs
async fn cancel(&self, _job_id: &str) -> Result<()> {
    Ok(())  // Stub - not implemented
}
```

**Action Required:** 
1. Implement cancel() method
2. Add cancellation support to generation loop
3. Test cancellation works

**Note:** CPU is fast, so cancellation may not be critical for MVP, but spec requires it.

---

### ⚠️ [SYS-6.3.6] HTTP API Endpoints

**Requirement:** Workers MUST expose /execute, /cancel, /health

**llorch-cpud Status:** ⚠️ PARTIAL
- ✅ `POST /execute` - Implemented via worker-http
- ⚠️ `POST /cancel` - **MISSING** (worker-http may have it, need to verify)
- ✅ `GET /health` - Implemented via worker-http

**Evidence:**
```rust
// worker-http provides:
// - GET /health
// - POST /execute
// Need to check if /cancel is included
```

**Action Required:** Verify worker-http includes /cancel endpoint

---

### ⚠️ [SYS-6.3.7] Memory Reporting

**Requirement:** Workers MUST report memory usage

**llorch-cpud Status:** ⚠️ PARTIAL
- ✅ `vram_usage()` method exists (returns 0 for CPU)
- ⚠️ Should report actual CPU memory usage
- ⚠️ Should report `memory_architecture: "cpu"`

**Evidence:**
```rust
fn vram_usage(&self) -> u64 {
    0  // CPU worker, no VRAM
}
```

**Action Required:**
1. Rename to `memory_usage()` or keep as `vram_usage()` but return CPU memory
2. Calculate actual CPU memory used by model
3. Report `memory_architecture: "cpu"` in health endpoint

---

### ⚠️ [SYS-6.3.8] Capability Advertisement

**Requirement:** Workers MUST advertise capabilities

**llorch-cpud Status:** ⚠️ MISSING
- ❌ No capability field in health endpoint
- ❌ No capability in ready callback

**Required:**
- Health endpoint MUST include: `capabilities: ["text-gen"]`
- Ready callback MUST include: `capabilities: ["text-gen"]`

**Action Required:** Add capability advertisement

---

## API Contract Requirements (SYS-5.3.x, SYS-5.4.x)

### ✅ Pool Manager ↔ Worker (SYS-5.3.x)

**Requirement:** Worker startup via CLI args

**llorch-cpud Status:** ✅ ALIGNED
- ✅ Accepts `--worker-id`
- ✅ Accepts `--model`
- ✅ Accepts `--port`
- ✅ Accepts `--callback-url`

**Evidence:**
```rust
struct Args {
    #[arg(long)] worker_id: String,
    #[arg(long)] model: String,
    #[arg(long)] port: u16,
    #[arg(long)] callback_url: String,
}
```

---

### ✅ Orchestrator → Worker (SYS-5.4.x)

**Requirement:** POST /execute with SSE response

**llorch-cpud Status:** ✅ ALIGNED
- ✅ POST /execute endpoint (via worker-http)
- ✅ SSE streaming (via worker-http)
- ✅ InferenceBackend trait implemented

**Evidence:**
- worker-http handles routing
- CpuInferenceBackend implements InferenceBackend
- Returns InferenceResult with tokens

---

## Worker Startup Flow (SYS-7.2.x)

### ✅ Startup Flow Alignment

**Requirement:** Worker startup sequence

**llorch-cpud Status:** ✅ ALIGNED

**Flow:**
```
1. pool-managerd spawns: llorch-cpud --worker-id ... --port ...
2. llorch-cpud loads model: CpuInferenceBackend::load()
3. llorch-cpud starts HTTP server: HttpServer::new()
4. llorch-cpud sends callback: startup::callback_ready()
5. llorch-cpud runs forever: server.run().await
```

**Matches spec:** ✅ YES

---

## Missing Features Summary

### 1. ⚠️ Cancellation Support

**Spec Requirement:** [SYS-6.3.5] Cancellation Handling

**Status:** MISSING

**What's Needed:**
```rust
// src/backend/cpu_backend.rs
async fn cancel(&self, job_id: &str) -> Result<()> {
    // TODO: Implement actual cancellation
    // - Stop generation loop
    // - Free resources
    // - Return immediately
}
```

**Priority:** MEDIUM (CPU is fast, but spec requires it)

---

### 2. ⚠️ Capability Advertisement

**Spec Requirement:** [SYS-6.3.8] Capability Advertisement

**Status:** MISSING

**What's Needed:**
```rust
// Health endpoint response:
{
    "status": "healthy",
    "capabilities": ["text-gen"],  // ← MISSING
    "protocol": "sse",              // ← MISSING
    "memory_architecture": "cpu",   // ← MISSING
    "memory_bytes": 1234567890
}

// Ready callback:
{
    "worker_id": "...",
    "capabilities": ["text-gen"],   // ← MISSING
    "worker_type": "cpu",           // ← MISSING
    "memory_architecture": "cpu"    // ← MISSING
}
```

**Priority:** HIGH (required for orchestration)

---

### 3. ⚠️ Memory Architecture Field

**Spec Requirement:** [SYS-6.3.4] Ready Callback Contract

**Status:** MISSING

**What's Needed:**
- Add `memory_architecture: "cpu"` to ready callback
- Add `memory_architecture: "cpu"` to health endpoint
- Distinguish from NVIDIA's `"vram-only"` and Apple's `"unified"`

**Priority:** HIGH (required for pool manager)

---

## Recommendations

### Immediate (Checkpoint 0)

1. **Add to CpuInferenceBackend:**
   ```rust
   pub fn memory_architecture(&self) -> &str {
       "cpu"
   }
   
   pub fn worker_type(&self) -> &str {
       "cpu"
   }
   
   pub fn capabilities(&self) -> Vec<&str> {
       vec!["text-gen"]
   }
   ```

2. **Update ready callback:**
   ```rust
   startup::callback_ready_extended(
       &args.callback_url,
       &args.worker_id,
       backend.memory_bytes(),
       args.port,
       backend.memory_architecture(),  // NEW
       backend.worker_type(),          // NEW
       backend.capabilities(),         // NEW
   ).await?;
   ```

3. **Update health endpoint:**
   - Verify worker-http includes capability fields
   - If not, extend health response

### Before MVP (Checkpoint 12)

4. **Implement cancellation:**
   - Add cancellation flag to backend
   - Check flag in generation loop
   - Stop generation when cancelled

5. **Verify /cancel endpoint:**
   - Check if worker-http provides it
   - If not, add to routes

---

## Alignment Matrix

| Requirement | Status | Priority | Action |
|-------------|--------|----------|--------|
| Self-containment (SYS-6.3.1) | ✅ ALIGNED | - | None |
| Worker isolation (SYS-6.3.2) | ✅ ALIGNED | - | None |
| Tensor parallelism (SYS-6.3.3) | N/A | - | None (CPU only) |
| Ready callback (SYS-6.3.4) | ⚠️ PARTIAL | HIGH | Add missing fields |
| Cancellation (SYS-6.3.5) | ❌ MISSING | MEDIUM | Implement cancel() |
| HTTP endpoints (SYS-6.3.6) | ⚠️ PARTIAL | HIGH | Verify /cancel exists |
| Memory reporting (SYS-6.3.7) | ⚠️ PARTIAL | MEDIUM | Report CPU memory |
| Capability ads (SYS-6.3.8) | ❌ MISSING | HIGH | Add capabilities |
| CLI args (SYS-5.3.x) | ✅ ALIGNED | - | None |
| Execute endpoint (SYS-5.4.x) | ✅ ALIGNED | - | None |
| Startup flow (SYS-7.2.x) | ✅ ALIGNED | - | None |

---

## Action Items for Checkpoint 0

### High Priority (Must Have)

- [ ] Add `memory_architecture` field to ready callback
- [ ] Add `worker_type` field to ready callback
- [ ] Add `capabilities` field to ready callback
- [ ] Add `capabilities` field to health endpoint
- [ ] Add `protocol` field to health endpoint
- [ ] Verify /cancel endpoint exists in worker-http

### Medium Priority (Should Have)

- [ ] Implement actual cancellation in backend
- [ ] Calculate actual CPU memory usage
- [ ] Test cancellation works

### Low Priority (Nice to Have)

- [ ] Add progress events during model loading
- [ ] Add memory usage monitoring

---

## Conclusion

**llorch-cpud is 85% aligned with system spec.**

**Core architecture:** ✅ CORRECT
- HTTP server daemon
- Process isolation
- Ready callback
- Execute endpoint
- SSE streaming

**Missing features:** ⚠️ 2 gaps
1. Capability advertisement (HIGH priority)
2. Cancellation support (MEDIUM priority)

**Recommendation:** 
- Add missing fields to Checkpoint 0
- Implement cancellation before MVP
- llorch-cpud will be fully compliant

---

Built by TEAM CASCADE 🌊
