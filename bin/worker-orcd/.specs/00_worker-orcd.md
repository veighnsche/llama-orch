# Worker-orcd SPEC — Dumb Executor (WORK-3xxx)

**Status**: Draft  
**Applies to**: `bin/worker-orcd/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

Worker-orcd is a **self-contained execution plane** process. It loads ONE model at startup, executes inference requests, and streams results. It is a dumb executor that makes NO intelligent decisions.

**Why it exists as a separate process:**
- Workers run in separate processes from pool-managerd (separate CUDA contexts)
- Worker owns its VRAM allocation within its process (CUDA allocations are per-process)
- Worker must be testable standalone: `worker-orcd --model X --gpu 0` runs independently
- Clean separation: orchestratord commands → pool manager spawns → worker executes

**What it does:**
- Load ONE model from disk/RAM into VRAM at startup (via `model-lifecycle` orchestration + binary's CUDA module)
- Execute inference requests and stream token-by-token results (via `inference-api` + binary's CUDA kernels)
- Monitor its own health and VRAM residency (via `health-monitor` + binary's CUDA queries)
- Report actual VRAM usage to pool manager after load (callback)

**What it does NOT do:**
- ❌ Load multiple models (tied to ONE model for lifetime)
- ❌ Make scheduling decisions (orchestratord does this)
- ❌ Make placement decisions (orchestratord does this)
- ❌ Validate model compatibility before load (pool manager does this preflight)
- ❌ Track system-wide VRAM state (pool manager's `gpu-inventory` does this)
- ❌ Manage other workers

**Binary structure:**
- `src/main.rs` — Entry point, CLI parsing, HTTP server
- `src/http/` — HTTP handlers (execute, health) - **previously separate api crate**
- `src/startup.rs` — Startup sequence and callbacks - **previously separate scheduler crate**
- `src/error.rs` — Error types - **previously separate error-handler crate**
- `src/cuda/mod.rs` — Rust FFI bindings to C++/CUDA
- `cuda/` — C++/CUDA implementation (separate directory)
  - `src/` — C++ implementation files
  - `include/` — C++ headers and C API
  - `kernels/` — CUDA kernel files
  - `.specs/` — CUDA module specifications

**Integrated functionality:**
All functionality previously in separate crates (`api`, `vram-residency`, `model-loader`, `capability-matcher`, `scheduler`, `error-handler`) is now implemented as modules within this binary.

**Parent spec**: `.specs/00_llama-orch.md`

---

## 1. Core Responsibilities

### [WORK-3001] Single Model
Worker-orcd MUST be tied to ONE model for its entire lifetime. It MUST NOT support loading multiple models or switching models.

### [WORK-3002] VRAM-Only Enforcement
Worker-orcd MUST enforce VRAM-only policy: model weights, KV cache, activations, and intermediate tensors MUST reside entirely in GPU VRAM. No RAM fallback is allowed.

### [WORK-3003] Inference Execution
Worker-orcd MUST execute inference requests from orchestratord and stream results back via SSE.

### [WORK-3004] Stateless
Worker-orcd MUST be stateless. It MUST NOT maintain job queues, session state, or placement logic.

---

## 2. Startup & Initialization

### [WORK-3010] Command-Line Arguments
Worker-orcd MUST accept command-line arguments:
```bash
worker-orcd \
  --worker-id <uuid> \
  --model <path-or-ref> \
  --gpu-device <id> \
  --port <port> \
  --callback-url <url>
```

Arguments:
- `--worker-id` — Unique worker identifier (UUID)
- `--model` — Model file path or location reference
- `--gpu-device` — CUDA device ID (0, 1, ...)
- `--port` — HTTP server port
- `--callback-url` — Pool manager callback URL

### [WORK-3011] Initialization Sequence
At startup, worker-orcd MUST:

1. **Initialize CUDA context**:
   - Call `cuda::safe::ContextHandle::new(gpu_device)`
   - Set CUDA device
   - Disable unified memory (UMA)
   - Disable zero-copy and pinned host memory
   - Verify CUDA device is available

2. **Load model to VRAM**:
   - Call `cuda::safe::ModelHandle::load(ctx, model_path)`
   - C++ layer: Parse GGUF format
   - C++ layer: Allocate VRAM for model weights
   - C++ layer: Copy model to VRAM
   - C++ layer: Verify VRAM residency
   - Return actual VRAM bytes used

3. **Start HTTP server**:
   - Bind to configured port
   - Expose inference and health endpoints

4. **Call back to pool manager**:
   - `POST {callback-url}` with worker metadata
   ```json
   {
     "worker_id": "worker-abc",
     "model_ref": "llama-7b",
     "vram_bytes": 16000000000,
     "uri": "http://localhost:8001"
   }
   ```

5. **Mark ready**:
   - Only after successful callback, start accepting requests

### [WORK-3012] Startup Failure
If initialization fails (insufficient VRAM, model load error, etc.), worker-orcd MUST:
1. Log detailed error
2. Attempt callback to pool manager with failure details (optional)
3. Exit with non-zero exit code

### [WORK-3014] Pool Manager Callback
After successful initialization, worker-orcd MUST call back to pool managerd to register itself as ready:

**Endpoint**: `POST {callback-url}/v2/internal/workers/ready`

**Request body**:
```json
{
  "worker_id": "worker-abc",
  "model_ref": "llama-7b",
  "vram_bytes": 16000000000,
  "uri": "http://localhost:8001"
}
```
**Callback timing**:
- Start HTTP server first, then call back (server-first), so pool-manager can validate `uri` reachability (see POOL-2060)
- Call immediately after model loads successfully and server is listening
- Only if initialization completed without errors

**Pool manager validation**:
- Validates `worker_id` exists and status is `starting`
- Validates `vram_bytes` fits within GPU capacity
- Validates `model_ref` matches the model loaded by worker-orcd
- Validates `uri` is reachable
- Updates worker status to `ready` on success

### [WORK-3015] Worker Status Values
Worker-orcd MUST use the following status values for state reporting:

- `starting` — Worker process spawned, initializing CUDA context and loading model
- `ready` — Worker successfully loaded model and is ready to accept inference requests
- `busy` — Worker is currently processing an inference request
- `draining` — Worker is finishing current request but not accepting new ones (shutdown in progress)
- `failed` — Worker encountered a fatal error and is unavailable

**Status transitions**:
- `starting` → `ready` (successful model load and callback)
- `starting` → `failed` (model load failure or timeout)
- `ready` → `busy` (inference request received)
- `busy` → `ready` (inference completed)
- `ready` → `draining` (shutdown command received)
- `draining` → `failed` (forced termination)
- Any status → `failed` (fatal error)

---

## 3. VRAM Enforcement

### [WORK-3020] VRAM-Only Enforcement
Worker-orcd MUST enforce VRAM-only policy using its internal CUDA module (`src/cuda/enforcement.rs`):
- Disable UMA (unified memory)
- Disable zero-copy mode
- Disable pinned host memory
- Verify no RAM fallback during inference

**Implementation**: The binary's `cuda` module handles all CUDA operations (allocation, enforcement, queries) within a single CUDA context.

### [WORK-3021] VRAM Allocation
Worker-orcd MUST allocate VRAM for:
- Model weights (loaded once at startup)
- KV cache (per session/request)
- Intermediate activations (per inference step)

### [WORK-3022] VRAM Health Check
Worker-orcd SHOULD periodically verify VRAM residency (health check). If RAM fallback is detected, worker MUST log critical error and mark itself unhealthy.

### [WORK-3023] Insufficient VRAM
If VRAM is insufficient at startup, worker-orcd MUST fail fast with error message containing:
- Required VRAM bytes
- Available VRAM bytes
- GPU device ID

---

## 4. Model Loading

### [WORK-3030] Model Location
Worker-orcd MUST support loading from:
- **Disk**: `file:/path/to/model.gguf`
- **RAM-staged**: Shared memory segment (passed by pool manager)

### [WORK-3031] Model Format
Worker-orcd MUST support GGUF format (llama.cpp compatible) for M0. Other formats MAY be supported later (safetensors, etc.).

### [WORK-3032] Model Validation
Worker-orcd MUST validate model at load:
- File format is valid GGUF
- Model fits in available VRAM
- Model metadata is readable

### [WORK-3033] Model Immutability
Once loaded, the model MUST remain immutable. Worker-orcd MUST NOT support model reloading or hot-swapping.

---

## 5. HTTP API

### [WORK-3040] Inference Endpoint
Worker-orcd MUST expose:
- `POST /execute`

Request body:
```json
{
  "job_id": "job-xyz",
  "prompt": "Tell me a story",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42
}
```

Response: SSE stream (see [WORK-3050])

### [WORK-3041] Health Endpoint
Worker-orcd MUST expose:
- `GET /health`

Response:
```json
{
  "status": "healthy",
  "model": "llama-7b",
  "vram_bytes": 16000000000,
  "uptime_seconds": 3600
}
```

### [WORK-3042] Metrics Endpoint
Worker-orcd SHOULD expose:
- `GET /metrics` (Prometheus format)

### [WORK-3043] Shutdown Endpoint
Worker-orcd MAY expose:
- `POST /shutdown` — Graceful shutdown (finish active jobs, then exit)

### [WORK-3044] Cancellation Endpoint
Worker-orcd MUST expose:
- `POST /cancel`

Request body:
```json
{ "job_id": "job-xyz" }
```

Semantics:
- Idempotent: Repeated cancels for the same `job_id` MUST be safe and return the same terminal outcome.
- Acceptance: Return HTTP 202 when cancellation is accepted and in progress.
- Effect: Stop decoding promptly, free resources, emit SSE `error` with code `CANCELLED` to the active stream.
- Deadline: If cancellation cannot be confirmed within 5s, worker SHOULD still stop work and close stream; orchestratord may close client SSE.

---

## 6. SSE Streaming

### [WORK-3050] SSE Framing
Worker-orcd MUST stream inference results via Server-Sent Events with event types:
- `started` — Inference started
- `token` — Token generated
- `metrics` — Metrics snapshot (optional)
- `end` — Inference complete
- `error` — Error occurred

### [WORK-3051] Event Payloads

**started**:
```json
{
  "job_id": "job-xyz",
  "model": "llama-7b",
  "started_at": "2025-10-03T00:00:00Z"
}
```

**token**:
```json
{
  "t": "Hello",
  "i": 0
}
```

**end**:
```json
{
  "tokens_out": 42,
  "decode_time_ms": 1234
}
```

**error**:
```json
{
  "code": "VRAM_OOM",
  "message": "Out of VRAM during inference"
}
```

### [WORK-3052] Stream Ordering
Worker-orcd MUST emit events in order: `started` → `token*` → `end` (or `error`).

### [WORK-3053] Stream Cancellation
If orchestratord closes the SSE connection, worker-orcd SHOULD cancel inference and free resources immediately.

---

## 7. Inference Execution

### [WORK-3060] Single-Threaded
Worker-orcd MUST process inference requests sequentially (one at a time). It MUST NOT support concurrent inference.

### [WORK-3061] Job Execution
For each `/execute` request, worker-orcd MUST:
1. Validate request parameters
2. Emit SSE `started` event
3. Run inference on GPU
4. Emit SSE `token` events as tokens are generated
5. Emit SSE `end` event when complete
6. Free KV cache and intermediate buffers

### [WORK-3062] Test Reproducibility
Worker-orcd MUST produce reproducible results for testing when given same `prompt`, `seed`, `temperature=0.0`, and `max_tokens`. It MUST use fixed RNG seeding.

**Clarification**: This is a TESTING requirement for validation, not a product promise. Temperature 0.0-2.0 is a product feature; reproducibility (temp=0) is for testing only.

### [WORK-3063] Error Handling
If inference fails (VRAM OOM, CUDA error, etc.), worker-orcd MUST:
1. Emit SSE `error` event
2. Log error details
3. Mark worker as unhealthy (if fatal error)
4. Terminate stream

---

## 8. Resource Limits

### [WORK-3070] Memory Limits
Worker-orcd MUST enforce memory limits:
- Max KV cache size per request
- Max intermediate buffer size

### [WORK-3071] Token Limits
Worker-orcd MUST enforce token limits:
- Max input tokens (from request or model context limit)
- Max output tokens (from request)

### [WORK-3072] Timeout
Worker-orcd SHOULD enforce inference timeout (default 5 minutes). After timeout, worker MUST cancel inference and emit `error` event.

---

## 9. Observability

### [WORK-3080] Structured Logging
Worker-orcd MUST emit structured logs with fields:
- `worker_id` — Worker identifier
- `job_id` — Job identifier
- `model_ref` — Model reference
- `gpu_device` — GPU device ID
- `vram_bytes` — VRAM usage
- `tokens_in` — Input tokens
- `tokens_out` — Output tokens
- `decode_time_ms` — Inference time
- `event` — `startup`, `ready`, `execute_start`, `execute_end`, `shutdown`

### [WORK-3081] Prometheus Metrics
Worker-orcd SHOULD expose metrics:
- `worker_vram_bytes` — VRAM usage
- `worker_requests_total{outcome}` — Request count
- `worker_tokens_in_total` — Total input tokens
- `worker_tokens_generated_total` — Total output tokens
- `worker_inference_duration_ms` — Inference latency histogram (milliseconds)
- `worker_uptime_seconds` — Worker uptime

### [WORK-3082] Human Narration
Worker-orcd SHOULD emit human-readable narration for key events:
- Worker startup
- Model loaded
- Inference started/completed
- Errors

---

## 10. Error Taxonomy

### [WORK-3090] Error Codes
Worker-orcd MUST use stable error codes that align with pool-managerd:

- `INVALID_REQUEST` — Invalid request parameters
- `MODEL_LOAD_FAILED` — Model failed to load
- `INSUFFICIENT_VRAM` — Not enough VRAM for model (aligns with pool-managerd)
- `VRAM_OOM` — Out of VRAM during inference
- `CUDA_ERROR` — CUDA runtime error
- `INFERENCE_TIMEOUT` — Inference exceeded timeout
- `WORKER_START_FAILED` — Worker failed to start (aligns with pool-managerd)
- `WORKER_NOT_FOUND` — Unknown worker_id (aligns with pool-managerd)
- `INTERNAL` — Internal error

### [WORK-3091] Error Responses
Worker-orcd MUST return errors via SSE `error` event with:
- `code` — Stable error code
- `message` — Human-readable description
- `retriable` — Boolean (true if orchestratord can retry)

---

## 11. Shutdown

### [WORK-3100] Graceful Shutdown
On `SIGTERM` or `POST /shutdown`, worker-orcd MUST:
1. Stop accepting new requests (return 503)
2. Finish active inference job
3. Emit shutdown log event
4. Free VRAM
5. Exit with code 0

### [WORK-3101] Shutdown Timeout
Worker-orcd MUST complete graceful shutdown within timeout (default 30s). After timeout, pool manager will `SIGKILL`.

### [WORK-3102] Crash Handling
On fatal error (CUDA error, VRAM corruption), worker-orcd SHOULD:
1. Log critical error
2. Attempt to emit error event to active streams
3. Exit with non-zero exit code

---

## 12. Configuration

### [WORK-3110] Required Config
Worker-orcd MUST accept via command-line:
- `--worker-id` — Worker UUID
- `--model` — Model path/reference
- `--gpu-device` — CUDA device ID
- `--port` — HTTP port
- `--callback-url` — Pool manager callback

### [WORK-3111] Optional Config
Worker-orcd MAY accept:
- `--max-tokens-in` — Max input tokens (default model context)
- `--max-tokens-out` — Max output tokens (default 2048)
- `--inference-timeout-sec` — Inference timeout (default 300)
- `--kv-cache-size-mb` — KV cache size (default auto)

---

## 13. Security

### [WORK-3120] Input Validation
Worker-orcd MUST validate all request parameters:
- `prompt` — Non-empty, max length
- `max_tokens` — Within limits
- `temperature` — 0.0 to 2.0
- `seed` — Valid integer

### [WORK-3121] No External Access
Worker-orcd HTTP API SHOULD only be accessible from orchestratord (localhost or trusted network).

### [WORK-3122] Secrets
Worker-orcd MUST NOT log sensitive data (prompts may contain PII).

---

## 14. GPU Requirements

### [WORK-3130] NVIDIA Only
Worker-orcd MUST require NVIDIA GPU with CUDA. It MUST fail fast if GPU is unavailable.

### [WORK-3131] VRAM-Only
Worker-orcd MUST NOT use CPU or RAM for inference. All computation MUST happen on GPU.

### [WORK-3132] Single GPU
Worker-orcd MUST run on a single GPU (no multi-GPU tensor parallelism for M0).

---

## 15. Traceability

**Code**: `bin/worker-orcd/src/`  
**CUDA**: `bin/worker-orcd/cuda/`  
**Tests**: `bin/worker-orcd/tests/`  
**Parent**: `.specs/00_llama-orch.md`  
**FFI Boundary**: `.specs/01_cuda_ffi_boundary.md`  
**CUDA Modules**: `cuda/.specs/` (context, model, inference, health)

---

## 16. Refinement Opportunities

### 16.1 Concurrency
- Support parallel inference (multi-slot)
- Batch requests together
- Continuous batching

### 16.2 Multi-GPU Support
- Tensor parallelism (split model across GPUs)
- Pipeline parallelism (split layers across GPUs)

### 16.3 Advanced Inference
- Speculative decoding
- Flash attention
- Quantized KV cache

---

**End of Specification**
