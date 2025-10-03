# M0: Worker-orcd Complete Specification

**Status**: Draft  
**Milestone**: M0 (v0.1.0) — Worker Haiku Test  
**Version**: 0.1.0  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Document Metadata

### 0.1 Purpose

This specification consolidates **ALL M0 requirements** for the `worker-orcd` binary. M0 is the foundational milestone that delivers a standalone GPU worker capable of loading a single model and executing inference with deterministic, VRAM-only operation.

**M0 Goal**: Prove the worker can load a model, execute inference, and stream results—standalone, without orchestrator or pool-manager dependencies.

**Success Criteria**: 
- Worker loads Qwen2.5-0.5B-Instruct (352MB GGUF) into VRAM
- Executes haiku generation prompt deterministically
- Streams tokens via SSE
- All operations VRAM-only (no RAM fallback)

### 0.2 Scope

**In Scope for M0**:
- ✅ Single worker binary (`worker-orcd`)
- ✅ Single model loading (GGUF format)
- ✅ Single GPU support (no tensor parallelism)
- ✅ VRAM-only enforcement
- ✅ HTTP API (execute, health, cancel)
- ✅ SSE streaming
- ✅ Deterministic inference (seeded RNG)
- ✅ CUDA FFI boundary (Rust ↔ C++/CUDA)
- ✅ Basic CUDA kernels (GEMM, RoPE, attention, RMSNorm, sampling)
- ✅ Standalone testing (no orchestrator required)

**Out of Scope for M0**:
- ❌ Pool manager integration (M1)
- ❌ Orchestrator integration (M2)
- ❌ Multi-GPU / tensor parallelism (M4)
- ❌ Multi-model support
- ❌ Advanced kernels (FlashAttention, continuous batching)
- ❌ Authentication/authorization
- ❌ Full metrics/observability (basic only)

### 0.3 Traceability System

**Traceability Hierarchy**:

```
00_llama-orch.md (LEADING SPEC)
├── SYS-X.Y.Z — System-level requirements (stable, cross-milestone)
│
└── 01_M0_worker_orcd.md (THIS SPEC)
    ├── M0-SYS-X.Y.Z — References to parent SYS requirements
    └── M0-W-NNNN — M0-specific worker requirements (granular)
```

**Numbering Convention**:
- **SYS-X.Y.Z**: System-level requirements from `00_llama-orch.md` (stable across milestones)
- **M0-SYS-X.Y.Z**: Direct reference to parent SYS requirement (indicates M0 implements this)
- **M0-W-NNNN**: M0-specific worker requirement (4-digit sequential, milestone-scoped)

**Example**:
- `SYS-6.3.1` — System requirement: "Worker Self-Containment"
- `M0-SYS-6.3.1` — M0 implements SYS-6.3.1
- `M0-W-1001` — M0-specific: "Worker MUST load Qwen2.5-0.5B-Instruct test model"

**Rationale**:
- Parent spec (`00_llama-orch.md`) defines stable system architecture
- Milestone specs add granular implementation requirements
- M0-SYS-* creates bidirectional traceability
- M0-W-* provides milestone-specific detail without polluting parent spec

### 0.4 Parent Spec References

**Leading Spec**: `bin/.specs/00_llama-orch.md`

**Worker Requirements in Parent Spec**:
- **SYS-6.3.x** — Worker-Orcd (Executor) [M0]
  - SYS-6.3.1 — Worker Self-Containment
  - SYS-6.3.2 — Worker Isolation
  - SYS-6.3.3 — Tensor Parallelism Design (M1+)
  - SYS-6.3.4 — Ready Callback Contract
  - SYS-6.3.5 — Cancellation Handling

**Foundational Concepts (Apply to M0)**:
- **SYS-2.1.x** — Model Reference Format [M0+]
- **SYS-2.2.x** — VRAM-Only Policy [M0+]
- **SYS-2.3.x** — Determinism Principles [M0+]
- **SYS-2.4.x** — Process Isolation Rationale [M0+]
- **SYS-2.5.x** — FFI Boundaries [M0+]

**Quality Attributes (Apply to M0)**:
- **SYS-8.1.x** — Determinism [M0+]
- **SYS-8.2.x** — Performance [M0+]

**Deployment Mode**:
- **SYS-3.1.x** — Home Mode (M0)

**Component Specs**:
- `bin/worker-orcd/.specs/00_worker-orcd.md` — Worker specification (WORK-3xxx)
- `bin/worker-orcd/.specs/01_cuda_ffi_boundary.md` — FFI boundary (CUDA-4xxx)
- `bin/worker-orcd/cuda/.specs/*.md` — CUDA module specs (CUDA-5xxx)

---

## 1. Architecture Overview

### 1.1 System Context

```
┌─────────────────────────────────────────────────────────────┐
│ RUST LAYER (src/*.rs)                                        │
│ • HTTP server (axum)                                         │
│ • CLI argument parsing                                       │
│ • SSE streaming                                              │
│ • Error handling and formatting                              │
│ • Logging and metrics                                        │
└────────────────────┬────────────────────────────────────────┘
                     │ FFI (unsafe extern "C")
┌────────────────────▼────────────────────────────────────────┐
│ C++/CUDA LAYER (cuda/src/*.cpp, *.cu)                       │
│ • CUDA context management                                    │
│ • VRAM allocation (cudaMalloc)                              │
│ • Model loading (GGUF → VRAM)                               │
│ • Inference execution (CUDA kernels)                         │
│ • VRAM residency checks                                      │
│ • All GPU operations                                         │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Process Model

#### [M0-SYS-6.3.1] Worker Self-Containment
**Parent**: SYS-6.3.1

Worker-orcd MUST operate as a self-contained process:
- MUST load exactly ONE model at startup (from disk to VRAM)
- MUST own VRAM allocation within its CUDA context
- MUST allocate all model resources in VRAM only (no RAM fallback)
- MUST execute inference requests received via HTTP
- MUST stream results via SSE (token-by-token)
- MUST monitor VRAM residency (self-health checks)

**M0 Clarification**: For M0, ready callback to pool manager is OPTIONAL (standalone mode). M1+ requires callback.

#### [M0-SYS-6.3.2] Worker Isolation
**Parent**: SYS-6.3.2

Each worker MUST run in a separate OS process. Workers MUST NOT share VRAM or CUDA contexts.

**Requirements**:
- Each worker MUST have its own OS process
- Each worker MUST have its own CUDA context
- Workers MUST NOT share VRAM pointers
- Worker MUST own complete VRAM lifecycle (allocate → use → free)

**M0 Testing**: Enables standalone worker testing (`worker-orcd --model X --gpu 0`).

### 1.3 Single Model Constraint

#### [M0-W-1001] Single Model Lifetime
Worker-orcd MUST be tied to ONE model for its entire lifetime. It MUST NOT support loading multiple models or switching models.

**Rationale**: Simplifies M0 implementation. Multi-model support deferred to post-M0.

**Verification**: Unit test MUST verify worker rejects second model load attempt.

#### [M0-W-1002] Model Immutability
Once loaded, the model MUST remain immutable. Worker-orcd MUST NOT support model reloading or hot-swapping.

**Verification**: Integration test MUST verify model remains in VRAM for worker lifetime.

---

## 2. VRAM-Only Policy

### 2.1 System Requirements

#### [M0-SYS-2.2.1] VRAM-Only Enforcement
**Parent**: SYS-2.2.1

The system MUST enforce VRAM-only policy: model weights, KV cache, activations, and intermediate tensors MUST reside entirely in GPU VRAM.

**Prohibited**:
- ❌ RAM fallback
- ❌ Unified memory (CUDA UMA)
- ❌ Zero-copy mode
- ❌ CPU inference fallback
- ❌ Disk swapping

### 2.2 CUDA Implementation

#### [M0-W-1010] CUDA Context Configuration
Worker-orcd MUST configure CUDA context to enforce VRAM-only operation:
- Disable Unified Memory (UMA) via `cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0)`
- Set cache config for compute via `cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)`
- Verify no host pointer fallback via `cudaPointerGetAttributes`

**Verification**: Unit test MUST verify UMA is disabled after context initialization.

**Spec Reference**: CUDA-5102 (context module)

#### [M0-W-1011] VRAM Allocation Tracking
Worker-orcd MUST track VRAM allocation:
- Model weights allocation (one-time at startup)
- KV cache allocation (per inference request)
- Intermediate buffer allocation (per inference step)

**Reporting**: Worker MUST report actual VRAM bytes used in ready callback (M1+) or health endpoint (M0).

**Spec Reference**: CUDA-5203 (model module)

#### [M0-W-1012] VRAM Residency Verification
Worker-orcd SHOULD periodically verify VRAM residency via `cudaPointerGetAttributes`:
- Verify pointer type is `cudaMemoryTypeDevice`
- Verify no host pointer exists (`hostPointer == nullptr`)
- If RAM fallback detected, worker MUST log critical error and mark itself unhealthy

**Frequency**: Check every 60 seconds (configurable).

**Spec Reference**: CUDA-5401 (health module)

### 2.3 Failure Handling

#### [M0-W-1020] Insufficient VRAM at Startup
If VRAM is insufficient at startup, worker-orcd MUST fail fast with error message containing:
- Required VRAM bytes
- Available VRAM bytes
- GPU device ID
- Model path

**Exit code**: Non-zero (1)

**Log level**: ERROR

**Verification**: Integration test MUST verify worker exits with proper error when VRAM insufficient.

#### [M0-W-1021] VRAM OOM During Inference
If VRAM exhausted during inference (KV cache allocation fails), worker MUST:
1. Emit SSE `error` event with code `VRAM_OOM`
2. Free partial allocations
3. Mark worker unhealthy
4. Continue accepting new requests (worker remains alive)

**Rationale**: Worker process remains alive to serve other requests. Individual job fails, not entire worker.

---

## 3. Determinism

### 3.1 System Requirements

#### [M0-SYS-2.3.1] Determinism Guarantee
**Parent**: SYS-2.3.1

The system MUST guarantee deterministic inference: same model + same seed + same prompt → same output.

**Requirements**:
- Sealed VRAM shards (worker-orcd)
- Pinned engine versions
- Deterministic sampling
- No non-deterministic operations

**Design Principle**: The system is designed AS IF deterministic, but acknowledges that underlying models may not guarantee determinism.

### 3.2 Implementation Requirements

#### [M0-W-1030] Seeded RNG
Worker-orcd MUST initialize RNG with provided seed:
```cpp
std::mt19937_64 rng_(seed);  // C++ implementation
```

**Seed source**: 
- Client provides `seed` in request
- If omitted, worker MUST generate seed and include in response

**Verification**: Unit test MUST verify identical outputs for identical seed.

**Spec Reference**: CUDA-5350 (inference module)

#### [M0-W-1031] Deterministic CUDA Kernels
All CUDA kernels MUST be deterministic:
- No atomics with race conditions
- No non-deterministic reductions
- Fixed execution order
- Disable non-deterministic cuBLAS algorithms

**cuBLAS Configuration**:
```cpp
cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);  // Disable Tensor Cores if non-deterministic
```

**Verification**: Property test MUST verify kernel determinism via repeated execution.

**Spec Reference**: CUDA-5351 (inference module)

#### [M0-W-1032] Temperature Scaling
Worker-orcd MUST apply temperature scaling deterministically:
```cpp
for (float& logit : host_logits) {
    logit /= config_.temperature;
}
```

**Temperature range**: 0.0 to 2.0
- 0.0 = greedy (argmax)
- 1.0 = no scaling
- >1.0 = more random

**Verification**: Unit test MUST verify temperature=0.0 produces greedy sampling.

**Spec Reference**: KERNEL-SAMPLE-003

### 3.3 Limitations

#### [M0-W-1040] Model-Level Non-Determinism
Worker-orcd acknowledges model-level limitations:
- Inference engines (llama.cpp) MAY have non-deterministic operations
- GPU hardware variations MAY produce different floating-point results
- Cross-worker/cross-GPU determinism is NOT guaranteed

**Documentation**: README MUST document which models/engines have been verified as deterministic.

**Spec Reference**: SYS-2.3.3

---

## 4. Process Isolation & FFI Boundaries

### 4.1 Process Isolation

#### [M0-SYS-2.4.1] Process Isolation Requirement
**Parent**: SYS-2.4.1

Workers MUST run in separate processes from pool managers.

**Why**: CUDA allocations are per-process. Workers need self-contained VRAM ownership within their CUDA context.

#### [M0-SYS-2.4.2] Worker Process Isolation
**Parent**: SYS-2.4.2

Workers MUST run in separate processes with isolated CUDA contexts.

**M0 Clarification**: For M0, worker runs standalone (no pool manager). M1+ adds pool manager spawning.

### 4.2 FFI Boundaries

#### [M0-SYS-2.5.1] FFI Boundary Enforcement
**Parent**: SYS-2.5.1

The system MUST enforce strict FFI boundaries:

**Worker** (CUDA only):
- MUST use CUDA Runtime API for VRAM allocation
- MUST allocate VRAM within its process CUDA context
- MUST use CUDA for compute operations
- MUST own VRAM lifecycle (allocate → use → free)

#### [M0-W-1050] Rust Layer Responsibilities
The Rust layer MUST handle:
- HTTP server (Axum)
- CLI argument parsing (clap)
- SSE streaming
- Error formatting (convert C++ errors to HTTP responses)
- Logging (tracing)
- Metrics (Prometheus, optional for M0)

The Rust layer MUST NOT:
- ❌ Call CUDA APIs directly
- ❌ Manage CUDA context
- ❌ Allocate VRAM
- ❌ Load model weights to GPU
- ❌ Execute inference
- ❌ Check VRAM residency

**Rationale**: CUDA context is per-process. Mixing Rust and C++ CUDA calls leads to context conflicts.

**Spec Reference**: CUDA-4011

#### [M0-W-1051] C++/CUDA Layer Responsibilities
The C++/CUDA layer MUST handle:
- CUDA context management (`cudaSetDevice`, `cudaStreamCreate`)
- VRAM allocation (`cudaMalloc`, `cudaFree`)
- Model loading (GGUF parsing, copy to VRAM)
- Inference execution (CUDA kernels)
- Health checks (VRAM residency verification)

**Spec Reference**: CUDA-4020

#### [M0-W-1052] C API Interface
The CUDA layer MUST expose a C API (not C++) for Rust FFI:

```c
// Opaque handle types
typedef struct CudaContext CudaContext;
typedef struct CudaModel CudaModel;
typedef struct InferenceResult InferenceResult;

// Context management
CudaContext* cuda_init(int gpu_device, int* error_code);
void cuda_destroy(CudaContext* ctx);

// Model loading
CudaModel* cuda_load_model(
    CudaContext* ctx,
    const char* model_path,
    uint64_t* vram_bytes_used,
    int* error_code
);
void cuda_unload_model(CudaModel* model);

// Inference
InferenceResult* cuda_inference_start(
    CudaModel* model,
    const char* prompt,
    int max_tokens,
    float temperature,
    uint64_t seed,
    int* error_code
);
bool cuda_inference_next_token(
    InferenceResult* result,
    char* token_out,
    int* token_index,
    int* error_code
);
void cuda_inference_free(InferenceResult* result);

// Health
bool cuda_check_vram_residency(CudaModel* model, int* error_code);
uint64_t cuda_get_vram_usage(CudaModel* model);
```

**Spec Reference**: CUDA-4030

---

## 5. Startup & Initialization

### 5.1 Command-Line Interface

#### [M0-W-1100] Required Arguments
Worker-orcd MUST accept command-line arguments:

```bash
worker-orcd \
  --worker-id <uuid> \
  --model <path> \
  --gpu-device <id> \
  --port <port>
```

**Arguments**:
- `--worker-id` — Unique worker identifier (UUID format)
- `--model` — Model file path (absolute path to .gguf file)
- `--gpu-device` — CUDA device ID (0, 1, 2, ...)
- `--port` — HTTP server port (1024-65535)

**M0 Note**: `--callback-url` is OPTIONAL for M0 (standalone mode). Required for M1+.

**Validation**:
- worker-id MUST be valid UUID
- model path MUST exist and be readable
- gpu-device MUST be valid device ID (0 to device_count-1)
- port MUST be available

**Spec Reference**: WORK-3010

#### [M0-W-1101] Optional Arguments
Worker-orcd MAY accept:
- `--max-tokens-in` — Max input tokens (default: model context length)
- `--max-tokens-out` — Max output tokens (default: 2048)
- `--inference-timeout-sec` — Inference timeout (default: 300)
- `--kv-cache-size-mb` — KV cache size (default: auto-calculate)

**Spec Reference**: WORK-3111

### 5.2 Initialization Sequence

#### [M0-W-1110] Startup Steps
At startup, worker-orcd MUST execute in order:

**1. Initialize CUDA context**:
```rust
let ctx = unsafe { cuda_init(gpu_device, &mut error_code) };
if ctx.is_null() {
    eprintln!("CUDA init failed: {}", error_code);
    std::process::exit(1);
}
```

**2. Load model to VRAM**:
```rust
let mut vram_bytes = 0;
let model = unsafe {
    cuda_load_model(ctx, model_path_cstr, &mut vram_bytes, &mut error_code)
};
if model.is_null() {
    eprintln!("Model load failed: {}", error_code);
    cuda_destroy(ctx);
    std::process::exit(1);
}
```

**3. Start HTTP server**:
```rust
let app = Router::new()
    .route("/execute", post(execute_handler))
    .route("/health", get(health_handler))
    .route("/cancel", post(cancel_handler));

let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
axum::serve(listener, app).await?;
```

**4. Mark ready**:
```rust
info!("Worker ready: worker_id={}, vram_bytes={}", worker_id, vram_bytes);
```

**M0 Note**: Steps 4 (callback) from parent spec is OPTIONAL for M0.

**Spec Reference**: WORK-3011

#### [M0-W-1111] Startup Failure Handling
If initialization fails, worker-orcd MUST:
1. Log detailed error with context
2. Clean up partial state (free CUDA resources)
3. Exit with non-zero exit code

**Error logging MUST include**:
- Error code
- Error message
- GPU device ID
- Model path
- VRAM available vs required (if applicable)

**Spec Reference**: WORK-3012

### 5.3 Performance Requirements

#### [M0-W-1120] Startup Latency Target
Worker startup SHOULD complete within 60 seconds measured from process start to ready state.

**Breakdown**:
- CUDA context init: <1s
- Model load to VRAM: <58s (varies by model size)
- HTTP server start: <1s

**M0 Test Model**: Qwen2.5-0.5B-Instruct (352MB) SHOULD load in <10s on modern GPU.

**Spec Reference**: SYS-8.2.1

---

## 6. Model Loading

### 6.1 Model Format

#### [M0-W-1200] GGUF Format Support
Worker-orcd MUST support GGUF (GPT-Generated Unified Format) for M0.

**GGUF Version**: Version 3

**Magic bytes**: `0x47475546` ("GGUF")

**Spec Reference**: WORK-3031, CUDA-5202

#### [M0-W-1201] Quantization Support
Worker-orcd MUST support Q4_0 quantization for M0.

**Dequantization**: Dequantize on load (convert Q4_0 to FP32 in VRAM).

**Future**: Q5_1, Q8_0, FP16 support deferred to M2+.

**Spec Reference**: WORKER-4701

### 6.2 Model Validation

#### [M0-W-1210] Pre-Load Validation
Worker-orcd MUST validate model before loading:
- File exists and is readable
- GGUF magic bytes are correct (`0x47475546`)
- GGUF version is supported (version 3)
- Tensor count is reasonable (<10,000)
- Total size fits in available VRAM

**Validation failure**: Exit with error code 1 and detailed message.

**Spec Reference**: CUDA-5251

#### [M0-W-1211] GGUF Header Parsing
Worker-orcd MUST parse GGUF header:

```cpp
struct GGUFHeader {
    uint32_t magic;              // 'GGUF' (0x47475546)
    uint32_t version;            // 3
    uint64_t tensor_count;       // Number of tensors
    uint64_t metadata_kv_count;  // Number of metadata entries
};
```

**Required metadata keys**:
- `general.architecture` — Model architecture ("llama", "gpt2", etc.)
- `general.name` — Model name
- `llama.context_length` — Context window size
- `llama.embedding_length` — Embedding dimensions
- `llama.block_count` — Number of layers

**Spec Reference**: CUDA-5280, CUDA-5281

### 6.3 VRAM Allocation

#### [M0-W-1220] Model Weights Allocation
Worker-orcd MUST allocate VRAM for model weights:

```cpp
size_t total_size = calculate_total_size();  // From GGUF tensors
weights_ = std::make_unique<DeviceMemory>(total_size);
vram_bytes_ = total_size;
```

**Alignment**: All tensors MUST be aligned to 256-byte boundaries for optimal GPU access.

**Spec Reference**: CUDA-5291

#### [M0-W-1221] Memory-Mapped I/O
Worker-orcd SHOULD use `mmap()` for efficient file reading without loading entire file into RAM.

**Rationale**: Reduces RAM usage, faster startup.

**Spec Reference**: CUDA-5260

#### [M0-W-1222] Chunked Transfer
Worker-orcd SHOULD copy model to VRAM in chunks (e.g., 1MB) to avoid large temporary buffers.

**Spec Reference**: CUDA-5261

### 6.4 Test Model

#### [M0-W-1230] Primary Test Model
M0 MUST use **Qwen2.5-0.5B-Instruct** as primary test model.

**Specifications**:
- Size: 352MB (Q4_K_M quantization)
- Format: GGUF
- Location: `.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf`
- Download: `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf`

**Rationale**: Small enough for fast testing, large enough to verify VRAM allocation.

---

## 7. HTTP API

### 7.1 Inference Endpoint

#### [M0-W-1300] POST /execute
Worker-orcd MUST expose inference endpoint:

**Request**:
```http
POST /execute HTTP/1.1
Content-Type: application/json

{
  "job_id": "job-xyz",
  "prompt": "Write a haiku about GPU computing",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42
}
```

**Response**: SSE stream (see §7.2)

**Validation**:
- `job_id` — MUST be non-empty string
- `prompt` — MUST be non-empty, max 32768 characters
- `max_tokens` — MUST be 1-2048
- `temperature` — MUST be 0.0-2.0
- `seed` — MUST be valid uint64

**Spec Reference**: WORK-3040

#### [M0-W-1301] Single-Threaded Execution
Worker-orcd MUST process inference requests sequentially (one at a time).

**Concurrency**: batch=1 (no concurrent inference)

**Rationale**: Simplifies M0 implementation. Concurrent inference deferred to M2+.

**Spec Reference**: WORK-3060, SYS-8.2.2

#### [M0-W-1302] Request Validation
Worker-orcd MUST validate all request parameters before starting inference.

**Validation failures**: Return HTTP 400 with error details.

**Spec Reference**: WORK-3120

### 7.2 SSE Streaming

#### [M0-W-1310] Event Types
Worker-orcd MUST stream inference results via Server-Sent Events:

**Event types**:
- `started` — Inference started
- `token` — Token generated
- `metrics` — Metrics snapshot (optional)
- `end` — Inference complete
- `error` — Error occurred

**Spec Reference**: WORK-3050

#### [M0-W-1311] Event Ordering
Worker-orcd MUST emit events in order:

```
started → token* → end
started → token* → error
```

**Exactly one terminal event** MUST be emitted per job (`end` or `error`).

**Spec Reference**: WORK-3052

#### [M0-W-1312] Event Payloads

**started**:
```json
{
  "job_id": "job-xyz",
  "model": "qwen2.5-0.5b-instruct",
  "started_at": "2025-10-03T20:00:00Z"
}
```

**token**:
```json
{
  "t": "GPU",
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

**Spec Reference**: WORK-3051

### 7.3 Health Endpoint

#### [M0-W-1320] GET /health
Worker-orcd MUST expose health endpoint:

**Response**:
```json
{
  "status": "healthy",
  "model": "qwen2.5-0.5b-instruct",
  "vram_bytes": 369098752,
  "uptime_seconds": 3600
}
```

**Status values**:
- `healthy` — Worker operational
- `unhealthy` — VRAM residency check failed or other critical error

**Spec Reference**: WORK-3041

### 7.4 Cancellation Endpoint

#### [M0-SYS-6.3.5] Cancellation Handling
**Parent**: SYS-6.3.5

Worker MUST handle cancellation requests promptly.

#### [M0-W-1330] POST /cancel
Worker-orcd MUST expose cancellation endpoint:

**Request**:
```json
{
  "job_id": "job-xyz"
}
```

**Response**: HTTP 202 Accepted

**Semantics**:
- Idempotent: Repeated cancels for same `job_id` are safe
- Stop decoding promptly (within 100ms)
- Free resources (VRAM buffers)
- Emit SSE `error` event with code `CANCELLED`

**Deadline**: Cancellation MUST complete within 5s.

**Spec Reference**: WORK-3044, SYS-6.3.5

### 7.5 Shutdown Endpoint (Optional)

#### [M0-W-1340] POST /shutdown
Worker-orcd MAY expose graceful shutdown endpoint:

**Request**: Empty body or `{}`

**Response**: HTTP 202 Accepted

**Semantics**:
- Stop accepting new requests (return 503)
- Finish active inference job
- Free VRAM
- Exit with code 0

**Timeout**: MUST complete within 30s (default).

**Alternative**: Worker MUST also handle SIGTERM for graceful shutdown.

**Spec Reference**: WORK-3043, WORK-3100

### 7.6 Metrics Endpoint (Optional)

#### [M0-W-1350] GET /metrics
Worker-orcd SHOULD expose Prometheus metrics endpoint:

**Response**: Prometheus text format

**Metrics** (basic set for M0):
- `worker_vram_bytes` — Current VRAM usage
- `worker_requests_total{outcome}` — Request count by outcome
- `worker_tokens_generated_total` — Total output tokens
- `worker_inference_duration_ms` — Inference latency histogram
- `worker_uptime_seconds` — Worker uptime

**M0 Note**: Full metrics implementation optional. Basic metrics sufficient.

**Spec Reference**: WORK-3042, WORK-3081

---

## 8. CUDA Implementation

### 8.1 Context Management

#### [M0-W-1400] Context Initialization
Worker-orcd MUST initialize CUDA context:

```cpp
Context::Context(int gpu_device) : device_(gpu_device) {
    // Check device exists
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    // Set device
    cudaSetDevice(gpu_device);
    
    // Get device properties
    cudaGetDeviceProperties(&props_, gpu_device);
    
    // Enforce VRAM-only mode
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0);  // Disable UMA
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
}
```

**Spec Reference**: CUDA-5101, CUDA-5120

#### [M0-W-1401] Context Cleanup
Worker-orcd MUST clean up CUDA context on shutdown:

```cpp
Context::~Context() {
    cudaDeviceReset();
}
```

**Spec Reference**: CUDA-5111

### 8.2 Model Loading

#### [M0-W-1410] Model Load Implementation
Worker-orcd MUST load model to VRAM:

```cpp
Model::Model(const Context& ctx, const std::string& path) {
    // 1. Memory-map file
    auto mapped_file = mmap_file(path);
    
    // 2. Parse GGUF format
    parse_gguf(mapped_file);
    
    // 3. Allocate VRAM
    allocate_vram();
    
    // 4. Copy weights to VRAM
    copy_to_vram(mapped_file);
    
    // 5. Unmap file
    munmap_file(mapped_file);
}
```

**Spec Reference**: CUDA-5240

### 8.3 Inference Execution

#### [M0-W-1420] Forward Pass
Worker-orcd MUST execute forward pass:

```cpp
void InferenceResult::run_forward_pass() {
    // 1. Embedding lookup
    embedding_kernel<<<grid, block, 0, stream_>>>(
        model_.weights(), prompt_tokens_.data(), embeddings_
    );
    
    // 2. Transformer layers
    for (int layer = 0; layer < model_.metadata().num_layers; ++layer) {
        // Self-attention
        attention_kernel<<<grid, block, 0, stream_>>>(
            model_.layer_weights(layer), embeddings_, kv_cache_->get(),
            current_token_, attention_output_
        );
        
        // Feed-forward
        ffn_kernel<<<grid, block, 0, stream_>>>(
            model_.layer_weights(layer), attention_output_, embeddings_
        );
    }
    
    // 3. Output projection
    output_kernel<<<grid, block, 0, stream_>>>(
        model_.output_weights(), embeddings_, logits_->get()
    );
    
    // 4. Synchronize
    cudaStreamSynchronize(stream_);
}
```

**Spec Reference**: CUDA-5321

#### [M0-W-1421] Token Sampling
Worker-orcd MUST sample next token:

```cpp
int InferenceResult::sample_token() {
    // Copy logits to host
    std::vector<float> host_logits(model_.metadata().vocab_size);
    cudaMemcpy(host_logits.data(), logits_->get(), 
               host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Apply temperature
    for (float& logit : host_logits) {
        logit /= config_.temperature;
    }
    
    // Softmax
    softmax(host_logits);
    
    // Greedy sampling (M0)
    return std::distance(host_logits.begin(), 
                        std::max_element(host_logits.begin(), host_logits.end()));
}
```

**Spec Reference**: CUDA-5321

### 8.4 CUDA Kernels

#### [M0-W-1430] Required Kernels
Worker-orcd MUST implement M0 kernel set:

**Kernels**:
- cuBLAS GEMM wrapper
- RoPE (llama variant)
- Naive attention (prefill + decode)
- RMSNorm
- Greedy sampling

**Spec Reference**: WORKER-4700

#### [M0-W-1431] Kernel Safety
All kernel launches MUST have:
- Bounds checking
- Dimension validation
- Error handling

**Spec Reference**: KERNEL-SAFE-001 through KERNEL-SAFE-004

---

## 9. Error Handling

### 9.1 Error Codes

#### [M0-W-1500] Stable Error Codes
Worker-orcd MUST use stable error codes:

**Error codes**:
- `INVALID_REQUEST` — Invalid request parameters
- `MODEL_LOAD_FAILED` — Model failed to load
- `INSUFFICIENT_VRAM` — Not enough VRAM for model
- `VRAM_OOM` — Out of VRAM during inference
- `CUDA_ERROR` — CUDA runtime error
- `INFERENCE_TIMEOUT` — Inference exceeded timeout
- `CANCELLED` — Job cancelled
- `INTERNAL` — Internal error

**Spec Reference**: WORK-3090

#### [M0-W-1501] CUDA Error Codes
CUDA layer MUST use integer error codes:

```cpp
enum CudaErrorCode {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_DEVICE_NOT_FOUND = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_INVALID_DEVICE = 3,
    CUDA_ERROR_MODEL_LOAD_FAILED = 4,
    CUDA_ERROR_INFERENCE_FAILED = 5,
    CUDA_ERROR_VRAM_RESIDENCY_FAILED = 6,
    CUDA_ERROR_KERNEL_LAUNCH_FAILED = 7,
    CUDA_ERROR_UNKNOWN = 99,
};
```

**Spec Reference**: CUDA-5040

### 9.2 Error Responses

#### [M0-W-1510] SSE Error Events
Worker-orcd MUST return errors via SSE `error` event:

```json
{
  "code": "VRAM_OOM",
  "message": "Out of VRAM during inference",
  "retriable": false
}
```

**Fields**:
- `code` — Stable error code
- `message` — Human-readable description
- `retriable` — Boolean (true if orchestratord can retry)

**Spec Reference**: WORK-3091

---

## 10. Performance Requirements

### 10.1 Latency Targets

#### [M0-W-1600] First Token Latency
Worker SHOULD emit first token within 100ms of receiving execute request.

**Measurement**: From HTTP POST accept to first SSE `token` event.

**M0 Target**: <100ms (p95)

**Spec Reference**: SYS-8.2.1, Performance audit in §14.1

#### [M0-W-1601] Token Generation Rate
Worker SHOULD generate tokens at 20-100 tokens/sec depending on model.

**M0 Baseline**: Qwen2.5-0.5B-Instruct on RTX 3090: ~60 tok/s

**Spec Reference**: SYS-8.2.1

#### [M0-W-1602] Per-Token Latency
Worker SHOULD emit tokens with inter-token latency of 10-50ms.

**Measurement**: Time between consecutive SSE `token` events.

**M0 Target**: 10-50ms (p95)

**Spec Reference**: Performance audit in §14.1

#### [M0-W-1603] Execute Endpoint Performance
Worker MUST parse execute request in <1ms.

**Optimization**: Use zero-copy JSON parsing, no allocations.

**Rationale**: Hot path optimization (performance audit comment in SYS-5.4.1).

#### [M0-W-1604] Health Endpoint Performance
Worker health endpoint MUST respond in <10ms.

**Implementation**: Use cached state, no CUDA calls in hot path.

**Rationale**: Avoid cascading timeouts in orchestrator health checks.

**Spec Reference**: Performance audit in §14.1

### 10.2 Cancellation Performance

#### [M0-W-1610] Cancellation Latency
Worker MUST stop inference within 100ms of receiving cancel request.

**Breakdown**:
- Detect cancel signal: <10ms
- Abort CUDA kernel: <50ms
- Free VRAM buffers: <40ms

**Total**: <100ms

**Spec Reference**: Performance audit comment in SYS-7.4.x

#### [M0-W-1611] Client Disconnect Detection
Worker MUST detect client disconnect and abort inference immediately.

**Detection**: Check SSE connection status every 10 tokens.

**Abort latency**: <100ms from disconnect to inference stopped.

**Rationale**: Don't waste GPU cycles on abandoned work.

**Spec Reference**: Performance audit in §14.1

### 10.3 Startup Performance

#### [M0-W-1620] Model Loading Time
Worker SHOULD complete model loading within 60s.

**M0 Target**: Qwen2.5-0.5B-Instruct (352MB) loads in <10s on modern GPU.

**Breakdown**:
- CUDA context init: <1s
- GGUF parsing: <1s
- VRAM allocation: <1s
- Copy to VRAM: <7s (352MB at ~50MB/s)

**Spec Reference**: SYS-8.2.1, Performance audit in §14.1

#### [M0-W-1621] Model Loading Progress
Worker SHOULD emit progress events during model loading.

**Progress points**: 0%, 25%, 50%, 75%, 100%

**Format**: Log events with `event="model_load_progress"` and `percent` field.

**Rationale**: Observability for long-running model loads.

**Spec Reference**: Performance audit in §14.1

### 10.4 Shutdown Performance

#### [M0-W-1630] Graceful Shutdown Deadline
Worker MUST complete graceful shutdown within 5s.

**Breakdown**:
- Stop accepting requests: <10ms
- Finish active inference: <4s
- Free VRAM: <1s

**Timeout**: After 5s, pool manager will SIGKILL.

**Spec Reference**: WORK-3101, Performance audit in §14.1

---

## 11. Build System

### 11.1 CUDA Feature Flag

#### [M0-W-1700] Opt-in CUDA Feature
Worker-orcd MUST support building with or without CUDA:

```bash
# Without CUDA (stub mode, development)
cargo build -p worker-orcd

# With CUDA (production)
cargo build -p worker-orcd --features cuda
```

**Spec Reference**: `.docs/CUDA_FEATURE_FLAG_IMPLEMENTATION.md`

#### [M0-W-1701] Local Configuration
Worker-orcd MUST support local configuration via `.llorch.toml`:

```toml
[build]
cuda = true
auto_detect_cuda = false
```

**Spec Reference**: `BUILD_CONFIGURATION.md`

#### [M0-W-1702] CMake Integration
Worker-orcd MUST use CMake to compile C++/CUDA code when `cuda` feature is enabled.

**Build script**: `build.rs`

**CMake config**: `cuda/CMakeLists.txt`

---

## 12. Testing Strategy

### 12.1 Haiku Test

#### [M0-W-1800] Haiku Generation Test
M0 success criteria MUST include haiku generation test:

**Test case**:
```rust
#[test]
fn test_haiku_generation() {
    let worker = start_worker("qwen2.5-0.5b-instruct-q4_k_m.gguf", 0);
    
    let response = worker.execute(ExecuteRequest {
        job_id: "test-haiku-001".to_string(),
        prompt: "Write a haiku about GPU computing".to_string(),
        max_tokens: 50,
        temperature: 0.7,
        seed: 42,
    });
    
    // Verify SSE stream
    assert_event!(response, "started");
    let tokens = collect_tokens!(response);
    assert!(tokens.len() > 0);
    assert_event!(response, "end");
    
    // Verify determinism
    let response2 = worker.execute(/* same request */);
    let tokens2 = collect_tokens!(response2);
    assert_eq!(tokens, tokens2);
}
```

**Verification**:
- Worker loads model successfully
- Generates haiku (3 lines)
- Streams tokens via SSE
- Deterministic with same seed
- All operations VRAM-only

**Spec Reference**: §14.1 M0 milestone in parent spec

### 12.2 Unit Tests

#### [M0-W-1810] CUDA Unit Tests
Each CUDA module MUST have unit tests:

**Tests**:
- Context initialization
- Model loading
- Inference execution
- Health checks
- Kernel validation

**Framework**: Google Test

**Location**: `cuda/tests/`

#### [M0-W-1811] Rust Unit Tests
Rust layer MUST have unit tests:

**Tests**:
- HTTP handlers (with mocked CUDA)
- Error conversion
- SSE streaming
- Request validation

**Framework**: Rust `#[test]`

**Location**: `tests/`

### 12.3 Integration Tests

#### [M0-W-1820] End-to-End Test
M0 MUST have integration test:

**Test flow**:
1. Start worker with test model
2. Send inference request
3. Verify SSE stream
4. Verify deterministic output
5. Verify VRAM-only operation
6. Shutdown gracefully

**Location**: `tests/integration_test.rs`

### 12.4 Performance Tests

#### [M0-W-1830] Performance Test Suite
M0 MUST include performance verification tests:

**Tests**:
1. **First token latency** — Measure POST /execute to first token (target <100ms p95)
2. **Per-token latency** — Measure inter-token timing (target 10-50ms p95)
3. **Model loading time** — Measure cold start to ready (target <60s)
4. **Health endpoint latency** — Measure /health response time (target <10ms p99)
5. **Graceful shutdown** — Measure SIGTERM to exit (target <5s)
6. **Client disconnect abort** — Measure SSE close to inference stop (target <100ms)
7. **Memory leak test** — Run 100 requests, verify VRAM returns to baseline

**Output**: Proof bundle with timing histograms (p50, p95, p99).

**Pass/Fail**: Automated comparison against targets with 10% tolerance.

**Spec Reference**: Performance audit in §14.1 of parent spec

### 12.5 Proof Bundle Requirements

#### [M0-W-1840] Proof Bundle Emission
All M0 tests MUST emit proof bundles per `libs/proof-bundle` standard:

**Location**: `bin/worker-orcd/.proof_bundle/<type>/<run_id>/`

**Contents**:
- `seeds.ndjson` — Test seeds for reproducibility
- `metadata.json` — Test metadata (model, GPU, CUDA version)
- `transcript.ndjson` — SSE event stream
- `timings.json` — Performance measurements
- `README.md` — Autogenerated header per PB-1012

**Environment variables**:
- `LLORCH_RUN_ID` — Run identifier
- `LLORCH_PROOF_DIR` — Override default location

**Spec Reference**: Memory about proof-bundle standard

---

## 13. Observability & Logging

### 13.1 Structured Logging

#### [M0-W-1900] Log Fields
Worker-orcd MUST emit structured logs with fields:

**Required fields**:
- `worker_id` — Worker identifier
- `job_id` — Job identifier (when applicable)
- `model_ref` — Model reference
- `gpu_device` — GPU device ID
- `event` — Event type

**Event types**:
- `startup` — Worker starting
- `model_load_start` — Model loading begins
- `model_load_progress` — Loading progress (0-100%)
- `model_load_complete` — Model loaded successfully
- `ready` — Worker ready for requests
- `execute_start` — Inference request received
- `execute_end` — Inference completed
- `error` — Error occurred
- `shutdown` — Worker shutting down

**Spec Reference**: WORK-3080

#### [M0-W-1901] Performance Metrics in Logs
Worker-orcd SHOULD include performance metrics in logs:

**Fields**:
- `vram_bytes` — VRAM usage
- `tokens_in` — Input tokens
- `tokens_out` — Output tokens
- `decode_time_ms` — Inference time
- `first_token_ms` — First token latency

**Spec Reference**: WORK-3080

#### [M0-W-1902] Sensitive Data Handling
Worker-orcd MUST NOT log sensitive data:

**Prohibited**:
- ❌ Raw prompts (may contain PII)
- ❌ Generated text (may contain sensitive output)
- ❌ API tokens or secrets

**Allowed**:
- ✅ Prompt hash (SHA-256)
- ✅ Prompt length (character count)
- ✅ Token counts

**Spec Reference**: WORK-3122

---

## 14. Gaps & Clarifications

### 14.1 Identified Gaps

The following gaps require clarification or implementation:

1. **Tokenization Library**: Spec mentions tokenization but doesn't specify library (llama.cpp tokenizer? sentencepiece? tiktoken?)
   - **Decision needed**: Which tokenizer for GGUF models?
   - **Impact**: Affects determinism and compatibility

2. **Detokenization**: How are token IDs converted to UTF-8 strings for SSE streaming?
   - **Decision needed**: Streaming detokenization strategy
   - **Impact**: Affects SSE event payload format

3. **Quantization Dequantization**: Q4_0 dequantization strategy not specified
   - **Decision needed**: Dequantize on load vs on-the-fly?
   - **M0 approach**: Dequantize on load (simpler, uses more VRAM)

4. **Attention Implementation**: Prefill vs decode optimization details missing
   - **Decision needed**: Separate kernels or unified?
   - **M0 approach**: Naive implementation, optimization deferred

5. **Sampling Strategy**: Top-k/top-p mentioned as optional—clarify M0 scope
   - **Decision**: M0 implements greedy sampling only
   - **Deferred**: Top-k/top-p to M2+

6. **Memory Alignment**: 256-byte alignment mentioned but not enforced in specs
   - **Decision needed**: Add alignment validation in model loading?
   - **Impact**: Performance optimization

7. **Stream Synchronization**: When to sync CUDA streams? After each kernel? Per token?
   - **Decision needed**: Synchronization strategy
   - **M0 approach**: Sync after full forward pass (per token)

8. **X-Deadline Header**: Should M0 parse deadline headers?
   - **Decision**: M0 SHOULD parse and log X-Deadline header (establish pattern)
   - **Enforcement**: Deferred to M2+ (M0 logs only)

9. **Request Timeout**: What happens if client disconnects during inference?
   - **Decision**: M0-W-1611 specifies abort on disconnect
   - **Implementation**: Check connection every 10 tokens

10. **Metrics Scraping**: Full Prometheus integration or basic only?
    - **Decision**: M0 implements basic metrics (optional)
    - **Full integration**: Deferred to M2+

### 14.2 Contradictions Resolved

**Contradiction 1**: Worker ready callback
- **Parent spec** (SYS-6.3.4): "Worker MUST issue ready callback"
- **M0 clarification**: Callback OPTIONAL for M0 (standalone mode), REQUIRED for M1+
- **Resolution**: M0-SYS-6.3.1 clarifies callback is M1+ feature

**Contradiction 2**: Worker concurrency
- **Parent spec** (SYS-8.2.2): "Worker concurrency for M0 MUST be 1"
- **Performance audit**: Mentions batching
- **Resolution**: M0-W-1301 confirms batch=1, batching deferred to M2+

**Contradiction 3**: Metrics endpoint
- **Parent spec** (WORK-3042): "Worker SHOULD expose /metrics"
- **M0 scope**: Full metrics deferred
- **Resolution**: M0 implements basic health endpoint only, full metrics M2+

### 14.3 Deferred to Post-M0

- FlashAttention (M2+)
- Continuous Batching (M2+)
- Tensor Parallelism (M4)
- Advanced Quantization (M2+)
- PagedAttention (M2+)
- Speculative Decoding (M3+)
- Authentication (M3+)
- Full Prometheus Metrics (M2+)

---

## 15. Acceptance Criteria

### 15.1 M0 Success Criteria

M0 is considered complete when:

1. ✅ Worker binary compiles successfully with `--features cuda`
2. ✅ Worker loads Qwen2.5-0.5B-Instruct (352MB) into VRAM
3. ✅ Worker accepts HTTP POST /execute request
4. ✅ Worker generates haiku deterministically (same seed → same output)
5. ✅ Worker streams tokens via SSE in correct order
6. ✅ Worker enforces VRAM-only (no RAM fallback detected)
7. ✅ Worker responds to GET /health with status
8. ✅ Worker handles POST /cancel gracefully
9. ✅ Worker shuts down gracefully on SIGTERM
10. ✅ All CUDA unit tests pass
11. ✅ All Rust unit tests pass
12. ✅ Integration test passes (end-to-end haiku generation)

### 15.2 Non-Goals for M0

- ❌ Pool manager integration
- ❌ Orchestrator integration
- ❌ Multi-model support
- ❌ Multi-GPU support
- ❌ Authentication/authorization
- ❌ Full metrics/observability
- ❌ Production-grade error recovery
- ❌ Advanced kernel optimizations

---

### 15.3 Performance Exit Criteria

M0 performance targets (from performance audit in parent §14.1):

1. ✅ First token latency p95 <100ms (measured over 10 haiku requests)
2. ✅ Per-token latency p95 <50ms (measured over 10 haiku requests)
3. ✅ Health endpoint p99 <10ms (measured over 100 requests)
4. ✅ Model loading time <60s (Qwen2.5-0.5B-Instruct cold start)
5. ✅ Graceful shutdown <5s (SIGTERM to exit)
6. ✅ Zero memory leaks (VRAM returns to baseline after 100 requests)
7. ✅ Client disconnect abort <100ms (SSE close to inference stop)

**Measurement**: Performance test suite (M0-W-1830) with proof bundle outputs.

**Tolerance**: 10% deviation acceptable for M0.

---

## 16. Traceability Matrix

| M0 Requirement | Parent Spec | Source | Status |
|----------------|-------------|--------|--------|
| M0-SYS-6.3.1 | SYS-6.3.1 | Worker Self-Containment | ✅ Specified |
| M0-SYS-6.3.2 | SYS-6.3.2 | Worker Isolation | ✅ Specified |
| M0-SYS-6.3.5 | SYS-6.3.5 | Cancellation Handling | ✅ Specified |
| M0-SYS-2.2.1 | SYS-2.2.1 | VRAM-Only Enforcement | ✅ Specified |
| M0-SYS-2.3.1 | SYS-2.3.1 | Determinism Guarantee | ✅ Specified |
| M0-SYS-2.4.1 | SYS-2.4.1 | Process Isolation | ✅ Specified |
| M0-SYS-2.5.1 | SYS-2.5.1 | FFI Boundary | ✅ Specified |
| M0-W-1001 | New | Single Model Lifetime | ✅ Specified |
| M0-W-1010 | CUDA-5102 | CUDA Context Config | ✅ Specified |
| M0-W-1030 | CUDA-5350 | Seeded RNG | ✅ Specified |
| M0-W-1100 | WORK-3010 | CLI Arguments | ✅ Specified |
| M0-W-1200 | WORK-3031 | GGUF Format | ✅ Specified |
| M0-W-1300 | WORK-3040 | POST /execute | ✅ Specified |
| M0-W-1310 | WORK-3050 | SSE Streaming | ✅ Specified |
| M0-W-1430 | WORKER-4700 | M0 Kernels | ✅ Specified |
| M0-W-1600 | SYS-8.2.1 | First Token Latency | ✅ Specified |
| M0-W-1700 | Custom | CUDA Feature Flag | ✅ Implemented |
| M0-W-1800 | Custom | Haiku Test | ✅ Specified |

---

## 17. API Completeness Checklist

### 17.1 HTTP Endpoints

| Endpoint | Method | M0 Status | Spec ID | Notes |
|----------|--------|-----------|---------|-------|
| `/execute` | POST | ✅ Required | M0-W-1300 | Inference execution |
| `/health` | GET | ✅ Required | M0-W-1320 | Health check |
| `/cancel` | POST | ✅ Required | M0-W-1330 | Job cancellation |
| `/shutdown` | POST | ⚠️ Optional | M0-W-1340 | Graceful shutdown |
| `/metrics` | GET | ⚠️ Optional | M0-W-1350 | Prometheus metrics |

**M0 API Surface**: 3 required endpoints, 2 optional.

### 17.2 SSE Event Types

| Event Type | M0 Status | Spec ID | Payload |
|------------|-----------|---------|----------|
| `started` | ✅ Required | M0-W-1312 | `{job_id, model, started_at}` |
| `token` | ✅ Required | M0-W-1312 | `{t, i}` |
| `end` | ✅ Required | M0-W-1312 | `{tokens_out, decode_time_ms}` |
| `error` | ✅ Required | M0-W-1312 | `{code, message}` |
| `metrics` | ⚠️ Optional | M0-W-1310 | Performance snapshot |

**M0 Event Surface**: 4 required event types, 1 optional.

### 17.3 CUDA FFI Functions

| Function | M0 Status | Spec ID | Purpose |
|----------|-----------|---------|----------|
| `cuda_init` | ✅ Required | M0-W-1400 | Initialize context |
| `cuda_destroy` | ✅ Required | M0-W-1401 | Cleanup context |
| `cuda_get_device_count` | ✅ Required | CUDA-5112 | Query devices |
| `cuda_load_model` | ✅ Required | M0-W-1410 | Load model to VRAM |
| `cuda_unload_model` | ✅ Required | CUDA-5211 | Free model |
| `cuda_model_get_vram_usage` | ✅ Required | CUDA-5212 | Query VRAM usage |
| `cuda_inference_start` | ✅ Required | CUDA-5310 | Start inference |
| `cuda_inference_next_token` | ✅ Required | CUDA-5311 | Generate token |
| `cuda_inference_free` | ✅ Required | CUDA-5312 | Free inference |
| `cuda_check_vram_residency` | ✅ Required | CUDA-5410 | Verify VRAM-only |
| `cuda_get_vram_usage` | ✅ Required | CUDA-5411 | Query VRAM |
| `cuda_get_process_vram_usage` | ✅ Required | CUDA-5411 | Query process VRAM |

**M0 FFI Surface**: 12 required C functions.

### 17.4 Error Codes

| Error Code | M0 Status | Spec ID | HTTP Status |
|------------|-----------|---------|-------------|
| `INVALID_REQUEST` | ✅ Required | M0-W-1500 | 400 |
| `MODEL_LOAD_FAILED` | ✅ Required | M0-W-1500 | 500 |
| `INSUFFICIENT_VRAM` | ✅ Required | M0-W-1500 | 503 |
| `VRAM_OOM` | ✅ Required | M0-W-1500 | 500 |
| `CUDA_ERROR` | ✅ Required | M0-W-1500 | 500 |
| `INFERENCE_TIMEOUT` | ✅ Required | M0-W-1500 | 504 |
| `CANCELLED` | ✅ Required | M0-W-1500 | 499 |
| `INTERNAL` | ✅ Required | M0-W-1500 | 500 |

**M0 Error Surface**: 8 stable error codes.

---

## 18. References

### 18.1 Parent Specifications

- `bin/.specs/00_llama-orch.md` — **LEADING SPEC** (system architecture)
- `bin/worker-orcd/.specs/00_worker-orcd.md` — Worker specification (WORK-3xxx)
- `bin/worker-orcd/.specs/01_cuda_ffi_boundary.md` — FFI boundary (CUDA-4xxx)

### 18.2 CUDA Module Specifications

- `bin/worker-orcd/cuda/.specs/00_cuda_overview.md` — CUDA overview (CUDA-5xxx)
- `bin/worker-orcd/cuda/.specs/01_context.md` — Context management (CUDA-5100)
- `bin/worker-orcd/cuda/.specs/02_model.md` — Model loading (CUDA-5200)
- `bin/worker-orcd/cuda/.specs/03_inference.md` — Inference execution (CUDA-5300)
- `bin/worker-orcd/cuda/.specs/04_health.md` — Health monitoring (CUDA-5400)
- `bin/worker-orcd/cuda/kernels/.specs/00_cuda-kernels.md` — CUDA kernels (WORKER-4700)

### 18.3 Documentation

- `bin/worker-orcd/README.md` — Worker overview
- `bin/worker-orcd/CUDA_FEATURE.md` — CUDA feature flag
- `.docs/CUDA_FEATURE_FLAG_IMPLEMENTATION.md` — Implementation guide
- `BUILD_CONFIGURATION.md` — Build configuration
- `.docs/testing/TEST_MODELS.md` — Test models

---

**End of M0 Specification**

**Status**: Draft — Ready for implementation  
**Next Steps**: 
1. Review gaps in §13.1
2. Implement CUDA modules (context, model, inference, health)
3. Implement Rust HTTP layer
4. Implement FFI boundary
5. Write unit tests
6. Write integration tests
7. Execute haiku test
