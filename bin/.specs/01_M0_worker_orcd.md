# M0: Worker-orcd Complete Specification

**Status**: Draft (Hybrid Scope - Performance Bundle Deferred)  
**Milestone**: M0 (v0.1.0) — Worker Haiku Test  
**Version**: 0.1.0  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)  
**Timeline**: 6-7 weeks (4-5 weeks foundation + 1-2 weeks architecture adapters)

---

## 0. Document Metadata

### 0.0 Scope Decision Summary (Hybrid Approach)

**Decision Date**: 2025-10-03  
**Approach**: Performance Bundle Deferral (Hybrid)  
**Rationale**: Balance faster delivery (4-5 weeks) with critical safety features

#### Scope Changes from Original M0

**DEFERRED to M1+ (14 items - Performance Bundle)**:
1. ✅ Prometheus metrics endpoint (M0-W-1350)
2. ✅ Performance metrics in logs (M0-W-1901)
3. ✅ Graceful shutdown endpoint (M0-W-1340)
4. ✅ Graceful shutdown performance target (M0-W-1630)
5. ✅ First token latency target (M0-W-1600)
6. ✅ Token generation rate target (M0-W-1601)
7. ✅ Per-token latency target (M0-W-1602)
8. ✅ Execute endpoint performance (M0-W-1603)
9. ✅ Health endpoint performance (M0-W-1604)
10. ✅ Cancellation latency target (M0-W-1610)
11. ✅ Client disconnect detection (M0-W-1611)
12. ✅ Model loading time target (M0-W-1620)
13. ✅ Performance test suite (M0-W-1830) - comprehensive perf validation
14. ✅ Deep CUDA determinism audit (kernel scheduling, atomics) (M0-W-1031)
15. ✅ Sensitive data handling in logs (M0-W-1902)

**Note**: M0 includes minimal same-device reproducibility check only (seeded RNG, temp=0)

**KEPT in M0 (18 items - Core + Critical Safety + Architecture Adapters)**:
1. ✅ All 3 models: Qwen2.5-0.5B, Phi-3-Mini, GPT-OSS-20B
2. ✅ All 3 quantization formats: Q4_K_M, MXFP4, Q4_0
3. ✅ 2 tokenizer backends: GGUF byte-BPE, tokenizer.json
4. ✅ Architecture adapters: LlamaInferenceAdapter + GPTInferenceAdapter (M0-W-1213, M0-W-1214, M0-W-1215)
5. ✅ Architecture detection from GGUF metadata (M0-W-1212)
6. ✅ GPT-specific kernels: LayerNorm, GELU, absolute pos embedding (M0-W-1432, M0-W-1433, M0-W-1434)
7. ✅ MXFP4 architecture-aware weight mapping (M0-W-1435)
8. ✅ Narration-core logging (replaces structured logging, basic events only)
9. ✅ Model load progress events (M0-W-1621) ← **CRITICAL** (user feedback)
10. ✅ VRAM OOM handling (M0-W-1021) ← **CRITICAL** (safety)
11. ✅ VRAM residency verification (M0-W-1012) ← **CRITICAL** (runtime leak detection)
12. ✅ Minimal same-device reproducibility check (seeded RNG, temp=0)
13. ✅ MXFP4 numerical correctness validation (M0-W-1822) ← **CRITICAL** (correctness)
14. ✅ Memory-mapped I/O (M0-W-1221) - REQUIRED for all models
15. ✅ Chunked VRAM transfer (M0-W-1222) - REQUIRED for all models
16. ✅ CUDA unit tests (functional only, no performance tests)
17. ✅ Kernel safety validation (M0-W-1431)
18. ✅ Temperature scaling 0.0-2.0 (M0-W-1032)

**ADDED in M0 Sprint 3 (Advanced Generation Parameters - 2025-10-04)**:
19. ✅ Top-P (nucleus) sampling (M0-W-1421) ← **COMPETITIVE PARITY**
20. ✅ Top-K sampling (M0-W-1421) ← **COMPETITIVE PARITY**
21. ✅ Repetition penalty (M0-W-1421) ← **COMPETITIVE PARITY**
22. ✅ Stop sequences (M0-W-1422) ← **CRITICAL** (structured output)
23. ✅ Min-P sampling (M0-W-1421, optional) ← **COMPETITIVE PARITY**
24. ✅ HTTP API extension with 5 new parameters (M0-W-1300)

**Rationale for Sprint 3 Expansion**:
- User expectations: APIs like OpenAI (10 params), llama.cpp (12 params), LM Studio (13 params) have these as standard
- Efficiency: Adding during sampling implementation (Sprint 3) is more efficient than deferring to M1
- Structured output: Stop sequences are critical for JSON, code generation
- Backward compatible: All new parameters optional with sensible defaults
- Timeline impact: +1 day to Sprint 3 (16 → 17 days)

**Analysis Reference**: See `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md` for comprehensive gap analysis

**REMOVED from Repo**:
- 🔥 Proof bundles (entire concept - all references to be removed)

#### Key Trade-offs

**Benefits**:
- ✅ All 3 models supported with correct architecture-specific pipelines
- ✅ Critical safety features retained (VRAM monitoring, OOM handling)
- ✅ User experience retained (model load progress events)
- ✅ All 3 quantization formats included (Q4_K_M, MXFP4, Q4_0)
- ✅ Clean architecture from day 1 (InferenceAdapter pattern)

**Deferred to M1**:
- ❌ Performance validation and benchmarking
- ❌ Reproducibility proof (implementation done, validation deferred)
- ❌ Graceful shutdown (rely on SIGTERM)
- ❌ Performance metrics collection

**Timeline Impact**:
- Foundation (Weeks 1-5): HTTP server, GGUF loader, tokenization, basic kernels
- Architecture Adapters (Weeks 6-7): InferenceAdapter pattern, Llama + GPT adapters, GPT-specific kernels
- Total: 6-7 weeks (includes +1-2 weeks for proper architecture support)

**Reference**: See `M0_RESOLUTION_CONTRADICTIONS.md` for full analysis

---

## 0. Document Metadata

### 0.1 Purpose

This specification consolidates **ALL M0 requirements** for the `worker-orcd` binary. M0 is the foundational milestone that delivers a standalone GPU worker capable of loading a single model and executing inference with deterministic, VRAM-only operation.

**M0 Goal**: Prove the worker can load a model, execute inference, and stream results—standalone, without orchestrator or pool-manager dependencies.

**Success Criteria**: 
- Worker loads Qwen2.5-0.5B-Instruct (352MB GGUF) into VRAM
- Executes a fixed haiku prompt with (seeded RNG, temperature=0) and produces identical token IDs across two runs on the same device
- Streams tokens via SSE
- All operations VRAM-only (no RAM fallback)

### 0.2 Scope

**In Scope for M0**:
- ✅ Single worker binary (`worker-orcd`)
- ✅ Single model loading (GGUF format)
- ✅ Single GPU support (no tensor parallelism)
- ✅ VRAM-only enforcement
- ✅ Quantized-only execution (Q4_K_M, MXFP4, Q4_0 - NO dequantization to FP32)
- ✅ HTTP API (execute, health, cancel)
- ✅ SSE streaming with UTF-8 boundary safety
- ✅ Tokenization: Two backends (`hf-json` for GPT-OSS-20B, `gguf-bpe` for Qwen/Phi-3)
- ✅ Architecture adapters: Llama-style (Qwen/Phi-3) and GPT-style (GPT-OSS-20B) inference pipelines
- ✅ Test reproducibility (seeded RNG, temp=0 for testing; temp 0.0-2.0 for production)
- ✅ CUDA FFI boundary (Rust ↔ C++/CUDA)
- ✅ Architecture-specific CUDA kernels:
  - Llama-style: RoPE, GQA attention, RMSNorm, SwiGLU
  - GPT-style: Absolute pos embedding, MHA attention, LayerNorm, GELU
  - Shared: GEMM (cuBLAS), embedding lookup, sampling
- ✅ Standalone testing (no orchestrator required)

**M0 Reference Target Models**:
1. **Qwen2.5-0.5B-Instruct** (GGUF, Q4_K_M) — Primary bring-up & smoke test
2. **Phi-3-Mini (~3.8B) Instruct** (GGUF, Q4_K_M) — Stretch target within 24 GB
3. **GPT-OSS-20B** (GGUF, MXFP4) — Trend-relevant large model

**Out of Scope for M0**:
- ❌ Pool manager integration (M1)
- ❌ Orchestrator integration (M2)
- ❌ Multi-GPU / tensor parallelism (M4)
- ❌ Multi-model support
- ❌ Advanced kernels (FlashAttention, continuous batching)
- ❌ Authentication/authorization
- ❌ Performance metrics/observability (deferred to M1 - hybrid scope)
- ❌ Performance test suite (deferred to M1 - hybrid scope)
- ❌ Graceful shutdown endpoint (deferred to M1 - hybrid scope)
- ❌ Client disconnect detection (deferred to M1 - hybrid scope)
- ❌ Reproducible kernels validation (implementation in M0, validation in M1)
- ❌ Proof bundles (removed from repo)

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

**Testing vs Product**: Temperature 0.0 (greedy) is used for reproducibility in TESTING, not as a product constraint. Production supports temperature 0.0-2.0.

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

**Implementation**:
```cpp
bool Health::check_vram_residency(const Model& model) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, model.weights());
    
    if (err != cudaSuccess) {
        return false;
    }
    
    // Verify pointer is device memory (not managed/host)
    if (attrs.type != cudaMemoryTypeDevice) {
        return false;
    }
    
    // Verify no host pointer (no UMA)
    if (attrs.hostPointer != nullptr) {
        return false;
    }
    
    return true;
}
```

**Process VRAM Usage Query**:
```cpp
uint64_t Health::get_process_vram_usage(const Context& ctx) {
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    
    if (err != cudaSuccess) {
        return 0;
    }
    
    return total_bytes - free_bytes;
}
```

**Spec Reference**: CUDA-5401, CUDA-5421 (health module)

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

## 3. Test Reproducibility (NOT a Product Guarantee)

### 3.1 System Requirements

#### [M0-SYS-2.3.1] Test Reproducibility (NOT a Product Guarantee)
**Parent**: SYS-2.3.1

The system MUST provide reproducibility for testing: same model + same seed + temp=0 + same prompt → same output (for validation only, NOT a product guarantee).

**Requirements**:
- Sealed VRAM shards (worker-orcd)
- Pinned engine versions
- Reproducible sampling for testing
- No non-deterministic operations in system code

**Design Principle**: The system provides test reproducibility (temp=0 + same seed → same output), but this is NOT a product promise. Models cannot guarantee deterministic behavior due to model architecture and hardware limitations. Temperature-based sampling (0.0-2.0) is the product feature.

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

#### [M0-W-1031] Reproducible CUDA Kernels for Testing
All CUDA kernels MUST be reproducible for testing:
- No atomics with race conditions
- No non-deterministic reductions
- Fixed execution order
- Disable non-deterministic cuBLAS algorithms

**cuBLAS Configuration**:
```cpp
cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);  // Disable Tensor Cores if non-deterministic
```

**Verification**: Property test MUST verify kernel reproducibility via repeated execution.

**Spec Reference**: CUDA-5351 (inference module)

#### [M0-W-1032] Temperature Scaling (Product Feature)
Worker-orcd MUST apply temperature scaling:
```cpp
for (float& logit : host_logits) {
    logit /= config_.temperature;
}
```

**Temperature range**: 0.0 to 2.0 (product feature)
- 0.0 = greedy (argmax) - used for testing reproducibility
- 1.0 = no scaling
- >1.0 = more random

**Verification**: Unit test MUST verify temperature=0.0 produces greedy sampling for test reproducibility.

**Spec Reference**: KERNEL-SAMPLE-003

### 3.3 Limitations

#### [M0-W-1040] Model-Level Non-Determinism
Worker-orcd acknowledges model-level limitations:
- Inference engines (llama.cpp) MAY have non-deterministic operations
- GPU hardware variations MAY produce different floating-point results
- Cross-worker/cross-GPU reproducibility is NOT guaranteed

**Documentation**: README MUST document which models/engines have been verified as reproducible for testing.

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
int cuda_get_device_count();

// Model loading
CudaModel* cuda_load_model(
    CudaContext* ctx,
    const char* model_path,
    uint64_t* vram_bytes_used,
    int* error_code
);
void cuda_unload_model(CudaModel* model);
uint64_t cuda_model_get_vram_usage(CudaModel* model);

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
    int token_buffer_size,
    int* token_index,
    int* error_code
);
void cuda_inference_free(InferenceResult* result);

// Health
bool cuda_check_vram_residency(CudaModel* model, int* error_code);
uint64_t cuda_get_vram_usage(CudaModel* model);
uint64_t cuda_get_process_vram_usage(CudaContext* ctx);
bool cuda_check_device_health(CudaContext* ctx, int* error_code);

// Error messages
const char* cuda_error_message(int error_code);
```

**Exception-to-Error-Code Pattern**:
```cpp
extern "C" CudaModel* cuda_load_model(..., int* error_code) {
    try {
        auto model = std::make_unique<Model>(...);
        *error_code = CUDA_SUCCESS;
        return reinterpret_cast<CudaModel*>(model.release());
    } catch (const CudaError& e) {
        *error_code = e.code();
        return nullptr;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    }
}
```

**Spec Reference**: CUDA-4030, CUDA-5040

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
    let error_msg = unsafe { 
        CStr::from_ptr(cuda_error_message(error_code)).to_string_lossy() 
    };
    eprintln!("CUDA init failed: {} (code: {})", error_msg, error_code);
    std::process::exit(1);
}
```

**CUDA Context Implementation** (C++):
```cpp
Context::Context(int gpu_device) : device_(gpu_device) {
    // Check device exists
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (gpu_device < 0 || gpu_device >= device_count) {
        throw CudaError(CUDA_ERROR_INVALID_DEVICE, "Invalid device ID");
    }
    
    // Set device
    cudaSetDevice(gpu_device);
    
    // Get device properties
    cudaGetDeviceProperties(&props_, gpu_device);
    
    // Enforce VRAM-only mode
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0);  // Disable UMA
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
}
```

**2. Load model to VRAM**:
```rust
let mut vram_bytes = 0;
let model = unsafe {
    cuda_load_model(ctx, model_path_cstr, &mut vram_bytes, &mut error_code)
};
if model.is_null() {
    let error_msg = unsafe { 
        CStr::from_ptr(cuda_error_message(error_code)).to_string_lossy() 
    };
    eprintln!("Model load failed: {} (code: {})", error_msg, error_code);
    unsafe { cuda_destroy(ctx); }
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

**Spec Reference**: WORK-3011, CUDA-5101, CUDA-5120

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

**GGUF Version**: Version 3 (required for MXFP4 tensor support)

**Magic bytes**: `0x47475546` ("GGUF")

**GGUF v3 Features**:
- MXFP4 tensor blocks (GPT-OSS-20B)
- Extended metadata for tokenizer configuration
- Improved tensor alignment specifications

**Spec Reference**: WORK-3031, CUDA-5202

#### [M0-W-1201] Quantized-Only Execution
Worker-orcd MUST execute models in their quantized formats for M0.

**Supported quantization formats**:
- **MXFP4** (GPT-OSS-20B primary)
- **Q4_K_M** (Qwen2.5-0.5B-Instruct, Phi-3-Mini; GPT-OSS-20B fallback)
- **Q4_0** (fallback compatibility)

**Loader Policy** (no dequantize-on-load):
Model weights remain quantized in VRAM (MXFP4, Q4_K_M, Q4_0). The loader MUST NOT materialize FP32 copies of weight tensors in device memory.

**Compute Policy** (on-the-fly dequantization):
Kernels dequantize weight tiles to registers/shared memory during compute and accumulate in FP16. This preserves quantized storage while enabling correct math in GEMM/attention. `/health.quant_kind` always reflects the stored format.

**Execution Details**:
- ✅ In-kernel dequantization for MXFP4 tiles/groups → registers/shared memory → FP16 accumulate
- ✅ FP16 accumulation for all matmul results
- ✅ FP16 KV cache precision (stated explicitly)

**MXFP4 Compute Path** (GPT-OSS-20B):
MXFP4 weights MUST be wired into all weight consumers:
1. **Embeddings**: MXFP4 embedding matrix lookup
2. **Attention**: Q/K/V projections, attention matmul, output projection (prefill + decode)
3. **FFN**: Up projection, activation, down projection
4. **LM Head**: Final logits projection

**KV Cache**: FP16 precision for all models

**Fallback Behavior**:
- If MXFP4 artifact unavailable for GPT-OSS-20B, accept Q4_K_M or Q4_0
- `/health.quant_kind` MUST reflect actually loaded quantization

**Rationale**: Matches local runtime behavior (LM Studio/llama.cpp GGUF execution) and aligns with GPT-OSS-20B guidance (~16 GB VRAM operation in MXFP4).

**Future**: Q5_1, Q8_0, INT8 support deferred to M2+.

**Spec Reference**: WORKER-4701, Management directive 2025-10-03

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

**Tensor Format Structure**:
```cpp
struct GGUFTensor {
    std::string name;
    uint32_t n_dimensions;
    uint64_t dimensions[4];
    uint32_t type;  // FP32, FP16, Q8_0, MXFP4, etc.
    uint64_t offset;
};
```

**Spec Reference**: CUDA-5280, CUDA-5281, CUDA-5282

#### [M0-W-1212] Architecture Detection from GGUF

Worker MUST detect model architecture from GGUF metadata and select appropriate InferenceAdapter.

**Detection Logic**:
```cpp
Architecture detect_architecture(const GGUFMetadata& metadata) {
    std::string arch = metadata.get_string("general.architecture");
    if (arch == "llama") return Architecture::Llama;  // Qwen, Phi-3
    if (arch == "gpt2" || arch == "gpt") return Architecture::GPT;  // GPT-OSS-20B
    throw std::runtime_error("Unsupported architecture: " + arch);
}
```

**Supported Architectures**:
- `llama` → LlamaInferenceAdapter (Qwen2.5-0.5B, Phi-3-Mini)
- `gpt2`/`gpt` → GPTInferenceAdapter (GPT-OSS-20B)

**Unsupported Architecture**: Worker MUST fail fast with clear error message.

**Spec Reference**: M0_ARCHITECTURAL_GAP_ANALYSIS.md (Gap 1)

### 6.3 VRAM Allocation

#### [M0-W-1220] Model Weights Allocation
Worker-orcd MUST allocate VRAM for model weights:

```cpp
size_t total_size = calculate_total_size();  // From GGUF tensors
weights_ = std::make_unique<DeviceMemory>(total_size);
vram_bytes_ = total_size;
```

**DeviceMemory RAII Wrapper**:
```cpp
class DeviceMemory {
public:
    explicit DeviceMemory(size_t bytes) {
        cudaError_t err = cudaMalloc(&ptr_, bytes);
        if (err != cudaSuccess) {
            throw CudaError(CUDA_ERROR_OUT_OF_MEMORY, "Failed to allocate VRAM");
        }
        size_ = bytes;
    }
    
    ~DeviceMemory() {
        if (ptr_) cudaFree(ptr_);
    }
    
    // Non-copyable, movable
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    DeviceMemory(DeviceMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    void* get() const { return ptr_; }
    size_t size() const { return size_; }
    
private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};
```

**Alignment**: All tensors MUST be aligned to 256-byte boundaries for optimal GPU access.

**VRAM-Only Residency**:
- Strict VRAM-only (no UMA/CPU spill)
- Fail fast on insufficient VRAM
- Keep process alive and report structured error (do NOT exit)
- Emit `VRAM_OOM` error via SSE if during inference

**Spec Reference**: CUDA-5291, CUDA-5222

#### [M0-W-1221] Memory-Mapped I/O (REQUIRED)
Worker-orcd MUST use `mmap()` for host I/O for all models to avoid full RAM copies and to standardize the loader across model sizes.

**Implementation**:
- Memory-map GGUF file for reading
- Parse headers and metadata from mapped region
- Stream tensor data directly from mmap to VRAM

**Rationale**: 
- Reduces RAM usage (critical for large models like GPT-OSS-20B)
- Faster startup
- Enables efficient chunked transfers
- Standardizes loader codepaths across all model sizes

**Spec Reference**: CUDA-5260

#### [M0-W-1222] Chunked H2D Transfer (REQUIRED)
Worker-orcd MUST copy model tensors to VRAM in bounded chunks for all models.

**Chunk Size**: 1MB (configurable, but 1MB default)

**Implementation**:
```cpp
const size_t CHUNK_SIZE = 1024 * 1024;  // 1MB
for (size_t offset = 0; offset < total_size; offset += CHUNK_SIZE) {
    size_t chunk_size = std::min(CHUNK_SIZE, total_size - offset);
    cudaError_t err = cudaMemcpy(
        static_cast<char*>(device_ptr) + offset,
        static_cast<const char*>(host_ptr) + offset,
        chunk_size,
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_MODEL_LOAD_FAILED, 
                       "Failed to copy chunk to VRAM");
    }
}
```

**Complete Model Loading Flow**:
```cpp
Model::Model(const Context& ctx, const std::string& path) {
    // 1. Memory-map file for efficient reading
    auto mapped_file = mmap_file(path);
    
    // 2. Parse GGUF format
    parse_gguf(mapped_file);
    
    // 3. Allocate VRAM
    allocate_vram();
    
    // 4. Copy weights to VRAM in chunks
    copy_weights_chunked(mapped_file);
    
    // 5. Unmap file
    munmap_file(mapped_file);
}
```

**Rationale**: Prevents RAM spikes during large model loading

**Spec Reference**: CUDA-5261, CUDA-5240

### 6.4 Test Models

#### [M0-W-1230] M0 Reference Target Models
M0 MUST support three reference target models for comprehensive validation.

---

##### Model 1: Qwen2.5-0.5B-Instruct (Primary)

**Purpose**: 🧪 Primary bring-up & smoke test

**Specifications**:
- **Name**: Qwen2.5-0.5B-Instruct
- **Format**: GGUF
- **Quantization**: Q4_K_M
- **Size**: 352 MB
- **VRAM Footprint**: ~400 MB (model + KV cache for 2K context)
- **Tokenizer**: GGUF byte-BPE (embedded in GGUF)
- **Context Length**: 32,768 (recommended test limit: 2,048)
- **License**: Apache 2.0
- **Location**: `.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf`
- **Download**: `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf`

**Test Coverage**:
- Fast unit tests
- Integration tests
- BDD scenarios
- Haiku generation test

---

##### Model 2: Phi-3-Mini (~3.8B) Instruct (Stretch)

**Purpose**: 📈 Stretch target within 24 GB; exercises longer context & tokenizer variety

**Specifications**:
- **Name**: Phi-3-Mini-4K-Instruct
- **Format**: GGUF
- **Quantization**: Q4_K_M
- **Size**: ~2.3 GB
- **VRAM Footprint**: ~3.5 GB (model + KV cache for 4K context)
- **Tokenizer**: GGUF byte-BPE (embedded in GGUF)
- **Context Length**: 4,096
- **License**: MIT
- **Location**: `.test-models/phi3/phi-3-mini-4k-instruct-q4_k_m.gguf`
- **Download**: `https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf`

**Test Coverage**:
- VRAM pressure tests
- Context length validation
- Tokenizer variety (non-Qwen)

---

##### Model 3: GPT-OSS-20B (Trend-Relevant) - MXFP4 Implementation

**Purpose**: 🌟 Trend-relevant large model; validates MXFP4 quantization path

**Specifications**:
- **Name**: GPT-OSS-20B
- **Format**: GGUF v3 (MXFP4 tensor support)
- **Quantization**: MXFP4 (primary); Q4_K_M/Q4_0 (fallback)
- **Size**: ~12 GB (MXFP4)
- **VRAM Footprint**: ~16 GB (model + KV cache, per OpenAI guidance)
- **Tokenizer**: HF tokenizers (Rust) loading tokenizer.json
- **Context Length**: 8,192 (recommended test limit: 2,048)
- **License**: Apache 2.0 (verify with OpenAI release)
- **Location**: `.test-models/gpt-oss-20b/gpt-oss-20b-mxfp4.gguf`
- **Download**: TBD (await OpenAI GGUF release or convert from native)

**Tokenizer Configuration**:
- **Backend**: `hf-json` (Hugging Face tokenizers crate)
- **Source**: `tokenizer.json` in model directory
- **Metadata Exposure**: See §8.2 Tokenization Strategy for full details

**MXFP4 Compute Requirements**:
- In-kernel dequant: MXFP4 tiles/groups → registers/shared memory → FP16 accumulate
- Weight consumers: Embeddings, Attention (Q/K/V, attn matmul, output proj), FFN (up/act/down), LM head
- KV cache: FP16 precision
- Streaming: UTF-8-safe SSE (buffer partial multibyte sequences)

**Loader Requirements**:
- GGUF v3 tensor support (MXFP4 blocks)
- Memory-mapped I/O (mmap for host I/O)
- Chunked H2D copies (1MB chunks)
- 256-byte alignment for device buffers
- Strict VRAM-only residency (no UMA/CPU spill)
- Fail fast on insufficient VRAM; keep process alive, report structured error

**Fallback Behavior**:
- If MXFP4 artifact unavailable, accept Q4_K_M or Q4_0
- `/health.quant_kind` reflects actually loaded quantization

**Test Coverage**:
- Large model validation
- MXFP4 quantization path
- HF tokenizer integration (tokenizer.json)
- 24 GB VRAM boundary test
- UTF-8 streaming with multibyte characters
- OOM recovery (intentional KV/context overflow)

---

**M0 Model Matrix**:

| Model | Size | VRAM | Quantization | Tokenizer | Primary Use |
|-------|------|------|--------------|-----------|-------------|
| Qwen2.5-0.5B | 352 MB | ~400 MB | Q4_K_M | GGUF byte-BPE | Smoke tests |
| Phi-3-Mini | 2.3 GB | ~3.5 GB | Q4_K_M | GGUF byte-BPE | Stretch tests |
| GPT-OSS-20B | 12 GB | ~16 GB | MXFP4 | tokenizer.json | Large model |

**Total VRAM**: All three models fit within 24 GB GPU (tested sequentially, not concurrently).

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
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "stop": ["\n\n", "END"],
  "seed": 42
}
```

**Response**: SSE stream (see §7.2)

**Validation**:
- `job_id` — MUST be non-empty string (required)
- `prompt` — MUST be non-empty, max 32768 characters (required)
- `max_tokens` — MUST be 1-2048 (optional, default: 2048)
- `temperature` — MUST be 0.0-2.0 (optional, default: 1.0)
- `top_p` — MUST be 0.0-1.0 (optional, default: 1.0, disabled)
- `top_k` — MUST be 0-vocab_size (optional, default: 0, disabled)
- `repetition_penalty` — MUST be 0.0-2.0 (optional, default: 1.0, disabled)
- `stop` — MUST be array of strings, max 4 sequences, each max 32 tokens (optional, default: [])
- `seed` — MUST be valid uint64 (optional, auto-generated if omitted)

**Parameter Semantics**:
- `top_p`: Nucleus sampling - keep tokens with cumulative probability >= top_p (1.0 = disabled)
- `top_k`: Top-k sampling - keep only top k tokens by probability (0 = disabled)
- `repetition_penalty`: Penalize tokens that appear in generation history (1.0 = disabled, >1.0 = penalize, <1.0 = encourage)
- `stop`: Stop generation when any sequence is matched (checked after each token)

**Backward Compatibility**: All new parameters (top_p, top_k, repetition_penalty, stop) are optional with defaults that preserve M0 baseline behavior.

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
  "model": "gpt-oss-20b",
  "resident": true,
  "quant_kind": "MXFP4",
  "vram_bytes_used": 16106127360,
  "tokenizer_kind": "hf-json",
  "vocab_size": 50257,
  "context_length": 2048,
  "uptime_seconds": 3600,
  "sm": 86,
  "cuda_runtime_version": "12.1"
}
```

**Status values**:
- `healthy` — Worker operational
- `unhealthy` — VRAM residency check failed or other critical error

**Required fields** (updated 2025-10-03):
- `resident` (bool) — VRAM residency status (true = all weights in VRAM)
- `quant_kind` (string) — Quantization format: `"MXFP4"` | `"Q4_K_M"` | `"Q4_0"`
- `vram_bytes_used` (int) — Current VRAM usage in bytes
- `tokenizer_kind` (string) — Backend type: `"gguf-bpe"` or `"hf-json"`
- `vocab_size` (int) — Vocabulary size
- `context_length` (int|null) — Max context length if known; else null

**Optional fields** (nice-to-have):
- `sm` (int) — Compute capability (e.g., 86 for SM_86)
- `cuda_runtime_version` (string) — CUDA runtime version

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

## 8. Tokenization Strategy

**Finalized**: 2025-10-03

### 8.1 Backend Architecture

Worker-orcd implements **two distinct tokenizer backends** with runtime selection:

#### [M0-W-1360] Tokenizer Backend Selection
Worker MUST select tokenizer backend at model load time based on model metadata:

**Backend Types**:
1. **`hf-json`** — Hugging Face tokenizer.json backend
2. **`gguf-bpe`** — Pure-Rust GGUF byte-BPE backend

**Selection Logic**:
```rust
match model_metadata.tokenizer_type {
    TokenizerType::HuggingFace => TokenizerBackend::HfJson,
    TokenizerType::GgufBpe => TokenizerBackend::GgufBpe,
}
```

### 8.2 HF-JSON Backend (GPT-OSS-20B)

#### [M0-W-1361] Hugging Face Tokenizers Crate
For GPT-OSS-20B, worker MUST use the Hugging Face `tokenizers` crate (Rust):

**Implementation**:
- Load tokenizer.json directly from model directory
- Use HF tokenizers API for encode/decode
- No Python or external binaries required

**Metadata Exposure** (added 2025-10-03):
Worker MUST expose tokenizer metadata from tokenizer.json:
- `eos_id`: End-of-sequence token ID
- `bos_id`: Begin-of-sequence token ID  
- `vocab_size`: Vocabulary size
- `model_max_context`: Maximum context length (if available in tokenizer.json)

**UTF-8 Streaming**:
- Enforce UTF-8-safe SSE streaming
- Buffer partial multibyte sequences
- Never emit invalid UTF-8
- Handle token boundaries that split UTF-8 codepoints

**Conformance Testing**:
- Golden encode/decode test vectors MUST be included (20-30 pairs)
- Vectors catch schema drift and ensure parity with upstream tokenizer
- Test vectors cover: BOS/EOS handling, special tokens, multibyte UTF-8, edge cases

**Example**:
```rust
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_file("tokenizer.json")?;

// Expose metadata
let eos_id = tokenizer.token_to_id("

### 8.3 GGUF-BPE Backend (Qwen2.5-0.5B, Phi-3-Mini)

#### [M0-W-1362] Pure-Rust GGUF Tokenizer
For Qwen2.5-0.5B and Phi-3-Mini, worker MUST implement pure-Rust GGUF tokenizer:

**Implementation**:
- Parse GGUF metadata to extract vocab and merges
- Run byte-level BPE entirely in Rust (no FFI, no llama.cpp)
- UTF-8 safe streaming decode with proper BOS/EOS handling
- Handle special tokens according to GGUF metadata

**GGUF Metadata Parsing**:
```rust
struct GgufTokenizer {
    vocab: Vec<String>,           // Token strings
    merges: Vec<(u32, u32)>,      // BPE merge pairs
    bos_token_id: Option<u32>,    // Begin-of-sequence
    eos_token_id: Option<u32>,    // End-of-sequence
    special_tokens: HashMap<String, u32>,
}
```

**Byte-Level BPE**:
- Encode: UTF-8 text → byte sequence → BPE merges → token IDs
- Decode: Token IDs → byte sequence → UTF-8 validation → text

**UTF-8 Safety**:
- Buffer partial bytes until complete UTF-8 codepoint
- Never emit invalid UTF-8 sequences
- Handle multi-byte characters correctly

### 8.4 Conformance Testing

#### [M0-W-1363] Test Vectors Required
Worker MUST include conformance test vectors for all three models:

**Test Coverage**:
1. **Qwen2.5-0.5B** (gguf-bpe):
   - Basic encode/decode
   - BOS/EOS token handling
   - Special tokens (e.g., `<|im_start|>`, `<|im_end|>`)
   - Multi-byte UTF-8 characters
   - Edge cases (empty string, very long sequences)

2. **Phi-3-Mini** (gguf-bpe):
   - Basic encode/decode
   - BOS/EOS token handling
   - Special tokens
   - UTF-8 edge cases

3. **GPT-OSS-20B** (hf-json):
   - Basic encode/decode
   - Parity with upstream HF tokenizer
   - Special tokens
   - Schema drift detection

**Test Format**:
```rust
struct TokenizerTestVector {
    text: String,
    expected_token_ids: Vec<u32>,
    expected_decoded: String,
    description: String,
}
```

### 8.5 Health Endpoint Integration

#### [M0-W-1364] Tokenizer Observability
Health endpoint MUST include tokenizer information (see §7.3):

**Required Fields**:
- `tokenizer_kind`: `"gguf-bpe"` or `"hf-json"`
- `vocab_size`: Vocabulary size (e.g., 151936 for Qwen2.5-0.5B)

**Purpose**:
- Debugging: Verify correct backend loaded
- Validation: Confirm vocab size matches expectations
- Monitoring: Track tokenizer configuration across deployments

### 8.6 Implementation Requirements

#### [M0-W-1365] No External Dependencies
Tokenization MUST be self-contained:

**Requirements**:
- ✅ No Python runtime required
- ✅ No external binaries (no llama.cpp)
- ✅ No FFI for tokenization (CUDA FFI only for inference)
- ✅ Pure Rust implementation for both backends
- ✅ Deterministic encode/decode across all platforms

**Rationale**:
- Simplifies deployment (single binary)
- Ensures reproducibility
- Reduces attack surface
- Enables static linking

---

## 8.7. Architecture Adapters

**Context**: M0 supports three models with fundamentally different architectures (Qwen/Phi-3 are Llama-style, GPT-OSS-20B is GPT-style). This requires architecture-specific inference pipelines.

**Reference**: See `M0_ARCHITECTURAL_GAP_ANALYSIS.md` for detailed analysis.

### 8.7.1 InferenceAdapter Pattern

#### [M0-W-1213] InferenceAdapter Interface

Worker MUST implement InferenceAdapter base class to abstract architecture-specific logic.

**Interface Design**:
```cpp
class InferenceAdapter {
public:
    virtual ~InferenceAdapter() = default;
    
    // Load weights from GGUF with architecture-specific mapping
    virtual void load_weights_from_gguf(
        const GGUFFile& gguf,
        DeviceMemory& vram_allocation
    ) = 0;
    
    // Run architecture-specific forward pass
    virtual void run_forward_pass(
        const ModelWeights& weights,
        const DeviceMemory& input_tokens,
        DeviceMemory& output_logits,
        KVCache& kv_cache,
        cudaStream_t stream
    ) = 0;
    
    // Get architecture name for logging/debugging
    virtual const char* architecture_name() const = 0;
};
```

**Factory Pattern**:
```cpp
std::unique_ptr<InferenceAdapter> create_adapter(Architecture arch) {
    switch (arch) {
        case Architecture::Llama:
            return std::make_unique<LlamaInferenceAdapter>();
        case Architecture::GPT:
            return std::make_unique<GPTInferenceAdapter>();
        default:
            throw std::runtime_error("Unsupported architecture");
    }
}
```

**Spec Reference**: M0_ARCHITECTURAL_GAP_ANALYSIS.md (Gap 2)

### 8.7.2 LlamaInferenceAdapter

#### [M0-W-1214] LlamaInferenceAdapter Implementation

Worker MUST implement LlamaInferenceAdapter for Llama-style models (Qwen2.5-0.5B, Phi-3-Mini).

**Architecture Characteristics**:
- **Position Encoding**: RoPE (Rotary Position Embedding)
- **Attention**: GQA (Grouped Query Attention)
- **Normalization**: RMSNorm
- **FFN Activation**: SwiGLU

**Forward Pass Pipeline**:
```cpp
void LlamaInferenceAdapter::run_forward_pass(...) {
    // 1. Embedding lookup
    embedding_kernel<<<...>>>(input_tokens, embeddings);
    
    // 2. Transformer layers
    for (int layer = 0; layer < num_layers; ++layer) {
        // Pre-attention norm
        rmsnorm_kernel<<<...>>>(embeddings, normed);
        
        // Apply RoPE to Q and K
        rope_kernel<<<...>>>(q, k, position_ids);
        
        // GQA attention (grouped K/V heads)
        gqa_attention_kernel<<<...>>>(q, k, v, attn_out, kv_cache);
        
        // Residual connection
        add_kernel<<<...>>>(embeddings, attn_out);
        
        // Pre-FFN norm
        rmsnorm_kernel<<<...>>>(attn_out, normed);
        
        // SwiGLU FFN (gate + up projections, swish activation, down projection)
        swiglu_ffn_kernel<<<...>>>(normed, ffn_out);
        
        // Residual connection
        add_kernel<<<...>>>(attn_out, ffn_out, embeddings);
    }
    
    // 3. Final norm + output projection
    rmsnorm_kernel<<<...>>>(embeddings, normed);
    output_kernel<<<...>>>(normed, logits);
}
```

**Weight Mapping** (Llama-style):
- `token_embd.weight` — Token embeddings
- `blk.{layer}.attn_norm.weight` — Pre-attention RMSNorm
- `blk.{layer}.attn_q.weight`, `attn_k.weight`, `attn_v.weight` — Q/K/V projections (GQA)
- `blk.{layer}.attn_output.weight` — Attention output projection
- `blk.{layer}.ffn_norm.weight` — Pre-FFN RMSNorm
- `blk.{layer}.ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` — SwiGLU FFN
- `output_norm.weight` — Final RMSNorm
- `output.weight` — LM head

**Spec Reference**: M0_ARCHITECTURAL_GAP_ANALYSIS.md (Gap 2a, Gap 4a)

### 8.7.3 GPTInferenceAdapter

#### [M0-W-1215] GPTInferenceAdapter Implementation

Worker MUST implement GPTInferenceAdapter for GPT-style models (GPT-OSS-20B).

**Architecture Characteristics**:
- **Position Encoding**: Absolute/Learned positional embeddings
- **Attention**: MHA (Multi-Head Attention)
- **Normalization**: LayerNorm
- **FFN Activation**: GELU

**Forward Pass Pipeline**:
```cpp
void GPTInferenceAdapter::run_forward_pass(...) {
    // 1. Token + position embeddings
    embedding_kernel<<<...>>>(input_tokens, token_emb);
    positional_embedding_kernel<<<...>>>(position_ids, pos_emb);
    add_kernel<<<...>>>(token_emb, pos_emb, embeddings);
    
    // 2. Transformer layers
    for (int layer = 0; layer < num_layers; ++layer) {
        // Pre-attention norm
        layernorm_kernel<<<...>>>(embeddings, normed);
        
        // MHA attention (all heads have unique K/V)
        mha_attention_kernel<<<...>>>(normed, attn_out, kv_cache);
        
        // Residual connection
        add_kernel<<<...>>>(embeddings, attn_out);
        
        // Pre-FFN norm
        layernorm_kernel<<<...>>>(attn_out, normed);
        
        // GELU FFN (fc1 + gelu + fc2)
        gelu_ffn_kernel<<<...>>>(normed, ffn_out);
        
        // Residual connection
        add_kernel<<<...>>>(attn_out, ffn_out, embeddings);
    }
    
    // 3. Final norm + output projection
    layernorm_kernel<<<...>>>(embeddings, normed);
    output_kernel<<<...>>>(normed, logits);
}
```

**Weight Mapping** (GPT-style):
- `wte` — Token embeddings
- `wpe` — Positional embeddings (learned)
- `h.{layer}.ln_1.weight`, `ln_1.bias` — Pre-attention LayerNorm
- `h.{layer}.attn.c_attn.weight` — Combined Q/K/V projection (or separate)
- `h.{layer}.attn.c_proj.weight` — Attention output projection
- `h.{layer}.ln_2.weight`, `ln_2.bias` — Pre-FFN LayerNorm
- `h.{layer}.mlp.c_fc.weight` — FFN up projection (fc1)
- `h.{layer}.mlp.c_proj.weight` — FFN down projection (fc2)
- `ln_f.weight`, `ln_f.bias` — Final LayerNorm
- `lm_head.weight` — LM head (or tied with wte)

**Spec Reference**: M0_ARCHITECTURAL_GAP_ANALYSIS.md (Gap 2b, Gap 4b)

### 8.7.4 Architecture-Specific Kernels

#### [M0-W-1432] LayerNorm Kernel (GPT)

Worker MUST implement LayerNorm kernel for GPT models.

**Algorithm**:
```cpp
__global__ void layernorm_kernel(
    const half* input,
    half* output,
    const half* gamma,  // scale
    const half* beta,   // bias
    int hidden_size,
    float eps = 1e-5
) {
    // 1. Compute mean
    float mean = compute_mean(input, hidden_size);
    
    // 2. Compute variance
    float variance = compute_variance(input, mean, hidden_size);
    
    // 3. Normalize
    float inv_std = rsqrtf(variance + eps);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (float(input[i]) - mean) * inv_std;
        output[i] = half(normalized * float(gamma[i]) + float(beta[i]));
    }
}
```

**Characteristics**:
- Two reduction passes (mean + variance)
- Learnable scale (γ) and bias (β)
- Numerical stability via epsilon

**Spec Reference**: M0_ARCHITECTURAL_GAP_ANALYSIS.md (Gap 3a)

#### [M0-W-1433] GELU Activation Kernel (GPT)

Worker MUST implement GELU activation kernel for GPT FFN.

**Algorithm**:
```cpp
__device__ float gelu(float x) {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const float c = sqrtf(2.0f / M_PI);
    float x_cubed = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x_cubed)));
}

__global__ void gelu_kernel(const half* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = half(gelu(float(input[idx])));
    }
}
```

**Spec Reference**: M0_ARCHITECTURAL_GAP_ANALYSIS.md (Gap 3c)

#### [M0-W-1434] Absolute Positional Embedding Kernel (GPT)

Worker MUST implement absolute positional embedding for GPT models.

**Algorithm**:
```cpp
__global__ void positional_embedding_kernel(
    const int* position_ids,
    const half* wpe,  // learned position embeddings
    half* output,
    int seq_len,
    int hidden_size
) {
    int pos_idx = blockIdx.x;  // position in sequence
    int hidden_idx = threadIdx.x;  // dimension
    
    if (pos_idx < seq_len && hidden_idx < hidden_size) {
        int pos = position_ids[pos_idx];
        output[pos_idx * hidden_size + hidden_idx] = wpe[pos * hidden_size + hidden_idx];
    }
}
```

**Spec Reference**: M0_ARCHITECTURAL_GAP_ANALYSIS.md (Gap 3b)

### 8.7.5 MXFP4 Architecture-Aware Handling

#### [M0-W-1435] MXFP4 Weight Mapping

Worker MUST handle MXFP4 quantization with architecture-specific weight structures.

**Llama-style MXFP4** (Qwen/Phi-3 if using MXFP4):
- Weights stored as: `token_embd`, `qkv_proj` (fused), `gate/up/down` projections
- Dequantization in: embedding lookup, attention Q/K/V, FFN kernels

**GPT-style MXFP4** (GPT-OSS-20B):
- Weights stored as: `wte`, `wpe`, separate `Q/K/V` or fused `c_attn`, `fc1/fc2`
- Dequantization in: embedding lookup, attention, FFN kernels

**In-Kernel Dequantization**:
```cpp
__global__ void mxfp4_gemm_kernel(
    const uint8_t* mxfp4_weights,  // Packed MXFP4
    const half* input,              // FP16 input
    half* output,                   // FP16 output
    int M, int N, int K
) {
    // 1. Load MXFP4 tile to shared memory
    // 2. Dequantize to FP16 in registers
    // 3. Perform GEMM with FP16 accumulation
    // 4. Write FP16 output
}
```

**Spec Reference**: M0_ARCHITECTURAL_GAP_ANALYSIS.md (Gap 4c)

---

## 9. CUDA Implementation

### 9.1 Context Management

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

### 9.2 Model Loading

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

**GGUF Header Parsing**:
```cpp
struct GGUFHeader {
    uint32_t magic;              // 'GGUF' (0x47475546)
    uint32_t version;            // 3
    uint64_t tensor_count;       // Number of tensors
    uint64_t metadata_kv_count;  // Number of metadata entries
};

void Model::parse_gguf(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw CudaError(CUDA_ERROR_MODEL_LOAD_FAILED, "Failed to open model file");
    }
    
    GGUFHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    if (header.magic != 0x47475546) {
        throw CudaError(CUDA_ERROR_MODEL_LOAD_FAILED, "Invalid GGUF magic bytes");
    }
    
    if (header.version != 3) {
        throw CudaError(CUDA_ERROR_MODEL_LOAD_FAILED, "Unsupported GGUF version");
    }
    
    parse_metadata(file, header.metadata_kv_count);
    parse_tensors(file, header.tensor_count);
}
```

**Model Metadata Structure**:
```cpp
struct ModelMetadata {
    std::string architecture;  // "llama", "gpt2", etc.
    std::string name;
    uint32_t context_length;
    uint32_t vocab_size;
    uint32_t embedding_length;
    uint32_t num_layers;
};
```

**Spec Reference**: CUDA-5240, CUDA-5280, CUDA-5281

### 9.3 Inference Execution

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
        
        // Check for errors after kernel launch
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw CudaError(CUDA_ERROR_KERNEL_LAUNCH_FAILED, cudaGetErrorString(err));
        }
        
        // Feed-forward
        ffn_kernel<<<grid, block, 0, stream_>>>(
            model_.layer_weights(layer), attention_output_, embeddings_
        );
    }
    
    // 3. Output projection
    output_kernel<<<grid, block, 0, stream_>>>(
        model_.output_weights(), embeddings_, logits_->get()
    );
    
    // 4. Synchronize and check for execution errors
    cudaError_t err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_INFERENCE_FAILED, cudaGetErrorString(err));
    }
}
```

**InferenceResult Class Design**:
```cpp
class InferenceResult {
public:
    InferenceResult(const Model& model, const std::string& prompt, const InferenceConfig& config);
    ~InferenceResult();
    
    // Non-copyable, non-movable (holds CUDA resources)
    InferenceResult(const InferenceResult&) = delete;
    InferenceResult& operator=(const InferenceResult&) = delete;
    InferenceResult(InferenceResult&&) = delete;
    InferenceResult& operator=(InferenceResult&&) = delete;
    
    bool next_token(std::string& token_out, int& token_index);
    bool is_done() const;
    
private:
    void tokenize_prompt(const std::string& prompt);
    void allocate_kv_cache();
    void run_forward_pass();
    int sample_token();
    std::string detokenize(int token_id);
    bool check_stop_sequences() const;
    
    const Model& model_;
    InferenceConfig config_;
    std::vector<int> prompt_tokens_;
    std::vector<int> token_history_;  // For repetition penalty and stop sequences
    int current_token_ = 0;
    bool stopped_by_sequence_ = false;
    std::unique_ptr<DeviceMemory> kv_cache_;
    std::unique_ptr<DeviceMemory> logits_;
    cudaStream_t stream_;
    std::mt19937_64 rng_;
};
```

**InferenceConfig Structure**:
```cpp
struct InferenceConfig {
    // Core parameters
    int max_tokens;
    float temperature = 1.0f;
    uint64_t seed;
    
    // Advanced sampling parameters (M0 Sprint 3 expansion)
    float top_p = 1.0f;              // Nucleus sampling (1.0 = disabled)
    int top_k = 0;                   // Top-k sampling (0 = disabled)
    float repetition_penalty = 1.0f; // Repetition penalty (1.0 = disabled)
    float min_p = 0.0f;              // Min-p sampling (0.0 = disabled, optional)
    
    // Stop sequences (tokenized, max 4 sequences)
    const int* stop_sequences[4] = {nullptr, nullptr, nullptr, nullptr};
    int stop_sequence_lengths[4] = {0, 0, 0, 0};
};
```

**Parameter Defaults**:
- All advanced parameters default to "disabled" state
- `top_p = 1.0` means no nucleus filtering
- `top_k = 0` means no top-k filtering
- `repetition_penalty = 1.0` means no penalty
- `min_p = 0.0` means no minimum probability threshold

**Spec Reference**: CUDA-5321, CUDA-5320

#### [M0-W-1421] Token Sampling
Worker-orcd MUST sample next token with advanced sampling parameters:

```cpp
int InferenceResult::sample_token() {
    // Copy logits to device-writable buffer (we'll modify them)
    float* d_logits_temp;
    cudaMalloc(&d_logits_temp, vocab_size * sizeof(float));
    cudaMemcpy(d_logits_temp, logits_->get(), 
               vocab_size * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Apply temperature scaling
    if (config_.temperature != 1.0f) {
        temperature_scale_kernel<<<grid, block>>>(d_logits_temp, vocab_size, config_.temperature);
    }
    
    // Apply repetition penalty (if enabled)
    if (config_.repetition_penalty != 1.0f && !token_history_.empty()) {
        apply_repetition_penalty_kernel<<<grid, block>>>(
            d_logits_temp, vocab_size,
            token_history_.data(), token_history_.size(),
            config_.repetition_penalty
        );
    }
    
    // Apply top-k filtering (if enabled)
    if (config_.top_k > 0) {
        apply_top_k_kernel<<<grid, block>>>(d_logits_temp, vocab_size, config_.top_k);
    }
    
    // Apply top-p (nucleus) filtering (if enabled)
    if (config_.top_p < 1.0f) {
        apply_top_p_kernel<<<grid, block>>>(d_logits_temp, vocab_size, config_.top_p);
    }
    
    // Apply min-p filtering (if enabled, optional)
    if (config_.min_p > 0.0f) {
        apply_min_p_kernel<<<grid, block>>>(d_logits_temp, vocab_size, config_.min_p);
    }
    
    // Sampling strategy based on temperature
    int token_id;
    if (config_.temperature == 0.0f) {
        // Greedy sampling (for testing reproducibility)
        token_id = greedy_sample_kernel(d_logits_temp, vocab_size);
    } else {
        // Stochastic sampling (for production use)
        float random_value = rng_.uniform();
        token_id = stochastic_sample_kernel(d_logits_temp, vocab_size, random_value);
    }
    
    cudaFree(d_logits_temp);
    
    // Add to history for repetition penalty
    token_history_.push_back(token_id);
    
    return token_id;
}
```

**Sampling Pipeline Order**:
1. Temperature scaling (always applied if temp != 1.0)
2. Repetition penalty (applied to history tokens)
3. Top-k filtering (keep only top k tokens)
4. Top-p filtering (nucleus sampling)
5. Min-p filtering (minimum probability threshold, optional)
6. Final sampling (greedy if temp=0, stochastic otherwise)

**Note**: Filters are applied in sequence. Each filter modifies logits in-place (zeroing out filtered tokens).

#### [M0-W-1422] Stop Sequence Detection
Worker-orcd MUST check for stop sequences after each token generation:

```cpp
bool InferenceResult::check_stop_sequences() const {
    // Check each configured stop sequence
    for (int seq_idx = 0; seq_idx < 4; ++seq_idx) {
        if (config_.stop_sequences[seq_idx] == nullptr) continue;
        
        int seq_len = config_.stop_sequence_lengths[seq_idx];
        if (token_history_.size() < seq_len) continue;
        
        // Check if last seq_len tokens match stop sequence
        bool match = true;
        for (int i = 0; i < seq_len; ++i) {
            int history_idx = token_history_.size() - seq_len + i;
            if (token_history_[history_idx] != config_.stop_sequences[seq_idx][i]) {
                match = false;
                break;
            }
        }
        
        if (match) return true;
    }
    
    return false;
}

bool InferenceResult::is_done() const {
    // Check max tokens limit
    if (current_token_ >= config_.max_tokens) return true;
    
    // Check stop sequences
    if (stopped_by_sequence_) return true;
    
    return false;
}
```

**Stop Sequence Workflow**:
1. After each token is generated, append to `token_history_`
2. Call `check_stop_sequences()` to check if any stop sequence matched
3. If matched, set `stopped_by_sequence_ = true` and terminate generation
4. Stop sequences are checked against tokenized IDs (not raw strings)

**Tokenization of Stop Sequences**:
- Client provides stop sequences as strings in HTTP request
- Worker MUST tokenize stop sequences using same tokenizer as prompt
- Tokenized sequences stored in `InferenceConfig.stop_sequences`
- Max 4 stop sequences, each max 32 tokens

**KV Cache Allocation**:
```cpp
void InferenceResult::allocate_kv_cache() {
    // Calculate KV cache size
    // For each layer: 2 (K and V) * context_length * hidden_dim
    size_t cache_size = 2 * 
                       model_.metadata().num_layers *
                       model_.metadata().context_length *
                       model_.metadata().embedding_length *
                       sizeof(float);
    
    kv_cache_ = std::make_unique<DeviceMemory>(cache_size);
    
    // Initialize to zero
    cudaMemset(kv_cache_->get(), 0, cache_size);
}
```

**KV Cache Layout** (per layer):
```
┌─────────────────────────────────────┐
│ Keys   [context_length × hidden_dim] │
├─────────────────────────────────────┤
│ Values [context_length × hidden_dim] │
└─────────────────────────────────────┘
```

**Spec Reference**: CUDA-5321, CUDA-5340, CUDA-5341

### 9.4 CUDA Kernels

#### [M0-W-1430] Required Kernels
Worker-orcd MUST implement M0 kernel set:

**Core Kernels**:
- cuBLAS GEMM wrapper
- RoPE (llama variant)
- Naive attention (prefill + decode)
- RMSNorm

**Sampling Kernels** (expanded in Sprint 3):
- Temperature scaling
- Greedy sampling (argmax)
- Stochastic sampling (softmax + CDF sampling)
- Top-k filtering
- Top-p (nucleus) filtering
- Repetition penalty
- Min-p filtering (optional)

**Kernel Organization**:
```
kernels/
├── attention.cu      # Attention mechanism kernels
├── matmul.cu         # Matrix multiplication kernels (cuBLAS wrapper)
├── sampling.cu       # Token sampling kernels (temperature, greedy, stochastic, top-k, top-p, repetition, min-p)
├── rope.cu           # Rotary position embeddings
├── normalization.cu  # RMSNorm, LayerNorm
└── common.cuh        # Shared kernel utilities
```

**Sampling Kernel Signatures**:
```cuda
// Temperature scaling
__global__ void temperature_scale_kernel(float* logits, int vocab_size, float temperature);

// Greedy sampling
__global__ void greedy_sample_kernel(const float* logits, int vocab_size, int* token_id);

// Stochastic sampling (softmax + sample)
__global__ void softmax_kernel(const float* logits, float* probs, int vocab_size);
__global__ void sample_from_cdf_kernel(const float* cdf, int vocab_size, float random_value, int* token_id);

// Advanced filters
__global__ void apply_top_k_kernel(float* logits, int vocab_size, int top_k);
__global__ void apply_top_p_kernel(float* logits, int vocab_size, float top_p);
__global__ void apply_repetition_penalty_kernel(float* logits, int vocab_size, const int* history, int history_len, float penalty);
__global__ void apply_min_p_kernel(float* logits, int vocab_size, float min_p);
```

**Kernel Launch Pattern**:
```cpp
void run_forward_pass(const Model& model, InferenceState& state) {
    dim3 grid(num_blocks);
    dim3 block(threads_per_block);
    
    attention_kernel<<<grid, block, 0, state.stream>>>(
        model.weights(),
        state.kv_cache(),
        state.output()
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_KERNEL_LAUNCH_FAILED, cudaGetErrorString(err));
    }
}
```

**Spec Reference**: WORKER-4700, CUDA-5030, CUDA-5031

#### [M0-W-1431] Kernel Safety
All kernel launches MUST have:
- Bounds checking
- Dimension validation
- Error handling after launch (`cudaGetLastError`)
- Error handling after sync (`cudaStreamSynchronize`)

**Spec Reference**: KERNEL-SAFE-001 through KERNEL-SAFE-004, CUDA-5371

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

**Error Message Retrieval**:
```cpp
extern "C" const char* cuda_error_message(int error_code) {
    switch (error_code) {
        case CUDA_SUCCESS: return "Success";
        case CUDA_ERROR_DEVICE_NOT_FOUND: return "CUDA device not found";
        case CUDA_ERROR_OUT_OF_MEMORY: return "Out of GPU memory";
        case CUDA_ERROR_INVALID_DEVICE: return "Invalid device ID";
        case CUDA_ERROR_MODEL_LOAD_FAILED: return "Model load failed";
        case CUDA_ERROR_INFERENCE_FAILED: return "Inference execution failed";
        case CUDA_ERROR_VRAM_RESIDENCY_FAILED: return "VRAM residency check failed";
        case CUDA_ERROR_KERNEL_LAUNCH_FAILED: return "Kernel launch failed";
        default: return "Unknown error";
    }
}
```

**CudaError Exception Class**:
```cpp
class CudaError : public std::exception {
public:
    CudaError(int code, const std::string& msg) 
        : code_(code), msg_(msg) {}
    
    int code() const { return code_; }
    const char* what() const noexcept override { return msg_.c_str(); }
    
private:
    int code_;
    std::string msg_;
};
```

**Spec Reference**: CUDA-5040, CUDA-5041

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

**Build script** (`build.rs`):
```rust
fn main() {
    let dst = cmake::Config::new("cuda")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();
    
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=worker_cuda");
    println!("cargo:rustc-link-lib=cudart");
}
```

**CMake config** (`cuda/CMakeLists.txt`):
```cmake
cmake_minimum_required(VERSION 3.18)
project(worker_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Source files
add_library(worker_cuda STATIC
    src/ffi.cpp
    src/context.cpp
    src/model.cpp
    src/inference.cu
    src/health.cpp
    src/errors.cpp
    kernels/attention.cu
    kernels/matmul.cu
    kernels/sampling.cu
    kernels/rope.cu
    kernels/normalization.cu
)

target_include_directories(worker_cuda PUBLIC include)
target_link_libraries(worker_cuda cudart cublas)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
```

**Spec Reference**: CUDA-5050, CUDA-5051

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

**Location**: `tests/unit/cuda/`

#### [M0-W-1811] Rust Unit Tests
Rust layer MUST have unit tests:

**Tests**:
- HTTP handlers (with mocked CUDA)
- Error conversion
- SSE streaming
- Request validation

**Framework**: Rust `#[test]`

**Location**: `tests/unit/rust/`

### 12.3 Integration Tests

#### [M0-W-1820] End-to-End Test
M0 MUST have integration test:

**Test flow**:
1. Start worker with test model
2. Send inference request
3. Verify SSE stream
4. Verify reproducible output (temp=0 for testing)
5. Verify VRAM-only operation
6. Shutdown gracefully

**Location**: `tests/integration/e2e_test.rs`

### 12.2.3 Numerical Correctness Tests

**Added**: 2025-10-03 (MXFP4 numerical correctness validation)

#### [M0-W-1822] MXFP4 Numerical-Correctness Micro-Goldens
Worker MUST pass MXFP4 numerical correctness test:

**Test**: Dequant→GEMM and small attention shape vs float reference within tolerance

**Coverage**:
- MXFP4 dequantization correctness
- GEMM with MXFP4 weights
- Attention computation with MXFP4
- FP16 accumulation validation

**Tolerance**: ±0.01 (1%) relative error for FP16 accumulation

**Note**: This validates **numerical correctness** (tolerance-based) and does NOT assert throughput/latency. The comprehensive performance suite (utilization, latency, p50-p99) remains deferred to M1+.

**Validation**:
```rust
#[test]
fn test_mxfp4_numerical_correctness() {
    // Test MXFP4 dequant + GEMM
    let mxfp4_weights = load_mxfp4_tensor("test_weights.bin");
    let input = create_test_input();
    let output = gemm_mxfp4(mxfp4_weights, input);
    let reference = gemm_fp32(dequant_to_fp32(mxfp4_weights), input);
    assert_close!(output, reference, tolerance = 0.01);
    
    // Test attention with MXFP4
    let attn_output = attention_mxfp4(/* ... */);
    let attn_reference = attention_fp32(/* ... */);
    assert_close!(attn_output, attn_reference, tolerance = 0.01);
}
```

**Location**: `tests/integration/mxfp4_numerical_goldens.rs`

### 12.2.4 Minimal Reproducibility Test

**Added**: 2025-10-03 (minimal same-device reproducibility check)

#### [M0-W-1826] Same-Device Reproducibility
Worker MUST pass minimal reproducibility test:

**Test**: Run twice with seeded RNG and temp=0, assert identical token IDs

**Validation**:
```rust
#[test]
fn test_repro_same_device() {
    let worker = start_worker("qwen2.5-0.5b-instruct-q4_k_m.gguf", 0);
    
    let request = ExecuteRequest {
        prompt: "Write a haiku about GPU computing".to_string(),
        max_tokens: 50,
        temperature: 0.0,  // Greedy sampling
        seed: Some(42),
    };
    
    // First run
    let tokens1 = collect_token_ids!(worker.execute(request.clone()));
    
    // Second run
    let tokens2 = collect_token_ids!(worker.execute(request.clone()));
    
    // Assert identical token IDs
    assert_eq!(tokens1, tokens2, "Same device, same seed, temp=0 must produce identical tokens");
}
```

**Location**: `tests/integration/repro_same_device.rs`

**Note**: Deep CUDA determinism audit (kernel scheduling, atomics) deferred to M1+

### 12.3.1 GPT-OSS-20B Integration Tests

**Added**: 2025-10-03 (MXFP4 implementation requirements)

#### [M0-W-1821] Tokenizer Conformance Test
Worker MUST pass tokenizer conformance test for GPT-OSS-20B:

**Test**: 20-30 text↔ids pairs from upstream tokenizer.json artifacts

**Coverage**:
- Basic encode/decode
- BOS/EOS token handling
- Special tokens
- Multibyte UTF-8 characters
- Edge cases (empty string, very long sequences)

**Validation**:
```rust
#[test]
fn test_gpt_oss_20b_tokenizer_conformance() {
    let tokenizer = load_hf_tokenizer("tokenizer.json");
    
    for vector in GOLDEN_VECTORS {
        let encoded = tokenizer.encode(&vector.text);
        assert_eq!(encoded, vector.expected_token_ids);
        
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, vector.expected_decoded);
    }
}
```

**Location**: `tests/integration/tokenizer_conformance_gpt_oss_20b.rs`

#### [M0-W-1823] Large Model Bring-Up Test
Worker MUST pass large model bring-up test for GPT-OSS-20B:

**Test**: Load GPT-OSS-20B (MXFP4), verify health endpoint, confirm VRAM envelope

**Validation**:
```rust
#[test]
fn test_gpt_oss_20b_bring_up() {
    let worker = start_worker("gpt-oss-20b-mxfp4.gguf", 0);
    
    // Verify health endpoint
    let health = worker.get_health();
    assert_eq!(health.quant_kind, "MXFP4");
    assert_eq!(health.tokenizer_kind, "hf-json");
    assert!(health.resident);
    
    // Verify VRAM envelope with tolerance
    let expected_vram = read_expected_vram_from_metadata(); // or config: 16_000_000_000
    let tolerance = 0.20; // ±20%
    assert!(within_tolerance(health.vram_bytes_used, expected_vram, tolerance),
        "VRAM usage {} outside expected range {}±20%", 
        health.vram_bytes_used, expected_vram);
    
    // Verify basic inference
    let response = worker.execute(/* simple prompt */);
    assert_event!(response, "started");
    assert!(collect_tokens!(response).len() > 0);
}
```

**Location**: `tests/integration/gpt_oss_20b_bring_up.rs`

#### [M0-W-1824] UTF-8 Streaming Test
Worker MUST pass UTF-8 streaming test with multibyte characters:

**Test**: Prompts with multibyte characters; assert no mojibake across token events

**Coverage**:
- Chinese characters (3-byte UTF-8)
- Emoji (4-byte UTF-8)
- Mixed ASCII + multibyte
- Token boundaries that split UTF-8 sequences

**Validation**:
```rust
#[test]
fn test_utf8_streaming_multibyte() {
    let worker = start_worker("gpt-oss-20b-mxfp4.gguf", 0);
    
    let prompts = vec![
        "你好世界",  // Chinese
        "Hello 👋 World 🌍",  // Emoji
        "Mixed: 日本語 English العربية",  // Mixed scripts
    ];
    
    for prompt in prompts {
        let response = worker.execute(ExecuteRequest {
            prompt: prompt.to_string(),
            max_tokens: 20,
            temperature: 0.7,
        });
        
        let tokens = collect_tokens!(response);
        let decoded = tokens.join("");
        
        // Verify no mojibake (all valid UTF-8)
        assert!(decoded.is_valid_utf8());
        assert!(!decoded.contains("�"));  // No replacement characters
    }
}
```

**Location**: `tests/integration/utf8_streaming.rs`

#### [M0-W-1825] OOM Recovery Test
Worker MUST pass OOM recovery test:

**Test**: Intentionally exceed KV/context → expect structured VRAM_OOM error and clean recovery

**Validation**:
```rust
#[test]
fn test_oom_recovery() {
    let worker = start_worker("gpt-oss-20b-mxfp4.gguf", 0);
    
    // Intentionally exceed context length
    let response = worker.execute(ExecuteRequest {
        prompt: "x".repeat(10000),  // Exceed context
        max_tokens: 10000,
        temperature: 0.7,
    });
    
    // Expect VRAM_OOM error
    assert_event!(response, "error", code = "VRAM_OOM");
    
    // Verify process still alive
    assert!(worker.is_alive());
    
    // Verify clean recovery (next request works)
    let response2 = worker.execute(ExecuteRequest {
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: 0.7,
    });
    
    assert_event!(response2, "started");
    assert!(collect_tokens!(response2).len() > 0);
}
```

**Location**: `tests/integration/oom_recovery.rs`

### 12.4 Performance Tests (DEFERRED to M1+)

#### [M0-W-1830] Performance Test Suite (DEFERRED)
**Status**: DEFERRED to M1+

Comprehensive performance test suite deferred to M1+:

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

### 13.1 Narration-Core Logging (Hybrid Scope)

**Scope Change**: Structured logging replaced with narration-core logging (basic events only, no performance metrics)

#### [M0-W-1900] Narration-Core Log Events (UPDATED)
Worker-orcd MUST emit narration-core logs with basic event tracking:

**Required context**:
- `worker_id` — Worker identifier
- `job_id` — Job identifier (when applicable)
- `model_ref` — Model reference
- `gpu_device` — GPU device ID
- `event` — Event type

**Event types** (basic narrative only):
- `startup` — Worker starting
- `model_load_start` — Model loading begins
- `model_load_progress` — Loading progress (0-100%) ← **KEPT** (critical UX)
- `model_load_complete` — Model loaded successfully
- `ready` — Worker ready for requests
- `execute_start` — Inference request received
- `execute_end` — Inference completed
- `error` — Error occurred
- `shutdown` — Worker shutting down

**Note**: No performance metrics fields (vram_bytes, decode_time_ms, etc.) - deferred to M1

**Spec Reference**: WORK-3080

#### [M0-W-1901] Performance Metrics in Logs (DEFERRED to M1)
**Status**: DEFERRED (Performance Bundle)

Performance metrics in logs deferred to M1:
- ❌ `vram_bytes` — VRAM usage
- ❌ `tokens_in` — Input tokens
- ❌ `tokens_out` — Output tokens
- ❌ `decode_time_ms` — Inference time
- ❌ `first_token_ms` — First token latency

**Rationale**: Part of performance bundle deferral for faster M0 delivery

#### [M0-W-1902] Sensitive Data Handling (DEFERRED to M1)
**Status**: DEFERRED (Performance Bundle)

Sensitive data redaction deferred to M1. M0 may log prompts for debugging purposes.

**M1 Requirements** (deferred):
- ❌ No raw prompts (may contain PII)
- ❌ No generated text (may contain sensitive output)
- ❌ No API tokens or secrets
- ✅ Prompt hash (SHA-256) only
- ✅ Prompt length (character count)
- ✅ Token counts

**M0 Behavior**: Basic logging without redaction (development/testing phase)

---

## 14. Gaps & Clarifications

### 14.1 Identified Gaps

The following gaps require clarification or implementation:

1. **Tokenization Library** ✅ RESOLVED (FINALIZED 2025-10-03)
   - **Decision**: Rust-side tokenizer with two distinct backends:
     - **`hf-json` backend** for GPT-OSS-20B:
       - Uses Hugging Face `tokenizers` crate (Rust)
       - Loads and runs tokenizer.json directly
       - Golden encode/decode test vectors to catch schema drift
       - Ensures parity with upstream tokenizer
     - **`gguf-bpe` backend** for Qwen2.5-0.5B and Phi-3-Mini:
       - Pure-Rust GGUF tokenizer implementation
       - Parses GGUF metadata (vocab + merges)
       - Runs byte-level BPE entirely in Rust
       - UTF-8 safe streaming decode with BOS/EOS handling
   - **Runtime Selection**: Worker selects backend at model load time based on model metadata
   - **Impact**: 
     - No Python or external binaries at runtime
     - Deterministic, self-contained tokenization across all targets
     - Clean separation between GGUF byte-BPE and tokenizer.json handling
     - No llama.cpp dependency
   - **Implementation**: Rust crate with pluggable backend trait (`TokenizerBackend`)
   - **Testing**: Conformance test vectors required for all three models

2. **Detokenization (SSE Streaming)** ✅ RESOLVED
   - **Decision**: Streaming UTF-8 boundary buffer in Rust:
     - Decode token IDs → raw bytes
     - Emit only valid UTF-8 per SSE event
     - Buffer partial bytes until complete codepoint
   - **Impact**: No mid-codepoint breaks over SSE; clean UTF-8 streaming
   - **Implementation**: UTF-8 validator with byte buffer in SSE handler

3. **Quantization / Execution** ✅ RESOLVED (UPDATED)
   - **Decision**: **M0 is quantized-only execution**
     - Load quantized weights (MXFP4, Q4_K_M, Q4_0) to VRAM as-is
     - Execute inference in quantized form
     - NO dequantization to FP32 on load
     - Minimal kernel path: blockwise 4-bit/8-bit → per-tile dequant in registers/shared → FP16 accumulate
   - **Impact**: Fits all three targets on 24 GB GPUs; aligns with GPT-OSS guidance and local runtime behavior (LM Studio/llama.cpp)
   - **Spec Change**: Remove all "dequantize on load to FP32" language from M0

4. **Attention Implementation** ✅ RESOLVED
   - **Decision**: Naive unified attention (prefill + decode same path) for M0
   - **Impact**: Simpler CUDA code; performance tuning deferred to M2+
   - **Implementation**: Single attention kernel handles both phases

5. **Sampling Strategy** ✅ RESOLVED
   - **Decision**: Temperature-based sampling for M0 (0.0-2.0 range)
     - Temperature = 0.0 → greedy (argmax) for testing reproducibility
     - Temperature > 0.0 → stochastic sampling for production use
   - **Impact**: Product feature (customers control creativity); top-k/top-p deferred to M2+
   - **Implementation**: Temperature scaling + sampling (greedy when temp=0, stochastic otherwise)

6. **Memory Alignment** ✅ RESOLVED
   - **Decision**: Enforce 256-byte alignment for VRAM tensors at load and validate
   - **Impact**: Predictable GPU performance; early failure on malformed GGUF
   - **Implementation**: Alignment check in `allocate_vram()` with error on mismatch

7. **Stream Synchronization** ✅ RESOLVED
   - **Decision**: Sync after each token's full forward pass in M0
   - **Impact**: Deterministic control flow; small latency trade-off acceptable for M0
   - **Implementation**: `cudaStreamSynchronize(stream_)` after output projection

8. **X-Deadline Header** ✅ RESOLVED
   - **Decision**: Parse and log X-Deadline header in M0; no enforcement yet
   - **Impact**: Establishes pattern; zero complexity for M0
   - **Implementation**: Extract header, log remaining time, emit metric

9. **Client Disconnect / Timeout** ✅ RESOLVED
   - **Decision**: Abort inference on disconnect; check every ≤10 tokens; free KV cache
   - **Impact**: No zombie jobs; prompt cleanup
   - **Implementation**: M0-W-1611 specifies connection check in token generation loop

10. **Metrics** ✅ RESOLVED
    - **Decision**: Basic optional metrics only for M0:
      - VRAM usage, token count, startup latency, health
      - Add labels: `quant_kind` (Q4_K_M/MXFP4/…)
      - For GPT-OSS-20B: `active_expert_count` (even if fixed)
    - **Impact**: Lightweight observability; full Prometheus exporter deferred to M2+
    - **Implementation**: M0-W-1350 optional /metrics endpoint

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

**Contradiction 4**: Determinism vs Temperature
- **Initial spec**: "Greedy sampling only for determinism"
- **Product reality**: Temperature is a customer-facing feature, NOT optional
- **Resolution**: M0 supports temperature 0.0-2.0; greedy (temp=0) is for TESTING reproducibility only, not a product constraint. Determinism is a testing tool, not a product promise.

### 14.3 Deferred to Post-M0

**Performance Bundle (Deferred to M1 - Hybrid Scope)**:
- Performance metrics in logs (M0-W-1901)
- Prometheus metrics endpoint (M0-W-1350)
- All performance targets (M0-W-1600 through M0-W-1620)
- Performance test suite (M0-W-1830)
- Graceful shutdown endpoint (M0-W-1340)
- Client disconnect detection (M0-W-1611)
- Reproducible kernels validation (M0-W-1031 validation only)
- Sensitive data handling (M0-W-1902)

**Advanced Features (Deferred to M2+)**:
- Top-k/top-p sampling (M2+) — M0 has temperature-based sampling only
- FlashAttention (M2+)
- Continuous Batching (M2+)
- Tensor Parallelism (M4)
- Advanced Quantization (M2+)
- PagedAttention (M2+)
- Speculative Decoding (M3+)
- Authentication (M3+)

**Removed from Repo**:
- Proof bundles (entire concept)

---

## 15. Acceptance Criteria

### 15.1 M0 Success Criteria

M0 is considered complete when:

**Per-Model Acceptance Criteria**:

For **each** of the three M0 reference models (Qwen2.5-0.5B, Phi-3-Mini, GPT-OSS-20B):

1. ✅ **Startup**: CUDA init, GGUF parse, quantized weights resident in VRAM; report `quant_kind`
2. ✅ **Inference**: Functional SSE token stream; UTF-8-safe streaming
3. ✅ **Health**: `/health` shows `resident=true`, VRAM bytes, `quant_kind`
4. ✅ **Progress**: Model load progress events (0-100%) ← **CRITICAL** (user feedback)
5. ✅ **Logs**: Narration-core events (startup, load, execute, error, shutdown)

**General M0 Success Criteria (Hybrid Scope)**:

1. ✅ Worker binary compiles successfully with `--features cuda`
2. ✅ Worker loads all three M0 models (sequentially) into VRAM in quantized form
3. ✅ Worker accepts HTTP POST /execute request
4. ✅ Worker generates haiku functionally (reproducibility implementation done, validation deferred)
4b. ✅ Worker supports temperature 0.0-2.0 for production use (stochastic sampling when temp>0)
5. ✅ Worker streams tokens via SSE with UTF-8 boundary safety
6. ✅ Worker enforces VRAM-only (no RAM fallback, no UMA detected)
7. ✅ VRAM residency verification (periodic checks) ← **CRITICAL** (runtime safety)
8. ✅ VRAM OOM handling (graceful error, not crash) ← **CRITICAL** (safety)
9. ✅ Worker responds to GET /health with status including `quant_kind`
10. ✅ Worker handles POST /cancel gracefully
11. ✅ Worker shuts down on SIGTERM (graceful shutdown endpoint deferred)
12. ✅ All CUDA unit tests pass (functional only, no performance tests)
13. ✅ All Rust unit tests pass
14. ✅ Integration test passes for all three models (functional validation)
15. ✅ Tokenization works for both backends:
    - `gguf-bpe` backend (Qwen2.5-0.5B, Phi-3-Mini)
    - `hf-json` backend (GPT-OSS-20B)
    - Conformance test vectors pass for all three models
16. ✅ Quantized execution verified (no FP32 dequant on load)
17. ✅ Model load progress events emit (0%, 25%, 50%, 75%, 100%)

### 15.2 Non-Goals for M0 (Hybrid Scope)

**Deferred to M1 (Performance Bundle)**:
- ❌ Performance metrics/observability
- ❌ Performance test suite
- ❌ Graceful shutdown endpoint
- ❌ Client disconnect detection
- ❌ Reproducible kernels validation
- ❌ Sensitive data handling
- ❌ Proof bundles

**Deferred to M2+**:
- ❌ Pool manager integration
- ❌ Orchestrator integration
- ❌ Multi-model support
- ❌ Multi-GPU support
- ❌ Authentication/authorization
- ❌ Production-grade error recovery
- ❌ Advanced kernel optimizations

---

### 15.3 Performance Exit Criteria (DEFERRED to M1)

**Status**: DEFERRED (Performance Bundle)

M0 performance targets deferred to M1:

1. ❌ First token latency p95 <100ms (deferred)
2. ❌ Per-token latency p95 <50ms (deferred)
3. ❌ Health endpoint p99 <10ms (deferred)
4. ❌ Model loading time <60s (deferred)
5. ❌ Graceful shutdown <5s (deferred)
6. ❌ Zero memory leaks (deferred)
7. ❌ Client disconnect abort <100ms (deferred)

**M0 Behavior**: Functional validation only, no performance benchmarking

**M1 Plan**: Comprehensive performance test suite with validation against targets

---

## 16. Traceability Matrix

| M0 Requirement | Parent Spec | Source | Status |
|----------------|-------------|--------|--------|
| M0-SYS-6.3.1 | SYS-6.3.1 | Worker Self-Containment | ✅ Specified |
| M0-SYS-6.3.2 | SYS-6.3.2 | Worker Isolation | ✅ Specified |
| M0-SYS-6.3.5 | SYS-6.3.5 | Cancellation Handling | ✅ Specified |
| M0-SYS-2.2.1 | SYS-2.2.1 | VRAM-Only Enforcement | ✅ Specified |
| M0-SYS-2.3.1 | SYS-2.3.1 | Test Reproducibility | ✅ Specified |
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

### 18.2 CUDA Module Specifications (Implementation Guides)

**Note**: The following CUDA specs provide detailed implementation guidance that complements this M0 spec. They contain concrete C++ class designs, RAII patterns, and implementation examples.

- `bin/worker-orcd/cuda/.specs/00_cuda_overview.md` — CUDA overview & patterns (CUDA-5xxx)
- `bin/worker-orcd/cuda/.specs/01_context.md` — Context management implementation (CUDA-5100)
- `bin/worker-orcd/cuda/.specs/02_model.md` — Model loading implementation (CUDA-5200)
- `bin/worker-orcd/cuda/.specs/03_inference.md` — Inference execution implementation (CUDA-5300)
- `bin/worker-orcd/cuda/.specs/04_health.md` — Health monitoring implementation (CUDA-5400)
- `bin/worker-orcd/cuda/.specs/GAP_ANALYSIS.md` — Gap analysis between old specs and M0
- `bin/worker-orcd/cuda/kernels/.specs/00_cuda-kernels.md` — CUDA kernels (WORKER-4700)

**Implementation Details**: The CUDA module specs provide:
- Complete C++ class hierarchies with RAII wrappers (DeviceMemory, CudaStream, Context)
- Exception-to-error-code conversion patterns for FFI
- GGUF parsing implementation with concrete structures
- KV cache allocation formulas and memory layouts
- Kernel error checking patterns
- C++ unit test examples with Google Test
- Rust FFI integration test examples

### 18.3 Documentation

- `bin/worker-orcd/README.md` — Worker overview
- `bin/worker-orcd/CUDA_FEATURE.md` — CUDA feature flag
- `.docs/CUDA_FEATURE_FLAG_IMPLEMENTATION.md` — Implementation guide
- `BUILD_CONFIGURATION.md` — Build configuration
- `.docs/testing/TEST_MODELS.md` — Test models

### 18.4 Scope Decision Documents

- `bin/.specs/M0_DEFERRAL_CANDIDATES.md` — Deferral analysis (28 candidates)
- `bin/.specs/M0_RESOLUTION_CONTRADICTIONS.md` — Contradiction resolution (hybrid approach)
- `bin/.specs/M0_PERFORMANCE_BUNDLE_ANALYSIS.md` — Performance bundle impact analysis

---

**End of M0 Specification**

**Status**: Draft (Hybrid Scope) — Ready for implementation  
**Scope**: Performance Bundle Deferred (14 items to M1)  
**Timeline**: 4-5 weeks (optimized from 6-8 weeks)

**Next Steps**: 
1. Remove proof bundle references from all specs and code
2. Implement narration-core logging (basic events only)
3. Implement CUDA modules (context, model, inference, health)
4. Implement Rust HTTP layer with critical features:
   - Model load progress events (0-100%)
   - VRAM residency verification (periodic checks)
   - VRAM OOM handling (graceful error)
5. Implement FFI boundary
6. Write CUDA unit tests (functional only, no performance tests)
7. Write Rust unit tests
8. Write integration tests for all 3 models
9. Execute haiku test (functional validation)

**Deferred to M1**:
- Performance test suite
- Performance metrics collection
- Reproducible kernels validation
- Graceful shutdown endpoint
- Client disconnect detection
- Sensitive data handling

**Key Trade-offs**:
- ✅ 2-3 weeks faster delivery
- ✅ Critical safety features retained (VRAM monitoring, OOM handling)
- ✅ User experience retained (progress events)
- ❌ Performance validation deferred to M1
