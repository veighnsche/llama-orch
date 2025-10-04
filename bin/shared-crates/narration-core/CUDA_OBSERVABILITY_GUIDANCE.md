# 🎀 CUDA/C++ Observability Guidance for M0 Worker

**From**: The Narration Core Team  
**To**: Foundation Team (worker-orcd M0 development)  
**Date**: 2025-10-04  
**Status**: Official Guidance

---

## Executive Summary

**TL;DR**: You do **NOT** need special CUDA/C++ observability functions for M0. All narration happens in the **Rust layer only**. The C++/CUDA layer remains pure compute with no logging.

---

## The Problem Statement

The M0 spec (`01_M0_worker_orcd.md`) has a clear FFI boundary:

```
┌─────────────────────────────────────────────────────────────┐
│ RUST LAYER (src/*.rs)                                        │
│ • HTTP server (axum)                                         │
│ • SSE streaming                                              │
│ • Error handling and formatting                              │
│ • Logging and metrics                                        │  ← OBSERVABILITY HERE
└────────────────────┬────────────────────────────────────────┘
                     │ FFI (unsafe extern "C")
┌────────────────────▼────────────────────────────────────────┐
│ C++/CUDA LAYER (cuda/src/*.cpp, *.cu)                       │
│ • CUDA context management                                    │
│ • VRAM allocation (cudaMalloc)                              │
│ • Model loading (GGUF → VRAM)                               │
│ • Inference execution (CUDA kernels)                         │
│ • VRAM residency checks                                      │
│ • All GPU operations                                         │  ← NO LOGGING HERE
└─────────────────────────────────────────────────────────────┘
```

**Question**: Do we need special C++/CUDA functions to serve observability?

**Answer**: **NO**. Keep the CUDA layer pure. All observability happens in Rust.

---

## Architecture Decision: Rust-Only Observability

### Rationale

**M0-W-1050** explicitly states:

> The Rust layer MUST handle:
> - HTTP server (Axum)
> - CLI argument parsing (clap)
> - SSE streaming
> - Error formatting (convert C++ errors to HTTP responses)
> - **Logging (tracing)**
> - Metrics (Prometheus, optional for M0)

**M0-W-1051** explicitly states:

> The C++/CUDA layer MUST handle:
> - CUDA context management
> - VRAM allocation
> - Model loading
> - Inference execution
> - Health checks (VRAM residency verification)

**Notice**: Logging is **NOT** in the C++/CUDA responsibilities.

### Why This Design?

1. **FFI Complexity**: Passing Rust strings/structs across FFI is error-prone
2. **Context Ownership**: Narration-core needs Rust context (correlation IDs, job IDs, etc.)
3. **Performance**: CUDA kernels should be pure compute (no I/O)
4. **Simplicity**: Single observability path = easier debugging
5. **Safety**: Rust's type system protects narration fields

---

## M0 Observability Pattern

### Pattern: Narrate at FFI Boundaries

All narration happens **immediately before/after** FFI calls in the Rust layer.

#### Example 1: Model Loading

**C++ Layer** (`cuda/src/model.cpp`):
```cpp
extern "C" CudaModel* cuda_load_model(
    CudaContext* ctx,
    const char* model_path,
    uint64_t* vram_bytes_used,
    int* error_code
) {
    try {
        auto model = std::make_unique<Model>(ctx, model_path);
        *vram_bytes_used = model->vram_bytes();
        *error_code = CUDA_SUCCESS;
        return reinterpret_cast<CudaModel*>(model.release());
    } catch (const CudaError& e) {
        *error_code = e.code();
        return nullptr;
    }
}
```

**Rust Layer** (`src/cuda_ffi.rs`):
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD};

pub fn load_model(
    ctx: *mut CudaContext,
    model_path: &str,
    correlation_id: &str,
) -> Result<*mut CudaModel, CudaError> {
    // Narrate BEFORE FFI call
    Narration::new(ACTOR_WORKER_ORCD, "model_load_start", model_path)
        .human(format!("Loading model from {}", model_path))
        .correlation_id(correlation_id)
        .emit();

    let path_cstr = CString::new(model_path)?;
    let mut vram_bytes = 0u64;
    let mut error_code = 0i32;

    let model = unsafe {
        cuda_load_model(ctx, path_cstr.as_ptr(), &mut vram_bytes, &mut error_code)
    };

    if model.is_null() {
        // Narrate ERROR
        let error_msg = get_cuda_error_message(error_code);
        Narration::new(ACTOR_WORKER_ORCD, "model_load_error", model_path)
            .human(format!("Failed to load model: {}", error_msg))
            .error_kind("ModelLoadFailed")
            .correlation_id(correlation_id)
            .emit_error();
        return Err(CudaError::from_code(error_code));
    }

    // Narrate SUCCESS
    Narration::new(ACTOR_WORKER_ORCD, "model_load_complete", model_path)
        .human(format!("Loaded model ({} MB VRAM)", vram_bytes / 1_000_000))
        .correlation_id(correlation_id)
        .emit();

    Ok(model)
}
```

**Key Points**:
- ✅ C++ layer is pure: returns error codes, no logging
- ✅ Rust layer narrates: before call, after success, after error
- ✅ Correlation IDs stay in Rust (no FFI complexity)
- ✅ Human-readable messages in Rust (string formatting is easy)

---

#### Example 2: Inference Execution

**C++ Layer** (`cuda/src/inference.cpp`):
```cpp
extern "C" bool cuda_inference_next_token(
    InferenceResult* result,
    char* token_out,
    int token_buffer_size,
    int* token_index,
    int* error_code
) {
    try {
        if (result->is_done()) {
            *error_code = CUDA_SUCCESS;
            return false;  // No more tokens
        }
        
        std::string token = result->next_token();
        strncpy(token_out, token.c_str(), token_buffer_size - 1);
        token_out[token_buffer_size - 1] = '\0';
        *token_index = result->current_token();
        *error_code = CUDA_SUCCESS;
        return true;
    } catch (const CudaError& e) {
        *error_code = e.code();
        return false;
    }
}
```

**Rust Layer** (`src/inference.rs`):
```rust
pub async fn stream_tokens(
    result: *mut InferenceResult,
    job_id: &str,
    correlation_id: &str,
) -> Result<impl Stream<Item = SseEvent>, CudaError> {
    // Narrate START
    Narration::new(ACTOR_WORKER_ORCD, "inference_start", job_id)
        .human(format!("Starting inference for job {}", job_id))
        .correlation_id(correlation_id)
        .job_id(job_id)
        .emit();

    let mut token_count = 0;
    let start_time = Instant::now();

    loop {
        let mut token_buffer = vec![0u8; 256];
        let mut token_index = 0;
        let mut error_code = 0;

        let has_more = unsafe {
            cuda_inference_next_token(
                result,
                token_buffer.as_mut_ptr() as *mut i8,
                token_buffer.len() as i32,
                &mut token_index,
                &mut error_code,
            )
        };

        if !has_more {
            if error_code != 0 {
                // Narrate ERROR
                let error_msg = get_cuda_error_message(error_code);
                Narration::new(ACTOR_WORKER_ORCD, "inference_error", job_id)
                    .human(format!("Inference failed: {}", error_msg))
                    .error_kind("InferenceFailed")
                    .correlation_id(correlation_id)
                    .job_id(job_id)
                    .emit_error();
                return Err(CudaError::from_code(error_code));
            }
            break;  // Done
        }

        token_count += 1;
        // Yield token via SSE...
    }

    // Narrate COMPLETE
    let duration_ms = start_time.elapsed().as_millis() as u64;
    Narration::new(ACTOR_WORKER_ORCD, "inference_complete", job_id)
        .human(format!("Completed inference: {} tokens in {} ms", token_count, duration_ms))
        .correlation_id(correlation_id)
        .job_id(job_id)
        .tokens_out(token_count)
        .duration_ms(duration_ms)
        .emit();

    Ok(stream)
}
```

**Key Points**:
- ✅ CUDA kernels run silently (pure compute)
- ✅ Rust narrates: start, error, complete
- ✅ Metrics (token count, duration) collected in Rust
- ✅ Correlation IDs never cross FFI boundary

---

### Pattern: CUDA Errors → Rust Narration

**C++ Layer**: Return error codes
```cpp
enum CudaErrorCode {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_OUT_OF_MEMORY = 1,
    CUDA_ERROR_INVALID_DEVICE = 2,
    CUDA_ERROR_MODEL_LOAD_FAILED = 3,
    CUDA_ERROR_INFERENCE_FAILED = 4,
    // ...
};
```

**Rust Layer**: Convert to narration
```rust
fn handle_cuda_error(
    error_code: i32,
    operation: &str,
    target: &str,
    correlation_id: &str,
) -> CudaError {
    let (error_kind, human_msg) = match error_code {
        1 => ("OutOfMemory", "VRAM exhausted"),
        2 => ("InvalidDevice", "Invalid GPU device"),
        3 => ("ModelLoadFailed", "Failed to load model"),
        4 => ("InferenceFailed", "Inference execution failed"),
        _ => ("Unknown", "Unknown CUDA error"),
    };

    Narration::new(ACTOR_WORKER_ORCD, operation, target)
        .human(format!("{}: {}", operation, human_msg))
        .error_kind(error_kind)
        .correlation_id(correlation_id)
        .emit_error();

    CudaError::from_code(error_code)
}
```

---

## M0 Narration Events (From Spec)

Per **M0-W-1900**, worker-orcd MUST emit these narration events:

| Event | When | Actor | Action | Target | Human Example |
|-------|------|-------|--------|--------|---------------|
| `startup` | Process starts | `worker-orcd` | `startup` | `worker-{id}` | "Worker starting on GPU 0" |
| `model_load_start` | Before `cuda_load_model()` | `worker-orcd` | `model_load_start` | `{model_path}` | "Loading model from /models/qwen.gguf" |
| `model_load_progress` | During load (0-100%) | `worker-orcd` | `model_load_progress` | `{model_path}` | "Loading model: 45% complete" |
| `model_load_complete` | After `cuda_load_model()` | `worker-orcd` | `model_load_complete` | `{model_path}` | "Loaded model (352 MB VRAM)" |
| `ready` | HTTP server started | `worker-orcd` | `ready` | `worker-{id}` | "Worker ready on port 8080" |
| `execute_start` | POST /execute received | `worker-orcd` | `execute_start` | `{job_id}` | "Starting inference for job-123" |
| `execute_end` | Inference complete | `worker-orcd` | `execute_end` | `{job_id}` | "Completed inference: 50 tokens in 150 ms" |
| `error` | Any error | `worker-orcd` | `{operation}` | `{target}` | "Failed to load model: VRAM exhausted" |
| `shutdown` | Process exits | `worker-orcd` | `shutdown` | `worker-{id}` | "Worker shutting down" |

**All events emitted from Rust layer only.**

---

## What About CUDA Kernel Debugging?

### Development: Use `printf()` in Kernels

For **development debugging only**, CUDA kernels can use `printf()`:

```cuda
__global__ void attention_kernel(...) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Attention: seq_len=%d, num_heads=%d\n", seq_len, num_heads);
    }
    // ... kernel logic
}
```

**Important**:
- ✅ Use `printf()` for **development debugging**
- ✅ Remove or `#ifdef DEBUG` guard before production
- ❌ Do NOT use `printf()` for observability/telemetry
- ❌ Do NOT try to call Rust narration from CUDA

### Production: Silent Kernels

In production, CUDA kernels should be **silent**:
- No `printf()`
- No logging
- Pure compute only
- Return via output buffers

**Why?**
- Performance: I/O in kernels is slow
- Clarity: All observability in one place (Rust)
- Safety: No mixed logging systems

---

## M0 Scope: Basic Events Only

Per the **hybrid scope decision**, M0 narration is **basic events only**:

**INCLUDED in M0**:
- ✅ Event lifecycle (startup, load, execute, error, shutdown)
- ✅ Correlation IDs
- ✅ Job IDs
- ✅ Model references
- ✅ GPU device IDs
- ✅ Human-readable messages

**DEFERRED to M1** (Performance Bundle):
- ❌ Performance metrics in logs (`vram_bytes`, `tokens_in`, `tokens_out`, `decode_time_ms`)
- ❌ Prometheus `/metrics` endpoint
- ❌ Sensitive data redaction

**M0 Behavior**: Basic logging without performance metrics (development/testing phase).

---

## Implementation Checklist

When implementing M0 worker observability:

### Rust Layer (src/*.rs)
- [ ] Use `observability_narration_core` v0.2.0 (builder pattern)
- [ ] Import constants: `ACTOR_WORKER_ORCD`, `ACTION_*`
- [ ] Narrate **before** FFI calls (e.g., `model_load_start`)
- [ ] Narrate **after** FFI success (e.g., `model_load_complete`)
- [ ] Narrate **after** FFI errors (e.g., `model_load_error`)
- [ ] Include correlation IDs in all narration
- [ ] Use `.emit()` for INFO, `.emit_error()` for ERROR, `.emit_warn()` for WARN
- [ ] Keep human messages ≤100 characters (ORCH-3305)

### C++/CUDA Layer (cuda/src/*.cpp, *.cu)
- [ ] Return error codes (not exceptions across FFI)
- [ ] Use `extern "C"` for all FFI functions
- [ ] No logging/narration in C++/CUDA
- [ ] Use `printf()` only for development debugging (remove before production)
- [ ] Keep kernels pure compute

### Testing
- [ ] Use `CaptureAdapter` to assert narration in tests
- [ ] Add `#[serial(capture_adapter)]` to tests
- [ ] Verify correlation IDs are propagated
- [ ] Test error paths emit proper narration

---

## Example: Complete Flow

### Scenario: Model Load with Error

**1. Rust calls FFI** (`src/cuda_ffi.rs`):
```rust
Narration::new(ACTOR_WORKER_ORCD, "model_load_start", model_path)
    .human(format!("Loading model from {}", model_path))
    .correlation_id(correlation_id)
    .emit();

let model = unsafe { cuda_load_model(ctx, path_cstr.as_ptr(), &mut vram_bytes, &mut error_code) };
```

**2. C++ attempts load** (`cuda/src/model.cpp`):
```cpp
extern "C" CudaModel* cuda_load_model(...) {
    try {
        // Check VRAM availability
        size_t free_bytes, total_bytes;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        
        if (free_bytes < required_bytes) {
            *error_code = CUDA_ERROR_OUT_OF_MEMORY;
            return nullptr;  // ← Error: insufficient VRAM
        }
        
        // ... load model
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    }
}
```

**3. Rust handles error** (`src/cuda_ffi.rs`):
```rust
if model.is_null() {
    let error_msg = match error_code {
        1 => "VRAM exhausted",
        _ => "Unknown error",
    };
    
    Narration::new(ACTOR_WORKER_ORCD, "model_load_error", model_path)
        .human(format!("Failed to load model: {}", error_msg))
        .error_kind("OutOfMemory")
        .correlation_id(correlation_id)
        .emit_error();
    
    return Err(CudaError::from_code(error_code));
}
```

**4. Logs show**:
```json
{
  "timestamp": "2025-10-04T18:09:47Z",
  "level": "INFO",
  "actor": "worker-orcd",
  "action": "model_load_start",
  "target": "/models/qwen.gguf",
  "human": "Loading model from /models/qwen.gguf",
  "correlation_id": "req-abc-123"
}
{
  "timestamp": "2025-10-04T18:09:48Z",
  "level": "ERROR",
  "actor": "worker-orcd",
  "action": "model_load_error",
  "target": "/models/qwen.gguf",
  "human": "Failed to load model: VRAM exhausted",
  "error_kind": "OutOfMemory",
  "correlation_id": "req-abc-123"
}
```

**Notice**:
- ✅ Clean narrative flow
- ✅ Correlation ID tracked
- ✅ Human-readable messages
- ✅ No C++ logging complexity

---

## FAQ

### Q: Can I use `tracing` in C++?

**A**: No. Rust's `tracing` crate is Rust-only. Use the pattern above (error codes → Rust narration).

### Q: What about CUDA kernel errors?

**A**: Check `cudaGetLastError()` in C++, return error code to Rust, narrate in Rust.

Example:
```cpp
extern "C" int cuda_run_kernel(...) {
    my_kernel<<<grid, block>>>(...);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return CUDA_ERROR_KERNEL_LAUNCH_FAILED;
    }
    
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return CUDA_ERROR_KERNEL_EXECUTION_FAILED;
    }
    
    return CUDA_SUCCESS;
}
```

```rust
let result = unsafe { cuda_run_kernel(...) };
if result != 0 {
    Narration::new(ACTOR_WORKER_ORCD, "kernel_error", "attention")
        .human("Attention kernel failed")
        .error_kind("KernelFailed")
        .emit_error();
}
```

### Q: What about model load progress (0-100%)?

**A**: C++ calls a Rust callback via FFI:

**C++ Layer**:
```cpp
typedef void (*ProgressCallback)(int percent);

extern "C" CudaModel* cuda_load_model_with_progress(
    CudaContext* ctx,
    const char* model_path,
    ProgressCallback callback,
    int* error_code
) {
    // ... loading
    callback(25);  // 25% done
    // ... more loading
    callback(50);  // 50% done
    // ...
}
```

**Rust Layer**:
```rust
extern "C" fn progress_callback(percent: i32) {
    Narration::new(ACTOR_WORKER_ORCD, "model_load_progress", "model")
        .human(format!("Loading model: {}% complete", percent))
        .emit();
}

let model = unsafe {
    cuda_load_model_with_progress(ctx, path, progress_callback, &mut error_code)
};
```

**Note**: This is the **only** case where C++ calls back to Rust for observability. Keep it simple.

### Q: Should I add cute/story fields in M0?

**A**: Optional! Cute/story fields are **always welcome** but not required for M0. Focus on solid `human` fields first.

Example with cute:
```rust
Narration::new(ACTOR_WORKER_ORCD, "model_load_complete", model_path)
    .human(format!("Loaded model (352 MB VRAM)"))
    .cute("Model is tucked safely into VRAM! Ready to generate! 🎉")
    .emit();
```

---

## Summary

### ✅ DO

- ✅ Emit all narration from **Rust layer only**
- ✅ Narrate **before/after** FFI calls
- ✅ Use error codes from C++, narrate in Rust
- ✅ Include correlation IDs in all events
- ✅ Keep C++/CUDA layer pure (compute only)
- ✅ Use `printf()` for development debugging (remove before production)

### ❌ DON'T

- ❌ Try to call Rust narration from C++/CUDA
- ❌ Pass Rust strings/structs across FFI for logging
- ❌ Add logging to CUDA kernels (production)
- ❌ Mix logging systems (Rust + C++)
- ❌ Forget correlation IDs

---

## References

- **M0 Spec**: `bin/.specs/01_M0_worker_orcd.md` (§13 Observability & Logging)
- **Narration Core**: `bin/shared-crates/narration-core/README.md`
- **Migration Guide**: `bin/worker-orcd/.plan/foundation-team/stories/FT-001-to-FT-010/NARRATION_V0.2.0_MIGRATION.md`
- **FFI Boundary**: M0-W-1050, M0-W-1051, M0-W-1052

---

*May your FFI boundaries be clean and your narration be delightful! 🎀*

— The Narration Core Team 💝
