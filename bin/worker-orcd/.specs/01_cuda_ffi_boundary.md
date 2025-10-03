# Worker CUDA FFI Boundary SPEC (CUDA-4xxx)

**Status**: Draft  
**Applies to**: `bin/worker-orcd/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

This spec defines the **clear boundary between Rust and CUDA** in the worker-orcd binary. It establishes which operations happen in Rust vs C++/CUDA and how they communicate.

**Core Principle**: CUDA context is per-process and single-threaded. All GPU operations MUST happen in C++/CUDA code within a single CUDA context. Rust code MUST NOT directly call CUDA APIs.

---

## 1. Architecture Overview

### [CUDA-4001] Single CUDA Context
The worker MUST maintain exactly ONE CUDA context for its entire lifetime. All GPU operations MUST happen within this context.

### [CUDA-4002] Language Boundary
```
┌─────────────────────────────────────────────────────────────┐
│ RUST LAYER (src/*.rs)                                        │
│ • HTTP server (axum)                                         │
│ • CLI argument parsing                                       │
│ • SSE streaming                                              │
│ • Error handling and formatting                              │
│ • Callbacks to pool manager                                  │
│ • Logging and metrics                                        │
└────────────────────┬────────────────────────────────────────┘
                     │ FFI (unsafe extern "C")
┌────────────────────▼────────────────────────────────────────┐
│ C++/CUDA LAYER (src/cuda/*.cpp, *.cu)                       │
│ • CUDA context management                                    │
│ • VRAM allocation (cudaMalloc)                              │
│ • Model loading (disk → VRAM)                               │
│ • Inference execution (CUDA kernels)                         │
│ • VRAM residency checks                                      │
│ • All GPU operations                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Rust Layer Responsibilities

### [CUDA-4010] Rust-Only Operations
The Rust layer MUST handle:
- **HTTP server**: Axum server, routing, request parsing
- **SSE streaming**: Server-Sent Events to orchestrator
- **CLI parsing**: Command-line arguments (clap)
- **Error formatting**: Convert C++ errors to HTTP responses
- **Callbacks**: HTTP POST to pool manager
- **Logging**: Tracing/structured logs
- **Metrics**: Prometheus metrics collection

### [CUDA-4011] Rust MUST NOT
The Rust layer MUST NOT:
- ❌ Call CUDA APIs directly (no `cudaMalloc`, `cudaMemcpy`, etc.)
- ❌ Manage CUDA context
- ❌ Allocate VRAM
- ❌ Load model weights to GPU
- ❌ Execute inference
- ❌ Check VRAM residency

**Rationale**: CUDA context is per-process. Mixing Rust and C++ CUDA calls leads to context conflicts.

---

## 3. C++/CUDA Layer Responsibilities

### [CUDA-4020] CUDA-Only Operations
The C++/CUDA layer MUST handle:
- **Context management**: Initialize CUDA context, set device
- **VRAM allocation**: `cudaMalloc`, `cudaFree`
- **Model loading**: Read GGUF → parse → copy to VRAM
- **Inference**: Run CUDA kernels, generate tokens
- **Health checks**: Verify VRAM residency
- **Memory management**: All GPU memory lifecycle

### [CUDA-4021] CUDA Layer Structure
```
src/cuda/
├── ffi.h              # C API exposed to Rust
├── context.cpp        # CUDA context management
├── model.cpp          # Model loading (GGUF → VRAM)
├── inference.cpp      # Inference execution
├── health.cpp         # VRAM health checks
└── errors.cpp         # Error code definitions
```

---

## 4. FFI Interface

### [CUDA-4030] C API Design
The CUDA layer MUST expose a **C API** (not C++) for Rust FFI:

```c
// src/cuda/ffi.h

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
    char* token_out,      // Buffer for token (UTF-8)
    int* token_index,
    int* error_code
);
void cuda_inference_free(InferenceResult* result);

// Health checks
bool cuda_check_vram_residency(CudaModel* model, int* error_code);
uint64_t cuda_get_vram_usage(CudaModel* model);

// Error handling
const char* cuda_error_message(int error_code);
```

### [CUDA-4031] Rust FFI Bindings
```rust
// src/cuda/mod.rs

#[repr(C)]
pub struct CudaContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CudaModel {
    _private: [u8; 0],
}

extern "C" {
    pub fn cuda_init(gpu_device: i32, error_code: *mut i32) -> *mut CudaContext;
    pub fn cuda_destroy(ctx: *mut CudaContext);
    
    pub fn cuda_load_model(
        ctx: *mut CudaContext,
        model_path: *const c_char,
        vram_bytes_used: *mut u64,
        error_code: *mut i32,
    ) -> *mut CudaModel;
    
    pub fn cuda_unload_model(model: *mut CudaModel);
    
    // ... rest of bindings
}
```

### [CUDA-4032] Safe Rust Wrappers
```rust
// src/cuda/safe.rs

pub struct CudaContextHandle {
    ptr: *mut CudaContext,
}

impl CudaContextHandle {
    pub fn new(gpu_device: i32) -> Result<Self, CudaError> {
        let mut error_code = 0;
        let ptr = unsafe { cuda_init(gpu_device, &mut error_code) };
        if ptr.is_null() {
            return Err(CudaError::from_code(error_code));
        }
        Ok(Self { ptr })
    }
}

impl Drop for CudaContextHandle {
    fn drop(&mut self) {
        unsafe { cuda_destroy(self.ptr) };
    }
}
```

---

## 5. Data Flow Examples

### [CUDA-4040] Startup Flow
```
1. main.rs:
   - Parse CLI args
   - Initialize tracing
   
2. main.rs → cuda::safe::CudaContextHandle::new(gpu_device)
   ↓ FFI
3. cuda/context.cpp:
   - cudaSetDevice(gpu_device)
   - Initialize CUDA context
   - Disable UMA, zero-copy
   - Return context handle
   ↓ FFI
4. main.rs:
   - Receive context handle
   
5. main.rs → cuda::safe::load_model(ctx, path)
   ↓ FFI
6. cuda/model.cpp:
   - Open file, mmap
   - Parse GGUF format
   - cudaMalloc for weights
   - cudaMemcpy to GPU
   - Return model handle + vram_bytes
   ↓ FFI
7. main.rs:
   - Receive model handle
   - POST callback to pool manager with vram_bytes
   - Start HTTP server
```

### [CUDA-4041] Inference Flow
```
1. HTTP handler receives POST /execute
   
2. handler.rs → cuda::safe::inference_start(model, prompt, ...)
   ↓ FFI
3. cuda/inference.cpp:
   - Tokenize prompt
   - Allocate KV cache in VRAM
   - Prepare inference state
   - Return inference handle
   ↓ FFI
4. handler.rs:
   - Receive inference handle
   - Start SSE stream
   
5. Loop: handler.rs → cuda::safe::inference_next_token(inference)
   ↓ FFI
6. cuda/inference.cpp:
   - Run CUDA kernels (forward pass)
   - Sample next token
   - Return token string
   ↓ FFI
7. handler.rs:
   - Receive token
   - Send SSE event to client
   - Repeat until done
   
8. handler.rs → cuda::safe::inference_free(inference)
   ↓ FFI
9. cuda/inference.cpp:
   - cudaFree KV cache
   - Clean up inference state
```

### [CUDA-4042] Health Check Flow
```
1. Health monitor timer fires
   
2. health.rs → cuda::safe::check_vram_residency(model)
   ↓ FFI
3. cuda/health.cpp:
   - Query CUDA memory info
   - Verify model pointers are in VRAM
   - Check for RAM fallback
   - Return true/false
   ↓ FFI
4. health.rs:
   - If false: mark worker unhealthy, trigger shutdown
   - If true: continue
```

---

## 6. Error Handling

### [CUDA-4050] Error Codes
C++ layer MUST use integer error codes:
```c
// src/cuda/errors.h
#define CUDA_SUCCESS 0
#define CUDA_ERROR_DEVICE_NOT_FOUND 1
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_INVALID_DEVICE 3
#define CUDA_ERROR_MODEL_LOAD_FAILED 4
#define CUDA_ERROR_INFERENCE_FAILED 5
#define CUDA_ERROR_VRAM_RESIDENCY_FAILED 6
```

### [CUDA-4051] Rust Error Conversion
```rust
pub enum CudaError {
    DeviceNotFound,
    OutOfMemory { requested: u64, available: u64 },
    InvalidDevice(i32),
    ModelLoadFailed(String),
    InferenceFailed(String),
    VramResidencyFailed,
}

impl CudaError {
    pub fn from_code(code: i32) -> Self {
        match code {
            1 => Self::DeviceNotFound,
            2 => Self::OutOfMemory { requested: 0, available: 0 },
            // ... etc
        }
    }
}
```

---

## 7. Memory Management

### [CUDA-4060] Ownership Rules
- **C++ owns GPU memory**: All `cudaMalloc` and `cudaFree` happen in C++
- **Rust owns handles**: Rust holds opaque pointers, calls C++ to free
- **No shared ownership**: Either Rust or C++ owns a resource, never both

### [CUDA-4061] Cleanup on Drop
```rust
impl Drop for CudaModelHandle {
    fn drop(&mut self) {
        unsafe { cuda_unload_model(self.ptr) };
    }
}
```

### [CUDA-4062] Panic Safety
Rust MUST NOT panic while holding CUDA resources. Use `catch_unwind` at FFI boundary if needed.

---

## 8. Build System

### [CUDA-4070] Build Configuration
```toml
# Cargo.toml
[package]
name = "worker-orcd"
build = "build.rs"

[build-dependencies]
cc = "1.0"
```

```rust
// build.rs
fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-std=c++17")
        .file("src/cuda/context.cpp")
        .file("src/cuda/model.cpp")
        .file("src/cuda/inference.cpp")
        .file("src/cuda/health.cpp")
        .file("src/cuda/errors.cpp")
        .compile("worker_cuda");
    
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
```

---

## 9. Testing Strategy

### [CUDA-4080] Unit Tests
- **Rust tests**: Test HTTP handlers, error conversion (mock CUDA layer)
- **C++ tests**: Test CUDA functions independently (Google Test)

### [CUDA-4081] Integration Tests
- **FFI tests**: Test Rust ↔ C++ boundary with real CUDA
- **End-to-end**: Full worker startup → inference → shutdown

### [CUDA-4082] Mocking
For Rust unit tests without GPU:
```rust
#[cfg(test)]
mod mock_cuda {
    pub fn cuda_init(...) -> *mut CudaContext {
        // Return mock pointer
    }
}
```

---

## 10. What This Eliminates

### [CUDA-4090] No Separate Crates for CUDA Operations
These are NOT separate crates, they're just organizational concepts within the binary:

- ❌ `model-lifecycle` crate → Just `src/cuda/model.cpp`
- ❌ `vram-policy` crate → Just `src/cuda/context.cpp` (enforcement flags)
- ❌ `health-monitor` crate → Just `src/cuda/health.cpp`

### [CUDA-4091] All Functionality Integrated

All functionality is now integrated into the main binary:

- ✅ `inference-api` → HTTP routing (integrated into `src/http/`)
- ✅ `error-handler` → Error type conversions (integrated into `src/error.rs`)
- ✅ `capability-matcher` → Pure metadata (integrated into main binary logic)
- ✅ `model-lifecycle` → Model loading (integrated into `src/cuda/model.cpp`)
- ✅ `vram-policy` → VRAM enforcement (integrated into `src/cuda/context.cpp`)
- ✅ `health-monitor` → Health checks (integrated into `src/cuda/health.cpp`)

All functionality is now modules within the single binary.

---

## 11. Directory Structure

### [CUDA-4100] Final Worker Structure
```
bin/worker-orcd/
├── Cargo.toml
├── build.rs                    # Build C++/CUDA code
├── src/
│   ├── main.rs                 # Entry point
│   ├── startup.rs              # Startup sequence
│   ├── cuda/
│   │   ├── mod.rs              # FFI declarations
│   │   ├── safe.rs             # Safe Rust wrappers
│   │   ├── ffi.h               # C API header
│   │   ├── context.cpp         # CUDA context management
│   │   ├── model.cpp           # Model loading
│   │   ├── inference.cpp       # Inference execution
│   │   ├── health.cpp          # Health checks
│   │   └── errors.cpp          # Error handling
│   ├── http/
│   │   ├── mod.rs
│   │   ├── execute.rs          # POST /execute handler
│   │   ├── health.rs           # GET /health handler
│   │   └── streaming.rs        # SSE streaming
│   └── error.rs                # Rust error types
├── tests/
│   ├── integration_test.rs     # Full worker tests
│   └── cuda/
│       └── test_model.cpp      # C++ unit tests
└── .specs/
    ├── 00_worker-orcd.md
    └── 01_cuda_ffi_boundary.md  # This spec
```

---

## 12. Migration Path

### [CUDA-4110] Current State
- All CUDA functionality integrated into main binary
- No separate crates for any operations
- Single hybrid Rust + C++/CUDA binary

### [CUDA-4111] Architecture Confirmed
- All CUDA code in `src/cuda/*.cpp`
- Rust FFI bindings in `src/cuda/mod.rs`
- HTTP API in `src/http/`
- All functionality integrated into single binary

### [CUDA-4112] Completed Actions
1. ✅ All worker-orcd-crates removed and functionality integrated
2. ✅ Specs updated to reference integrated modules instead of crates
3. ✅ C++/CUDA layer implemented
4. ✅ Rust FFI bindings implemented
5. ✅ Binary architecture completed

---

## 13. Traceability

**Code**: `bin/worker-orcd/src/cuda/`  
**Tests**: `bin/worker-orcd/tests/cuda/`  
**Parent**: `bin/worker-orcd/.specs/00_worker-orcd.md`  
**Spec IDs**: CUDA-4001 to CUDA-4112

---

**End of Specification**
