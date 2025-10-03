# CUDA Implementation Overview (CUDA-5xxx)

**Status**: Draft  
**Applies to**: `bin/worker-orcd/cuda/`  
**Language**: C++17 + CUDA  
**Conformance**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

This spec provides an overview of the CUDA implementation within worker-orcd. It defines the architecture, module responsibilities, and how the C++/CUDA code integrates with the Rust binary.

**Parent spec**: `bin/worker-orcd/.specs/01_cuda_ffi_boundary.md`

---

## 1. Architecture Principles

### [CUDA-5001] Single Responsibility per Module
Each CUDA module MUST have a single, well-defined responsibility:
- `context` — CUDA device and context management
- `model` — Model loading and VRAM allocation
- `inference` — Inference execution and token generation
- `health` — VRAM residency and health checks

### [CUDA-5002] C API Boundary
All modules MUST expose functionality through a C API in `include/worker_cuda.h`. Internal C++ implementation MAY use modern C++ features (RAII, exceptions, templates).

### [CUDA-5003] Error Handling
- C++ code MAY use exceptions internally
- C API MUST use error codes (no exceptions across FFI)
- All errors MUST be convertible to stable error codes

### [CUDA-5004] Memory Management
- All CUDA memory MUST be managed via RAII wrappers
- No raw `cudaMalloc`/`cudaFree` outside of RAII classes
- Opaque handles passed to Rust MUST own their resources

---

## 2. Module Responsibilities

### [CUDA-5010] Context Module
**File**: `src/context.cpp`, `include/context.hpp`

**Responsibilities**:
- Initialize CUDA device
- Create CUDA context
- Set device flags (disable UMA, zero-copy)
- Query device capabilities
- Manage context lifetime

**C API**:
```c
CudaContext* cuda_init(int gpu_device, int* error_code);
void cuda_destroy(CudaContext* ctx);
int cuda_get_device_count();
```

**Spec**: `01_context.md`

---

### [CUDA-5011] Model Module
**File**: `src/model.cpp`, `include/model.hpp`

**Responsibilities**:
- Parse GGUF format
- Allocate VRAM for model weights
- Copy model from disk/RAM to VRAM
- Track VRAM usage
- Manage model lifetime

**C API**:
```c
CudaModel* cuda_load_model(
    CudaContext* ctx,
    const char* model_path,
    uint64_t* vram_bytes_used,
    int* error_code
);
void cuda_unload_model(CudaModel* model);
uint64_t cuda_get_vram_usage(CudaModel* model);
```

**Spec**: `02_model.md`

---

### [CUDA-5012] Inference Module
**File**: `src/inference.cu`, `include/inference.hpp`, `kernels/*.cu`

**Responsibilities**:
- Tokenize input prompt
- Allocate KV cache in VRAM
- Execute forward pass (CUDA kernels)
- Sample next token
- Stream tokens one-by-one
- Free inference resources

**C API**:
```c
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
```

**Spec**: `03_inference.md`

---

### [CUDA-5013] Health Module
**File**: `src/health.cpp`, `include/health.hpp`

**Responsibilities**:
- Verify VRAM residency (no RAM fallback)
- Check for CUDA errors
- Monitor memory leaks
- Report VRAM usage

**C API**:
```c
bool cuda_check_vram_residency(CudaModel* model, int* error_code);
uint64_t cuda_get_vram_usage(CudaModel* model);
bool cuda_check_device_health(CudaContext* ctx, int* error_code);
```

**Spec**: `04_health.md`

---

## 3. Internal C++ Architecture

### [CUDA-5020] RAII Wrappers
All CUDA resources MUST be wrapped in RAII classes:

```cpp
namespace worker_cuda {

class DeviceMemory {
public:
    DeviceMemory(size_t bytes);
    ~DeviceMemory() { cudaFree(ptr_); }
    void* get() const { return ptr_; }
private:
    void* ptr_;
};

class CudaStream {
public:
    CudaStream();
    ~CudaStream() { cudaStreamDestroy(stream_); }
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_;
};

} // namespace worker_cuda
```

### [CUDA-5021] Exception Safety
Internal C++ code MAY throw exceptions. C API functions MUST catch all exceptions and convert to error codes:

```cpp
extern "C" CudaModel* cuda_load_model(..., int* error_code) {
    try {
        auto model = std::make_unique<Model>(...);
        *error_code = 0;
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

### [CUDA-5022] Opaque Handles
Handles passed to Rust MUST be opaque pointers to C++ objects:

```cpp
// C API (worker_cuda.h)
typedef struct CudaModel CudaModel;

// C++ implementation (model.hpp)
namespace worker_cuda {
    class Model {
        // ... implementation
    };
}

// FFI implementation (ffi.cpp)
extern "C" CudaModel* cuda_load_model(...) {
    auto model = std::make_unique<worker_cuda::Model>(...);
    return reinterpret_cast<CudaModel*>(model.release());
}
```

---

## 4. CUDA Kernels

### [CUDA-5030] Kernel Organization
CUDA kernels MUST be organized by operation:

```
kernels/
├── attention.cu      # Attention mechanism kernels
├── matmul.cu         # Matrix multiplication kernels
├── sampling.cu       # Token sampling kernels
├── rope.cu           # Rotary position embeddings
└── common.cuh        # Shared kernel utilities
```

### [CUDA-5031] Kernel Launch Pattern
```cpp
// inference.cu
void run_forward_pass(const Model& model, InferenceState& state) {
    // Launch attention kernel
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
        throw CudaError(err);
    }
}
```

---

## 5. Error Handling

### [CUDA-5040] Error Code Definitions
```cpp
// errors.hpp
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

### [CUDA-5041] Error Messages
```cpp
// errors.cpp
extern "C" const char* cuda_error_message(int error_code) {
    switch (error_code) {
        case CUDA_SUCCESS: return "Success";
        case CUDA_ERROR_DEVICE_NOT_FOUND: return "CUDA device not found";
        case CUDA_ERROR_OUT_OF_MEMORY: return "Out of GPU memory";
        // ... etc
        default: return "Unknown error";
    }
}
```

---

## 6. Build System

### [CUDA-5050] CMake Configuration
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
)

target_include_directories(worker_cuda PUBLIC include)
target_link_libraries(worker_cuda cudart)

# Tests
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
```

### [CUDA-5051] Cargo Integration
```rust
// build.rs
fn main() {
    let dst = cmake::Config::new("cuda")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();
    
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=worker_cuda");
    println!("cargo:rustc-link-lib=cudart");
}
```

---

## 7. Testing

### [CUDA-5060] Unit Tests (C++)
```cpp
// tests/test_context.cpp
#include <gtest/gtest.h>
#include "context.hpp"

TEST(ContextTest, InitializeDevice) {
    int error_code;
    auto ctx = cuda_init(0, &error_code);
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(error_code, 0);
    cuda_destroy(ctx);
}
```

### [CUDA-5061] Integration Tests (Rust)
```rust
// tests/cuda_integration.rs
#[test]
fn test_load_model() {
    let ctx = unsafe { cuda_init(0, &mut 0) };
    assert!(!ctx.is_null());
    
    let mut vram_bytes = 0;
    let model = unsafe {
        cuda_load_model(ctx, c"model.gguf".as_ptr(), &mut vram_bytes, &mut 0)
    };
    assert!(!model.is_null());
    assert!(vram_bytes > 0);
}
```

---

## 8. Dependencies

### [CUDA-5070] Required
- **CUDA Toolkit**: 11.8+ or 12.x
- **CMake**: 3.18+
- **C++ Compiler**: GCC 9+, Clang 10+, or MSVC 2019+

### [CUDA-5071] Optional
- **Google Test**: For C++ unit tests
- **GGML**: For GGUF parsing (may vendor or implement)

---

## 9. Performance Considerations

### [CUDA-5080] Memory Coalescing
Kernels MUST access global memory in coalesced patterns for optimal bandwidth.

### [CUDA-5081] Stream Parallelism
Use CUDA streams for overlapping computation and memory transfers where possible.

### [CUDA-5082] Kernel Fusion
Fuse small kernels to reduce launch overhead.

---

## 10. Traceability

**Code**: `bin/worker-orcd/cuda/src/`, `bin/worker-orcd/cuda/include/`  
**Tests**: `bin/worker-orcd/cuda/tests/`  
**Parent**: `bin/worker-orcd/.specs/01_cuda_ffi_boundary.md`  
**Spec IDs**: CUDA-5001 to CUDA-5082

---

## 11. Related Specs

- **Context Management**: `01_context.md` (CUDA-5100 to CUDA-5199)
- **Model Loading**: `02_model.md` (CUDA-5200 to CUDA-5299)
- **Inference Execution**: `03_inference.md` (CUDA-5300 to CUDA-5399)
- **Health Monitoring**: `04_health.md` (CUDA-5400 to CUDA-5499)

---

**End of Specification**
