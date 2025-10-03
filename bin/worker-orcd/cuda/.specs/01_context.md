# CUDA Context Management (CUDA-5100)

**Status**: Draft  
**Module**: `context`  
**Files**: `src/context.cpp`, `include/context.hpp`  
**Conformance**: RFC-2119

---

## 0. Scope

The context module manages CUDA device initialization, context creation, and device configuration for the worker process.

**Parent**: `00_cuda_overview.md`

---

## 1. Responsibilities

### [CUDA-5101] Device Initialization
The module MUST initialize a CUDA device and create a CUDA context for the worker process.

### [CUDA-5102] VRAM-Only Enforcement
The module MUST configure device flags to enforce VRAM-only operation:
- Disable Unified Memory (UMA)
- Disable zero-copy mode
- Disable pinned host memory fallback

### [CUDA-5103] Single Context
The module MUST maintain exactly ONE CUDA context per worker process.

---

## 2. C API

### [CUDA-5110] Initialization
```c
// Initialize CUDA context on specified device
// Returns: Opaque context handle, or NULL on error
// error_code: Set to error code (0 = success)
CudaContext* cuda_init(int gpu_device, int* error_code);
```

### [CUDA-5111] Cleanup
```c
// Destroy CUDA context and free resources
void cuda_destroy(CudaContext* ctx);
```

### [CUDA-5112] Device Query
```c
// Get number of available CUDA devices
int cuda_get_device_count();

// Get device properties
void cuda_get_device_properties(
    int device,
    CudaDeviceProperties* props,
    int* error_code
);
```

---

## 3. C++ Implementation

### [CUDA-5120] Context Class
```cpp
// include/context.hpp
namespace worker_cuda {

class Context {
public:
    // Initialize CUDA device
    explicit Context(int gpu_device);
    ~Context();
    
    // Non-copyable, movable
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = default;
    Context& operator=(Context&&) = default;
    
    // Get CUDA device ID
    int device() const { return device_; }
    
    // Get device properties
    const cudaDeviceProp& properties() const { return props_; }
    
private:
    int device_;
    cudaDeviceProp props_;
};

} // namespace worker_cuda
```

### [CUDA-5121] Implementation
```cpp
// src/context.cpp
namespace worker_cuda {

Context::Context(int gpu_device) : device_(gpu_device) {
    // Check device exists
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_DEVICE_NOT_FOUND, 
                       "Failed to get device count");
    }
    
    if (gpu_device < 0 || gpu_device >= device_count) {
        throw CudaError(CUDA_ERROR_INVALID_DEVICE,
                       "Invalid device ID");
    }
    
    // Set device
    err = cudaSetDevice(gpu_device);
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_INVALID_DEVICE,
                       "Failed to set device");
    }
    
    // Get device properties
    err = cudaGetDeviceProperties(&props_, gpu_device);
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_DEVICE_NOT_FOUND,
                       "Failed to get device properties");
    }
    
    // Enforce VRAM-only mode
    // Disable managed memory (UMA)
    err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0);
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_INVALID_DEVICE,
                       "Failed to disable managed memory");
    }
    
    // Set cache config for compute
    err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_INVALID_DEVICE,
                       "Failed to set cache config");
    }
}

Context::~Context() {
    // Reset device
    cudaDeviceReset();
}

} // namespace worker_cuda
```

---

## 4. FFI Implementation

### [CUDA-5130] C API Wrapper
```cpp
// src/ffi.cpp
extern "C" {

CudaContext* cuda_init(int gpu_device, int* error_code) {
    try {
        auto ctx = std::make_unique<worker_cuda::Context>(gpu_device);
        *error_code = CUDA_SUCCESS;
        return reinterpret_cast<CudaContext*>(ctx.release());
    } catch (const worker_cuda::CudaError& e) {
        *error_code = e.code();
        return nullptr;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    }
}

void cuda_destroy(CudaContext* ctx) {
    if (ctx) {
        auto ptr = reinterpret_cast<worker_cuda::Context*>(ctx);
        delete ptr;
    }
}

int cuda_get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

} // extern "C"
```

---

## 5. Error Handling

### [CUDA-5140] Error Codes
- `CUDA_SUCCESS` (0) — Initialization successful
- `CUDA_ERROR_DEVICE_NOT_FOUND` (1) — No CUDA device found
- `CUDA_ERROR_INVALID_DEVICE` (3) — Invalid device ID

### [CUDA-5141] Error Recovery
If initialization fails, the module MUST:
1. Clean up any partial state
2. Return NULL pointer
3. Set error_code to appropriate value

---

## 6. Testing

### [CUDA-5150] Unit Tests
```cpp
// tests/test_context.cpp
TEST(ContextTest, InitializeValidDevice) {
    int error_code;
    auto ctx = cuda_init(0, &error_code);
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(error_code, CUDA_SUCCESS);
    cuda_destroy(ctx);
}

TEST(ContextTest, InitializeInvalidDevice) {
    int error_code;
    auto ctx = cuda_init(999, &error_code);
    ASSERT_EQ(ctx, nullptr);
    ASSERT_EQ(error_code, CUDA_ERROR_INVALID_DEVICE);
}

TEST(ContextTest, GetDeviceCount) {
    int count = cuda_get_device_count();
    ASSERT_GT(count, 0);
}
```

---

## 7. Traceability

**Code**: `cuda/src/context.cpp`, `cuda/include/context.hpp`  
**Tests**: `cuda/tests/test_context.cpp`  
**Parent**: `00_cuda_overview.md`  
**Spec IDs**: CUDA-5101 to CUDA-5150

---

**End of Specification**
