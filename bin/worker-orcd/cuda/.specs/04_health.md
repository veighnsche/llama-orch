# CUDA Health Monitoring (CUDA-5400)

**Status**: Draft  
**Module**: `health`  
**Files**: `src/health.cpp`, `include/health.hpp`  
**Conformance**: RFC-2119

---

## 0. Scope

The health module verifies VRAM residency, detects memory leaks, and monitors GPU health. It ensures the model remains in VRAM and no RAM fallback occurs.

**Parent**: `00_cuda_overview.md`

---

## 1. Responsibilities

### [CUDA-5401] VRAM Residency Verification
The module MUST verify that model weights remain in GPU VRAM (no RAM fallback).

### [CUDA-5402] Memory Leak Detection
The module SHOULD detect CUDA memory leaks.

### [CUDA-5403] GPU Health Checks
The module SHOULD monitor GPU health (temperature, errors).

---

## 2. C API

### [CUDA-5410] VRAM Residency Check
```c
// Verify model is still in VRAM (no RAM fallback)
// Returns: true if model in VRAM, false if corrupted/swapped
bool cuda_check_vram_residency(CudaModel* model, int* error_code);
```

### [CUDA-5411] VRAM Usage Query
```c
// Get current VRAM usage for model
uint64_t cuda_get_vram_usage(CudaModel* model);

// Get total VRAM allocated by process
uint64_t cuda_get_process_vram_usage(CudaContext* ctx);
```

### [CUDA-5412] Device Health
```c
// Check GPU health (temperature, ECC errors)
bool cuda_check_device_health(CudaContext* ctx, int* error_code);

// Get device temperature
int cuda_get_device_temperature(CudaContext* ctx);
```

---

## 3. C++ Implementation

### [CUDA-5420] Health Class
```cpp
// include/health.hpp
namespace worker_cuda {

struct HealthStatus {
    bool vram_resident;
    uint64_t vram_bytes_used;
    uint64_t vram_bytes_free;
    int temperature_celsius;
    bool has_ecc_errors;
};

class Health {
public:
    // Check VRAM residency
    static bool check_vram_residency(const Model& model);
    
    // Get VRAM usage
    static uint64_t get_vram_usage(const Model& model);
    static uint64_t get_process_vram_usage(const Context& ctx);
    
    // Check device health
    static HealthStatus check_device_health(const Context& ctx);
    
private:
    static bool verify_pointer_in_vram(const void* ptr, size_t size);
};

} // namespace worker_cuda
```

### [CUDA-5421] VRAM Residency Implementation
```cpp
// src/health.cpp
namespace worker_cuda {

bool Health::check_vram_residency(const Model& model) {
    // Get pointer attributes
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(
        &attrs,
        model.weights()
    );
    
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

uint64_t Health::get_vram_usage(const Model& model) {
    return model.vram_bytes();
}

uint64_t Health::get_process_vram_usage(const Context& ctx) {
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    
    if (err != cudaSuccess) {
        return 0;
    }
    
    return total_bytes - free_bytes;
}

HealthStatus Health::check_device_health(const Context& ctx) {
    HealthStatus status = {};
    
    // Get VRAM info
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    status.vram_bytes_used = total_bytes - free_bytes;
    status.vram_bytes_free = free_bytes;
    
    // Get temperature (NVML required)
    // Note: This requires NVML, not CUDA Runtime
    // Pool manager should handle this via gpu-inventory
    status.temperature_celsius = -1;  // Not available via CUDA Runtime
    
    // Check for ECC errors
    cudaError_t err = cudaDeviceGetAttribute(
        &status.has_ecc_errors,
        cudaDevAttrEccEnabled,
        ctx.device()
    );
    
    status.vram_resident = true;  // Assume true if we got here
    
    return status;
}

} // namespace worker_cuda
```

---

## 4. FFI Implementation

### [CUDA-5430] C API Wrapper
```cpp
// src/ffi.cpp
extern "C" {

bool cuda_check_vram_residency(CudaModel* model, int* error_code) {
    try {
        auto model_ptr = reinterpret_cast<worker_cuda::Model*>(model);
        bool resident = worker_cuda::Health::check_vram_residency(*model_ptr);
        *error_code = CUDA_SUCCESS;
        return resident;
    } catch (const worker_cuda::CudaError& e) {
        *error_code = e.code();
        return false;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return false;
    }
}

uint64_t cuda_get_vram_usage(CudaModel* model) {
    if (!model) return 0;
    auto model_ptr = reinterpret_cast<worker_cuda::Model*>(model);
    return worker_cuda::Health::get_vram_usage(*model_ptr);
}

uint64_t cuda_get_process_vram_usage(CudaContext* ctx) {
    if (!ctx) return 0;
    auto ctx_ptr = reinterpret_cast<worker_cuda::Context*>(ctx);
    return worker_cuda::Health::get_process_vram_usage(*ctx_ptr);
}

bool cuda_check_device_health(CudaContext* ctx, int* error_code) {
    try {
        auto ctx_ptr = reinterpret_cast<worker_cuda::Context*>(ctx);
        auto status = worker_cuda::Health::check_device_health(*ctx_ptr);
        *error_code = CUDA_SUCCESS;
        return status.vram_resident && !status.has_ecc_errors;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return false;
    }
}

} // extern "C"
```

---

## 5. Error Handling

### [CUDA-5440] Error Codes
- `CUDA_SUCCESS` (0) — Health check passed
- `CUDA_ERROR_VRAM_RESIDENCY_FAILED` (6) — Model not in VRAM

---

## 6. Testing

### [CUDA-5450] Unit Tests
```cpp
// tests/test_health.cpp
TEST(HealthTest, CheckVramResidency) {
    auto ctx = create_test_context();
    auto model = load_test_model(ctx);
    
    int error_code;
    bool resident = cuda_check_vram_residency(model, &error_code);
    
    ASSERT_TRUE(resident);
    ASSERT_EQ(error_code, CUDA_SUCCESS);
}

TEST(HealthTest, GetVramUsage) {
    auto model = load_test_model();
    
    uint64_t usage = cuda_get_vram_usage(model);
    ASSERT_GT(usage, 0);
}

TEST(HealthTest, GetProcessVramUsage) {
    auto ctx = create_test_context();
    
    uint64_t usage = cuda_get_process_vram_usage(ctx);
    ASSERT_GT(usage, 0);
}
```

---

## 7. Notes

### [CUDA-5460] NVML vs CUDA Runtime
- **CUDA Runtime**: Can check VRAM residency, memory usage
- **NVML**: Required for temperature, power, utilization
- **Decision**: Worker uses CUDA Runtime only; pool manager uses NVML for system-wide monitoring

---

## 8. Traceability

**Code**: `cuda/src/health.cpp`, `cuda/include/health.hpp`  
**Tests**: `cuda/tests/test_health.cpp`  
**Parent**: `00_cuda_overview.md`  
**Spec IDs**: CUDA-5401 to CUDA-5460

---

**End of Specification**
