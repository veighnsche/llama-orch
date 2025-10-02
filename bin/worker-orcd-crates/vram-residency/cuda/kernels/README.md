# VRAM Residency CUDA Kernels

**Purpose**: CUDA kernels for VRAM operations and integrity verification  
**Security Tier**: TIER 1 (Critical)  
**Last Updated**: 2025-10-02

---

## Overview

This directory contains CUDA C++ kernels for vram-residency operations:

- **vram_ops.cu** - VRAM allocation, deallocation, and memory copy
- **digest_compute.cu** - GPU-accelerated SHA-256 digest computation (optional)

---

## Kernels

### 1. `vram_ops.cu` - VRAM Operations

**Purpose**: Safe wrappers around CUDA memory operations

**Functions**:
- `vram_malloc(size_t bytes)` - Allocate VRAM
- `vram_free(void* ptr)` - Deallocate VRAM
- `vram_memcpy_h2d(void* dst, const void* src, size_t bytes)` - Host to device
- `vram_memcpy_d2h(void* dst, const void* src, size_t bytes)` - Device to host
- `vram_get_info(size_t* free, size_t* total)` - Query VRAM capacity

**Security**:
- Bounds checking before all operations
- Error code validation
- No raw pointer exposure

---

### 2. `digest_compute.cu` - GPU SHA-256 (Optional)

**Purpose**: GPU-accelerated digest computation for large models

**Functions**:
- `gpu_sha256(const void* data, size_t bytes, uint8_t* digest)` - Compute SHA-256 on GPU

**Note**: This is optional. CPU-based SHA-256 is sufficient for most cases.

---

## Build System

CUDA kernels are compiled separately and linked into the Rust binary.

### Build Script

See `build.rs` in the crate root for CUDA compilation.

### Requirements

- CUDA Toolkit 11.0+
- nvcc compiler
- CUDA-capable GPU (Compute Capability 6.0+)

---

## Security Considerations

### TIER 1 Requirements

1. **Bounds Checking**
   - All operations validate size parameters
   - Overflow detection before allocation
   - Null pointer checks

2. **Error Handling**
   - All CUDA errors mapped to error codes
   - No silent failures
   - Fail-fast on driver errors

3. **No Raw Pointers**
   - All pointers wrapped in SafeCudaPtr
   - Automatic cleanup via RAII
   - No pointer arithmetic exposed

---

## Testing

CUDA kernels are tested via:
- Unit tests in Rust (via FFI)
- Integration tests with real GPU
- Bounds checking tests
- Error injection tests

---

## References

- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- CUDA Runtime API: https://docs.nvidia.com/cuda/cuda-runtime-api/
- Security Audit: `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md`
