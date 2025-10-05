# Future Work - Architecture Changes

**Purpose**: Document planned architecture changes that are currently stubbed  
**Status**: Planning phase  
**Owner**: Foundation-Alpha

---

## Overview

The worker-orcd codebase contains several `TODO(ARCH-CHANGE)` markers indicating planned architecture changes. These are **intentional stubs** - the current implementation provides the correct interface while deferring actual CUDA implementation to a future phase.

**Why stubs?**
1. **Interface-first design**: Lock the API before implementation
2. **Parallel development**: Allow model teams to integrate while CUDA work continues
3. **Testing infrastructure**: Build tests against stable interfaces
4. **Risk mitigation**: Validate design before committing to implementation

---

## CUDA FFI Implementation (Phase 3)

**File**: `src/cuda_ffi/mod.rs`  
**Priority**: High  
**Estimated Effort**: 2 weeks  
**Dependencies**: CUDA toolkit, GPU hardware

### Current State

All CUDA operations are stubbed with correct error handling and type signatures:

```rust
// Current: Stub implementation
pub fn copy_to_device(&self, data: &[u8]) -> Result<SafeCudaPtr> {
    if data.is_empty() {
        return Err(CudaError::InvalidParameter("Empty data".to_string()));
    }
    
    // TODO(ARCH-CHANGE): Implement actual CUDA memcpy
    Ok(SafeCudaPtr { ptr: 0xDEADBEEF, len: data.len() })
}
```

### Required Changes

#### 1. CUDA Memory Copy (Lines 102, 133)

**Current**: Returns dummy pointers  
**Required**: Actual `cudaMemcpy` calls

```rust
// Future implementation
pub fn copy_to_device(&self, data: &[u8]) -> Result<SafeCudaPtr> {
    if data.is_empty() {
        return Err(CudaError::InvalidParameter("Empty data".to_string()));
    }
    
    let mut device_ptr: *mut c_void = std::ptr::null_mut();
    
    // Allocate device memory
    let status = unsafe {
        cudaMalloc(&mut device_ptr as *mut *mut c_void, data.len())
    };
    
    if status != cudaSuccess {
        return Err(CudaError::AllocationFailed(data.len()));
    }
    
    // Copy to device
    let status = unsafe {
        cudaMemcpy(
            device_ptr,
            data.as_ptr() as *const c_void,
            data.len(),
            cudaMemcpyHostToDevice
        )
    };
    
    if status != cudaSuccess {
        unsafe { cudaFree(device_ptr); }
        return Err(CudaError::MemcpyFailed("Host to device".to_string()));
    }
    
    Ok(SafeCudaPtr {
        ptr: device_ptr as usize,
        len: data.len(),
    })
}
```

**Testing Requirements**:
- [ ] Test with various data sizes (1 byte to 1 GB)
- [ ] Test error handling (out of memory)
- [ ] Test concurrent copies
- [ ] Benchmark copy performance

#### 2. CUDA Memory Deallocation (Line 152)

**Current**: Logs but doesn't free  
**Required**: Actual `cudaFree` call

```rust
// Future implementation
impl Drop for SafeCudaPtr {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let status = unsafe {
                cudaFree(self.ptr as *mut c_void)
            };
            
            if status != cudaSuccess {
                // Don't panic in Drop, but log error
                tracing::error!(
                    ptr = self.ptr,
                    len = self.len,
                    "Failed to free CUDA memory: {:?}",
                    status
                );
            } else {
                tracing::debug!(
                    ptr = self.ptr,
                    len = self.len,
                    "Freed CUDA memory"
                );
            }
        }
    }
}
```

**Testing Requirements**:
- [ ] Verify no memory leaks with Valgrind
- [ ] Test Drop in error paths
- [ ] Test Drop with multiple allocations
- [ ] Verify cleanup on panic

#### 3. CUDA Context Initialization (Line 195)

**Current**: Returns success without initialization  
**Required**: Actual device selection and context setup

```rust
// Future implementation
pub fn new(device_id: i32, vram_limit_bytes: usize) -> Result<Self> {
    tracing::info!(
        device_id = device_id,
        vram_limit_bytes = vram_limit_bytes,
        "Initializing CUDA context"
    );
    
    // Set device
    let status = unsafe { cudaSetDevice(device_id) };
    if status != cudaSuccess {
        return Err(CudaError::InitializationFailed(
            format!("Failed to set device {}: {:?}", device_id, status)
        ));
    }
    
    // Initialize cuBLAS
    let mut cublas_handle: cublasHandle_t = std::ptr::null_mut();
    let status = unsafe { cublasCreate_v2(&mut cublas_handle) };
    if status != CUBLAS_STATUS_SUCCESS {
        return Err(CudaError::InitializationFailed(
            format!("Failed to create cuBLAS handle: {:?}", status)
        ));
    }
    
    // Validate GPU availability
    let mut device_props: cudaDeviceProp = unsafe { std::mem::zeroed() };
    let status = unsafe { cudaGetDeviceProperties(&mut device_props, device_id) };
    if status != cudaSuccess {
        unsafe { cublasDestroy_v2(cublas_handle); }
        return Err(CudaError::InitializationFailed(
            format!("Failed to get device properties: {:?}", status)
        ));
    }
    
    tracing::info!(
        device_name = std::str::from_utf8(&device_props.name).unwrap_or("unknown"),
        compute_capability = format!("{}.{}", device_props.major, device_props.minor),
        "CUDA context initialized"
    );
    
    Ok(Self {
        device_id,
        vram_limit_bytes,
        cublas_handle: Some(cublas_handle),
    })
}
```

**Testing Requirements**:
- [ ] Test with valid device IDs
- [ ] Test with invalid device IDs
- [ ] Test with multiple contexts
- [ ] Verify cuBLAS handle creation

#### 4. CUDA Memory Allocation (Line 211)

**Current**: Returns dummy pointer  
**Required**: Actual `cudaMalloc` call

```rust
// Future implementation
pub fn allocate(&self, size: usize) -> Result<SafeCudaPtr> {
    if size > self.vram_limit_bytes {
        return Err(CudaError::AllocationFailed(size));
    }
    
    let mut device_ptr: *mut c_void = std::ptr::null_mut();
    
    let status = unsafe {
        cudaMalloc(&mut device_ptr as *mut *mut c_void, size)
    };
    
    if status != cudaSuccess {
        return Err(CudaError::AllocationFailed(size));
    }
    
    tracing::debug!(
        ptr = device_ptr as usize,
        size = size,
        "Allocated CUDA memory"
    );
    
    Ok(SafeCudaPtr {
        ptr: device_ptr as usize,
        len: size,
    })
}
```

**Testing Requirements**:
- [ ] Test allocation sizes (1 byte to max VRAM)
- [ ] Test allocation failure handling
- [ ] Test fragmentation scenarios
- [ ] Benchmark allocation performance

#### 5. VRAM Query (Lines 228, 236)

**Current**: Returns hardcoded 24GB  
**Required**: Actual `cudaMemGetInfo` call

```rust
// Future implementation
pub fn get_free_vram(&self) -> Result<usize> {
    let mut free_bytes: usize = 0;
    let mut total_bytes: usize = 0;
    
    let status = unsafe {
        cudaMemGetInfo(&mut free_bytes, &mut total_bytes)
    };
    
    if status != cudaSuccess {
        return Err(CudaError::QueryFailed(
            format!("Failed to query VRAM: {:?}", status)
        ));
    }
    
    Ok(free_bytes)
}

pub fn get_total_vram(&self) -> Result<usize> {
    let mut free_bytes: usize = 0;
    let mut total_bytes: usize = 0;
    
    let status = unsafe {
        cudaMemGetInfo(&mut free_bytes, &mut total_bytes)
    };
    
    if status != cudaSuccess {
        return Err(CudaError::QueryFailed(
            format!("Failed to query VRAM: {:?}", status)
        ));
    }
    
    Ok(total_bytes)
}
```

**Testing Requirements**:
- [ ] Test VRAM query accuracy
- [ ] Test after allocations
- [ ] Test with multiple contexts
- [ ] Verify against nvidia-smi

---

## Implementation Plan

### Phase 1: CUDA Bindings (Week 1)
- [ ] Add CUDA FFI declarations
- [ ] Create safe wrappers for CUDA functions
- [ ] Add error code conversions
- [ ] Write unit tests for bindings

### Phase 2: Memory Management (Week 2)
- [ ] Implement `cudaMalloc` / `cudaFree`
- [ ] Implement `cudaMemcpy` (H2D, D2H, D2D)
- [ ] Implement `SafeCudaPtr` with real cleanup
- [ ] Add memory leak tests

### Phase 3: Context Management (Week 3)
- [ ] Implement device selection
- [ ] Implement cuBLAS initialization
- [ ] Implement context cleanup
- [ ] Add multi-GPU support

### Phase 4: Integration & Testing (Week 4)
- [ ] Integration tests with real GPU
- [ ] Performance benchmarks
- [ ] Memory leak validation
- [ ] Documentation updates

---

## Testing Strategy

### Unit Tests
- Test each CUDA function wrapper independently
- Mock CUDA errors to test error handling
- Verify memory safety with Miri (where possible)

### Integration Tests
- Test full allocation/copy/free cycles
- Test error recovery
- Test concurrent operations
- Test multi-GPU scenarios

### Performance Tests
- Benchmark memory copy bandwidth
- Benchmark allocation/deallocation overhead
- Compare with theoretical peak performance
- Profile with NVIDIA Nsight

### Validation Tests
- Run with CUDA-MEMCHECK
- Run with Valgrind (host memory)
- Run with AddressSanitizer
- Verify no memory leaks over 1000 iterations

---

## Dependencies

### Required
- CUDA Toolkit 11.8+ (for CUDA 11.8 compatibility)
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- cuBLAS library
- CUDA runtime library

### Optional
- NVIDIA Nsight Systems (profiling)
- NVIDIA Nsight Compute (kernel profiling)
- cuda-gdb (debugging)
- CUDA-MEMCHECK (memory validation)

---

## Risks & Mitigations

### Risk 1: CUDA Version Compatibility
**Impact**: Code may not work on older/newer CUDA versions  
**Mitigation**: 
- Target CUDA 11.8 (widely available)
- Use feature flags for version-specific code
- Test on multiple CUDA versions

### Risk 2: Memory Leaks
**Impact**: VRAM exhaustion over time  
**Mitigation**:
- Comprehensive Drop implementations
- Memory leak tests in CI
- Regular validation with tools

### Risk 3: Multi-GPU Complexity
**Impact**: Context management becomes complex  
**Mitigation**:
- Start with single-GPU support
- Design for multi-GPU from start
- Add multi-GPU as Phase 5

### Risk 4: Performance Issues
**Impact**: Slower than expected  
**Mitigation**:
- Benchmark early and often
- Profile with NVIDIA tools
- Optimize hot paths first

---

## Success Criteria

- [ ] All CUDA operations implemented
- [ ] All tests passing on real GPU
- [ ] No memory leaks detected
- [ ] Performance within 90% of theoretical peak
- [ ] Documentation complete
- [ ] Integration tests passing
- [ ] Multi-GPU support (Phase 5)

---

## References

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [Rust CUDA Bindings](https://github.com/Rust-GPU/Rust-CUDA)

---

**Last Updated**: 2025-10-05  
**Status**: Planning phase - stubs intentional  
**Next Review**: Before Sprint 6

---
Built by Foundation-Alpha 🏗️
