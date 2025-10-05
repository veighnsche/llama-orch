# Next Steps: GGUF Dequantization Integration

**Status**: CUDA kernels complete, awaiting Rust FFI integration  
**Priority**: High (performance-critical path)  
**Team**: Llama Team  

## Completed ✅

1. ✅ Q6_K CUDA kernel (`kernels/q6_k_dequant.cu`)
2. ✅ Q5_0 CUDA kernel (`kernels/q5_0_dequant.cu`)
3. ✅ Q8_0 CUDA kernel (`kernels/q8_0_dequant.cu`)
4. ✅ FFI header (`kernels/gguf_dequant.cuh`)
5. ✅ CMake integration
6. ✅ Test suite (`tests/test_q6k_dequant.cu`)
7. ✅ Engineering rules update (`CODING_STANDARDS.md`)

## Remaining Work

### 1. Rust FFI Bindings (High Priority)

**File**: `bin/worker-orcd/src/cuda/gguf_dequant.rs` (new)

```rust
// Safe Rust wrapper for GGUF dequantization kernels

use crate::cuda::CudaError;
use half::f16;

extern "C" {
    fn q6k_dequant_launch(
        output: *mut f16,
        input: *const u8,
        num_elements: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    
    fn q5_0_dequant_launch(
        output: *mut f16,
        input: *const u8,
        num_elements: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    
    fn q8_0_dequant_launch(
        output: *mut f16,
        input: *const u8,
        num_elements: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

/// Dequantize Q6_K on GPU
pub fn dequantize_q6k_gpu(
    input: &[u8],
    num_elements: usize,
    stream: Option<CudaStream>,
) -> Result<DeviceBuffer<f16>, CudaError> {
    // Validate input
    if num_elements % 256 != 0 {
        return Err(CudaError::InvalidValue("num_elements must be multiple of 256"));
    }
    
    // Allocate device buffers
    let d_input = DeviceBuffer::from_slice(input)?;
    let mut d_output = DeviceBuffer::alloc(num_elements)?;
    
    // Launch kernel
    let stream_ptr = stream.map(|s| s.as_ptr()).unwrap_or(std::ptr::null_mut());
    let err = unsafe {
        q6k_dequant_launch(
            d_output.as_mut_ptr(),
            d_input.as_ptr(),
            num_elements as i32,
            stream_ptr,
        )
    };
    
    CudaError::from_code(err)?;
    Ok(d_output)
}

// Similar for Q5_0 and Q8_0...
```

**Tasks**:
- [ ] Create `src/cuda/gguf_dequant.rs`
- [ ] Add FFI declarations
- [ ] Implement safe wrappers
- [ ] Add error handling
- [ ] Add documentation

### 2. Integration with Weight Loading (High Priority)

**File**: `bin/worker-orcd/src/model/qwen_weight_loader.cpp` (modify)

Replace CPU dequantization with GPU kernel calls:

```cpp
// Before (CPU):
// auto weights_fp16 = worker_gguf::dequantize_q6_k(quantized_data, num_elements);
// cudaMemcpy(d_weights, weights_fp16.data(), ...);

// After (GPU):
cudaMemcpy(d_quantized, quantized_data, compressed_size, cudaMemcpyHostToDevice);
q6k_dequant_launch(d_weights, d_quantized, num_elements, stream);
cudaFree(d_quantized);
```

**Tasks**:
- [ ] Update `qwen_weight_loader.cpp` to use GPU dequant
- [ ] Remove CPU dequant calls
- [ ] Add stream synchronization
- [ ] Update error handling

### 3. Deprecate CPU Implementation (Per destructive-actions.md)

**Files to clean up**:
- `bin/worker-crates/worker-gguf/src/q6_k_dequant.rs` - Mark deprecated
- `bin/worker-crates/worker-gguf/src/q5_0_dequant.rs` - Mark deprecated
- `bin/worker-crates/worker-gguf/src/q8_0_dequant.rs` - Mark deprecated

**Approach**:
```rust
#[deprecated(
    since = "0.2.0",
    note = "Use CUDA GPU dequantization instead (100× faster). See bin/worker-orcd/cuda/kernels/q6_k_dequant.cu"
)]
pub fn dequantize_q6_k(input: &[u8], num_elements: usize) -> Vec<f16> {
    // Keep for fallback/testing only
}
```

**Tasks**:
- [ ] Add deprecation warnings
- [ ] Update documentation
- [ ] Add migration guide
- [ ] Schedule removal for v0.3.0

### 4. End-to-End Integration Test (High Priority)

**File**: `bin/worker-orcd/tests/test_gguf_e2e.rs` (new)

```rust
#[test]
fn test_q6k_dequant_e2e() {
    // Load real GGUF model
    let model_path = "test_data/qwen-2.5-0.5b-q6k.gguf";
    let metadata = GGUFMetadata::from_file(model_path)?;
    
    // Load quantized weights
    let tensors = metadata.parse_tensors(model_path)?;
    let weight_tensor = tensors.iter()
        .find(|t| t.name == "model.layers.0.attn.q_proj.weight")
        .unwrap();
    
    // Dequantize on GPU
    let d_weights = dequantize_q6k_gpu(
        &weight_tensor.data,
        weight_tensor.num_elements,
        None,
    )?;
    
    // Verify against CPU reference
    let cpu_weights = dequantize_q6_k(&weight_tensor.data, weight_tensor.num_elements);
    let gpu_weights = d_weights.to_host()?;
    
    for (i, (cpu, gpu)) in cpu_weights.iter().zip(gpu_weights.iter()).enumerate() {
        assert!((cpu.to_f32() - gpu.to_f32()).abs() < 0.01, 
            "Mismatch at index {}: CPU={}, GPU={}", i, cpu, gpu);
    }
}
```

**Tasks**:
- [ ] Create end-to-end test
- [ ] Add test GGUF model to `test_data/`
- [ ] Verify numerical accuracy
- [ ] Benchmark performance improvement

### 5. Performance Benchmarking (Medium Priority)

**File**: `bin/worker-orcd/benches/bench_gguf_dequant.rs` (new)

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_q6k_cpu(c: &mut Criterion) {
    let input = vec![0u8; 210 * 1000]; // 1000 blocks
    c.bench_function("q6k_dequant_cpu", |b| {
        b.iter(|| dequantize_q6_k(black_box(&input), 256_000))
    });
}

fn bench_q6k_gpu(c: &mut Criterion) {
    let input = vec![0u8; 210 * 1000]; // 1000 blocks
    c.bench_function("q6k_dequant_gpu", |b| {
        b.iter(|| dequantize_q6k_gpu(black_box(&input), 256_000, None))
    });
}

criterion_group!(benches, bench_q6k_cpu, bench_q6k_gpu);
criterion_main!(benches);
```

**Tasks**:
- [ ] Create benchmark suite
- [ ] Measure CPU vs GPU performance
- [ ] Document speedup metrics
- [ ] Add to CI pipeline

### 6. Documentation Updates (Low Priority)

**Files to update**:
- `bin/worker-orcd/README.md` - Add GGUF dequant section
- `bin/worker-crates/worker-gguf/README.md` - Mark CPU impl as deprecated
- `docs/PERFORMANCE.md` - Add dequant benchmarks

**Tasks**:
- [ ] Update README files
- [ ] Add architecture diagrams
- [ ] Document API changes
- [ ] Add migration guide

## Timeline Estimate

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Rust FFI bindings | High | 4 hours | None |
| Weight loader integration | High | 2 hours | FFI bindings |
| E2E integration test | High | 3 hours | Weight loader |
| Deprecate CPU impl | Medium | 1 hour | E2E test passing |
| Performance benchmarks | Medium | 2 hours | E2E test passing |
| Documentation | Low | 2 hours | All above |

**Total**: ~14 hours (2 days)

## Success Criteria

- ✅ All CUDA tests pass (`cuda_tests`)
- ✅ E2E test passes with real GGUF model
- ✅ GPU dequant is 50-100× faster than CPU
- ✅ Numerical accuracy within 0.01 of CPU reference
- ✅ No memory leaks (valgrind clean)
- ✅ CI pipeline green

## Risks & Mitigations

**Risk**: FFI boundary safety issues
- **Mitigation**: Extensive testing, bounds checking, error handling

**Risk**: Memory leaks in device buffers
- **Mitigation**: RAII wrappers, automated cleanup, valgrind testing

**Risk**: Numerical accuracy drift
- **Mitigation**: Reference comparison tests, tolerance validation

**Risk**: Performance regression on small tensors
- **Mitigation**: Benchmark suite, CPU fallback for small sizes

## References

- CUDA Kernels: `bin/worker-orcd/cuda/kernels/q6_k_dequant.cu`
- FFI Header: `bin/worker-orcd/cuda/kernels/gguf_dequant.cuh`
- Test Suite: `bin/worker-orcd/cuda/tests/test_q6k_dequant.cu`
- Engineering Rules: `CODING_STANDARDS.md` (Rust/CUDA/C++ section)
- Port Summary: `bin/worker-orcd/.plan/GGUF_DEQUANT_CUDA_PORT.md`

---

**Next Action**: Implement Rust FFI bindings in `src/cuda/gguf_dequant.rs`
