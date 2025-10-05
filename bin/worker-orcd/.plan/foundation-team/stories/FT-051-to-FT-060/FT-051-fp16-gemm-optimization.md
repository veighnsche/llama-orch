# FT-051: FP16 GEMM Optimization

**Team**: Foundation-Alpha  
**Sprint**: Sprint 6 - FP16 Optimization (Post-M0)  
**Size**: M (2 days)  
**Priority**: High (Post-M0)  
**Spec Ref**: M0-W-1430, DEFERRED_WORK_BACKLOG.md

---

## ⚠️ Prerequisites

**Requires M0 completion:**
- Working FP32 GEMM implementation in `cuda/kernels/gemm.cu`
- Functional cuBLAS integration
- Performance baseline measurements

**Current State**: `gemm.cu` is a stub. Implement FP32 baseline first.

---

## Story Description

Implement FP16 matrix multiplication using cuBLAS for all transformer operations (attention Q·K^T, attention·V, FFN layers). FP16 GEMM reduces memory bandwidth by 50% and improves throughput on modern GPUs (Ampere, Ada).

---

## Acceptance Criteria

- [ ] Implement `cuda_gemm_fp16()` wrapper around `cublasGemmEx`
- [ ] Support both FP16 input/output and mixed precision (FP16 compute, FP32 accumulate)
- [ ] Add runtime precision selection (FP32/FP16/mixed)
- [ ] Validate numerical accuracy against FP32 baseline (tolerance: 1e-3)
- [ ] Benchmark FP16 vs FP32 GEMM performance (expect 1.5-2x speedup)
- [ ] Unit tests for all GEMM variants (prefill, decode, FFN)
- [ ] Integration with attention and FFN kernels
- [ ] Memory bandwidth profiling (expect ~50% reduction)

---

## Dependencies

**Upstream**: FT-050 (Haiku test, Day 76)  
**Downstream**: FT-052 (FP16 attention kernels)

---

## Technical Details

### Current State

`cuda/kernels/gemm.cu` is a stub with TODO comments. Only FP32 path exists.

### Implementation Plan

**Phase 1: cuBLAS FP16 Wrapper** (Day 1)

```cpp
// cuda/kernels/gemm.cu

extern "C" {

/**
 * FP16 GEMM using cuBLAS
 * 
 * Computes: C = alpha * A @ B + beta * C
 * 
 * @param handle cuBLAS handle (reused across calls)
 * @param M Rows of A and C
 * @param N Columns of B and C
 * @param K Columns of A, rows of B
 * @param A Device pointer to A [M, K] (FP16)
 * @param B Device pointer to B [K, N] (FP16)
 * @param C Device pointer to C [M, N] (FP16)
 * @param alpha Scalar multiplier (default 1.0)
 * @param beta Scalar multiplier for C (default 0.0)
 * @param use_tensor_cores Enable Tensor Core acceleration
 * @return 0 on success, error code on failure
 */
int cuda_gemm_fp16(
    cublasHandle_t handle,
    int M, int N, int K,
    const half* A,
    const half* B,
    half* C,
    float alpha,
    float beta,
    bool use_tensor_cores
) {
    // Validate dimensions
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "GEMM: Invalid dimensions M=%d, N=%d, K=%d\n", M, N, K);
        return -1;
    }
    
    // Check for integer overflow
    if ((long long)M * N > INT_MAX || (long long)K * N > INT_MAX) {
        fprintf(stderr, "GEMM: Dimension overflow\n");
        return -1;
    }
    
    // Set math mode for Tensor Cores
    if (use_tensor_cores) {
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    } else {
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    
    // Call cublasGemmEx with FP16 compute
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        A, CUDA_R_16F, M,
        B, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_16F, M,
        CUBLAS_COMPUTE_16F,  // FP16 compute
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "GEMM: cublasGemmEx failed with status %d\n", status);
        return -1;
    }
    
    return 0;
}

/**
 * Mixed precision GEMM (FP16 input, FP32 accumulate)
 * 
 * Higher numerical accuracy than pure FP16, still faster than FP32.
 */
int cuda_gemm_mixed(
    cublasHandle_t handle,
    int M, int N, int K,
    const half* A,
    const half* B,
    half* C,
    float alpha,
    float beta
) {
    // Validate dimensions
    if (M <= 0 || N <= 0 || K <= 0) {
        return -1;
    }
    
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        A, CUDA_R_16F, M,
        B, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_16F, M,
        CUBLAS_COMPUTE_32F,  // FP32 accumulate for accuracy
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Mixed GEMM failed with status %d\n", status);
        return -1;
    }
    
    return 0;
}

} // extern "C"
```

**Phase 2: Integration with Attention** (Day 1)

Update `cuda/kernels/gqa_attention.cu` to use FP16 GEMM:

```cpp
// Replace naive attention with cuBLAS GEMM calls
// 1. Q @ K^T -> scores [batch, num_heads, seq_q, seq_k]
cuda_gemm_fp16(handle, seq_q, seq_k, head_dim, Q, K_T, scores, 1.0f, 0.0f, true);

// 2. scores @ V -> output [batch, num_heads, seq_q, head_dim]
cuda_gemm_fp16(handle, seq_q, head_dim, seq_k, scores, V, output, 1.0f, 0.0f, true);
```

**Phase 3: Numerical Validation** (Day 2)

```cpp
// cuda/tests/test_gemm_fp16.cu

void test_gemm_fp16_accuracy() {
    // Compare FP16 vs FP32 GEMM
    // Tolerance: max absolute error < 1e-3
    
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float fp32_val = h_C_fp32[i];
        float fp16_val = __half2float(h_C_fp16[i]);
        float error = fabsf(fp32_val - fp16_val);
        max_error = fmaxf(max_error, error);
    }
    
    assert(max_error < 1e-3);
}
```

**Phase 4: Performance Benchmarking** (Day 2)

```cpp
// Benchmark suite
void benchmark_gemm_precision() {
    // Test cases:
    // - Small (M=1, N=896, K=896) - decode phase
    // - Medium (M=32, N=896, K=896) - prefill
    // - Large (M=128, N=4864, K=896) - FFN
    
    // Metrics:
    // - Throughput (GFLOPS)
    // - Memory bandwidth (GB/s)
    // - Latency (ms)
    
    // Expected results:
    // - FP16: 1.5-2x faster than FP32
    // - Mixed: 1.3-1.5x faster than FP32
    // - Memory bandwidth: ~50% reduction
}
```

---

## Files to Create/Modify

**Create**:
- `cuda/tests/test_gemm_fp16.cu` - FP16 GEMM unit tests
- `cuda/tests/benchmark_gemm.cu` - Performance benchmarks

**Modify**:
- `cuda/kernels/gemm.cu` - Add FP16 GEMM implementations
- `cuda/kernels/gqa_attention.cu` - Integrate FP16 GEMM
- `cuda/include/gemm.h` - Add FP16 function declarations
- `src/cuda/inference.rs` - Add precision selection API

---

## Testing Strategy

### Unit Tests (8 tests)

1. **test_gemm_fp16_basic** - Small matrix (32x32)
2. **test_gemm_fp16_large** - Large matrix (1024x1024)
3. **test_gemm_fp16_non_square** - Rectangular matrices
4. **test_gemm_mixed_accuracy** - Validate FP32 accumulate
5. **test_gemm_fp16_numerical** - Compare vs FP32 baseline
6. **test_gemm_tensor_cores** - Verify Tensor Core usage
7. **test_gemm_edge_cases** - M=1, N=1, K=1
8. **test_gemm_overflow** - Large dimension validation

### Integration Tests (3 tests)

1. **test_attention_fp16** - Full attention with FP16 GEMM
2. **test_ffn_fp16** - FFN layers with FP16
3. **test_inference_fp16** - End-to-end inference

### Performance Tests (3 benchmarks)

1. **bench_gemm_decode** - Single token (M=1)
2. **bench_gemm_prefill** - Batch tokens (M=32, 128)
3. **bench_gemm_ffn** - FFN dimensions

---

## Performance Targets

| Operation | FP32 (ms) | FP16 (ms) | Speedup | Memory BW Reduction |
|-----------|-----------|-----------|---------|---------------------|
| Decode GEMM (1x896x896) | 0.05 | 0.03 | 1.67x | 50% |
| Prefill GEMM (32x896x896) | 0.8 | 0.5 | 1.6x | 50% |
| FFN GEMM (128x4864x896) | 5.0 | 3.0 | 1.67x | 50% |

**Note**: Actual speedup depends on GPU architecture (Ampere/Ada have better FP16 performance).

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing (8 tests)
- [ ] Integration tests passing (3 tests)
- [ ] Performance benchmarks complete
- [ ] Numerical accuracy validated (< 1e-3 error)
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-05

---

## References

- cuBLAS GEMM documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmEx
- Tensor Core programming guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-cores
- DEFERRED_WORK_BACKLOG.md: FT-021 (FP16 Sampling Support)

---
Built by Foundation-Alpha 🏗️
