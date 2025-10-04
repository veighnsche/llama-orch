# FT-016: cuBLAS GEMM Wrapper

**Team**: Foundation-Alpha  
**Sprint**: Sprint 3 - Shared Kernels  
**Size**: M (2 days)  
**Days**: 30 - 31  
**Spec Ref**: M0-W-1430, M0-W-1031, CUDA-5030

---

## Story Description

Implement wrapper around cuBLAS GEMM (General Matrix Multiply) for matrix operations in transformer layers. This is the computational workhorse for attention and FFN layers, shared across all architectures.

---

## Acceptance Criteria

- [ ] cuBLAS handle initialized and managed per context
- [ ] GEMM wrapper supports FP16 (half precision) operations
- [ ] Wrapper supports transposed and non-transposed matrices
- [ ] Deterministic mode enabled (CUBLAS_PEDANTIC_MATH) for reproducibility
- [ ] Unit tests validate correctness against CPU reference
- [ ] Integration tests validate with real model dimensions
- [ ] Error handling for cuBLAS errors
- [ ] Performance benchmarks (TFLOPS) for common sizes
- [ ] Support for batched GEMM (future-proof)

---

## Dependencies

### Upstream (Blocks This Story)
- FT-013: Device memory RAII (Expected completion: Day 26)

### Downstream (This Story Blocks)
- FT-017: Temperature scaling needs GEMM for logits projection
- Llama/GPT teams need GEMM for attention and FFN

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/include/cublas_wrapper.h` - cuBLAS wrapper interface
- `bin/worker-orcd/cuda/src/cublas_wrapper.cpp` - Wrapper implementation
- `bin/worker-orcd/cuda/tests/cublas_test.cu` - Unit tests
- `bin/worker-orcd/cuda/CMakeLists.txt` - Link cublas library

### Key Interfaces
```cpp
// cublas_wrapper.h
#ifndef WORKER_CUBLAS_WRAPPER_H
#define WORKER_CUBLAS_WRAPPER_H

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "cuda_error.h"

namespace worker {

class CublasHandle {
public:
    /**
     * Create cuBLAS handle.
     * 
     * @param deterministic If true, enable deterministic mode
     * @throws CudaError if creation fails
     */
    explicit CublasHandle(bool deterministic = true);
    
    /**
     * Destroy cuBLAS handle.
     */
    ~CublasHandle();
    
    // Non-copyable, non-movable
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    
    /**
     * Get raw cuBLAS handle.
     */
    cublasHandle_t get() const { return handle_; }
    
    /**
     * Set CUDA stream for cuBLAS operations.
     */
    void set_stream(cudaStream_t stream);
    
private:
    cublasHandle_t handle_;
};

/**
 * GEMM operation: C = alpha * op(A) * op(B) + beta * C
 * 
 * Matrix dimensions:
 *   A: [M, K] if transA == false, [K, M] if transA == true
 *   B: [K, N] if transB == false, [N, K] if transB == true
 *   C: [M, N]
 * 
 * @param handle cuBLAS handle
 * @param transA Transpose A matrix
 * @param transB Transpose B matrix
 * @param M Number of rows in op(A) and C
 * @param N Number of columns in op(B) and C
 * @param K Number of columns in op(A) and rows in op(B)
 * @param alpha Scalar multiplier for A*B
 * @param A Device pointer to matrix A
 * @param lda Leading dimension of A
 * @param B Device pointer to matrix B
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C
 * @param C Device pointer to matrix C (input/output)
 * @param ldc Leading dimension of C
 * @throws CudaError if GEMM fails
 */
void gemm_fp16(
    const CublasHandle& handle,
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    int lda,
    const half* B,
    int ldb,
    float beta,
    half* C,
    int ldc
);

/**
 * Simplified GEMM: C = A * B (no transpose, alpha=1, beta=0)
 */
void gemm_simple_fp16(
    const CublasHandle& handle,
    int M,
    int N,
    int K,
    const half* A,
    const half* B,
    half* C
);

} // namespace worker

#endif // WORKER_CUBLAS_WRAPPER_H

// cublas_wrapper.cpp
#include "cublas_wrapper.h"
#include <cuda_runtime.h>

namespace worker {

CublasHandle::CublasHandle(bool deterministic) {
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CudaError(
            CUDA_ERROR_UNKNOWN,
            "Failed to create cuBLAS handle: " + std::to_string(status)
        );
    }
    
    if (deterministic) {
        // Enable deterministic mode (disable Tensor Cores if non-deterministic)
        status = cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH);
        if (status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle_);
            throw CudaError(
                CUDA_ERROR_UNKNOWN,
                "Failed to set deterministic math mode: " + std::to_string(status)
            );
        }
    }
}

CublasHandle::~CublasHandle() {
    if (handle_) {
        cublasDestroy(handle_);
    }
}

void CublasHandle::set_stream(cudaStream_t stream) {
    cublasStatus_t status = cublasSetStream(handle_, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CudaError(
            CUDA_ERROR_UNKNOWN,
            "Failed to set cuBLAS stream: " + std::to_string(status)
        );
    }
}

void gemm_fp16(
    const CublasHandle& handle,
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    int lda,
    const half* B,
    int ldb,
    float beta,
    half* C,
    int ldc
) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    // cuBLAS uses column-major layout, so we swap A and B
    // C = alpha * op(A) * op(B) + beta * C
    // becomes
    // C^T = alpha * op(B)^T * op(A)^T + beta * C^T
    cublasStatus_t status = cublasGemmEx(
        handle.get(),
        opB, opA,  // Swapped
        N, M, K,   // Swapped M and N
        &alpha,
        B, CUDA_R_16F, ldb,
        A, CUDA_R_16F, lda,
        &beta,
        C, CUDA_R_16F, ldc,
        CUBLAS_COMPUTE_32F,  // Use FP32 accumulation
        CUBLAS_GEMM_DEFAULT
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CudaError(
            CUDA_ERROR_KERNEL_LAUNCH_FAILED,
            "cuBLAS GEMM failed: " + std::to_string(status)
        );
    }
}

void gemm_simple_fp16(
    const CublasHandle& handle,
    int M,
    int N,
    int K,
    const half* A,
    const half* B,
    half* C
) {
    gemm_fp16(handle, false, false, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

} // namespace worker

// Unit tests
// cuda/tests/cublas_test.cu
#include <gtest/gtest.h>
#include "cublas_wrapper.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace worker;

class CublasTest : public ::testing::Test {
protected:
    void SetUp() override {
        handle = std::make_unique<CublasHandle>(true);
    }
    
    std::unique_ptr<CublasHandle> handle;
};

TEST_F(CublasTest, HandleCreation) {
    EXPECT_NE(handle->get(), nullptr);
}

TEST_F(CublasTest, SimpleMatrixMultiply) {
    // Test C = A * B
    // A: [2, 3], B: [3, 2], C: [2, 2]
    int M = 2, N = 2, K = 3;
    
    // Host matrices
    std::vector<half> h_A = {
        __float2half(1.0f), __float2half(2.0f), __float2half(3.0f),
        __float2half(4.0f), __float2half(5.0f), __float2half(6.0f)
    };
    std::vector<half> h_B = {
        __float2half(1.0f), __float2half(2.0f),
        __float2half(3.0f), __float2half(4.0f),
        __float2half(5.0f), __float2half(6.0f)
    };
    std::vector<half> h_C(M * N);
    
    // Device matrices
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // Perform GEMM
    gemm_simple_fp16(*handle, M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Expected result:
    // C[0,0] = 1*1 + 2*3 + 3*5 = 22
    // C[0,1] = 1*2 + 2*4 + 3*6 = 28
    // C[1,0] = 4*1 + 5*3 + 6*5 = 49
    // C[1,1] = 4*2 + 5*4 + 6*6 = 64
    EXPECT_NEAR(__half2float(h_C[0]), 22.0f, 0.1f);
    EXPECT_NEAR(__half2float(h_C[1]), 28.0f, 0.1f);
    EXPECT_NEAR(__half2float(h_C[2]), 49.0f, 0.1f);
    EXPECT_NEAR(__half2float(h_C[3]), 64.0f, 0.1f);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(CublasTest, IdentityMatrix) {
    // Test C = I * A = A
    int M = 3, N = 3, K = 3;
    
    // Identity matrix
    std::vector<half> h_I(M * K, __float2half(0.0f));
    for (int i = 0; i < M; ++i) {
        h_I[i * K + i] = __float2half(1.0f);
    }
    
    // Test matrix
    std::vector<half> h_A = {
        __float2half(1.0f), __float2half(2.0f), __float2half(3.0f),
        __float2half(4.0f), __float2half(5.0f), __float2half(6.0f),
        __float2half(7.0f), __float2half(8.0f), __float2half(9.0f)
    };
    std::vector<half> h_C(M * N);
    
    half *d_I, *d_A, *d_C;
    cudaMalloc(&d_I, M * K * sizeof(half));
    cudaMalloc(&d_A, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_I, h_I.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    gemm_simple_fp16(*handle, M, N, K, d_I, d_A, d_C);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Result should equal A
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(__half2float(h_C[i]), __half2float(h_A[i]), 0.01f);
    }
    
    cudaFree(d_I);
    cudaFree(d_A);
    cudaFree(d_C);
}

TEST_F(CublasTest, LargeDimensions) {
    // Test with realistic transformer dimensions
    int M = 512, N = 512, K = 512;  // Typical hidden dimension
    
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<half> h_C(M * N);
    
    // Initialize with small values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = __float2half(0.01f);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = __float2half(0.01f);
    }
    
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    gemm_simple_fp16(*handle, M, N, K, d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate TFLOPS
    double flops = 2.0 * M * N * K;  // 2 ops per multiply-add
    double tflops = (flops / (milliseconds / 1000.0)) / 1e12;
    
    std::cout << "GEMM [" << M << "x" << N << "x" << K << "]: "
              << milliseconds << " ms, "
              << tflops << " TFLOPS" << std::endl;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
```

### Implementation Notes
- cuBLAS uses column-major layout (Fortran convention), so we swap A/B and M/N
- FP32 accumulation (CUBLAS_COMPUTE_32F) for numerical stability
- Deterministic mode (CUBLAS_PEDANTIC_MATH) ensures reproducibility
- Leading dimensions (lda, ldb, ldc) handle strided memory access
- Error checking after every cuBLAS call
- Performance benchmarking for optimization validation

---

## Testing Strategy

### Unit Tests
- Test handle creation succeeds
- Test simple matrix multiply (2x3 * 3x2)
- Test identity matrix multiplication
- Test large dimensions (512x512x512)
- Test deterministic mode (same inputs → same outputs)

### Integration Tests
- Test with real transformer dimensions (e.g., 768x768x768)
- Test performance benchmarks (TFLOPS)
- Test memory efficiency (no unnecessary copies)

### Manual Verification
1. Run unit tests: `./build/tests/cublas_test`
2. Profile with nvprof: `nvprof --metrics flop_count_sp ./build/tests/cublas_test`
3. Verify TFLOPS matches GPU specs

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (5+ tests)
- [ ] Integration tests passing (3+ tests)
- [ ] Documentation updated (cuBLAS wrapper docs)
- [ ] Performance benchmarks documented
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §9.4 Required Kernels (M0-W-1430)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §3.2 Reproducible CUDA Kernels (M0-W-1031)
- cuBLAS Documentation: https://docs.nvidia.com/cuda/cublas/

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **cuBLAS handle created**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "cublas_init",
       target: format!("GPU{}", device_id),
       device: Some(format!("GPU{}", device_id)),
       human: format!("cuBLAS handle created (deterministic={})", deterministic),
       ..Default::default()
   });
   ```

2. **GEMM operation completed**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "gemm_complete",
       target: format!("{}x{}x{}", M, N, K),
       device: Some(format!("GPU{}", device_id)),
       duration_ms: Some(elapsed.as_millis() as u64),
       human: format!("GEMM [{}x{}x{}] completed: {} ms ({:.2} TFLOPS)", M, N, K, elapsed.as_millis(), tflops),
       ..Default::default()
   });
   ```

3. **GEMM failure**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "gemm_complete",
       target: format!("{}x{}x{}", M, N, K),
       error_kind: Some("cublas_error".to_string()),
       human: format!("cuBLAS GEMM failed: {}", error),
       ..Default::default()
   });
   ```

**Why this matters**: GEMM is the computational workhorse of transformers. Narration helps track performance (TFLOPS) and diagnose cuBLAS errors.

---
*Narration guidance added by Narration-Core Team 🎀*
