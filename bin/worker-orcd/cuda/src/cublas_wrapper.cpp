/**
 * cuBLAS GEMM Wrapper Implementation
 * 
 * Implements RAII wrapper for cuBLAS handle and GEMM operations.
 * 
 * Spec: M0-W-1430, M0-W-1031, CUDA-5030
 * Story: FT-016
 */

#include "../include/cublas_wrapper.h"
#include "cuda_error.h"
#include <string>

namespace worker {

CublasHandle::CublasHandle(bool deterministic) {
    // Create cuBLAS handle
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CudaError(
            CUDA_ERROR_UNKNOWN,
            "Failed to create cuBLAS handle: status=" + std::to_string(status)
        );
    }
    
    if (deterministic) {
        // Enable deterministic mode (disable Tensor Cores if non-deterministic)
        // CUBLAS_PEDANTIC_MATH ensures bit-exact reproducibility
        status = cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH);
        if (status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle_);
            throw CudaError(
                CUDA_ERROR_UNKNOWN,
                "Failed to set deterministic math mode: status=" + std::to_string(status)
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
            "Failed to set cuBLAS stream: status=" + std::to_string(status)
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
    // Convert transpose flags to cuBLAS operations
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    // cuBLAS uses column-major layout (Fortran convention)
    // We use row-major layout (C convention)
    // To compute C = A * B in row-major, we compute C^T = B^T * A^T in column-major
    // This swaps A/B, M/N, and transposes
    cublasStatus_t status = cublasGemmEx(
        handle.get(),
        opB, opA,        // Swapped operations
        N, M, K,         // Swapped M and N
        &alpha,
        B, CUDA_R_16F, ldb,  // B first
        A, CUDA_R_16F, lda,  // A second
        &beta,
        C, CUDA_R_16F, ldc,
        CUBLAS_COMPUTE_32F,  // Use FP32 accumulation for numerical stability
        CUBLAS_GEMM_DEFAULT
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CudaError::kernel_launch_failed(
            "cuBLAS GEMM failed: status=" + std::to_string(status)
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
    // C = A * B (no transpose, alpha=1, beta=0)
    // A: [M, K], B: [K, N], C: [M, N]
    // Leading dimensions: lda=K, ldb=N, ldc=N
    gemm_fp16(handle, false, false, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

} // namespace worker

// ---
// Built by Foundation-Alpha 🏗️
