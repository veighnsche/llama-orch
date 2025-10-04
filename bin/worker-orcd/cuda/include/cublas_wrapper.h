/**
 * cuBLAS GEMM Wrapper
 * 
 * Provides RAII wrapper for cuBLAS handle and simplified GEMM interface.
 * 
 * Features:
 * - Automatic handle lifecycle management
 * - Deterministic mode for reproducibility (CUBLAS_PEDANTIC_MATH)
 * - FP16 operations with FP32 accumulation
 * - Simplified API for common GEMM patterns
 * - Error handling for all cuBLAS operations
 * 
 * Spec: M0-W-1430, M0-W-1031, CUDA-5030
 * Story: FT-016
 */

#ifndef WORKER_CUBLAS_WRAPPER_H
#define WORKER_CUBLAS_WRAPPER_H

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cuda_error.h"

namespace worker {

/**
 * RAII wrapper for cuBLAS handle.
 * 
 * Manages cuBLAS handle lifecycle and configuration.
 * 
 * Features:
 * - Automatic cleanup in destructor
 * - Non-copyable (unique ownership)
 * - Deterministic mode support (CUBLAS_PEDANTIC_MATH)
 * - Stream configuration
 * 
 * Example:
 * ```cpp
 * CublasHandle handle(true);  // deterministic=true
 * 
 * // Use for GEMM operations
 * gemm_simple_fp16(handle, M, N, K, A, B, C);
 * 
 * // Handle automatically destroyed
 * ```
 */
class CublasHandle {
public:
    /**
     * Create cuBLAS handle.
     * 
     * @param deterministic If true, enable deterministic mode (CUBLAS_PEDANTIC_MATH)
     * @throws CudaError if creation fails
     * 
     * Deterministic mode:
     * - Disables Tensor Cores if they produce non-deterministic results
     * - Ensures same inputs → same outputs (bit-exact)
     * - Required for M0 reproducibility (M0-W-1031)
     */
    explicit CublasHandle(bool deterministic = true);
    
    /**
     * Destroy cuBLAS handle.
     * 
     * Automatically called when object goes out of scope.
     */
    ~CublasHandle();
    
    // Non-copyable, non-movable (handle is tied to CUDA context)
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    CublasHandle(CublasHandle&&) = delete;
    CublasHandle& operator=(CublasHandle&&) = delete;
    
    /**
     * Get raw cuBLAS handle.
     * 
     * @return cuBLAS handle for use with cuBLAS API
     */
    cublasHandle_t get() const { return handle_; }
    
    /**
     * Set CUDA stream for cuBLAS operations.
     * 
     * @param stream CUDA stream (0 = default stream)
     * @throws CudaError if setting stream fails
     */
    void set_stream(cudaStream_t stream);

private:
    cublasHandle_t handle_ = nullptr;
};

/**
 * General Matrix Multiply (GEMM): C = alpha * op(A) * op(B) + beta * C
 * 
 * Performs matrix multiplication using cuBLAS with FP16 inputs and FP32 accumulation.
 * 
 * Matrix dimensions:
 *   A: [M, K] if transA == false, [K, M] if transA == true
 *   B: [K, N] if transB == false, [N, K] if transB == true
 *   C: [M, N] (always)
 * 
 * Memory layout: Row-major (C convention)
 * cuBLAS layout: Column-major (Fortran convention) - handled internally
 * 
 * Example: C = A * B (no transpose)
 * ```cpp
 * // A: [2, 3], B: [3, 4], C: [2, 4]
 * gemm_fp16(handle, false, false, 2, 4, 3, 1.0f, A, 3, B, 4, 0.0f, C, 4);
 * ```
 * 
 * @param handle cuBLAS handle
 * @param transA Transpose A matrix
 * @param transB Transpose B matrix
 * @param M Number of rows in op(A) and C
 * @param N Number of columns in op(B) and C
 * @param K Number of columns in op(A) and rows in op(B)
 * @param alpha Scalar multiplier for A*B
 * @param A Device pointer to matrix A
 * @param lda Leading dimension of A (stride between rows)
 * @param B Device pointer to matrix B
 * @param ldb Leading dimension of B (stride between rows)
 * @param beta Scalar multiplier for C
 * @param C Device pointer to matrix C (input/output)
 * @param ldc Leading dimension of C (stride between rows)
 * @throws CudaError::kernel_launch_failed if GEMM fails
 * 
 * Spec: M0-W-1430 (GEMM Kernel), M0-W-1031 (Deterministic Mode)
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
 * 
 * Common case: simple matrix multiplication without scaling or accumulation.
 * 
 * Matrix dimensions:
 *   A: [M, K]
 *   B: [K, N]
 *   C: [M, N]
 * 
 * Example:
 * ```cpp
 * // A: [2, 3], B: [3, 4], C: [2, 4]
 * gemm_simple_fp16(handle, 2, 4, 3, A, B, C);
 * ```
 * 
 * @param handle cuBLAS handle
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param A Device pointer to matrix A [M, K]
 * @param B Device pointer to matrix B [K, N]
 * @param C Device pointer to matrix C [M, N] (output)
 * @throws CudaError::kernel_launch_failed if GEMM fails
 * 
 * Spec: M0-W-1430 (GEMM Kernel)
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

// ---
// Built by Foundation-Alpha 🏗️
