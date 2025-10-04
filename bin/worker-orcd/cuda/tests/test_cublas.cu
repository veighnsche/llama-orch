/**
 * cuBLAS GEMM Wrapper Unit Tests
 * 
 * Tests cuBLAS handle management and GEMM operations.
 * 
 * Spec: M0-W-1430, M0-W-1031, CUDA-5030
 * Story: FT-016
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/cublas_wrapper.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace worker;

// ============================================================================
// Test Fixture
// ============================================================================

class CublasTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA device
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
        cudaSetDevice(0);
        
        // Create cuBLAS handle
        handle = std::make_unique<CublasHandle>(true);  // deterministic=true
    }
    
    std::unique_ptr<CublasHandle> handle;
};

// ============================================================================
// Handle Management Tests
// ============================================================================

/**
 * Test: Handle creation succeeds
 * 
 * Spec: M0-W-1430 (cuBLAS GEMM Wrapper)
 * Critical: Foundation of all GEMM operations
 */
TEST_F(CublasTest, HandleCreationSucceeds) {
    EXPECT_NE(handle->get(), nullptr) << "cuBLAS handle should be non-null";
}

/**
 * Test: Deterministic mode enabled
 * 
 * Spec: M0-W-1031 (Reproducible CUDA Kernels)
 * Critical: Reproducibility requirement
 */
TEST_F(CublasTest, DeterministicModeEnabled) {
    // Create handle with deterministic=true
    CublasHandle det_handle(true);
    EXPECT_NE(det_handle.get(), nullptr);
    
    // Note: Cannot query math mode via cuBLAS API
    // Determinism will be verified in functional tests
}

/**
 * Test: Handle can set stream
 * 
 * Critical: Stream management for async operations
 */
TEST_F(CublasTest, HandleCanSetStream) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Should not throw
    EXPECT_NO_THROW(handle->set_stream(stream));
    
    cudaStreamDestroy(stream);
}

// ============================================================================
// Basic GEMM Tests
// ============================================================================

/**
 * Test: Simple matrix multiply (2x3 * 3x2)
 * 
 * Spec: M0-W-1430 (cuBLAS GEMM Wrapper)
 * Critical: Core GEMM correctness
 */
TEST_F(CublasTest, SimpleMatrixMultiply) {
    // Test C = A * B
    // A: [2, 3], B: [3, 2], C: [2, 2]
    int M = 2, N = 2, K = 3;
    
    // Host matrices (row-major)
    // A = [1, 2, 3]
    //     [4, 5, 6]
    std::vector<half> h_A = {
        __float2half(1.0f), __float2half(2.0f), __float2half(3.0f),
        __float2half(4.0f), __float2half(5.0f), __float2half(6.0f)
    };
    
    // B = [1, 2]
    //     [3, 4]
    //     [5, 6]
    std::vector<half> h_B = {
        __float2half(1.0f), __float2half(2.0f),
        __float2half(3.0f), __float2half(4.0f),
        __float2half(5.0f), __float2half(6.0f)
    };
    
    std::vector<half> h_C(M * N, __float2half(-999.0f));  // Sentinel
    
    // Device matrices
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // Perform GEMM
    gemm_simple_fp16(*handle, M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Expected result:
    // C = [1*1 + 2*3 + 3*5,  1*2 + 2*4 + 3*6] = [22, 28]
    //     [4*1 + 5*3 + 6*5,  4*2 + 5*4 + 6*6]   [49, 64]
    EXPECT_NEAR(__half2float(h_C[0]), 22.0f, 0.1f) << "C[0,0]";
    EXPECT_NEAR(__half2float(h_C[1]), 28.0f, 0.1f) << "C[0,1]";
    EXPECT_NEAR(__half2float(h_C[2]), 49.0f, 0.1f) << "C[1,0]";
    EXPECT_NEAR(__half2float(h_C[3]), 64.0f, 0.1f) << "C[1,1]";
    
    // Verify sentinel was overwritten (GEMM ran)
    EXPECT_NE(__half2float(h_C[0]), -999.0f) << "GEMM should overwrite sentinel";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/**
 * Test: Identity matrix multiplication
 * 
 * Critical: Validates GEMM preserves data
 */
TEST_F(CublasTest, IdentityMatrixMultiplication) {
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
        EXPECT_NEAR(__half2float(h_C[i]), __half2float(h_A[i]), 0.01f) << "Index " << i;
    }
    
    cudaFree(d_I);
    cudaFree(d_A);
    cudaFree(d_C);
}

/**
 * Test: Zero matrix multiplication
 * 
 * Edge case: A * 0 = 0
 */
TEST_F(CublasTest, ZeroMatrixMultiplication) {
    int M = 2, N = 2, K = 2;
    
    std::vector<half> h_A = {
        __float2half(1.0f), __float2half(2.0f),
        __float2half(3.0f), __float2half(4.0f)
    };
    std::vector<half> h_B(K * N, __float2half(0.0f));  // Zero matrix
    std::vector<half> h_C(M * N);
    
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    gemm_simple_fp16(*handle, M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Result should be all zeros
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(__half2float(h_C[i]), 0.0f, 0.01f) << "Index " << i;
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Real-World Dimensions Tests
// ============================================================================

/**
 * Test: Large dimensions (512x512x512)
 * 
 * Spec: M0-W-1430 (cuBLAS GEMM Wrapper)
 * Critical: Real-world scale
 */
TEST_F(CublasTest, LargeDimensions) {
    // Test with realistic transformer dimensions
    int M = 512, N = 512, K = 512;
    
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
    
    // Copy result back
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Expected result: each element = K * 0.01 * 0.01 = 512 * 0.0001 = 0.0512
    float expected = K * 0.01f * 0.01f;
    EXPECT_NEAR(__half2float(h_C[0]), expected, 0.01f) << "GEMM result incorrect";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * Test: Qwen2.5-0.5B dimensions
 * 
 * Critical: Real model dimensions
 */
TEST_F(CublasTest, QwenDimensions) {
    // Qwen2.5-0.5B: hidden_dim=896, intermediate_size=4864
    // Test FFN up-projection: [batch, hidden_dim] * [hidden_dim, intermediate_size]
    int M = 1;     // batch_size
    int N = 4864;  // intermediate_size
    int K = 896;   // hidden_dim
    
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
    
    gemm_simple_fp16(*handle, M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Expected: K * 0.01 * 0.01 = 896 * 0.0001 = 0.0896
    float expected = K * 0.01f * 0.01f;
    EXPECT_NEAR(__half2float(h_C[0]), expected, 0.02f) << "Qwen FFN projection";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/**
 * Test: GPT-OSS-20B dimensions
 * 
 * Critical: Large model dimensions
 */
TEST_F(CublasTest, GPTDimensions) {
    // GPT-OSS-20B: hidden_dim=2048, intermediate_size=8192
    // Test attention Q*K^T: [batch, seq_len, hidden_dim] * [batch, hidden_dim, seq_len]
    // Simplified: [1, 64] * [64, 1] for testing
    int M = 1;
    int N = 1;
    int K = 2048;  // hidden_dim
    
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<half> h_C(M * N);
    
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = __float2half(0.001f);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = __float2half(0.001f);
    }
    
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    gemm_simple_fp16(*handle, M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Expected: K * 0.001 * 0.001 = 2048 * 0.000001 = 0.002048
    float expected = K * 0.001f * 0.001f;
    EXPECT_NEAR(__half2float(h_C[0]), expected, 0.001f) << "GPT attention projection";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Determinism Tests
// ============================================================================

/**
 * Test: Deterministic mode (same inputs → same outputs)
 * 
 * Spec: M0-W-1031 (Reproducible CUDA Kernels)
 * Critical: Reproducibility requirement
 */
TEST_F(CublasTest, DeterministicGEMM) {
    int M = 128, N = 128, K = 128;
    
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    
    // Initialize with pseudo-random values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = __float2half((i % 100) * 0.01f);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = __float2half((i % 100) * 0.01f);
    }
    
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // Run GEMM 3 times
    std::vector<std::vector<half>> results(3);
    for (int run = 0; run < 3; ++run) {
        gemm_simple_fp16(*handle, M, N, K, d_A, d_B, d_C);
        cudaDeviceSynchronize();
        
        results[run].resize(M * N);
        cudaMemcpy(results[run].data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    }
    
    // Verify all runs produce identical results (bit-exact)
    for (int run = 1; run < 3; ++run) {
        for (int i = 0; i < M * N; ++i) {
            float val0 = __half2float(results[0][i]);
            float valN = __half2float(results[run][i]);
            EXPECT_FLOAT_EQ(val0, valN) << "Run " << run << ", index " << i << " differs";
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Transpose Tests
// ============================================================================

/**
 * Test: Transpose A matrix
 * 
 * Critical: Attention requires Q*K^T
 */
TEST_F(CublasTest, TransposeA) {
    // Test C = A^T * B
    // A: [3, 2] (transposed to [2, 3]), B: [3, 2], C: [2, 2]
    int M = 2, N = 2, K = 3;
    
    // A = [1, 2]  (stored as [3, 2])
    //     [3, 4]
    //     [5, 6]
    std::vector<half> h_A = {
        __float2half(1.0f), __float2half(2.0f),
        __float2half(3.0f), __float2half(4.0f),
        __float2half(5.0f), __float2half(6.0f)
    };
    
    // B = [1, 2]
    //     [3, 4]
    //     [5, 6]
    std::vector<half> h_B = {
        __float2half(1.0f), __float2half(2.0f),
        __float2half(3.0f), __float2half(4.0f),
        __float2half(5.0f), __float2half(6.0f)
    };
    
    std::vector<half> h_C(M * N);
    
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, K * M * sizeof(half));  // A is [3, 2]
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_A, h_A.data(), K * M * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // C = A^T * B
    gemm_fp16(*handle, true, false, M, N, K, 1.0f, d_A, M, d_B, N, 0.0f, d_C, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Expected result:
    // A^T = [1, 3, 5]
    //       [2, 4, 6]
    // C = A^T * B = [1*1+3*3+5*5, 1*2+3*4+5*6] = [35, 44]
    //               [2*1+4*3+6*5, 2*2+4*4+6*6]   [44, 56]
    EXPECT_NEAR(__half2float(h_C[0]), 35.0f, 0.2f) << "C[0,0]";
    EXPECT_NEAR(__half2float(h_C[1]), 44.0f, 0.2f) << "C[0,1]";
    EXPECT_NEAR(__half2float(h_C[2]), 44.0f, 0.2f) << "C[1,0]";
    EXPECT_NEAR(__half2float(h_C[3]), 56.0f, 0.2f) << "C[1,1]";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/**
 * Test: Transpose B matrix
 * 
 * Critical: Some projections require B^T
 */
TEST_F(CublasTest, TransposeB) {
    // Test C = A * B^T
    // A: [2, 3], B: [2, 3] (transposed to [3, 2]), C: [2, 2]
    int M = 2, N = 2, K = 3;
    
    std::vector<half> h_A = {
        __float2half(1.0f), __float2half(2.0f), __float2half(3.0f),
        __float2half(4.0f), __float2half(5.0f), __float2half(6.0f)
    };
    
    // B = [1, 3, 5]  (stored as [2, 3])
    //     [2, 4, 6]
    std::vector<half> h_B = {
        __float2half(1.0f), __float2half(3.0f), __float2half(5.0f),
        __float2half(2.0f), __float2half(4.0f), __float2half(6.0f)
    };
    
    std::vector<half> h_C(M * N);
    
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, N * K * sizeof(half));  // B is [2, 3]
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(half), cudaMemcpyHostToDevice);
    
    // C = A * B^T
    gemm_fp16(*handle, false, true, M, N, K, 1.0f, d_A, K, d_B, K, 0.0f, d_C, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Expected result:
    // B^T = [1, 2]
    //       [3, 4]
    //       [5, 6]
    // C = A * B^T = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    //               [4*1+5*3+6*5, 4*2+5*4+6*6]   [49, 64]
    EXPECT_NEAR(__half2float(h_C[0]), 22.0f, 0.1f) << "C[0,0]";
    EXPECT_NEAR(__half2float(h_C[1]), 28.0f, 0.1f) << "C[0,1]";
    EXPECT_NEAR(__half2float(h_C[2]), 49.0f, 0.1f) << "C[1,0]";
    EXPECT_NEAR(__half2float(h_C[3]), 64.0f, 0.1f) << "C[1,1]";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Alpha/Beta Scaling Tests
// ============================================================================

/**
 * Test: Alpha scaling (C = alpha * A * B)
 * 
 * Critical: Attention scaling requires alpha
 */
TEST_F(CublasTest, AlphaScaling) {
    int M = 2, N = 2, K = 2;
    
    std::vector<half> h_A = {
        __float2half(1.0f), __float2half(2.0f),
        __float2half(3.0f), __float2half(4.0f)
    };
    std::vector<half> h_B = {
        __float2half(1.0f), __float2half(0.0f),
        __float2half(0.0f), __float2half(1.0f)
    };
    std::vector<half> h_C(M * N);
    
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // C = 2.0 * A * B
    float alpha = 2.0f;
    gemm_fp16(*handle, false, false, M, N, K, alpha, d_A, K, d_B, N, 0.0f, d_C, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Expected: 2.0 * [1, 2; 3, 4] = [2, 4; 6, 8]
    EXPECT_NEAR(__half2float(h_C[0]), 2.0f, 0.1f) << "C[0,0]";
    EXPECT_NEAR(__half2float(h_C[1]), 4.0f, 0.1f) << "C[0,1]";
    EXPECT_NEAR(__half2float(h_C[2]), 6.0f, 0.1f) << "C[1,0]";
    EXPECT_NEAR(__half2float(h_C[3]), 8.0f, 0.1f) << "C[1,1]";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/**
 * Test: Beta accumulation (C = A * B + beta * C)
 * 
 * Critical: Residual connections require beta
 */
TEST_F(CublasTest, BetaAccumulation) {
    int M = 2, N = 2, K = 2;
    
    std::vector<half> h_A = {
        __float2half(1.0f), __float2half(0.0f),
        __float2half(0.0f), __float2half(1.0f)
    };
    std::vector<half> h_B = {
        __float2half(1.0f), __float2half(2.0f),
        __float2half(3.0f), __float2half(4.0f)
    };
    std::vector<half> h_C_init = {
        __float2half(10.0f), __float2half(20.0f),
        __float2half(30.0f), __float2half(40.0f)
    };
    std::vector<half> h_C(M * N);
    
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C_init.data(), M * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // C = A * B + 1.0 * C
    // A * B = I * B = B
    // C = B + C_init
    float alpha = 1.0f;
    float beta = 1.0f;
    gemm_fp16(*handle, false, false, M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Expected: [1+10, 2+20; 3+30, 4+40] = [11, 22; 33, 44]
    EXPECT_NEAR(__half2float(h_C[0]), 11.0f, 0.1f) << "C[0,0]";
    EXPECT_NEAR(__half2float(h_C[1]), 22.0f, 0.1f) << "C[0,1]";
    EXPECT_NEAR(__half2float(h_C[2]), 33.0f, 0.1f) << "C[1,0]";
    EXPECT_NEAR(__half2float(h_C[3]), 44.0f, 0.1f) << "C[1,1]";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Performance Benchmark Tests
// ============================================================================

/**
 * Test: Performance benchmark for transformer dimensions
 * 
 * Spec: M0-W-1430 (cuBLAS GEMM Wrapper)
 * Critical: Performance baseline
 */
TEST_F(CublasTest, PerformanceBenchmark768) {
    // BERT/GPT-2 hidden dimension
    int M = 768, N = 768, K = 768;
    
    std::vector<half> h_A(M * K, __float2half(0.01f));
    std::vector<half> h_B(K * N, __float2half(0.01f));
    
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // Warmup
    gemm_simple_fp16(*handle, M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
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
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (milliseconds / 1000.0)) / 1e12;
    
    std::cout << "GEMM [" << M << "x" << N << "x" << K << "]: "
              << milliseconds << " ms, "
              << tflops << " TFLOPS" << std::endl;
    
    // Sanity check: should complete in reasonable time
    EXPECT_LT(milliseconds, 1000.0f) << "GEMM should complete within 1 second";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ---
// Built by Foundation-Alpha 🏗️
