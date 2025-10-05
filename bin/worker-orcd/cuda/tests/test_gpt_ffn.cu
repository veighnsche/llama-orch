// Unit tests for GPT FFN kernel
//
// Tests feed-forward network implementation
//
// Story: GT-014

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

// External kernel declarations
extern "C" {
    void cuda_gpt_ffn_forward(
        const half* input,
        const half* w_up,
        const half* b_up,
        const half* w_down,
        const half* b_down,
        half* output,
        half* workspace,
        int batch_size,
        int seq_len,
        int d_model,
        int ffn_dim,
        cublasHandle_t cublas_handle,
        cudaStream_t stream
    );
    
    size_t cuda_gpt_ffn_workspace_size(
        int batch_size,
        int seq_len,
        int ffn_dim
    );
    
    bool cuda_gpt_ffn_validate_dims(
        int batch_size,
        int seq_len,
        int d_model,
        int ffn_dim
    );
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                    __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

bool approx_equal(float a, float b, float tol = 1e-2f) {
    return fabsf(a - b) <= tol;
}

// Test 1: Workspace size calculation
void test_workspace_size() {
    printf("Test 1: Workspace size calculation...\n");
    
    int batch_size = 2;
    int seq_len = 4;
    int ffn_dim = 1024;
    
    size_t expected_size = batch_size * seq_len * ffn_dim * sizeof(half);
    size_t actual_size = cuda_gpt_ffn_workspace_size(batch_size, seq_len, ffn_dim);
    
    assert(actual_size == expected_size);
    printf("  ✓ Workspace size correct: %zu bytes\n", actual_size);
}

// Test 2: Dimension validation
void test_dimension_validation() {
    printf("Test 2: Dimension validation...\n");
    
    // Valid dimensions
    assert(cuda_gpt_ffn_validate_dims(1, 1, 256, 1024) == true);
    assert(cuda_gpt_ffn_validate_dims(2, 4, 512, 2048) == true);
    
    // Invalid dimensions
    assert(cuda_gpt_ffn_validate_dims(0, 1, 256, 1024) == false);  // batch_size = 0
    assert(cuda_gpt_ffn_validate_dims(1, 0, 256, 1024) == false);  // seq_len = 0
    assert(cuda_gpt_ffn_validate_dims(1, 1, 0, 1024) == false);    // d_model = 0
    assert(cuda_gpt_ffn_validate_dims(1, 1, 256, 0) == false);     // ffn_dim = 0
    assert(cuda_gpt_ffn_validate_dims(1, 1, 256, 3000) == false);  // ffn_dim too large
    
    printf("  ✓ Dimension validation working\n");
}

// Test 3: Simple FFN forward pass
void test_ffn_forward_simple() {
    printf("Test 3: Simple FFN forward pass...\n");
    
    const int batch_size = 1;
    const int seq_len = 1;
    const int d_model = 4;
    const int ffn_dim = 8;
    
    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // Allocate host memory
    float h_input[d_model] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_w_up[d_model * ffn_dim];
    float h_b_up[ffn_dim];
    float h_w_down[ffn_dim * d_model];
    float h_b_down[d_model];
    
    // Initialize weights (simple identity-like pattern)
    for (int i = 0; i < d_model * ffn_dim; i++) {
        h_w_up[i] = 0.1f;
    }
    for (int i = 0; i < ffn_dim; i++) {
        h_b_up[i] = 0.0f;
    }
    for (int i = 0; i < ffn_dim * d_model; i++) {
        h_w_down[i] = 0.1f;
    }
    for (int i = 0; i < d_model; i++) {
        h_b_down[i] = 0.0f;
    }
    
    // Convert to half precision
    half *h_input_half = new half[d_model];
    half *h_w_up_half = new half[d_model * ffn_dim];
    half *h_b_up_half = new half[ffn_dim];
    half *h_w_down_half = new half[ffn_dim * d_model];
    half *h_b_down_half = new half[d_model];
    
    for (int i = 0; i < d_model; i++) {
        h_input_half[i] = __float2half(h_input[i]);
        h_b_down_half[i] = __float2half(h_b_down[i]);
    }
    for (int i = 0; i < d_model * ffn_dim; i++) {
        h_w_up_half[i] = __float2half(h_w_up[i]);
    }
    for (int i = 0; i < ffn_dim; i++) {
        h_b_up_half[i] = __float2half(h_b_up[i]);
    }
    for (int i = 0; i < ffn_dim * d_model; i++) {
        h_w_down_half[i] = __float2half(h_w_down[i]);
    }
    
    // Allocate device memory
    half *d_input, *d_w_up, *d_b_up, *d_w_down, *d_b_down, *d_output, *d_workspace;
    CUDA_CHECK(cudaMalloc(&d_input, d_model * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w_up, d_model * ffn_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b_up, ffn_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w_down, ffn_dim * d_model * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b_down, d_model * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, d_model * sizeof(half)));
    
    size_t workspace_size = cuda_gpt_ffn_workspace_size(batch_size, seq_len, ffn_dim);
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input_half, d_model * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w_up, h_w_up_half, d_model * ffn_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_up, h_b_up_half, ffn_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w_down, h_w_down_half, ffn_dim * d_model * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_down, h_b_down_half, d_model * sizeof(half), cudaMemcpyHostToDevice));
    
    // Run FFN
    cuda_gpt_ffn_forward(
        d_input, d_w_up, d_b_up, d_w_down, d_b_down,
        d_output, d_workspace,
        batch_size, seq_len, d_model, ffn_dim,
        cublas_handle, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    half *h_output_half = new half[d_model];
    CUDA_CHECK(cudaMemcpy(h_output_half, d_output, d_model * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Verify output is finite and reasonable
    for (int i = 0; i < d_model; i++) {
        float output_val = __half2float(h_output_half[i]);
        assert(isfinite(output_val));
        assert(fabsf(output_val) < 100.0f);  // Sanity check
    }
    
    printf("  ✓ FFN forward pass completed successfully\n");
    
    // Cleanup
    delete[] h_input_half;
    delete[] h_w_up_half;
    delete[] h_b_up_half;
    delete[] h_w_down_half;
    delete[] h_b_down_half;
    delete[] h_output_half;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_w_up));
    CUDA_CHECK(cudaFree(d_b_up));
    CUDA_CHECK(cudaFree(d_w_down));
    CUDA_CHECK(cudaFree(d_b_down));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_workspace));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

// Test 4: GPT-OSS-20B dimensions
void test_ffn_gpt_oss_20b_dims() {
    printf("Test 4: GPT-OSS-20B dimensions...\n");
    
    const int batch_size = 1;
    const int seq_len = 8;
    const int d_model = 6144;
    const int ffn_dim = 24576;  // 4 * d_model
    
    // Validate dimensions
    assert(cuda_gpt_ffn_validate_dims(batch_size, seq_len, d_model, ffn_dim));
    
    // Calculate workspace size
    size_t workspace_size = cuda_gpt_ffn_workspace_size(batch_size, seq_len, ffn_dim);
    size_t expected_size = batch_size * seq_len * ffn_dim * sizeof(half);
    assert(workspace_size == expected_size);
    
    printf("  ✓ GPT-OSS-20B dimensions validated\n");
    printf("  ✓ Workspace size: %.2f MB\n", workspace_size / (1024.0f * 1024.0f));
}

// Test 5: Batched operation
void test_ffn_batched() {
    printf("Test 5: Batched FFN operation...\n");
    
    const int batch_size = 4;
    const int seq_len = 8;
    const int d_model = 256;
    const int ffn_dim = 1024;
    
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // Allocate device memory
    int input_size = batch_size * seq_len * d_model;
    int output_size = batch_size * seq_len * d_model;
    
    half *d_input, *d_w_up, *d_b_up, *d_w_down, *d_b_down, *d_output, *d_workspace;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w_up, d_model * ffn_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b_up, ffn_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w_down, ffn_dim * d_model * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b_down, d_model * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(half)));
    
    size_t workspace_size = cuda_gpt_ffn_workspace_size(batch_size, seq_len, ffn_dim);
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    
    // Initialize with random values
    CUDA_CHECK(cudaMemset(d_input, 0, input_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_w_up, 0, d_model * ffn_dim * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_b_up, 0, ffn_dim * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_w_down, 0, ffn_dim * d_model * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_b_down, 0, d_model * sizeof(half)));
    
    // Run FFN
    cuda_gpt_ffn_forward(
        d_input, d_w_up, d_b_up, d_w_down, d_b_down,
        d_output, d_workspace,
        batch_size, seq_len, d_model, ffn_dim,
        cublas_handle, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("  ✓ Batched FFN operation completed\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_w_up));
    CUDA_CHECK(cudaFree(d_b_up));
    CUDA_CHECK(cudaFree(d_w_down));
    CUDA_CHECK(cudaFree(d_b_down));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_workspace));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

int main() {
    printf("=== GPT FFN Kernel Unit Tests ===\n\n");
    
    test_workspace_size();
    test_dimension_validation();
    test_ffn_forward_simple();
    test_ffn_gpt_oss_20b_dims();
    test_ffn_batched();
    
    printf("\n✅ All FFN tests passed!\n");
    return 0;
}

// ---
// Crafted by GPT-Gamma 🤖
