// Unit tests for MHA attention kernel
//
// Tests Multi-Head Attention implementation
//
// Story: GT-017, GT-018, GT-020

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

extern "C" {
    void cuda_mha_attention_prefill(
        const half* q, const half* k, const half* v,
        half* output, half* scores_workspace, half* attn_workspace,
        int batch_size, int num_heads, int seq_len_q, int seq_len_k, int head_dim,
        cublasHandle_t cublas_handle, cudaStream_t stream
    );
    
    void cuda_mha_attention_decode(
        const half* q, const half* k_cache, const half* v_cache,
        half* output, half* scores_workspace, half* attn_workspace,
        int batch_size, int num_heads, int current_len, int head_dim,
        cublasHandle_t cublas_handle, cudaStream_t stream
    );
    
    size_t cuda_mha_workspace_size(
        int batch_size, int num_heads, int seq_len_q, int seq_len_k
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

void test_workspace_size() {
    printf("Test 1: MHA workspace size calculation...\n");
    
    int batch_size = 2;
    int num_heads = 8;
    int seq_len_q = 16;
    int seq_len_k = 16;
    
    size_t workspace_size = cuda_mha_workspace_size(batch_size, num_heads, seq_len_q, seq_len_k);
    
    // Should be 2 * (batch * heads * seq_q * seq_k * sizeof(half))
    size_t expected = 2 * batch_size * num_heads * seq_len_q * seq_len_k * sizeof(half);
    assert(workspace_size == expected);
    
    printf("  ✓ Workspace size: %zu bytes\n", workspace_size);
}

void test_mha_prefill_simple() {
    printf("Test 2: MHA prefill simple case...\n");
    
    const int batch_size = 1;
    const int num_heads = 2;
    const int seq_len = 4;
    const int head_dim = 8;
    
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // Allocate host memory
    int q_size = batch_size * num_heads * seq_len * head_dim;
    int k_size = batch_size * num_heads * seq_len * head_dim;
    int v_size = batch_size * num_heads * seq_len * head_dim;
    int out_size = batch_size * num_heads * seq_len * head_dim;
    
    half *h_q = new half[q_size];
    half *h_k = new half[k_size];
    half *h_v = new half[v_size];
    
    // Initialize with simple values
    for (int i = 0; i < q_size; i++) {
        h_q[i] = __float2half(0.1f);
        h_k[i] = __float2half(0.1f);
        h_v[i] = __float2half(1.0f);
    }
    
    // Allocate device memory
    half *d_q, *d_k, *d_v, *d_output;
    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, k_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v, v_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, out_size * sizeof(half)));
    
    // Workspace
    size_t workspace_size = cuda_mha_workspace_size(batch_size, num_heads, seq_len, seq_len);
    half *d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    
    half *d_scores = d_workspace;
    half *d_attn = d_workspace + (workspace_size / (2 * sizeof(half)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_q, h_q, q_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, k_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, v_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Run MHA
    cuda_mha_attention_prefill(
        d_q, d_k, d_v, d_output, d_scores, d_attn,
        batch_size, num_heads, seq_len, seq_len, head_dim,
        cublas_handle, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    half *h_output = new half[out_size];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Verify output is finite
    for (int i = 0; i < out_size; i++) {
        float val = __half2float(h_output[i]);
        assert(isfinite(val));
    }
    
    printf("  ✓ MHA prefill completed successfully\n");
    
    // Cleanup
    delete[] h_q;
    delete[] h_k;
    delete[] h_v;
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_workspace));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

void test_mha_decode() {
    printf("Test 3: MHA decode (autoregressive)...\n");
    
    const int batch_size = 1;
    const int num_heads = 4;
    const int current_len = 8;
    const int head_dim = 16;
    
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // Allocate memory
    int q_size = batch_size * num_heads * 1 * head_dim;  // Single token
    int cache_size = batch_size * num_heads * current_len * head_dim;
    int out_size = batch_size * num_heads * 1 * head_dim;
    
    half *d_q, *d_k_cache, *d_v_cache, *d_output;
    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_cache, cache_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v_cache, cache_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, out_size * sizeof(half)));
    
    // Initialize
    CUDA_CHECK(cudaMemset(d_q, 0, q_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_k_cache, 0, cache_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_v_cache, 0, cache_size * sizeof(half)));
    
    // Workspace
    size_t workspace_size = cuda_mha_workspace_size(batch_size, num_heads, 1, current_len);
    half *d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    
    half *d_scores = d_workspace;
    half *d_attn = d_workspace + (workspace_size / (2 * sizeof(half)));
    
    // Run MHA decode
    cuda_mha_attention_decode(
        d_q, d_k_cache, d_v_cache, d_output, d_scores, d_attn,
        batch_size, num_heads, current_len, head_dim,
        cublas_handle, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Verify
    half *h_output = new half[out_size];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < out_size; i++) {
        float val = __half2float(h_output[i]);
        assert(isfinite(val));
    }
    
    printf("  ✓ MHA decode completed successfully\n");
    
    // Cleanup
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_v_cache));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_workspace));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

void test_gpt_oss_20b_dimensions() {
    printf("Test 4: GPT-OSS-20B dimensions...\n");
    
    const int batch_size = 1;
    const int num_heads = 64;
    const int seq_len = 128;
    const int head_dim = 96;  // 6144 / 64
    
    // Calculate workspace size
    size_t workspace_size = cuda_mha_workspace_size(batch_size, num_heads, seq_len, seq_len);
    
    printf("  ✓ Workspace for GPT-OSS-20B: %.2f MB\n", workspace_size / (1024.0f * 1024.0f));
    
    // Verify reasonable size
    assert(workspace_size > 0);
    assert(workspace_size < 1024 * 1024 * 1024);  // < 1GB
}

void test_mha_batched() {
    printf("Test 5: Batched MHA operation...\n");
    
    const int batch_size = 4;
    const int num_heads = 8;
    const int seq_len = 16;
    const int head_dim = 32;
    
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    int q_size = batch_size * num_heads * seq_len * head_dim;
    
    half *d_q, *d_k, *d_v, *d_output;
    CUDA_CHECK(cudaMalloc(&d_q, q_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, q_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v, q_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, q_size * sizeof(half)));
    
    CUDA_CHECK(cudaMemset(d_q, 0, q_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_k, 0, q_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_v, 0, q_size * sizeof(half)));
    
    size_t workspace_size = cuda_mha_workspace_size(batch_size, num_heads, seq_len, seq_len);
    half *d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    
    half *d_scores = d_workspace;
    half *d_attn = d_workspace + (workspace_size / (2 * sizeof(half)));
    
    cuda_mha_attention_prefill(
        d_q, d_k, d_v, d_output, d_scores, d_attn,
        batch_size, num_heads, seq_len, seq_len, head_dim,
        cublas_handle, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("  ✓ Batched MHA operation completed\n");
    
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_workspace));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

int main() {
    printf("=== MHA Attention Unit Tests ===\n\n");
    
    test_workspace_size();
    test_mha_prefill_simple();
    test_mha_decode();
    test_gpt_oss_20b_dimensions();
    test_mha_batched();
    
    printf("\n✅ All MHA tests passed!\n");
    return 0;
}

// ---
// Crafted by GPT-Gamma 🤖
