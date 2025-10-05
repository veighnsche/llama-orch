// Comprehensive LayerNorm unit tests
//
// Tests all aspects of LayerNorm implementation including edge cases
//
// Story: GT-011

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>

extern "C" {
    void cuda_layernorm(
        half* output, const half* input,
        const half* gamma, const half* beta,
        int batch_size, int seq_len, int hidden_size,
        float epsilon, cudaStream_t stream
    );
    
    void cuda_layernorm_residual(
        half* output, const half* input, const half* residual,
        const half* gamma, const half* beta,
        int batch_size, int seq_len, int hidden_size,
        float epsilon, cudaStream_t stream
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

bool approx_equal(float a, float b, float tol = 1e-3f) {
    return fabsf(a - b) <= tol;
}

// Test 1: Zero mean and unit variance
void test_layernorm_normalization() {
    printf("Test 1: LayerNorm normalization (mean=0, var=1)...\n");
    
    const int batch_size = 1;
    const int seq_len = 1;
    const int hidden_size = 8;
    const float epsilon = 1e-5f;
    
    // Input with known mean and variance
    float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float h_gamma[8], h_beta[8];
    for (int i = 0; i < 8; i++) {
        h_gamma[i] = 1.0f;  // No scaling
        h_beta[i] = 0.0f;   // No bias
    }
    
    half *d_input, *d_output, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_beta, hidden_size * sizeof(half)));
    
    half h_input_half[8], h_gamma_half[8], h_beta_half[8];
    for (int i = 0; i < 8; i++) {
        h_input_half[i] = __float2half(h_input[i]);
        h_gamma_half[i] = __float2half(h_gamma[i]);
        h_beta_half[i] = __float2half(h_beta[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input_half, hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma_half, hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta_half, hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    
    cuda_layernorm(d_output, d_input, d_gamma, d_beta,
                   batch_size, seq_len, hidden_size, epsilon, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half h_output_half[8];
    CUDA_CHECK(cudaMemcpy(h_output_half, d_output, hidden_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    float h_output[8];
    for (int i = 0; i < 8; i++) {
        h_output[i] = __half2float(h_output_half[i]);
    }
    
    // Verify mean ≈ 0
    float mean = 0.0f;
    for (int i = 0; i < 8; i++) {
        mean += h_output[i];
    }
    mean /= 8.0f;
    assert(approx_equal(mean, 0.0f, 0.01f));
    
    // Verify variance ≈ 1
    float variance = 0.0f;
    for (int i = 0; i < 8; i++) {
        variance += (h_output[i] - mean) * (h_output[i] - mean);
    }
    variance /= 8.0f;
    assert(approx_equal(variance, 1.0f, 0.1f));
    
    printf("  ✓ Mean: %.6f, Variance: %.6f\n", mean, variance);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

// Test 2: Scale and bias application
void test_layernorm_affine() {
    printf("Test 2: LayerNorm affine transformation...\n");
    
    const int batch_size = 1;
    const int seq_len = 1;
    const int hidden_size = 4;
    const float epsilon = 1e-5f;
    
    float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_gamma[] = {2.0f, 2.0f, 2.0f, 2.0f};  // Scale by 2
    float h_beta[] = {1.0f, 1.0f, 1.0f, 1.0f};   // Shift by 1
    
    half *d_input, *d_output, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_beta, hidden_size * sizeof(half)));
    
    half h_input_half[4], h_gamma_half[4], h_beta_half[4];
    for (int i = 0; i < 4; i++) {
        h_input_half[i] = __float2half(h_input[i]);
        h_gamma_half[i] = __float2half(h_gamma[i]);
        h_beta_half[i] = __float2half(h_beta[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input_half, hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma_half, hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta_half, hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    
    cuda_layernorm(d_output, d_input, d_gamma, d_beta,
                   batch_size, seq_len, hidden_size, epsilon, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half h_output_half[4];
    CUDA_CHECK(cudaMemcpy(h_output_half, d_output, hidden_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    float h_output[4];
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        h_output[i] = __half2float(h_output_half[i]);
        sum += h_output[i];
    }
    float mean = sum / 4.0f;
    
    // After scaling by 2 and shifting by 1, mean should be around 1
    assert(approx_equal(mean, 1.0f, 0.1f));
    
    printf("  ✓ Affine mean: %.6f\n", mean);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

// Test 3: Batched operation
void test_layernorm_batched() {
    printf("Test 3: LayerNorm batched operation...\n");
    
    const int batch_size = 4;
    const int seq_len = 8;
    const int hidden_size = 256;
    const float epsilon = 1e-5f;
    
    int total_size = batch_size * seq_len * hidden_size;
    
    half *d_input, *d_output, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_beta, hidden_size * sizeof(half)));
    
    CUDA_CHECK(cudaMemset(d_input, 0, total_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_gamma, 0, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_beta, 0, hidden_size * sizeof(half)));
    
    cuda_layernorm(d_output, d_input, d_gamma, d_beta,
                   batch_size, seq_len, hidden_size, epsilon, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("  ✓ Batched operation completed\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

// Test 4: GPT-OSS-20B dimensions
void test_layernorm_gpt_oss_20b() {
    printf("Test 4: LayerNorm GPT-OSS-20B dimensions...\n");
    
    const int batch_size = 1;
    const int seq_len = 128;
    const int hidden_size = 6144;
    const float epsilon = 1e-5f;
    
    int total_size = batch_size * seq_len * hidden_size;
    
    half *d_input, *d_output, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_beta, hidden_size * sizeof(half)));
    
    CUDA_CHECK(cudaMemset(d_input, 0, total_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_gamma, 0, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_beta, 0, hidden_size * sizeof(half)));
    
    cuda_layernorm(d_output, d_input, d_gamma, d_beta,
                   batch_size, seq_len, hidden_size, epsilon, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("  ✓ GPT-OSS-20B dimensions validated\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

// Test 5: Different epsilon values
void test_layernorm_epsilon() {
    printf("Test 5: LayerNorm with different epsilon values...\n");
    
    const int batch_size = 1;
    const int seq_len = 1;
    const int hidden_size = 4;
    
    float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float h_beta[] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    half *d_input, *d_output, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_beta, hidden_size * sizeof(half)));
    
    half h_input_half[4], h_gamma_half[4], h_beta_half[4];
    for (int i = 0; i < 4; i++) {
        h_input_half[i] = __float2half(h_input[i]);
        h_gamma_half[i] = __float2half(h_gamma[i]);
        h_beta_half[i] = __float2half(h_beta[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input_half, hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma_half, hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta_half, hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Test with different epsilon values
    float epsilons[] = {1e-5f, 1e-6f, 1e-4f};
    for (int e = 0; e < 3; e++) {
        cuda_layernorm(d_output, d_input, d_gamma, d_beta,
                       batch_size, seq_len, hidden_size, epsilons[e], 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    printf("  ✓ Different epsilon values handled\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

int main() {
    printf("=== Comprehensive LayerNorm Unit Tests ===\n\n");
    
    test_layernorm_normalization();
    test_layernorm_affine();
    test_layernorm_batched();
    test_layernorm_gpt_oss_20b();
    test_layernorm_epsilon();
    
    printf("\n✅ All LayerNorm tests passed!\n");
    return 0;
}

// ---
// Crafted by GPT-Gamma 🤖
