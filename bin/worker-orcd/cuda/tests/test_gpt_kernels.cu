// Unit tests for GPT-specific CUDA kernels
//
// Tests LayerNorm, GELU, and positional embedding kernels
//
// Story: GT-011, GT-013, GT-016

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

// External kernel declarations
extern "C" {
    void cuda_layernorm(
        half* output, const half* input,
        const half* gamma, const half* beta,
        int batch_size, int seq_len, int hidden_size,
        float epsilon, cudaStream_t stream
    );
    
    void cuda_gelu(
        half* output, const half* input,
        int size, cudaStream_t stream
    );
    
    void cuda_add_positional_embedding(
        half* output, const half* token_emb, const half* pos_emb,
        int batch_size, int seq_len, int hidden_size,
        cudaStream_t stream
    );
}

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Helper function to compare floats with tolerance
bool approx_equal(float a, float b, float tol = 1e-3f) {
    return fabsf(a - b) <= tol;
}

// Test 1: LayerNorm with known input
void test_layernorm_basic() {
    printf("Test 1: LayerNorm basic...\n");
    
    const int batch_size = 1;
    const int seq_len = 1;
    const int hidden_size = 4;
    const float epsilon = 1e-5f;
    
    // Input: [1.0, 2.0, 3.0, 4.0]
    // Mean = 2.5, Variance = 1.25
    // Normalized: [-1.34, -0.45, 0.45, 1.34] (approximately)
    float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};  // No scaling
    float h_beta[] = {0.0f, 0.0f, 0.0f, 0.0f};   // No bias
    
    // Allocate device memory
    half *d_input, *d_output, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_beta, hidden_size * sizeof(half)));
    
    // Convert to half and copy to device
    half h_input_half[4], h_gamma_half[4], h_beta_half[4];
    for (int i = 0; i < 4; i++) {
        h_input_half[i] = __float2half(h_input[i]);
        h_gamma_half[i] = __float2half(h_gamma[i]);
        h_beta_half[i] = __float2half(h_beta[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input_half, hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma_half, hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta_half, hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Run kernel
    cuda_layernorm(d_output, d_input, d_gamma, d_beta,
                   batch_size, seq_len, hidden_size, epsilon, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    half h_output_half[4];
    CUDA_CHECK(cudaMemcpy(h_output_half, d_output, hidden_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Verify: output should be normalized (mean=0, variance=1)
    float h_output[4];
    for (int i = 0; i < 4; i++) {
        h_output[i] = __half2float(h_output_half[i]);
    }
    
    // Check mean is close to 0
    float mean = (h_output[0] + h_output[1] + h_output[2] + h_output[3]) / 4.0f;
    assert(approx_equal(mean, 0.0f, 0.01f));
    
    // Check variance is close to 1
    float variance = 0.0f;
    for (int i = 0; i < 4; i++) {
        variance += (h_output[i] - mean) * (h_output[i] - mean);
    }
    variance /= 4.0f;
    assert(approx_equal(variance, 1.0f, 0.1f));
    
    printf("  ✓ LayerNorm basic test passed\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

// Test 2: GELU activation
void test_gelu_activation() {
    printf("Test 2: GELU activation...\n");
    
    const int size = 5;
    
    // Test inputs: [-2.0, -1.0, 0.0, 1.0, 2.0]
    float h_input[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    
    // Expected GELU outputs (approximate):
    // GELU(-2.0) ≈ -0.0454
    // GELU(-1.0) ≈ -0.1587
    // GELU(0.0) = 0.0
    // GELU(1.0) ≈ 0.8413
    // GELU(2.0) ≈ 1.9545
    float expected[] = {-0.0454f, -0.1587f, 0.0f, 0.8413f, 1.9545f};
    
    // Allocate device memory
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    // Convert to half and copy to device
    half h_input_half[5];
    for (int i = 0; i < size; i++) {
        h_input_half[i] = __float2half(h_input[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input_half, size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Run kernel
    cuda_gelu(d_output, d_input, size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    half h_output_half[5];
    CUDA_CHECK(cudaMemcpy(h_output_half, d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Verify outputs
    for (int i = 0; i < size; i++) {
        float output = __half2float(h_output_half[i]);
        assert(approx_equal(output, expected[i], 0.01f));
    }
    
    printf("  ✓ GELU activation test passed\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Test 3: Positional embedding addition
void test_positional_embedding() {
    printf("Test 3: Positional embedding...\n");
    
    const int batch_size = 2;
    const int seq_len = 3;
    const int hidden_size = 4;
    
    // Token embeddings: all 1.0
    float h_token_emb[batch_size * seq_len * hidden_size];
    for (int i = 0; i < batch_size * seq_len * hidden_size; i++) {
        h_token_emb[i] = 1.0f;
    }
    
    // Position embeddings: position index as value
    float h_pos_emb[seq_len * hidden_size];
    for (int pos = 0; pos < seq_len; pos++) {
        for (int h = 0; h < hidden_size; h++) {
            h_pos_emb[pos * hidden_size + h] = (float)pos;
        }
    }
    
    // Allocate device memory
    half *d_token_emb, *d_pos_emb, *d_output;
    CUDA_CHECK(cudaMalloc(&d_token_emb, batch_size * seq_len * hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_pos_emb, seq_len * hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * seq_len * hidden_size * sizeof(half)));
    
    // Convert to half and copy to device
    half *h_token_emb_half = new half[batch_size * seq_len * hidden_size];
    half *h_pos_emb_half = new half[seq_len * hidden_size];
    
    for (int i = 0; i < batch_size * seq_len * hidden_size; i++) {
        h_token_emb_half[i] = __float2half(h_token_emb[i]);
    }
    for (int i = 0; i < seq_len * hidden_size; i++) {
        h_pos_emb_half[i] = __float2half(h_pos_emb[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_token_emb, h_token_emb_half, 
                         batch_size * seq_len * hidden_size * sizeof(half), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_emb, h_pos_emb_half, 
                         seq_len * hidden_size * sizeof(half), 
                         cudaMemcpyHostToDevice));
    
    // Run kernel
    cuda_add_positional_embedding(d_output, d_token_emb, d_pos_emb,
                                  batch_size, seq_len, hidden_size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    half *h_output_half = new half[batch_size * seq_len * hidden_size];
    CUDA_CHECK(cudaMemcpy(h_output_half, d_output, 
                         batch_size * seq_len * hidden_size * sizeof(half), 
                         cudaMemcpyDeviceToHost));
    
    // Verify: output should be token_emb + pos_emb
    // For batch 0, pos 0: 1.0 + 0.0 = 1.0
    // For batch 0, pos 1: 1.0 + 1.0 = 2.0
    // For batch 0, pos 2: 1.0 + 2.0 = 3.0
    // Same for batch 1
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < hidden_size; h++) {
                int idx = (b * seq_len + s) * hidden_size + h;
                float output = __half2float(h_output_half[idx]);
                float expected = 1.0f + (float)s;
                assert(approx_equal(output, expected, 0.01f));
            }
        }
    }
    
    printf("  ✓ Positional embedding test passed\n");
    
    // Cleanup
    delete[] h_token_emb_half;
    delete[] h_pos_emb_half;
    delete[] h_output_half;
    CUDA_CHECK(cudaFree(d_token_emb));
    CUDA_CHECK(cudaFree(d_pos_emb));
    CUDA_CHECK(cudaFree(d_output));
}

// Test 4: LayerNorm with scale and bias
void test_layernorm_affine() {
    printf("Test 4: LayerNorm with affine transform...\n");
    
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
    
    // After normalization, scaling by 2 and shifting by 1
    // Mean should be around 1.0, not 0.0
    float h_output[4];
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        h_output[i] = __half2float(h_output_half[i]);
        sum += h_output[i];
    }
    float mean = sum / 4.0f;
    
    // Mean should be close to beta (1.0) since normalized mean is 0
    assert(approx_equal(mean, 1.0f, 0.1f));
    
    printf("  ✓ LayerNorm affine transform test passed\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

int main() {
    printf("=== GPT Kernel Unit Tests ===\n\n");
    
    test_layernorm_basic();
    test_gelu_activation();
    test_positional_embedding();
    test_layernorm_affine();
    
    printf("\n✅ All tests passed!\n");
    return 0;
}

// ---
// Crafted by GPT-Gamma 🤖
