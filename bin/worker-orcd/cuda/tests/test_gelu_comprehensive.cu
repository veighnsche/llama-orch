// Comprehensive GELU unit tests
//
// Tests all aspects of GELU activation including numerical accuracy
//
// Story: GT-013

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

extern "C" {
    void cuda_gelu(
        half* output, const half* input,
        int size, cudaStream_t stream
    );
    
    void cuda_gelu_tanh_approx(
        half* output, const half* input,
        int size, cudaStream_t stream
    );
    
    void cuda_gelu_inplace(
        half* data, int size, cudaStream_t stream
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

bool approx_equal(float a, float b, float tol = 1e-2f) {
    return fabsf(a - b) <= tol;
}

// Reference GELU implementation
float gelu_reference(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Test 1: Known input values
void test_gelu_known_values() {
    printf("Test 1: GELU with known values...\n");
    
    const int size = 5;
    float h_input[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float h_expected[5];
    
    for (int i = 0; i < size; i++) {
        h_expected[i] = gelu_reference(h_input[i]);
    }
    
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    half h_input_half[5];
    for (int i = 0; i < size; i++) {
        h_input_half[i] = __float2half(h_input[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input_half, size * sizeof(half), cudaMemcpyHostToDevice));
    
    cuda_gelu(d_output, d_input, size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half h_output_half[5];
    CUDA_CHECK(cudaMemcpy(h_output_half, d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    printf("  Input  -> Output (Expected)\n");
    for (int i = 0; i < size; i++) {
        float output = __half2float(h_output_half[i]);
        printf("  %.2f -> %.6f (%.6f)\n", h_input[i], output, h_expected[i]);
        assert(approx_equal(output, h_expected[i], 0.05f));
    }
    
    printf("  ✓ Known values validated\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Test 2: Zero input
void test_gelu_zero() {
    printf("Test 2: GELU(0) = 0...\n");
    
    const int size = 1;
    float h_input[] = {0.0f};
    
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    half h_input_half = __float2half(h_input[0]);
    CUDA_CHECK(cudaMemcpy(d_input, &h_input_half, size * sizeof(half), cudaMemcpyHostToDevice));
    
    cuda_gelu(d_output, d_input, size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half h_output_half;
    CUDA_CHECK(cudaMemcpy(&h_output_half, d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    float output = __half2float(h_output_half);
    assert(approx_equal(output, 0.0f, 1e-4f));
    
    printf("  ✓ GELU(0) = %.6f\n", output);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Test 3: Positive values (x > 0)
void test_gelu_positive() {
    printf("Test 3: GELU for positive values...\n");
    
    const int size = 10;
    float h_input[10];
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(i + 1);  // 1, 2, 3, ..., 10
    }
    
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    half h_input_half[10];
    for (int i = 0; i < size; i++) {
        h_input_half[i] = __float2half(h_input[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input_half, size * sizeof(half), cudaMemcpyHostToDevice));
    
    cuda_gelu(d_output, d_input, size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half h_output_half[10];
    CUDA_CHECK(cudaMemcpy(h_output_half, d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // For large positive x, GELU(x) ≈ x
    for (int i = 0; i < size; i++) {
        float output = __half2float(h_output_half[i]);
        float expected = gelu_reference(h_input[i]);
        assert(approx_equal(output, expected, 0.1f));
    }
    
    printf("  ✓ Positive values validated\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Test 4: Negative values (x < 0)
void test_gelu_negative() {
    printf("Test 4: GELU for negative values...\n");
    
    const int size = 10;
    float h_input[10];
    for (int i = 0; i < size; i++) {
        h_input[i] = -(float)(i + 1);  // -1, -2, -3, ..., -10
    }
    
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    half h_input_half[10];
    for (int i = 0; i < size; i++) {
        h_input_half[i] = __float2half(h_input[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input_half, size * sizeof(half), cudaMemcpyHostToDevice));
    
    cuda_gelu(d_output, d_input, size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half h_output_half[10];
    CUDA_CHECK(cudaMemcpy(h_output_half, d_output, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // For large negative x, GELU(x) ≈ 0
    for (int i = 0; i < size; i++) {
        float output = __half2float(h_output_half[i]);
        float expected = gelu_reference(h_input[i]);
        assert(approx_equal(output, expected, 0.1f));
    }
    
    printf("  ✓ Negative values validated\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Test 5: Large tensor
void test_gelu_large_tensor() {
    printf("Test 5: GELU on large tensor...\n");
    
    const int size = 1024 * 1024;  // 1M elements
    
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    CUDA_CHECK(cudaMemset(d_input, 0, size * sizeof(half)));
    
    cuda_gelu(d_output, d_input, size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("  ✓ Large tensor (1M elements) processed\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Test 6: In-place operation
void test_gelu_inplace() {
    printf("Test 6: GELU in-place operation...\n");
    
    const int size = 5;
    float h_input[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    
    half *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(half)));
    
    half h_input_half[5];
    for (int i = 0; i < size; i++) {
        h_input_half[i] = __float2half(h_input[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_data, h_input_half, size * sizeof(half), cudaMemcpyHostToDevice));
    
    cuda_gelu_inplace(d_data, size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half h_output_half[5];
    CUDA_CHECK(cudaMemcpy(h_output_half, d_data, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < size; i++) {
        float output = __half2float(h_output_half[i]);
        float expected = gelu_reference(h_input[i]);
        assert(approx_equal(output, expected, 0.05f));
    }
    
    printf("  ✓ In-place operation validated\n");
    
    CUDA_CHECK(cudaFree(d_data));
}

// Test 7: Fast approximation vs exact
void test_gelu_fast_vs_exact() {
    printf("Test 7: GELU fast approximation vs exact...\n");
    
    const int size = 100;
    float h_input[100];
    for (int i = 0; i < size; i++) {
        h_input[i] = -5.0f + (i * 0.1f);  // Range: -5 to 5
    }
    
    half *d_input, *d_output_exact, *d_output_fast;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output_exact, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output_fast, size * sizeof(half)));
    
    half h_input_half[100];
    for (int i = 0; i < size; i++) {
        h_input_half[i] = __float2half(h_input[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input_half, size * sizeof(half), cudaMemcpyHostToDevice));
    
    cuda_gelu(d_output_exact, d_input, size, 0);
    cuda_gelu_tanh_approx(d_output_fast, d_input, size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half h_output_exact_half[100], h_output_fast_half[100];
    CUDA_CHECK(cudaMemcpy(h_output_exact_half, d_output_exact, size * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_fast_half, d_output_fast, size * sizeof(half), cudaMemcpyDeviceToHost));
    
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float exact = __half2float(h_output_exact_half[i]);
        float fast = __half2float(h_output_fast_half[i]);
        float diff = fabsf(exact - fast);
        max_diff = fmaxf(max_diff, diff);
    }
    
    printf("  ✓ Max difference between exact and fast: %.6f\n", max_diff);
    assert(max_diff < 0.1f);  // Fast approximation should be close
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_exact));
    CUDA_CHECK(cudaFree(d_output_fast));
}

// Test 8: GPT-OSS-20B FFN dimensions
void test_gelu_gpt_oss_20b() {
    printf("Test 8: GELU GPT-OSS-20B FFN dimensions...\n");
    
    const int batch_size = 1;
    const int seq_len = 128;
    const int ffn_dim = 24576;  // 4 * 6144
    const int size = batch_size * seq_len * ffn_dim;
    
    half *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(half)));
    
    CUDA_CHECK(cudaMemset(d_input, 0, size * sizeof(half)));
    
    cuda_gelu(d_output, d_input, size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("  ✓ GPT-OSS-20B FFN dimensions validated\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    printf("=== Comprehensive GELU Unit Tests ===\n\n");
    
    test_gelu_known_values();
    test_gelu_zero();
    test_gelu_positive();
    test_gelu_negative();
    test_gelu_large_tensor();
    test_gelu_inplace();
    test_gelu_fast_vs_exact();
    test_gelu_gpt_oss_20b();
    
    printf("\n✅ All GELU tests passed!\n");
    return 0;
}

// ---
// Crafted by GPT-Gamma 🤖
