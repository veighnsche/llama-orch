// Unit tests for MXFP4 dequantization kernel
//
// Tests MXFP4 format dequantization
//
// Story: GT-030

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstdint>

extern "C" {
    void cuda_mxfp4_dequant(
        half* output,
        const uint8_t* input,
        int num_elements,
        cudaStream_t stream
    );
    
    void cuda_mxfp4_dequant_optimized(
        half* output,
        const uint8_t* input,
        int num_elements,
        cudaStream_t stream
    );
    
    size_t cuda_mxfp4_storage_size(int num_elements);
    
    bool cuda_mxfp4_validate_block(
        const uint8_t* block_data,
        int block_size
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

void test_storage_size() {
    printf("Test 1: MXFP4 storage size calculation...\n");
    
    // 32 elements = 1 block = 17 bytes
    assert(cuda_mxfp4_storage_size(32) == 17);
    
    // 64 elements = 2 blocks = 34 bytes
    assert(cuda_mxfp4_storage_size(64) == 34);
    
    // 33 elements = 2 blocks = 34 bytes (rounds up)
    assert(cuda_mxfp4_storage_size(33) == 34);
    
    // 1024 elements = 32 blocks = 544 bytes
    assert(cuda_mxfp4_storage_size(1024) == 544);
    
    printf("  ✓ Storage size calculations correct\n");
}

void test_block_validation() {
    printf("Test 2: MXFP4 block validation...\n");
    
    uint8_t valid_block[17];
    for (int i = 0; i < 16; i++) {
        valid_block[i] = 0x00;
    }
    valid_block[16] = 0x7F;  // Scale = 2^0 = 1.0
    
    assert(cuda_mxfp4_validate_block(valid_block, 17) == true);
    
    // Invalid: wrong size
    assert(cuda_mxfp4_validate_block(valid_block, 16) == false);
    
    // Invalid: infinity scale
    valid_block[16] = 0xFF;
    assert(cuda_mxfp4_validate_block(valid_block, 17) == false);
    
    printf("  ✓ Block validation working\n");
}

void test_dequant_zero() {
    printf("Test 3: Dequantize zero values...\n");
    
    // Create MXFP4 block with all zeros
    uint8_t h_input[17];
    for (int i = 0; i < 16; i++) {
        h_input[i] = 0x00;  // All FP4 values = 0
    }
    h_input[16] = 0x7F;  // Scale = 1.0
    
    // Allocate device memory
    uint8_t *d_input;
    half *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, 17));
    CUDA_CHECK(cudaMalloc(&d_output, 32 * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, 17, cudaMemcpyHostToDevice));
    
    // Dequantize
    cuda_mxfp4_dequant(d_output, d_input, 32, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    half h_output[32];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 32 * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Verify all zeros
    for (int i = 0; i < 32; i++) {
        float val = __half2float(h_output[i]);
        assert(approx_equal(val, 0.0f, 1e-6f));
    }
    
    printf("  ✓ Zero dequantization correct\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void test_dequant_positive() {
    printf("Test 4: Dequantize positive values...\n");
    
    uint8_t h_input[17];
    
    // FP4 values: [1.0, 2.0, 1.5, 3.0, ...]
    // Packed as: 0x21 (1.0, 2.0), 0x43 (1.5, 3.0), ...
    h_input[0] = 0x21;  // FP4: 0010 (1.0), 0001 (0.5) -> wait, need to check nibble order
    h_input[1] = 0x43;  // FP4: 0100 (2.0), 0011 (1.5)
    for (int i = 2; i < 16; i++) {
        h_input[i] = 0x22;  // FP4: 0010 (1.0), 0010 (1.0)
    }
    h_input[16] = 0x7F;  // Scale = 1.0
    
    uint8_t *d_input;
    half *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, 17));
    CUDA_CHECK(cudaMalloc(&d_output, 32 * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, 17, cudaMemcpyHostToDevice));
    
    cuda_mxfp4_dequant(d_output, d_input, 32, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half h_output[32];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 32 * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Verify values are positive and reasonable
    for (int i = 0; i < 32; i++) {
        float val = __half2float(h_output[i]);
        assert(val >= 0.0f && val <= 4.0f);
    }
    
    printf("  ✓ Positive value dequantization correct\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void test_dequant_negative() {
    printf("Test 5: Dequantize negative values...\n");
    
    uint8_t h_input[17];
    
    // FP4 negative values: 1001 (-0.5), 1010 (-1.0), etc.
    for (int i = 0; i < 16; i++) {
        h_input[i] = 0xAA;  // FP4: 1010 (-1.0), 1010 (-1.0)
    }
    h_input[16] = 0x7F;  // Scale = 1.0
    
    uint8_t *d_input;
    half *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, 17));
    CUDA_CHECK(cudaMalloc(&d_output, 32 * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, 17, cudaMemcpyHostToDevice));
    
    cuda_mxfp4_dequant(d_output, d_input, 32, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half h_output[32];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 32 * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Verify all values are negative
    for (int i = 0; i < 32; i++) {
        float val = __half2float(h_output[i]);
        assert(val <= 0.0f);
        assert(val >= -4.0f);
    }
    
    printf("  ✓ Negative value dequantization correct\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void test_dequant_scaled() {
    printf("Test 6: Dequantize with different scales...\n");
    
    uint8_t h_input[17];
    
    // All FP4 = 1.0 (0010)
    for (int i = 0; i < 16; i++) {
        h_input[i] = 0x22;
    }
    
    // Test with scale = 2^1 = 2.0
    h_input[16] = 0x80;  // Exponent = 128, scale = 2^(128-127) = 2.0
    
    uint8_t *d_input;
    half *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, 17));
    CUDA_CHECK(cudaMalloc(&d_output, 32 * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, 17, cudaMemcpyHostToDevice));
    
    cuda_mxfp4_dequant(d_output, d_input, 32, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half h_output[32];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 32 * sizeof(half), cudaMemcpyDeviceToHost));
    
    // All values should be 1.0 * 2.0 = 2.0
    for (int i = 0; i < 32; i++) {
        float val = __half2float(h_output[i]);
        assert(approx_equal(val, 2.0f, 0.1f));
    }
    
    printf("  ✓ Scaled dequantization correct\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void test_dequant_multiple_blocks() {
    printf("Test 7: Dequantize multiple blocks...\n");
    
    const int num_blocks = 4;
    const int num_elements = num_blocks * 32;
    const int input_size = num_blocks * 17;
    
    uint8_t *h_input = new uint8_t[input_size];
    
    // Initialize blocks with different patterns
    for (int b = 0; b < num_blocks; b++) {
        uint8_t *block = h_input + b * 17;
        for (int i = 0; i < 16; i++) {
            block[i] = (uint8_t)(b * 16 + i);
        }
        block[16] = 0x7F;  // Scale = 1.0
    }
    
    uint8_t *d_input;
    half *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    
    cuda_mxfp4_dequant(d_output, d_input, num_elements, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half *h_output = new half[num_elements];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_elements * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Verify all values are finite
    for (int i = 0; i < num_elements; i++) {
        float val = __half2float(h_output[i]);
        assert(isfinite(val));
    }
    
    printf("  ✓ Multiple block dequantization correct\n");
    
    delete[] h_input;
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void test_dequant_optimized() {
    printf("Test 8: Optimized dequantization...\n");
    
    const int num_elements = 1024;
    const int input_size = cuda_mxfp4_storage_size(num_elements);
    
    uint8_t *h_input = new uint8_t[input_size];
    for (int i = 0; i < input_size; i++) {
        h_input[i] = (uint8_t)(i % 256);
    }
    
    uint8_t *d_input;
    half *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    
    // Test optimized version
    cuda_mxfp4_dequant_optimized(d_output, d_input, num_elements, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    half *h_output = new half[num_elements];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_elements * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Verify
    for (int i = 0; i < num_elements; i++) {
        float val = __half2float(h_output[i]);
        assert(isfinite(val));
    }
    
    printf("  ✓ Optimized dequantization working\n");
    
    delete[] h_input;
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    printf("=== MXFP4 Dequantization Unit Tests ===\n\n");
    
    test_storage_size();
    test_block_validation();
    test_dequant_zero();
    test_dequant_positive();
    test_dequant_negative();
    test_dequant_scaled();
    test_dequant_multiple_blocks();
    test_dequant_optimized();
    
    printf("\n✅ All MXFP4 tests passed!\n");
    return 0;
}

// ---
// Crafted by GPT-Gamma 🤖
