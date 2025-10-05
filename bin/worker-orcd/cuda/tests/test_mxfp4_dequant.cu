// Unit tests for MXFP4 dequantization kernel
//
// Tests MXFP4 format dequantization
//
// Story: GT-030

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

TEST(MXFP4Dequant, StorageSize) {
    // 32 elements = 1 block = 17 bytes
    EXPECT_EQ(cuda_mxfp4_storage_size(32), 17);
    
    // 64 elements = 2 blocks = 34 bytes
    EXPECT_EQ(cuda_mxfp4_storage_size(64), 34);
    
    // 33 elements = 2 blocks = 34 bytes (rounds up)
    EXPECT_EQ(cuda_mxfp4_storage_size(33), 34);
    
    // 1024 elements = 32 blocks = 544 bytes
    EXPECT_EQ(cuda_mxfp4_storage_size(1024), 544);
}

TEST(MXFP4Dequant, BlockValidation) {
    uint8_t valid_block[17];
    for (int i = 0; i < 16; i++) {
        valid_block[i] = 0x00;
    }
    valid_block[16] = 0x7F;  // Scale = 2^0 = 1.0
    
    EXPECT_TRUE(cuda_mxfp4_validate_block(valid_block, 17));
    
    // Invalid: wrong size
    EXPECT_FALSE(cuda_mxfp4_validate_block(valid_block, 16));
    
    // Invalid: infinity scale
    valid_block[16] = 0xFF;
    EXPECT_FALSE(cuda_mxfp4_validate_block(valid_block, 17));
}

TEST(MXFP4Dequant, DequantZero) {
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
        EXPECT_TRUE(approx_equal(val, 0.0f, 1e-6f));
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

TEST(MXFP4Dequant, DequantPositive) {
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
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 4.0f);
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

TEST(MXFP4Dequant, DequantNegative) {
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
        EXPECT_LE(val, 0.0f);
        EXPECT_GE(val, -4.0f);
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

TEST(MXFP4Dequant, DequantScaled) {
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
        EXPECT_TRUE(approx_equal(val, 2.0f, 0.1f));
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

TEST(MXFP4Dequant, DequantMultipleBlocks) {
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
        EXPECT_TRUE(isfinite(val));
    }
    
    delete[] h_input;
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

TEST(MXFP4Dequant, DequantOptimized) {
    const int num_elements = 1024;
    const int input_size = cuda_mxfp4_storage_size(num_elements);
    
    uint8_t *h_input = new uint8_t[input_size];
    for (int i = 0; i < input_size; i++) {
        h_input[i] = (uint8_t)(i % 256);
    }
    
    // Fix scale factors (every 17th byte) to avoid infinity
    int num_blocks = (num_elements + 31) / 32;
    for (int b = 0; b < num_blocks; b++) {
        h_input[b * 17 + 16] = 0x7F;  // Scale = 1.0 (valid)
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
        EXPECT_TRUE(isfinite(val));
    }
    
    delete[] h_input;
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// ---
// Crafted by GPT-Gamma 🤖
// Converted to GTest by Cascade 🌊
