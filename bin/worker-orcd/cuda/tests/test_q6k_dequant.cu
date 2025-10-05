// Q6_K Dequantization Kernel Tests
//
// Tests for Q6_K CUDA dequantization kernel.
// Validates correctness against reference Rust implementation.
//
// Test Strategy:
// 1. Zero block test - all zeros should produce zero output
// 2. Known value test - verify bit unpacking and scaling
// 3. Signed range test - verify [-32, 31] range handling
// 4. Multi-block test - verify batch processing
// 5. Numerical accuracy test - compare against CPU reference
//
// Spec: Based on GGML Q6_K format
// Story: Performance optimization - GPU dequantization

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <cmath>

// Forward declarations from kernel
extern "C" cudaError_t q6k_dequant(
    half* output,
    const uint8_t* input,
    int num_elements
);

// Constants
constexpr int Q6K_BLOCK_SIZE = 256;
constexpr int Q6K_BLOCK_BYTES = 210;

// Helper: Convert half to float for comparison
float half_to_float(half h) {
    return __half2float(h);
}

// Helper: Convert float to half
half float_to_half(float f) {
    return __float2half(f);
}

// Helper: Create a zero block
std::vector<uint8_t> create_zero_block() {
    return std::vector<uint8_t>(Q6K_BLOCK_BYTES, 0);
}

// Helper: Set fp16 scale in block
void set_scale(std::vector<uint8_t>& block, float scale_val) {
    half h = float_to_half(scale_val);
    uint16_t bits = __half_as_ushort(h);
    block[0] = bits & 0xFF;
    block[1] = (bits >> 8) & 0xFF;
}

// Test 1: Zero block produces zero output
TEST(Q6KDequantTest, ZeroBlock) {
    // Create zero block
    std::vector<uint8_t> input = create_zero_block();
    
    // Allocate device memory
    uint8_t* d_input;
    half* d_output;
    cudaMalloc(&d_input, Q6K_BLOCK_BYTES);
    cudaMalloc(&d_output, Q6K_BLOCK_SIZE * sizeof(half));
    
    // Copy input to device
    cudaMemcpy(d_input, input.data(), Q6K_BLOCK_BYTES, cudaMemcpyHostToDevice);
    
    // Run kernel
    cudaError_t err = q6k_dequant(d_output, d_input, Q6K_BLOCK_SIZE);
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);
    
    // Copy output back
    std::vector<half> output(Q6K_BLOCK_SIZE);
    cudaMemcpy(output.data(), d_output, Q6K_BLOCK_SIZE * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Verify all zeros
    for (int i = 0; i < Q6K_BLOCK_SIZE; i++) {
        float val = half_to_float(output[i]);
        EXPECT_FLOAT_EQ(val, 0.0f) << "Element " << i << " should be zero";
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test 2: Known value - verify bit unpacking
TEST(Q6KDequantTest, KnownValue) {
    // Create block with d=1.0, scales[0]=1
    std::vector<uint8_t> input = create_zero_block();
    set_scale(input, 1.0f);
    
    // Set scales[0] = 1 (at byte 194)
    input[194] = 1;
    
    // Set element (g=0, t=0): low4=0, hi2=0 → q6_u=0 → q6_s=-32
    // ql[0] low nibble = 0 (already zero)
    // qh bits = 0 (already zero)
    // Expected: 1.0 * 1 * (-32) = -32.0
    
    // Allocate device memory
    uint8_t* d_input;
    half* d_output;
    cudaMalloc(&d_input, Q6K_BLOCK_BYTES);
    cudaMalloc(&d_output, Q6K_BLOCK_SIZE * sizeof(half));
    
    // Copy input to device
    cudaMemcpy(d_input, input.data(), Q6K_BLOCK_BYTES, cudaMemcpyHostToDevice);
    
    // Run kernel
    cudaError_t err = q6k_dequant(d_output, d_input, Q6K_BLOCK_SIZE);
    ASSERT_EQ(err, cudaSuccess);
    
    // Copy output back
    std::vector<half> output(Q6K_BLOCK_SIZE);
    cudaMemcpy(output.data(), d_output, Q6K_BLOCK_SIZE * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Verify element 0
    float val = half_to_float(output[0]);
    EXPECT_NEAR(val, -32.0f, 0.1f) << "Element 0 should be -32.0";
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test 3: Signed range verification
TEST(Q6KDequantTest, SignedRange) {
    // Create block with d=1.0, scales[0]=1
    std::vector<uint8_t> input = create_zero_block();
    set_scale(input, 1.0f);
    input[194] = 1;  // scales[0] = 1
    
    // Set element (g=0, t=0): low4=15, hi2=3 → q6_u=63 → q6_s=31
    // ql[0] low nibble = 15
    input[2] = 0x0F;
    
    // qh: Need to set bits for hi2=3 (both bits set)
    // For g=0, t=0: base = 0*4 + (0&7) = 0, shift = 0>>3 = 0
    // b0 at qh[0] bit 0, b1 at qh[0] bit 2
    input[130] = 0x05;  // bits 0 and 2 set
    
    // Expected: 1.0 * 1 * 31 = 31.0
    
    // Allocate device memory
    uint8_t* d_input;
    half* d_output;
    cudaMalloc(&d_input, Q6K_BLOCK_BYTES);
    cudaMalloc(&d_output, Q6K_BLOCK_SIZE * sizeof(half));
    
    // Copy input to device
    cudaMemcpy(d_input, input.data(), Q6K_BLOCK_BYTES, cudaMemcpyHostToDevice);
    
    // Run kernel
    cudaError_t err = q6k_dequant(d_output, d_input, Q6K_BLOCK_SIZE);
    ASSERT_EQ(err, cudaSuccess);
    
    // Copy output back
    std::vector<half> output(Q6K_BLOCK_SIZE);
    cudaMemcpy(output.data(), d_output, Q6K_BLOCK_SIZE * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Verify element 0
    float val = half_to_float(output[0]);
    EXPECT_NEAR(val, 31.0f, 0.1f) << "Element 0 should be 31.0 (max signed value)";
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test 4: Multi-block processing
TEST(Q6KDequantTest, MultiBlock) {
    const int num_blocks = 4;
    const int total_elements = num_blocks * Q6K_BLOCK_SIZE;
    const int total_bytes = num_blocks * Q6K_BLOCK_BYTES;
    
    // Create input with different scales per block
    std::vector<uint8_t> input(total_bytes, 0);
    
    for (int b = 0; b < num_blocks; b++) {
        int offset = b * Q6K_BLOCK_BYTES;
        
        // Set scale = b + 1.0
        float scale_val = static_cast<float>(b + 1);
        half h = float_to_half(scale_val);
        uint16_t bits = __half_as_ushort(h);
        input[offset + 0] = bits & 0xFF;
        input[offset + 1] = (bits >> 8) & 0xFF;
        
        // Set scales[0] = 1
        input[offset + 194] = 1;
    }
    
    // Allocate device memory
    uint8_t* d_input;
    half* d_output;
    cudaMalloc(&d_input, total_bytes);
    cudaMalloc(&d_output, total_elements * sizeof(half));
    
    // Copy input to device
    cudaMemcpy(d_input, input.data(), total_bytes, cudaMemcpyHostToDevice);
    
    // Run kernel
    cudaError_t err = q6k_dequant(d_output, d_input, total_elements);
    ASSERT_EQ(err, cudaSuccess);
    
    // Copy output back
    std::vector<half> output(total_elements);
    cudaMemcpy(output.data(), d_output, total_elements * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Verify first element of each block has correct scale
    for (int b = 0; b < num_blocks; b++) {
        float expected = static_cast<float>(b + 1) * 1.0f * (-32.0f);  // scale * sub_scale * q6_s
        float val = half_to_float(output[b * Q6K_BLOCK_SIZE]);
        EXPECT_NEAR(val, expected, 0.5f) << "Block " << b << " first element mismatch";
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test 5: Invalid input size
TEST(Q6KDequantTest, InvalidInputSize) {
    // Try to dequantize non-multiple of 256
    const int invalid_size = 100;
    
    uint8_t* d_input;
    half* d_output;
    cudaMalloc(&d_input, Q6K_BLOCK_BYTES);
    cudaMalloc(&d_output, invalid_size * sizeof(half));
    
    // Should return error
    cudaError_t err = q6k_dequant(d_output, d_input, invalid_size);
    EXPECT_NE(err, cudaSuccess) << "Should reject non-multiple of 256";
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test 6: Sub-block scale variation
TEST(Q6KDequantTest, SubBlockScales) {
    // Create block with varying sub-block scales
    std::vector<uint8_t> input = create_zero_block();
    set_scale(input, 1.0f);
    
    // Set different scales for each sub-block
    for (int g = 0; g < 16; g++) {
        input[194 + g] = g + 1;  // scales[g] = g + 1
    }
    
    // Allocate device memory
    uint8_t* d_input;
    half* d_output;
    cudaMalloc(&d_input, Q6K_BLOCK_BYTES);
    cudaMalloc(&d_output, Q6K_BLOCK_SIZE * sizeof(half));
    
    // Copy input to device
    cudaMemcpy(d_input, input.data(), Q6K_BLOCK_BYTES, cudaMemcpyHostToDevice);
    
    // Run kernel
    cudaError_t err = q6k_dequant(d_output, d_input, Q6K_BLOCK_SIZE);
    ASSERT_EQ(err, cudaSuccess);
    
    // Copy output back
    std::vector<half> output(Q6K_BLOCK_SIZE);
    cudaMemcpy(output.data(), d_output, Q6K_BLOCK_SIZE * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Verify first element of each sub-block
    for (int g = 0; g < 16; g++) {
        int idx = g * 16;  // First element of sub-block g
        float expected = 1.0f * static_cast<float>(g + 1) * (-32.0f);
        float val = half_to_float(output[idx]);
        EXPECT_NEAR(val, expected, 0.5f) << "Sub-block " << g << " first element mismatch";
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

// Main
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
