/**
 * Residual Connection Kernel Tests - LT-014
 * 
 * Unit tests for residual connection kernel.
 * 
 * Tests cover:
 * - Basic residual addition
 * - In-place operation
 * - Out-of-place operation
 * - Different tensor shapes
 * - Dimension validation
 * 
 * Spec: M0-W-1214
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

// External C function
extern "C" int cuda_residual_forward(
    half* output,
    const half* input,
    const half* residual,
    int batch_size,
    int seq_len,
    int hidden_dim,
    bool in_place
);

class ResidualKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    half* allocate_device(size_t elements) {
        half* ptr;
        cudaMalloc(&ptr, elements * sizeof(half));
        return ptr;
    }
    
    void free_device(half* ptr) {
        cudaFree(ptr);
    }
    
    void copy_to_device(half* dst, const std::vector<float>& src) {
        std::vector<half> host(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            host[i] = __float2half(src[i]);
        }
        cudaMemcpy(dst, host.data(), src.size() * sizeof(half), cudaMemcpyHostToDevice);
    }
    
    std::vector<float> copy_from_device(const half* src, size_t elements) {
        std::vector<half> host(elements);
        cudaMemcpy(host.data(), src, elements * sizeof(half), cudaMemcpyDeviceToHost);
        
        std::vector<float> result(elements);
        for (size_t i = 0; i < elements; ++i) {
            result[i] = __half2float(host[i]);
        }
        return result;
    }
};

// Test: Basic residual addition
TEST_F(ResidualKernelTest, BasicResidualAddition) {
    int batch = 1, seq_len = 1, hidden_dim = 4;
    int size = batch * seq_len * hidden_dim;
    
    half* d_input = allocate_device(size);
    half* d_residual = allocate_device(size);
    half* d_output = allocate_device(size);
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> residual = {0.5f, 0.5f, 0.5f, 0.5f};
    
    copy_to_device(d_input, input);
    copy_to_device(d_residual, residual);
    
    int result = cuda_residual_forward(
        d_output, d_input, d_residual,
        batch, seq_len, hidden_dim, false
    );
    
    EXPECT_EQ(result, 0);
    
    auto output = copy_from_device(d_output, size);
    
    EXPECT_NEAR(output[0], 1.5f, 0.01f);
    EXPECT_NEAR(output[1], 2.5f, 0.01f);
    EXPECT_NEAR(output[2], 3.5f, 0.01f);
    EXPECT_NEAR(output[3], 4.5f, 0.01f);
    
    free_device(d_input);
    free_device(d_residual);
    free_device(d_output);
}

// Test: In-place operation
TEST_F(ResidualKernelTest, InPlaceOperation) {
    int batch = 1, seq_len = 1, hidden_dim = 4;
    int size = batch * seq_len * hidden_dim;
    
    half* d_output = allocate_device(size);
    half* d_residual = allocate_device(size);
    
    std::vector<float> initial = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> residual = {0.5f, 0.5f, 0.5f, 0.5f};
    
    copy_to_device(d_output, initial);
    copy_to_device(d_residual, residual);
    
    // In-place: output += residual
    int result = cuda_residual_forward(
        d_output, nullptr, d_residual,
        batch, seq_len, hidden_dim, true
    );
    
    EXPECT_EQ(result, 0);
    
    auto output = copy_from_device(d_output, size);
    
    EXPECT_NEAR(output[0], 1.5f, 0.01f);
    EXPECT_NEAR(output[1], 2.5f, 0.01f);
    
    free_device(d_output);
    free_device(d_residual);
}

// Test: Out-of-place operation
TEST_F(ResidualKernelTest, OutOfPlaceOperation) {
    int batch = 1, seq_len = 1, hidden_dim = 4;
    int size = batch * seq_len * hidden_dim;
    
    half* d_input = allocate_device(size);
    half* d_residual = allocate_device(size);
    half* d_output = allocate_device(size);
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> residual = {0.5f, 0.5f, 0.5f, 0.5f};
    
    copy_to_device(d_input, input);
    copy_to_device(d_residual, residual);
    
    int result = cuda_residual_forward(
        d_output, d_input, d_residual,
        batch, seq_len, hidden_dim, false
    );
    
    EXPECT_EQ(result, 0);
    
    // Verify input unchanged
    auto input_check = copy_from_device(d_input, size);
    EXPECT_NEAR(input_check[0], 1.0f, 0.01f);
    
    free_device(d_input);
    free_device(d_residual);
    free_device(d_output);
}

// Test: Different tensor shapes
TEST_F(ResidualKernelTest, DifferentShapes) {
    std::vector<std::tuple<int, int, int>> shapes = {
        {1, 1, 896},    // Qwen hidden dim
        {1, 128, 64},   // Prefill sequence
        {4, 1, 3072},   // Phi-3 hidden dim, batch=4
    };
    
    for (auto [batch, seq_len, hidden_dim] : shapes) {
        int size = batch * seq_len * hidden_dim;
        
        half* d_input = allocate_device(size);
        half* d_residual = allocate_device(size);
        half* d_output = allocate_device(size);
        
        std::vector<float> input(size, 1.0f);
        std::vector<float> residual(size, 0.5f);
        
        copy_to_device(d_input, input);
        copy_to_device(d_residual, residual);
        
        int result = cuda_residual_forward(
            d_output, d_input, d_residual,
            batch, seq_len, hidden_dim, false
        );
        
        EXPECT_EQ(result, 0) << "Failed for shape (" << batch << ", " 
                             << seq_len << ", " << hidden_dim << ")";
        
        free_device(d_input);
        free_device(d_residual);
        free_device(d_output);
    }
}

// Test: Invalid dimensions
TEST_F(ResidualKernelTest, InvalidDimensions) {
    half* dummy = allocate_device(1);
    
    // Negative batch size
    int result1 = cuda_residual_forward(
        dummy, dummy, dummy,
        -1, 1, 4, false
    );
    EXPECT_NE(result1, 0);
    
    // Zero hidden_dim
    int result2 = cuda_residual_forward(
        dummy, dummy, dummy,
        1, 1, 0, false
    );
    EXPECT_NE(result2, 0);
    
    free_device(dummy);
}

// Test: Vectorized path (even hidden_dim)
TEST_F(ResidualKernelTest, VectorizedPath) {
    int batch = 1, seq_len = 1, hidden_dim = 8;  // Even dimension
    int size = batch * seq_len * hidden_dim;
    
    half* d_input = allocate_device(size);
    half* d_residual = allocate_device(size);
    half* d_output = allocate_device(size);
    
    std::vector<float> input(size, 2.0f);
    std::vector<float> residual(size, 3.0f);
    
    copy_to_device(d_input, input);
    copy_to_device(d_residual, residual);
    
    int result = cuda_residual_forward(
        d_output, d_input, d_residual,
        batch, seq_len, hidden_dim, false
    );
    
    EXPECT_EQ(result, 0);
    
    auto output = copy_from_device(d_output, size);
    
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(output[i], 5.0f, 0.01f);
    }
    
    free_device(d_input);
    free_device(d_residual);
    free_device(d_output);
}
