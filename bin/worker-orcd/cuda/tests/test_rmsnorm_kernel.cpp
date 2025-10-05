/**
 * RMSNorm Kernel Tests - LT-013
 * 
 * Unit tests for Root Mean Square Normalization kernel.
 * 
 * Tests cover:
 * - Basic RMSNorm application
 * - Numerical stability
 * - Different hidden dimensions
 * - Dimension validation
 * 
 * Spec: M0-W-1214, M0-W-1430
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

// External C function
extern "C" int cuda_rmsnorm_forward(
    half* output,
    const half* input,
    const half* weight,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps
);

class RMSNormKernelTest : public ::testing::Test {
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

// Test: Basic RMSNorm
TEST_F(RMSNormKernelTest, BasicRMSNorm) {
    int batch = 1, seq_len = 1, hidden_dim = 4;
    float eps = 1e-6f;
    
    int size = batch * seq_len * hidden_dim;
    
    half* d_input = allocate_device(size);
    half* d_weight = allocate_device(hidden_dim);
    half* d_output = allocate_device(size);
    
    // Input: [1, 2, 3, 4]
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f};
    
    copy_to_device(d_input, input);
    copy_to_device(d_weight, weight);
    
    int result = cuda_rmsnorm_forward(
        d_output, d_input, d_weight,
        batch, seq_len, hidden_dim, eps
    );
    
    EXPECT_EQ(result, 0);
    
    auto output = copy_from_device(d_output, size);
    
    // Compute expected RMS: sqrt(mean([1, 4, 9, 16])) = sqrt(7.5) ≈ 2.74
    float expected_rms = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + eps);
    
    // Output should be input / rms
    EXPECT_NEAR(output[0], 1.0f / expected_rms, 0.01f);
    EXPECT_NEAR(output[1], 2.0f / expected_rms, 0.01f);
    
    free_device(d_input);
    free_device(d_weight);
    free_device(d_output);
}

// Test: RMSNorm with weight scaling
TEST_F(RMSNormKernelTest, WeightScaling) {
    int batch = 1, seq_len = 1, hidden_dim = 4;
    float eps = 1e-6f;
    
    int size = batch * seq_len * hidden_dim;
    
    half* d_input = allocate_device(size);
    half* d_weight = allocate_device(hidden_dim);
    half* d_output = allocate_device(size);
    
    std::vector<float> input = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> weight = {2.0f, 2.0f, 2.0f, 2.0f};
    
    copy_to_device(d_input, input);
    copy_to_device(d_weight, weight);
    
    cuda_rmsnorm_forward(
        d_output, d_input, d_weight,
        batch, seq_len, hidden_dim, eps
    );
    
    auto output = copy_from_device(d_output, size);
    
    // RMS of [1,1,1,1] = 1.0
    // Output = (input / 1.0) * weight = 1.0 * 2.0 = 2.0
    EXPECT_NEAR(output[0], 2.0f, 0.01f);
    EXPECT_NEAR(output[1], 2.0f, 0.01f);
    
    free_device(d_input);
    free_device(d_weight);
    free_device(d_output);
}

// Test: Numerical stability (small values)
TEST_F(RMSNormKernelTest, NumericalStabilitySmallValues) {
    int batch = 1, seq_len = 1, hidden_dim = 4;
    float eps = 1e-6f;
    
    int size = batch * seq_len * hidden_dim;
    
    half* d_input = allocate_device(size);
    half* d_weight = allocate_device(hidden_dim);
    half* d_output = allocate_device(size);
    
    // Very small values
    std::vector<float> input = {1e-4f, 1e-4f, 1e-4f, 1e-4f};
    std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f};
    
    copy_to_device(d_input, input);
    copy_to_device(d_weight, weight);
    
    int result = cuda_rmsnorm_forward(
        d_output, d_input, d_weight,
        batch, seq_len, hidden_dim, eps
    );
    
    EXPECT_EQ(result, 0);
    
    // Should not crash or produce NaN
    auto output = copy_from_device(d_output, size);
    for (float val : output) {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
    
    free_device(d_input);
    free_device(d_weight);
    free_device(d_output);
}

// Test: Different hidden dimensions
TEST_F(RMSNormKernelTest, DifferentHiddenDimensions) {
    std::vector<int> dims = {896, 3072, 4096};
    
    for (int hidden_dim : dims) {
        int batch = 1, seq_len = 1;
        float eps = 1e-6f;
        
        int size = batch * seq_len * hidden_dim;
        
        half* d_input = allocate_device(size);
        half* d_weight = allocate_device(hidden_dim);
        half* d_output = allocate_device(size);
        
        std::vector<float> input(size, 1.0f);
        std::vector<float> weight(hidden_dim, 1.0f);
        
        copy_to_device(d_input, input);
        copy_to_device(d_weight, weight);
        
        int result = cuda_rmsnorm_forward(
            d_output, d_input, d_weight,
            batch, seq_len, hidden_dim, eps
        );
        
        EXPECT_EQ(result, 0) << "Failed for hidden_dim=" << hidden_dim;
        
        free_device(d_input);
        free_device(d_weight);
        free_device(d_output);
    }
}

// Test: Invalid dimensions
TEST_F(RMSNormKernelTest, InvalidDimensions) {
    half* dummy = allocate_device(1);
    
    // Negative dimensions
    int result1 = cuda_rmsnorm_forward(
        dummy, dummy, dummy,
        -1, 1, 4, 1e-6f
    );
    EXPECT_NE(result1, 0);
    
    // Zero epsilon
    int result2 = cuda_rmsnorm_forward(
        dummy, dummy, dummy,
        1, 1, 4, 0.0f
    );
    EXPECT_NE(result2, 0);
    
    // Negative epsilon
    int result3 = cuda_rmsnorm_forward(
        dummy, dummy, dummy,
        1, 1, 4, -1e-6f
    );
    EXPECT_NE(result3, 0);
    
    free_device(dummy);
}

// Test: Batch processing
TEST_F(RMSNormKernelTest, BatchProcessing) {
    int batch = 4, seq_len = 2, hidden_dim = 8;
    float eps = 1e-6f;
    
    int size = batch * seq_len * hidden_dim;
    
    half* d_input = allocate_device(size);
    half* d_weight = allocate_device(hidden_dim);
    half* d_output = allocate_device(size);
    
    std::vector<float> input(size, 1.0f);
    std::vector<float> weight(hidden_dim, 1.0f);
    
    copy_to_device(d_input, input);
    copy_to_device(d_weight, weight);
    
    int result = cuda_rmsnorm_forward(
        d_output, d_input, d_weight,
        batch, seq_len, hidden_dim, eps
    );
    
    EXPECT_EQ(result, 0);
    
    free_device(d_input);
    free_device(d_weight);
    free_device(d_output);
}
