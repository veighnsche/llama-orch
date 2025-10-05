/**
 * SwiGLU FFN Kernel Tests - LT-017
 * 
 * Unit tests for SwiGLU (Swish-Gated Linear Unit) activation kernel.
 * 
 * Tests cover:
 * - SiLU activation
 * - Element-wise multiply
 * - Vectorized path
 * - Different FFN dimensions
 * - Numerical correctness
 * 
 * Spec: M0-W-1214
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

// External C function
extern "C" int cuda_swiglu_activation(
    half* output,
    const half* gate,
    const half* up,
    int batch_size,
    int seq_len,
    int ffn_dim
);

class SwiGLUTest : public ::testing::Test {
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
    
    // Reference SiLU implementation
    float silu(float x) {
        return x / (1.0f + expf(-x));
    }
};

// Test: Basic SwiGLU activation
TEST_F(SwiGLUTest, BasicActivation) {
    int batch = 1, seq_len = 1, ffn_dim = 4;
    int size = batch * seq_len * ffn_dim;
    
    half* d_gate = allocate_device(size);
    half* d_up = allocate_device(size);
    half* d_output = allocate_device(size);
    
    std::vector<float> gate = {1.0f, 2.0f, -1.0f, 0.0f};
    std::vector<float> up = {0.5f, 0.5f, 0.5f, 0.5f};
    
    copy_to_device(d_gate, gate);
    copy_to_device(d_up, up);
    
    int result = cuda_swiglu_activation(
        d_output, d_gate, d_up,
        batch, seq_len, ffn_dim
    );
    
    EXPECT_EQ(result, 0);
    
    auto output = copy_from_device(d_output, size);
    
    // Verify SiLU(gate) * up
    EXPECT_NEAR(output[0], silu(1.0f) * 0.5f, 0.01f);
    EXPECT_NEAR(output[1], silu(2.0f) * 0.5f, 0.01f);
    EXPECT_NEAR(output[2], silu(-1.0f) * 0.5f, 0.01f);
    EXPECT_NEAR(output[3], silu(0.0f) * 0.5f, 0.01f);
    
    free_device(d_gate);
    free_device(d_up);
    free_device(d_output);
}

// Test: SiLU activation properties
TEST_F(SwiGLUTest, SiLUProperties) {
    int batch = 1, seq_len = 1, ffn_dim = 2;
    int size = batch * seq_len * ffn_dim;
    
    half* d_gate = allocate_device(size);
    half* d_up = allocate_device(size);
    half* d_output = allocate_device(size);
    
    // Test: SiLU(0) ≈ 0
    std::vector<float> gate = {0.0f, 0.0f};
    std::vector<float> up = {1.0f, 1.0f};
    
    copy_to_device(d_gate, gate);
    copy_to_device(d_up, up);
    
    cuda_swiglu_activation(d_output, d_gate, d_up, batch, seq_len, ffn_dim);
    
    auto output = copy_from_device(d_output, size);
    
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    EXPECT_NEAR(output[0], 0.0f, 0.01f);
    
    free_device(d_gate);
    free_device(d_up);
    free_device(d_output);
}

// Test: Different FFN dimensions (Qwen, Phi-3)
TEST_F(SwiGLUTest, DifferentFFNDimensions) {
    std::vector<int> ffn_dims = {4864, 10240};  // Qwen, Phi-3
    
    for (int ffn_dim : ffn_dims) {
        int batch = 1, seq_len = 1;
        int size = batch * seq_len * ffn_dim;
        
        half* d_gate = allocate_device(size);
        half* d_up = allocate_device(size);
        half* d_output = allocate_device(size);
        
        std::vector<float> gate(size, 1.0f);
        std::vector<float> up(size, 1.0f);
        
        copy_to_device(d_gate, gate);
        copy_to_device(d_up, up);
        
        int result = cuda_swiglu_activation(
            d_output, d_gate, d_up,
            batch, seq_len, ffn_dim
        );
        
        EXPECT_EQ(result, 0) << "Failed for ffn_dim=" << ffn_dim;
        
        free_device(d_gate);
        free_device(d_up);
        free_device(d_output);
    }
}

// Test: Vectorized path (even ffn_dim)
TEST_F(SwiGLUTest, VectorizedPath) {
    int batch = 1, seq_len = 1, ffn_dim = 8;  // Even dimension
    int size = batch * seq_len * ffn_dim;
    
    half* d_gate = allocate_device(size);
    half* d_up = allocate_device(size);
    half* d_output = allocate_device(size);
    
    std::vector<float> gate(size, 2.0f);
    std::vector<float> up(size, 3.0f);
    
    copy_to_device(d_gate, gate);
    copy_to_device(d_up, up);
    
    int result = cuda_swiglu_activation(
        d_output, d_gate, d_up,
        batch, seq_len, ffn_dim
    );
    
    EXPECT_EQ(result, 0);
    
    auto output = copy_from_device(d_output, size);
    
    // Verify silu(2.0) * 3.0
    float expected = silu(2.0f) * 3.0f;
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(output[i], expected, 0.01f);
    }
    
    free_device(d_gate);
    free_device(d_up);
    free_device(d_output);
}

// Test: Invalid dimensions
TEST_F(SwiGLUTest, InvalidDimensions) {
    half* dummy = allocate_device(1);
    
    // Negative dimensions
    int result1 = cuda_swiglu_activation(
        dummy, dummy, dummy,
        -1, 1, 4
    );
    EXPECT_NE(result1, 0);
    
    // Zero ffn_dim
    int result2 = cuda_swiglu_activation(
        dummy, dummy, dummy,
        1, 1, 0
    );
    EXPECT_NE(result2, 0);
    
    free_device(dummy);
}

// Test: Batch processing
TEST_F(SwiGLUTest, BatchProcessing) {
    int batch = 4, seq_len = 2, ffn_dim = 16;
    int size = batch * seq_len * ffn_dim;
    
    half* d_gate = allocate_device(size);
    half* d_up = allocate_device(size);
    half* d_output = allocate_device(size);
    
    std::vector<float> gate(size, 1.0f);
    std::vector<float> up(size, 1.0f);
    
    copy_to_device(d_gate, gate);
    copy_to_device(d_up, up);
    
    int result = cuda_swiglu_activation(
        d_output, d_gate, d_up,
        batch, seq_len, ffn_dim
    );
    
    EXPECT_EQ(result, 0);
    
    free_device(d_gate);
    free_device(d_up);
    free_device(d_output);
}
