/**
 * RoPE Kernel Tests - LT-012
 * 
 * Unit tests for Rotary Position Embedding kernel.
 * 
 * Tests cover:
 * - Basic RoPE application
 * - Different frequency bases
 * - GQA support
 * - Dimension validation
 * - Numerical correctness
 * 
 * Spec: M0-W-1214, M0-W-1430
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

// External C function
extern "C" int cuda_rope_forward(
    half* q_out,
    half* k_out,
    const half* q_in,
    const half* k_in,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float freq_base,
    int rope_dim
);

class RoPEKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        cudaSetDevice(0);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // Helper: Allocate device memory
    half* allocate_device(size_t elements) {
        half* ptr;
        cudaMalloc(&ptr, elements * sizeof(half));
        return ptr;
    }
    
    // Helper: Free device memory
    void free_device(half* ptr) {
        cudaFree(ptr);
    }
    
    // Helper: Copy to device
    void copy_to_device(half* dst, const std::vector<float>& src) {
        std::vector<half> host(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            host[i] = __float2half(src[i]);
        }
        cudaMemcpy(dst, host.data(), src.size() * sizeof(half), cudaMemcpyHostToDevice);
    }
    
    // Helper: Copy from device
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

// Test: Basic RoPE application (single position)
TEST_F(RoPEKernelTest, BasicRoPESinglePosition) {
    int batch = 1, seq_len = 1, num_heads = 1, num_kv_heads = 1, head_dim = 4;
    float freq_base = 10000.0f;
    int rope_dim = head_dim;
    
    int q_size = batch * seq_len * num_heads * head_dim;
    int k_size = batch * seq_len * num_kv_heads * head_dim;
    
    // Allocate device memory
    half* d_q_in = allocate_device(q_size);
    half* d_k_in = allocate_device(k_size);
    half* d_q_out = allocate_device(q_size);
    half* d_k_out = allocate_device(k_size);
    
    // Initialize input (simple values)
    std::vector<float> q_in = {1.0f, 0.0f, 1.0f, 0.0f};
    copy_to_device(d_q_in, q_in);
    copy_to_device(d_k_in, q_in);
    
    // Apply RoPE
    int result = cuda_rope_forward(
        d_q_out, d_k_out, d_q_in, d_k_in,
        batch, seq_len, num_heads, num_kv_heads,
        head_dim, freq_base, rope_dim
    );
    
    EXPECT_EQ(result, 0);
    
    // Verify output
    auto q_out = copy_from_device(d_q_out, q_size);
    
    // At position 0, rotation should be identity (cos=1, sin=0)
    EXPECT_NEAR(q_out[0], 1.0f, 0.01f);
    EXPECT_NEAR(q_out[1], 0.0f, 0.01f);
    
    // Cleanup
    free_device(d_q_in);
    free_device(d_k_in);
    free_device(d_q_out);
    free_device(d_k_out);
}

// Test: RoPE with multiple positions
TEST_F(RoPEKernelTest, RoPEMultiplePositions) {
    int batch = 1, seq_len = 4, num_heads = 1, num_kv_heads = 1, head_dim = 4;
    float freq_base = 10000.0f;
    int rope_dim = head_dim;
    
    int q_size = batch * seq_len * num_heads * head_dim;
    
    half* d_q_in = allocate_device(q_size);
    half* d_k_in = allocate_device(q_size);
    half* d_q_out = allocate_device(q_size);
    half* d_k_out = allocate_device(q_size);
    
    // Initialize with ones
    std::vector<float> q_in(q_size, 1.0f);
    copy_to_device(d_q_in, q_in);
    copy_to_device(d_k_in, q_in);
    
    int result = cuda_rope_forward(
        d_q_out, d_k_out, d_q_in, d_k_in,
        batch, seq_len, num_heads, num_kv_heads,
        head_dim, freq_base, rope_dim
    );
    
    EXPECT_EQ(result, 0);
    
    free_device(d_q_in);
    free_device(d_k_in);
    free_device(d_q_out);
    free_device(d_k_out);
}

// Test: RoPE with different frequency bases
TEST_F(RoPEKernelTest, DifferentFrequencyBases) {
    int batch = 1, seq_len = 2, num_heads = 1, num_kv_heads = 1, head_dim = 4;
    int rope_dim = head_dim;
    
    int q_size = batch * seq_len * num_heads * head_dim;
    
    half* d_q_in = allocate_device(q_size);
    half* d_k_in = allocate_device(q_size);
    half* d_q_out = allocate_device(q_size);
    half* d_k_out = allocate_device(q_size);
    
    std::vector<float> q_in(q_size, 1.0f);
    copy_to_device(d_q_in, q_in);
    copy_to_device(d_k_in, q_in);
    
    // Test with standard base
    int result1 = cuda_rope_forward(
        d_q_out, d_k_out, d_q_in, d_k_in,
        batch, seq_len, num_heads, num_kv_heads,
        head_dim, 10000.0f, rope_dim
    );
    EXPECT_EQ(result1, 0);
    
    // Test with extended context base
    int result2 = cuda_rope_forward(
        d_q_out, d_k_out, d_q_in, d_k_in,
        batch, seq_len, num_heads, num_kv_heads,
        head_dim, 1000000.0f, rope_dim
    );
    EXPECT_EQ(result2, 0);
    
    free_device(d_q_in);
    free_device(d_k_in);
    free_device(d_q_out);
    free_device(d_k_out);
}

// Test: GQA support (num_heads != num_kv_heads)
TEST_F(RoPEKernelTest, GQASupport) {
    int batch = 1, seq_len = 2, num_heads = 8, num_kv_heads = 2, head_dim = 64;
    float freq_base = 10000.0f;
    int rope_dim = head_dim;
    
    int q_size = batch * seq_len * num_heads * head_dim;
    int k_size = batch * seq_len * num_kv_heads * head_dim;
    
    half* d_q_in = allocate_device(q_size);
    half* d_k_in = allocate_device(k_size);
    half* d_q_out = allocate_device(q_size);
    half* d_k_out = allocate_device(k_size);
    
    std::vector<float> q_in(q_size, 1.0f);
    std::vector<float> k_in(k_size, 1.0f);
    copy_to_device(d_q_in, q_in);
    copy_to_device(d_k_in, k_in);
    
    int result = cuda_rope_forward(
        d_q_out, d_k_out, d_q_in, d_k_in,
        batch, seq_len, num_heads, num_kv_heads,
        head_dim, freq_base, rope_dim
    );
    
    EXPECT_EQ(result, 0);
    
    free_device(d_q_in);
    free_device(d_k_in);
    free_device(d_q_out);
    free_device(d_k_out);
}

// Test: Invalid dimensions
TEST_F(RoPEKernelTest, InvalidDimensions) {
    half* dummy = allocate_device(1);
    
    // Negative dimensions
    int result1 = cuda_rope_forward(
        dummy, dummy, dummy, dummy,
        -1, 1, 1, 1, 4, 10000.0f, 4
    );
    EXPECT_NE(result1, 0);
    
    // Odd head_dim
    int result2 = cuda_rope_forward(
        dummy, dummy, dummy, dummy,
        1, 1, 1, 1, 3, 10000.0f, 3
    );
    EXPECT_NE(result2, 0);
    
    // rope_dim > head_dim
    int result3 = cuda_rope_forward(
        dummy, dummy, dummy, dummy,
        1, 1, 1, 1, 4, 10000.0f, 8
    );
    EXPECT_NE(result3, 0);
    
    free_device(dummy);
}

// Test: Rotation preserves magnitude
TEST_F(RoPEKernelTest, RotationPreservesMagnitude) {
    int batch = 1, seq_len = 1, num_heads = 1, num_kv_heads = 1, head_dim = 4;
    float freq_base = 10000.0f;
    int rope_dim = head_dim;
    
    int q_size = batch * seq_len * num_heads * head_dim;
    
    half* d_q_in = allocate_device(q_size);
    half* d_k_in = allocate_device(q_size);
    half* d_q_out = allocate_device(q_size);
    half* d_k_out = allocate_device(q_size);
    
    // Input with known magnitude
    std::vector<float> q_in = {3.0f, 4.0f, 1.0f, 0.0f};  // magnitude = 5.0
    copy_to_device(d_q_in, q_in);
    copy_to_device(d_k_in, q_in);
    
    cuda_rope_forward(
        d_q_out, d_k_out, d_q_in, d_k_in,
        batch, seq_len, num_heads, num_kv_heads,
        head_dim, freq_base, rope_dim
    );
    
    auto q_out = copy_from_device(d_q_out, q_size);
    
    // Compute magnitude of output
    float mag_out = std::sqrt(q_out[0]*q_out[0] + q_out[1]*q_out[1]);
    float mag_in = std::sqrt(q_in[0]*q_in[0] + q_in[1]*q_in[1]);
    
    // Rotation should preserve magnitude
    EXPECT_NEAR(mag_out, mag_in, 0.1f);
    
    free_device(d_q_in);
    free_device(d_k_in);
    free_device(d_q_out);
    free_device(d_k_out);
}
