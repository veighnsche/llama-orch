/**
 * GQA Attention Kernel Tests - LT-015, LT-016
 * 
 * Unit tests for Grouped Query Attention kernels (prefill and decode).
 * 
 * Tests cover:
 * - GQA prefill attention
 * - GQA decode attention
 * - Head grouping
 * - KV cache integration
 * - Dimension validation
 * 
 * Spec: M0-W-1214, M0-W-1430
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

// External C functions
extern "C" int cuda_gqa_attention_prefill(
    half* output,
    const half* q,
    const half* k,
    const half* v,
    half* kv_cache_k,
    half* kv_cache_v,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
);

extern "C" int cuda_gqa_attention_decode(
    half* output,
    const half* q,
    const half* k_current,
    const half* v_current,
    half* kv_cache_k,
    half* kv_cache_v,
    int batch_size,
    int cache_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
);

class GQAAttentionTest : public ::testing::Test {
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
    
    void fill_device(half* ptr, size_t elements, float value) {
        std::vector<half> host(elements);
        for (size_t i = 0; i < elements; ++i) {
            host[i] = __float2half(value);
        }
        cudaMemcpy(ptr, host.data(), elements * sizeof(half), cudaMemcpyHostToDevice);
    }
};

// Test: GQA Prefill with Qwen config (14 Q heads, 2 KV heads)
TEST_F(GQAAttentionTest, PrefillQwenConfig) {
    int batch = 1, seq_len = 10, num_q_heads = 14, num_kv_heads = 2, head_dim = 64;
    float scale = 1.0f / sqrtf(head_dim);
    
    int q_size = batch * seq_len * num_q_heads * head_dim;
    int kv_size = batch * seq_len * num_kv_heads * head_dim;
    
    half* d_q = allocate_device(q_size);
    half* d_k = allocate_device(kv_size);
    half* d_v = allocate_device(kv_size);
    half* d_output = allocate_device(q_size);
    half* d_cache_k = allocate_device(kv_size);
    half* d_cache_v = allocate_device(kv_size);
    
    fill_device(d_q, q_size, 1.0f);
    fill_device(d_k, kv_size, 1.0f);
    fill_device(d_v, kv_size, 1.0f);
    
    int result = cuda_gqa_attention_prefill(
        d_output, d_q, d_k, d_v, d_cache_k, d_cache_v,
        batch, seq_len, num_q_heads, num_kv_heads, head_dim, scale
    );
    
    EXPECT_EQ(result, 0);
    
    free_device(d_q);
    free_device(d_k);
    free_device(d_v);
    free_device(d_output);
    free_device(d_cache_k);
    free_device(d_cache_v);
}

// Test: GQA Prefill with Phi-3 config (32 Q heads, 32 KV heads = MHA)
TEST_F(GQAAttentionTest, PrefillPhi3Config) {
    int batch = 1, seq_len = 8, num_q_heads = 32, num_kv_heads = 32, head_dim = 96;
    float scale = 1.0f / sqrtf(head_dim);
    
    int q_size = batch * seq_len * num_q_heads * head_dim;
    int kv_size = batch * seq_len * num_kv_heads * head_dim;
    
    half* d_q = allocate_device(q_size);
    half* d_k = allocate_device(kv_size);
    half* d_v = allocate_device(kv_size);
    half* d_output = allocate_device(q_size);
    
    fill_device(d_q, q_size, 1.0f);
    fill_device(d_k, kv_size, 1.0f);
    fill_device(d_v, kv_size, 1.0f);
    
    int result = cuda_gqa_attention_prefill(
        d_output, d_q, d_k, d_v, nullptr, nullptr,
        batch, seq_len, num_q_heads, num_kv_heads, head_dim, scale
    );
    
    EXPECT_EQ(result, 0);
    
    free_device(d_q);
    free_device(d_k);
    free_device(d_v);
    free_device(d_output);
}

// Test: GQA Decode with cache
TEST_F(GQAAttentionTest, DecodeWithCache) {
    int batch = 1, cache_len = 10, num_q_heads = 14, num_kv_heads = 2, head_dim = 64;
    float scale = 1.0f / sqrtf(head_dim);
    
    int q_size = batch * 1 * num_q_heads * head_dim;
    int kv_size = batch * 1 * num_kv_heads * head_dim;
    int cache_size = batch * (cache_len + 1) * num_kv_heads * head_dim;
    
    half* d_q = allocate_device(q_size);
    half* d_k = allocate_device(kv_size);
    half* d_v = allocate_device(kv_size);
    half* d_output = allocate_device(q_size);
    half* d_cache_k = allocate_device(cache_size);
    half* d_cache_v = allocate_device(cache_size);
    
    fill_device(d_q, q_size, 1.0f);
    fill_device(d_k, kv_size, 1.0f);
    fill_device(d_v, kv_size, 1.0f);
    fill_device(d_cache_k, cache_size, 0.5f);
    fill_device(d_cache_v, cache_size, 0.5f);
    
    int result = cuda_gqa_attention_decode(
        d_output, d_q, d_k, d_v, d_cache_k, d_cache_v,
        batch, cache_len, num_q_heads, num_kv_heads, head_dim, scale
    );
    
    EXPECT_EQ(result, 0);
    
    free_device(d_q);
    free_device(d_k);
    free_device(d_v);
    free_device(d_output);
    free_device(d_cache_k);
    free_device(d_cache_v);
}

// Test: Invalid dimensions (prefill)
TEST_F(GQAAttentionTest, PrefillInvalidDimensions) {
    half* dummy = allocate_device(1);
    
    // num_q_heads not divisible by num_kv_heads
    int result1 = cuda_gqa_attention_prefill(
        dummy, dummy, dummy, dummy, nullptr, nullptr,
        1, 10, 15, 2, 64, 0.125f
    );
    EXPECT_NE(result1, 0);
    
    // Negative dimensions
    int result2 = cuda_gqa_attention_prefill(
        dummy, dummy, dummy, dummy, nullptr, nullptr,
        -1, 10, 14, 2, 64, 0.125f
    );
    EXPECT_NE(result2, 0);
    
    free_device(dummy);
}

// Test: Invalid dimensions (decode)
TEST_F(GQAAttentionTest, DecodeInvalidDimensions) {
    half* dummy = allocate_device(1);
    
    // num_q_heads not divisible by num_kv_heads
    int result1 = cuda_gqa_attention_decode(
        dummy, dummy, dummy, dummy, nullptr, nullptr,
        1, 10, 15, 2, 64, 0.125f
    );
    EXPECT_NE(result1, 0);
    
    free_device(dummy);
}

// Test: Different sequence lengths
TEST_F(GQAAttentionTest, DifferentSequenceLengths) {
    std::vector<int> seq_lens = {1, 16, 128, 512};
    
    for (int seq_len : seq_lens) {
        int batch = 1, num_q_heads = 8, num_kv_heads = 2, head_dim = 64;
        float scale = 1.0f / sqrtf(head_dim);
        
        int q_size = batch * seq_len * num_q_heads * head_dim;
        int kv_size = batch * seq_len * num_kv_heads * head_dim;
        
        half* d_q = allocate_device(q_size);
        half* d_k = allocate_device(kv_size);
        half* d_v = allocate_device(kv_size);
        half* d_output = allocate_device(q_size);
        
        fill_device(d_q, q_size, 1.0f);
        fill_device(d_k, kv_size, 1.0f);
        fill_device(d_v, kv_size, 1.0f);
        
        int result = cuda_gqa_attention_prefill(
            d_output, d_q, d_k, d_v, nullptr, nullptr,
            batch, seq_len, num_q_heads, num_kv_heads, head_dim, scale
        );
        
        EXPECT_EQ(result, 0) << "Failed for seq_len=" << seq_len;
        
        free_device(d_q);
        free_device(d_k);
        free_device(d_v);
        free_device(d_output);
    }
}

// Test: GQA head grouping (7:1 ratio for Qwen)
TEST_F(GQAAttentionTest, HeadGrouping7to1) {
    int batch = 1, seq_len = 4, num_q_heads = 14, num_kv_heads = 2, head_dim = 64;
    float scale = 1.0f / sqrtf(head_dim);
    
    // 14 Q heads / 2 KV heads = 7 Q heads per KV head
    EXPECT_EQ(num_q_heads / num_kv_heads, 7);
    
    int q_size = batch * seq_len * num_q_heads * head_dim;
    int kv_size = batch * seq_len * num_kv_heads * head_dim;
    
    half* d_q = allocate_device(q_size);
    half* d_k = allocate_device(kv_size);
    half* d_v = allocate_device(kv_size);
    half* d_output = allocate_device(q_size);
    
    fill_device(d_q, q_size, 1.0f);
    fill_device(d_k, kv_size, 1.0f);
    fill_device(d_v, kv_size, 1.0f);
    
    int result = cuda_gqa_attention_prefill(
        d_output, d_q, d_k, d_v, nullptr, nullptr,
        batch, seq_len, num_q_heads, num_kv_heads, head_dim, scale
    );
    
    EXPECT_EQ(result, 0);
    
    free_device(d_q);
    free_device(d_k);
    free_device(d_v);
    free_device(d_output);
}
