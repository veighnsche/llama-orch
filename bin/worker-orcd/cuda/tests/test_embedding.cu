/**
 * Embedding Lookup Kernel Unit Tests
 * 
 * Tests embedding lookup kernel correctness, edge cases, and error handling.
 * 
 * Spec: M0-W-1430, CUDA-5030
 * Story: FT-015
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../kernels/embedding.cuh"
#include <vector>
#include <cmath>

using namespace worker::kernels;

// ============================================================================
// Test Fixture
// ============================================================================

class EmbeddingKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA device
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
        cudaSetDevice(0);
        
        // Allocate device memory
        cudaMalloc(&d_token_ids, batch_size * sizeof(int));
        cudaMalloc(&d_weight_matrix_fp16, vocab_size * hidden_dim * sizeof(half));
        cudaMalloc(&d_embeddings_fp16, batch_size * hidden_dim * sizeof(half));
        cudaMalloc(&d_weight_matrix_fp32, vocab_size * hidden_dim * sizeof(float));
        cudaMalloc(&d_embeddings_fp32, batch_size * hidden_dim * sizeof(float));
    }
    
    void TearDown() override {
        cudaFree(d_token_ids);
        cudaFree(d_weight_matrix_fp16);
        cudaFree(d_embeddings_fp16);
        cudaFree(d_weight_matrix_fp32);
        cudaFree(d_embeddings_fp32);
    }
    
    // Default dimensions
    int batch_size = 4;
    int hidden_dim = 128;
    int vocab_size = 1000;
    
    // Device pointers
    int* d_token_ids = nullptr;
    half* d_weight_matrix_fp16 = nullptr;
    half* d_embeddings_fp16 = nullptr;
    float* d_weight_matrix_fp32 = nullptr;
    float* d_embeddings_fp32 = nullptr;
};

// ============================================================================
// Basic Functionality Tests (FP16)
// ============================================================================

/**
 * Test: Basic embedding lookup with known values (FP16)
 * 
 * Spec: M0-W-1430 (Embedding Lookup Kernel)
 * Critical: Core kernel functionality must be correct
 */
TEST_F(EmbeddingKernelTest, BasicLookupFP16) {
    // Host data
    std::vector<int> h_token_ids = {0, 1, 2, 3};
    std::vector<half> h_weight_matrix(vocab_size * hidden_dim);
    
    // Initialize weights: token_id i → embedding value i*0.1
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < hidden_dim; ++j) {
            h_weight_matrix[i * hidden_dim + j] = __float2half(i * 0.1f);
        }
    }
    
    // Copy to device
    cudaMemcpy(d_token_ids, h_token_ids.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix_fp16, h_weight_matrix.data(), vocab_size * hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    // Launch kernel
    launch_embedding_lookup_fp16(d_token_ids, d_weight_matrix_fp16, d_embeddings_fp16, 
                                  batch_size, hidden_dim, vocab_size);
    cudaDeviceSynchronize();
    
    // Copy result back
    std::vector<half> h_embeddings(batch_size * hidden_dim);
    cudaMemcpy(h_embeddings.data(), d_embeddings_fp16, batch_size * hidden_dim * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Verify each token's embedding
    for (int i = 0; i < batch_size; ++i) {
        float expected = i * 0.1f;
        float actual = __half2float(h_embeddings[i * hidden_dim]);
        EXPECT_NEAR(actual, expected, 0.01f) << "Token " << i << " embedding mismatch";
        
        // Verify all dimensions have same value
        for (int j = 0; j < hidden_dim; ++j) {
            float dim_value = __half2float(h_embeddings[i * hidden_dim + j]);
            EXPECT_NEAR(dim_value, expected, 0.01f) << "Token " << i << ", dim " << j;
        }
    }
}

/**
 * Test: Basic embedding lookup with known values (FP32)
 * 
 * Spec: M0-W-1430 (Embedding Lookup Kernel)
 * Critical: FP32 variant must work correctly
 */
TEST_F(EmbeddingKernelTest, BasicLookupFP32) {
    std::vector<int> h_token_ids = {5, 10, 15, 20};
    std::vector<float> h_weight_matrix(vocab_size * hidden_dim);
    
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < hidden_dim; ++j) {
            h_weight_matrix[i * hidden_dim + j] = i * 0.1f;
        }
    }
    
    cudaMemcpy(d_token_ids, h_token_ids.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix_fp32, h_weight_matrix.data(), vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    
    launch_embedding_lookup_fp32(d_token_ids, d_weight_matrix_fp32, d_embeddings_fp32,
                                  batch_size, hidden_dim, vocab_size);
    cudaDeviceSynchronize();
    
    std::vector<float> h_embeddings(batch_size * hidden_dim);
    cudaMemcpy(h_embeddings.data(), d_embeddings_fp32, batch_size * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < batch_size; ++i) {
        float expected = h_token_ids[i] * 0.1f;
        float actual = h_embeddings[i * hidden_dim];
        EXPECT_NEAR(actual, expected, 0.001f) << "Token " << i;
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/**
 * Test: Out-of-bounds token IDs return zero
 * 
 * Spec: M0-W-1430 (Embedding Lookup Kernel)
 * Critical: Prevents crashes and undefined behavior
 */
TEST_F(EmbeddingKernelTest, OutOfBoundsTokenIDReturnsZero) {
    std::vector<int> h_token_ids = {0, vocab_size + 10, -1, 1};  // Two invalid IDs
    std::vector<half> h_weight_matrix(vocab_size * hidden_dim, __float2half(1.0f));
    
    cudaMemcpy(d_token_ids, h_token_ids.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix_fp16, h_weight_matrix.data(), vocab_size * hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_embedding_lookup_fp16(d_token_ids, d_weight_matrix_fp16, d_embeddings_fp16,
                                  batch_size, hidden_dim, vocab_size);
    cudaDeviceSynchronize();
    
    std::vector<half> h_embeddings(batch_size * hidden_dim);
    cudaMemcpy(h_embeddings.data(), d_embeddings_fp16, batch_size * hidden_dim * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Token 0 should be valid (1.0)
    EXPECT_NEAR(__half2float(h_embeddings[0]), 1.0f, 0.01f) << "Token 0 should be valid";
    
    // Token 1 (out of bounds) should be zero
    EXPECT_NEAR(__half2float(h_embeddings[hidden_dim]), 0.0f, 0.01f) << "OOB token should be zero";
    
    // Token 2 (negative) should be zero
    EXPECT_NEAR(__half2float(h_embeddings[2 * hidden_dim]), 0.0f, 0.01f) << "Negative token should be zero";
    
    // Token 3 should be valid (1.0)
    EXPECT_NEAR(__half2float(h_embeddings[3 * hidden_dim]), 1.0f, 0.01f) << "Token 3 should be valid";
}

/**
 * Test: Negative token IDs return zero
 * 
 * Critical: Defensive programming
 */
TEST_F(EmbeddingKernelTest, NegativeTokenIDReturnsZero) {
    std::vector<int> h_token_ids = {-1, -100, -5, -999};
    std::vector<half> h_weight_matrix(vocab_size * hidden_dim, __float2half(1.0f));
    
    cudaMemcpy(d_token_ids, h_token_ids.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix_fp16, h_weight_matrix.data(), vocab_size * hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_embedding_lookup_fp16(d_token_ids, d_weight_matrix_fp16, d_embeddings_fp16,
                                  batch_size, hidden_dim, vocab_size);
    cudaDeviceSynchronize();
    
    std::vector<half> h_embeddings(batch_size * hidden_dim);
    cudaMemcpy(h_embeddings.data(), d_embeddings_fp16, batch_size * hidden_dim * sizeof(half), cudaMemcpyDeviceToHost);
    
    // All embeddings should be zero
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < hidden_dim; ++j) {
            float value = __half2float(h_embeddings[i * hidden_dim + j]);
            EXPECT_NEAR(value, 0.0f, 0.01f) << "Token " << i << ", dim " << j << " should be zero";
        }
    }
}

/**
 * Test: Large hidden dimensions (>256)
 * 
 * Spec: M0-W-1430 (Embedding Lookup Kernel)
 * Critical: Real models have large hidden dims (e.g., 1024, 2048, 4096)
 */
TEST_F(EmbeddingKernelTest, LargeHiddenDim) {
    // Test with hidden_dim = 1024 (requires multiple blocks per token)
    int large_hidden_dim = 1024;
    
    // Reallocate device memory
    cudaFree(d_weight_matrix_fp16);
    cudaFree(d_embeddings_fp16);
    cudaMalloc(&d_weight_matrix_fp16, vocab_size * large_hidden_dim * sizeof(half));
    cudaMalloc(&d_embeddings_fp16, batch_size * large_hidden_dim * sizeof(half));
    
    std::vector<int> h_token_ids = {5, 10, 15, 20};
    std::vector<half> h_weight_matrix(vocab_size * large_hidden_dim);
    
    // Initialize weights
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < large_hidden_dim; ++j) {
            h_weight_matrix[i * large_hidden_dim + j] = __float2half(i * 0.1f);
        }
    }
    
    cudaMemcpy(d_token_ids, h_token_ids.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix_fp16, h_weight_matrix.data(), vocab_size * large_hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_embedding_lookup_fp16(d_token_ids, d_weight_matrix_fp16, d_embeddings_fp16,
                                  batch_size, large_hidden_dim, vocab_size);
    cudaDeviceSynchronize();
    
    std::vector<half> h_embeddings(batch_size * large_hidden_dim);
    cudaMemcpy(h_embeddings.data(), d_embeddings_fp16, batch_size * large_hidden_dim * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Verify all dimensions are correct
    for (int i = 0; i < batch_size; ++i) {
        float expected = h_token_ids[i] * 0.1f;
        for (int j = 0; j < large_hidden_dim; ++j) {
            float actual = __half2float(h_embeddings[i * large_hidden_dim + j]);
            EXPECT_NEAR(actual, expected, 0.01f) << "Token " << i << ", dim " << j;
        }
    }
}

/**
 * Test: Single token (batch_size = 1)
 * 
 * Critical: Edge case for inference
 */
TEST_F(EmbeddingKernelTest, SingleToken) {
    int single_batch = 1;
    
    std::vector<int> h_token_ids = {42};
    std::vector<half> h_weight_matrix(vocab_size * hidden_dim);
    
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < hidden_dim; ++j) {
            h_weight_matrix[i * hidden_dim + j] = __float2half(i * 0.1f);
        }
    }
    
    cudaMemcpy(d_token_ids, h_token_ids.data(), single_batch * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix_fp16, h_weight_matrix.data(), vocab_size * hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_embedding_lookup_fp16(d_token_ids, d_weight_matrix_fp16, d_embeddings_fp16,
                                  single_batch, hidden_dim, vocab_size);
    cudaDeviceSynchronize();
    
    std::vector<half> h_embeddings(single_batch * hidden_dim);
    cudaMemcpy(h_embeddings.data(), d_embeddings_fp16, single_batch * hidden_dim * sizeof(half), cudaMemcpyDeviceToHost);
    
    float expected = 42 * 0.1f;
    float actual = __half2float(h_embeddings[0]);
    EXPECT_NEAR(actual, expected, 0.01f);
}

/**
 * Test: Empty batch (batch_size = 0)
 * 
 * Critical: Defensive programming - should not crash
 */
TEST_F(EmbeddingKernelTest, EmptyBatch) {
    int empty_batch = 0;
    
    // Should not crash with batch_size = 0
    launch_embedding_lookup_fp16(d_token_ids, d_weight_matrix_fp16, d_embeddings_fp16,
                                  empty_batch, hidden_dim, vocab_size);
    cudaDeviceSynchronize();
    
    // No assertions needed - just verify no crash
    SUCCEED();
}

// ============================================================================
// Real-World Dimensions Tests
// ============================================================================

/**
 * Test: Qwen2.5-0.5B dimensions
 * 
 * Critical: Real vocabulary size and hidden dimension
 */
TEST_F(EmbeddingKernelTest, QwenDimensions) {
    int qwen_vocab_size = 151936;  // Qwen2.5-0.5B vocabulary
    int qwen_hidden_dim = 896;     // Qwen2.5-0.5B hidden dimension
    int qwen_batch = 8;
    
    // Allocate for Qwen dimensions
    int* d_qwen_token_ids;
    half* d_qwen_weights;
    half* d_qwen_embeddings;
    
    cudaMalloc(&d_qwen_token_ids, qwen_batch * sizeof(int));
    cudaMalloc(&d_qwen_weights, qwen_vocab_size * qwen_hidden_dim * sizeof(half));
    cudaMalloc(&d_qwen_embeddings, qwen_batch * qwen_hidden_dim * sizeof(half));
    
    std::vector<int> h_token_ids = {0, 100, 1000, 10000, 50000, 100000, 150000, 151935};
    std::vector<half> h_weights(qwen_vocab_size * qwen_hidden_dim);
    
    // Initialize weights
    for (int i = 0; i < qwen_vocab_size; ++i) {
        for (int j = 0; j < qwen_hidden_dim; ++j) {
            h_weights[i * qwen_hidden_dim + j] = __float2half((i % 1000) * 0.001f);
        }
    }
    
    cudaMemcpy(d_qwen_token_ids, h_token_ids.data(), qwen_batch * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qwen_weights, h_weights.data(), qwen_vocab_size * qwen_hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_embedding_lookup_fp16(d_qwen_token_ids, d_qwen_weights, d_qwen_embeddings,
                                  qwen_batch, qwen_hidden_dim, qwen_vocab_size);
    cudaDeviceSynchronize();
    
    std::vector<half> h_embeddings(qwen_batch * qwen_hidden_dim);
    cudaMemcpy(h_embeddings.data(), d_qwen_embeddings, qwen_batch * qwen_hidden_dim * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Verify embeddings
    for (int i = 0; i < qwen_batch; ++i) {
        float expected = (h_token_ids[i] % 1000) * 0.001f;
        float actual = __half2float(h_embeddings[i * qwen_hidden_dim]);
        EXPECT_NEAR(actual, expected, 0.01f) << "Token " << i;
    }
    
    // Cleanup
    cudaFree(d_qwen_token_ids);
    cudaFree(d_qwen_weights);
    cudaFree(d_qwen_embeddings);
}

/**
 * Test: GPT-OSS-20B dimensions
 * 
 * Critical: Large model dimensions
 */
TEST_F(EmbeddingKernelTest, GPTDimensions) {
    int gpt_vocab_size = 50257;   // GPT-2/GPT-OSS vocabulary
    int gpt_hidden_dim = 2048;    // GPT-OSS-20B hidden dimension
    int gpt_batch = 4;
    
    int* d_gpt_token_ids;
    half* d_gpt_weights;
    half* d_gpt_embeddings;
    
    cudaMalloc(&d_gpt_token_ids, gpt_batch * sizeof(int));
    cudaMalloc(&d_gpt_weights, gpt_vocab_size * gpt_hidden_dim * sizeof(half));
    cudaMalloc(&d_gpt_embeddings, gpt_batch * gpt_hidden_dim * sizeof(half));
    
    std::vector<int> h_token_ids = {0, 1000, 25000, 50256};
    std::vector<half> h_weights(gpt_vocab_size * gpt_hidden_dim);
    
    for (int i = 0; i < gpt_vocab_size; ++i) {
        for (int j = 0; j < gpt_hidden_dim; ++j) {
            h_weights[i * gpt_hidden_dim + j] = __float2half(i * 0.0001f);
        }
    }
    
    cudaMemcpy(d_gpt_token_ids, h_token_ids.data(), gpt_batch * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpt_weights, h_weights.data(), gpt_vocab_size * gpt_hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_embedding_lookup_fp16(d_gpt_token_ids, d_gpt_weights, d_gpt_embeddings,
                                  gpt_batch, gpt_hidden_dim, gpt_vocab_size);
    cudaDeviceSynchronize();
    
    std::vector<half> h_embeddings(gpt_batch * gpt_hidden_dim);
    cudaMemcpy(h_embeddings.data(), d_gpt_embeddings, gpt_batch * gpt_hidden_dim * sizeof(half), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < gpt_batch; ++i) {
        float expected = h_token_ids[i] * 0.0001f;
        float actual = __half2float(h_embeddings[i * gpt_hidden_dim]);
        EXPECT_NEAR(actual, expected, 0.001f) << "Token " << i;
    }
    
    cudaFree(d_gpt_token_ids);
    cudaFree(d_gpt_weights);
    cudaFree(d_gpt_embeddings);
}

// ============================================================================
// Property Tests
// ============================================================================

/**
 * Test: Embedding lookup is deterministic
 * 
 * Property: Same inputs → same outputs (every time)
 */
TEST_F(EmbeddingKernelTest, DeterministicLookup) {
    std::vector<int> h_token_ids = {7, 13, 42, 99};
    std::vector<half> h_weight_matrix(vocab_size * hidden_dim);
    
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < hidden_dim; ++j) {
            h_weight_matrix[i * hidden_dim + j] = __float2half(i * 0.1f + j * 0.01f);
        }
    }
    
    cudaMemcpy(d_token_ids, h_token_ids.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_matrix_fp16, h_weight_matrix.data(), vocab_size * hidden_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    // Run kernel 5 times
    std::vector<std::vector<half>> results(5);
    for (int run = 0; run < 5; ++run) {
        launch_embedding_lookup_fp16(d_token_ids, d_weight_matrix_fp16, d_embeddings_fp16,
                                      batch_size, hidden_dim, vocab_size);
        cudaDeviceSynchronize();
        
        results[run].resize(batch_size * hidden_dim);
        cudaMemcpy(results[run].data(), d_embeddings_fp16, batch_size * hidden_dim * sizeof(half), cudaMemcpyDeviceToHost);
    }
    
    // Verify all runs produce identical results
    for (int run = 1; run < 5; ++run) {
        for (int i = 0; i < batch_size * hidden_dim; ++i) {
            float val0 = __half2float(results[0][i]);
            float valN = __half2float(results[run][i]);
            EXPECT_FLOAT_EQ(val0, valN) << "Run " << run << ", index " << i << " differs";
        }
    }
}

// ---
// Built by Foundation-Alpha 🏗️
