/**
 * Inference Pipeline Integration Tests - GT-057
 * 
 * Tests the complete inference pipeline:
 * - Weight loading
 * - Transformer forward pass
 * - Sampling
 * - FFI interface
 * 
 * Spec: M0-W-1420, M0-W-1430
 */

#include <gtest/gtest.h>
#include "../src/ffi_inference.cpp"
#include "../src/model/qwen_weight_loader.h"
#include "../src/transformer/qwen_transformer.h"
#include <cuda_runtime.h>
#include <fstream>

// External sampling function
extern "C" int cuda_sample_token(
    float* logits,
    uint32_t vocab_size,
    float temperature,
    uint32_t top_k,
    float top_p,
    uint64_t seed
);

class InferencePipelineTest : public ::testing::Test {
protected:
    const char* model_path = "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    worker::model::QwenModel* model = nullptr;
    InferenceContext* inf_ctx = nullptr;
    
    void SetUp() override {
        // Check if model file exists
        std::ifstream test_file(model_path);
        if (!test_file) {
            GTEST_SKIP() << "Model file not found: " << model_path;
        }
        test_file.close();
        
        // Initialize CUDA
        cudaSetDevice(0);
        
        // Load model
        worker::model::QwenConfig config;
        config.vocab_size = 151936;
        config.hidden_dim = 896;
        config.num_layers = 24;
        config.num_heads = 14;
        config.num_kv_heads = 2;
        config.context_length = 32768;
        
        model = worker::model::QwenWeightLoader::load(model_path, config);
        ASSERT_NE(model, nullptr);
    }
    
    void TearDown() override {
        if (inf_ctx) {
            cuda_inference_free(inf_ctx);
        }
        if (model) {
            delete model;
        }
    }
};

// ============================================================================
// Test 1: Inference Context Initialization
// ============================================================================

TEST_F(InferencePipelineTest, InferenceContextInit) {
    int error = 0;
    inf_ctx = cuda_inference_init(
        model,
        151936,  // vocab_size
        896,     // hidden_dim
        24,      // num_layers
        14,      // num_heads
        2,       // num_kv_heads
        64,      // head_dim
        4864,    // ffn_dim
        32768,   // context_length
        &error
    );
    
    EXPECT_EQ(error, 0);
    EXPECT_NE(inf_ctx, nullptr);
}

// ============================================================================
// Test 2: Single Token Generation
// ============================================================================

TEST_F(InferencePipelineTest, GenerateSingleToken) {
    int error = 0;
    inf_ctx = cuda_inference_init(
        model, 151936, 896, 24, 14, 2, 64, 4864, 32768, &error
    );
    ASSERT_EQ(error, 0);
    
    // Generate token with greedy sampling (temperature = 0)
    uint32_t next_token = cuda_inference_generate_token(
        inf_ctx,
        151643,  // BOS token
        0.0f,    // temperature (greedy)
        0,       // top_k
        0.0f,    // top_p
        42,      // seed
        &error
    );
    
    EXPECT_EQ(error, 0);
    EXPECT_LT(next_token, 151936u);  // Valid token ID
}

// ============================================================================
// Test 3: Multiple Token Generation
// ============================================================================

TEST_F(InferencePipelineTest, GenerateMultipleTokens) {
    int error = 0;
    inf_ctx = cuda_inference_init(
        model, 151936, 896, 24, 14, 2, 64, 4864, 32768, &error
    );
    ASSERT_EQ(error, 0);
    
    uint32_t current_token = 151643;  // BOS
    std::vector<uint32_t> generated_tokens;
    
    for (int i = 0; i < 10; i++) {
        uint32_t next_token = cuda_inference_generate_token(
            inf_ctx, current_token, 0.7f, 0, 0.0f, 42, &error
        );
        
        EXPECT_EQ(error, 0);
        EXPECT_LT(next_token, 151936u);
        
        generated_tokens.push_back(next_token);
        current_token = next_token;
    }
    
    EXPECT_EQ(generated_tokens.size(), 10u);
}

// ============================================================================
// Test 4: KV Cache Reset
// ============================================================================

TEST_F(InferencePipelineTest, KVCacheReset) {
    int error = 0;
    inf_ctx = cuda_inference_init(
        model, 151936, 896, 24, 14, 2, 64, 4864, 32768, &error
    );
    ASSERT_EQ(error, 0);
    
    // Generate some tokens
    uint32_t token1 = cuda_inference_generate_token(
        inf_ctx, 151643, 0.7f, 0, 0.0f, 42, &error
    );
    EXPECT_EQ(error, 0);
    
    // Reset cache
    cuda_inference_reset(inf_ctx);
    
    // Generate again - should work
    uint32_t token2 = cuda_inference_generate_token(
        inf_ctx, 151643, 0.7f, 0, 0.0f, 42, &error
    );
    EXPECT_EQ(error, 0);
    
    // With same seed, should get same token
    EXPECT_EQ(token1, token2);
}

// ============================================================================
// Test 5: Temperature Sampling
// ============================================================================

TEST_F(InferencePipelineTest, TemperatureSampling) {
    int error = 0;
    inf_ctx = cuda_inference_init(
        model, 151936, 896, 24, 14, 2, 64, 4864, 32768, &error
    );
    ASSERT_EQ(error, 0);
    
    // Test different temperatures
    std::vector<float> temperatures = {0.0f, 0.5f, 0.7f, 1.0f};
    
    for (float temp : temperatures) {
        cuda_inference_reset(inf_ctx);
        
        uint32_t token = cuda_inference_generate_token(
            inf_ctx, 151643, temp, 0, 0.0f, 42, &error
        );
        
        EXPECT_EQ(error, 0);
        EXPECT_LT(token, 151936u);
    }
}

// ============================================================================
// Test 6: Reproducibility with Seed
// ============================================================================

TEST_F(InferencePipelineTest, ReproducibilityWithSeed) {
    int error = 0;
    inf_ctx = cuda_inference_init(
        model, 151936, 896, 24, 14, 2, 64, 4864, 32768, &error
    );
    ASSERT_EQ(error, 0);
    
    // Generate sequence with seed 42
    cuda_inference_reset(inf_ctx);
    std::vector<uint32_t> tokens1;
    uint32_t current = 151643;
    for (int i = 0; i < 5; i++) {
        current = cuda_inference_generate_token(
            inf_ctx, current, 0.7f, 0, 0.0f, 42, &error
        );
        tokens1.push_back(current);
    }
    
    // Generate again with same seed
    cuda_inference_reset(inf_ctx);
    std::vector<uint32_t> tokens2;
    current = 151643;
    for (int i = 0; i < 5; i++) {
        current = cuda_inference_generate_token(
            inf_ctx, current, 0.7f, 0, 0.0f, 42, &error
        );
        tokens2.push_back(current);
    }
    
    // Should be identical
    EXPECT_EQ(tokens1, tokens2);
}

// ============================================================================
// Test 7: Top-k Sampling
// ============================================================================

TEST_F(InferencePipelineTest, TopKSampling) {
    int error = 0;
    inf_ctx = cuda_inference_init(
        model, 151936, 896, 24, 14, 2, 64, 4864, 32768, &error
    );
    ASSERT_EQ(error, 0);
    
    // Test with top_k = 50
    uint32_t token = cuda_inference_generate_token(
        inf_ctx, 151643, 0.7f, 50, 0.0f, 42, &error
    );
    
    EXPECT_EQ(error, 0);
    EXPECT_LT(token, 151936u);
}

// ============================================================================
// Test 8: Top-p Sampling
// ============================================================================

TEST_F(InferencePipelineTest, TopPSampling) {
    int error = 0;
    inf_ctx = cuda_inference_init(
        model, 151936, 896, 24, 14, 2, 64, 4864, 32768, &error
    );
    ASSERT_EQ(error, 0);
    
    // Test with top_p = 0.9
    uint32_t token = cuda_inference_generate_token(
        inf_ctx, 151643, 0.7f, 0, 0.9f, 42, &error
    );
    
    EXPECT_EQ(error, 0);
    EXPECT_LT(token, 151936u);
}

// ============================================================================
// Test 9: Error Handling - Invalid Token
// ============================================================================

TEST_F(InferencePipelineTest, ErrorHandlingInvalidToken) {
    int error = 0;
    inf_ctx = cuda_inference_init(
        model, 151936, 896, 24, 14, 2, 64, 4864, 32768, &error
    );
    ASSERT_EQ(error, 0);
    
    // Try with invalid token (beyond vocab)
    uint32_t token = cuda_inference_generate_token(
        inf_ctx, 999999, 0.7f, 0, 0.0f, 42, &error
    );
    
    // Should still work (embedding lookup will handle it)
    // or return error - depends on implementation
    EXPECT_LT(token, 151936u);
}

// ============================================================================
// Test 10: Memory Cleanup
// ============================================================================

TEST_F(InferencePipelineTest, MemoryCleanup) {
    int error = 0;
    
    // Create and destroy multiple contexts
    for (int i = 0; i < 3; i++) {
        InferenceContext* ctx = cuda_inference_init(
            model, 151936, 896, 24, 14, 2, 64, 4864, 32768, &error
        );
        EXPECT_EQ(error, 0);
        EXPECT_NE(ctx, nullptr);
        
        cuda_inference_free(ctx);
    }
    
    // Should not leak memory
    // (Can verify with cuda-memcheck or nvidia-smi)
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ---
// Crafted by GPT-Gamma 🤖
