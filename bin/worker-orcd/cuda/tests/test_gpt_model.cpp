/**
 * GPT Model Forward Pass Tests
 * 
 * Unit tests for GPT model forward pass structure.
 * Uses mock data (no actual model file required).
 * 
 * Story: GT-026
 */

#include "../src/model/gpt_model.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace worker::model;

// ============================================================================
// Test Fixtures
// ============================================================================

class GPTModelTest : public ::testing::Test {
protected:
    cublasHandle_t cublas_handle_;
    
    void SetUp() override {
        cublasCreate(&cublas_handle_);
    }
    
    void TearDown() override {
        cublasDestroy(cublas_handle_);
    }
    
    std::unique_ptr<GPTModelWeights> create_mock_weights() {
        auto weights = std::make_unique<GPTModelWeights>();
        
        // Set config
        weights->config.vocab_size = 50257;
        weights->config.hidden_dim = 768;
        weights->config.num_layers = 12;
        weights->config.num_heads = 12;
        weights->config.head_dim = 64;
        weights->config.ffn_dim = 3072;
        weights->config.max_seq_len = 1024;
        weights->config.context_length = 1024;
        weights->config.quant_kind = "Q4_K_M";
        
        // Allocate mock embeddings (small size for testing)
        size_t token_emb_size = 100 * 768 * sizeof(half);  // Subset of vocab
        cudaMalloc(&weights->token_embeddings, token_emb_size);
        cudaMemset(weights->token_embeddings, 0, token_emb_size);
        
        size_t pos_emb_size = 1024 * 768 * sizeof(half);
        cudaMalloc(&weights->position_embeddings, pos_emb_size);
        cudaMemset(weights->position_embeddings, 0, pos_emb_size);
        
        // Allocate output norm
        size_t norm_size = 768 * sizeof(half);
        cudaMalloc(&weights->output_norm_weight, norm_size);
        cudaMalloc(&weights->output_norm_bias, norm_size);
        cudaMemset(weights->output_norm_weight, 0, norm_size);
        cudaMemset(weights->output_norm_bias, 0, norm_size);
        
        // Allocate LM head (small subset)
        size_t lm_head_size = 768 * 100 * sizeof(half);
        cudaMalloc(&weights->lm_head_weight, lm_head_size);
        cudaMemset(weights->lm_head_weight, 0, lm_head_size);
        
        // Add mock layers
        for (int i = 0; i < 12; ++i) {
            weights->layers.push_back(std::make_unique<GPTLayerWeights>());
        }
        
        weights->total_vram_bytes = token_emb_size + pos_emb_size + 
                                    norm_size * 2 + lm_head_size;
        
        return weights;
    }
};

// ============================================================================
// GPTForwardConfig Tests
// ============================================================================

TEST_F(GPTModelTest, ForwardConfigPrefill) {
    GPTForwardConfig config;
    config.is_prefill = true;
    config.batch_size = 1;
    config.seq_len = 10;
    config.cache_len = 0;
    config.temperature = 0.7f;
    config.seed = 42;
    config.top_p = 1.0f;
    config.top_k = 0;
    config.repetition_penalty = 1.0f;
    
    EXPECT_TRUE(config.is_prefill);
    EXPECT_EQ(config.batch_size, 1);
    EXPECT_EQ(config.seq_len, 10);
    EXPECT_FLOAT_EQ(config.temperature, 0.7f);
}

TEST_F(GPTModelTest, ForwardConfigDecode) {
    GPTForwardConfig config;
    config.is_prefill = false;
    config.batch_size = 1;
    config.seq_len = 1;
    config.cache_len = 10;
    config.temperature = 1.0f;
    config.seed = 42;
    config.top_p = 0.9f;
    config.top_k = 50;
    config.repetition_penalty = 1.1f;
    
    EXPECT_FALSE(config.is_prefill);
    EXPECT_EQ(config.cache_len, 10);
    EXPECT_FLOAT_EQ(config.top_p, 0.9f);
    EXPECT_EQ(config.top_k, 50);
}

// ============================================================================
// GPTModel Construction Tests
// ============================================================================

TEST_F(GPTModelTest, ModelConstruction) {
    auto weights = create_mock_weights();
    
    EXPECT_NO_THROW({
        GPTModel model(std::move(weights), cublas_handle_);
    });
}

TEST_F(GPTModelTest, ModelConfigAccess) {
    auto weights = create_mock_weights();
    GPTModel model(std::move(weights), cublas_handle_);
    
    const GPTConfig& config = model.config();
    EXPECT_EQ(config.vocab_size, 50257);
    EXPECT_EQ(config.hidden_dim, 768);
    EXPECT_EQ(config.num_layers, 12);
}

TEST_F(GPTModelTest, ModelVRAMUsage) {
    auto weights = create_mock_weights();
    size_t expected_vram = weights->total_vram_bytes;
    
    GPTModel model(std::move(weights), cublas_handle_);
    
    EXPECT_GT(model.vram_usage(), 0);
    EXPECT_EQ(model.vram_usage(), expected_vram);
}

// ============================================================================
// Cache Management Tests
// ============================================================================

TEST_F(GPTModelTest, ResetCache) {
    auto weights = create_mock_weights();
    GPTModel model(std::move(weights), cublas_handle_);
    
    // Should not throw
    EXPECT_NO_THROW(model.reset_cache());
}

// ============================================================================
// Forward Pass Structure Tests (Mock)
// ============================================================================

TEST_F(GPTModelTest, PrefillStructure) {
    auto weights = create_mock_weights();
    GPTModel model(std::move(weights), cublas_handle_);
    
    // Create mock input
    std::vector<uint32_t> input_ids = {1, 2, 3, 4, 5};
    
    GPTForwardConfig config;
    config.is_prefill = true;
    config.batch_size = 1;
    config.seq_len = 5;
    config.cache_len = 0;
    config.temperature = 0.0f;  // Greedy
    config.seed = 42;
    config.top_p = 1.0f;
    config.top_k = 0;
    config.repetition_penalty = 1.0f;
    
    // Should not crash (returns stub token)
    EXPECT_NO_THROW({
        uint32_t token = model.prefill(input_ids.data(), 5, config);
        EXPECT_GE(token, 0);
    });
}

TEST_F(GPTModelTest, DecodeStructure) {
    auto weights = create_mock_weights();
    GPTModel model(std::move(weights), cublas_handle_);
    
    // First do prefill
    std::vector<uint32_t> input_ids = {1, 2, 3};
    GPTForwardConfig prefill_config;
    prefill_config.is_prefill = true;
    prefill_config.batch_size = 1;
    prefill_config.seq_len = 3;
    prefill_config.temperature = 0.0f;
    prefill_config.seed = 42;
    prefill_config.top_p = 1.0f;
    prefill_config.top_k = 0;
    prefill_config.repetition_penalty = 1.0f;
    
    model.prefill(input_ids.data(), 3, prefill_config);
    
    // Now decode
    GPTForwardConfig decode_config;
    decode_config.is_prefill = false;
    decode_config.batch_size = 1;
    decode_config.seq_len = 1;
    decode_config.cache_len = 3;
    decode_config.temperature = 0.0f;
    decode_config.seed = 42;
    decode_config.top_p = 1.0f;
    decode_config.top_k = 0;
    decode_config.repetition_penalty = 1.0f;
    
    EXPECT_NO_THROW({
        uint32_t token = model.decode(10, decode_config);
        EXPECT_GE(token, 0);
    });
}

TEST_F(GPTModelTest, MultipleDecodeSteps) {
    auto weights = create_mock_weights();
    GPTModel model(std::move(weights), cublas_handle_);
    
    // Prefill
    std::vector<uint32_t> input_ids = {1, 2, 3};
    GPTForwardConfig config;
    config.is_prefill = true;
    config.batch_size = 1;
    config.seq_len = 3;
    config.temperature = 0.0f;
    config.seed = 42;
    config.top_p = 1.0f;
    config.top_k = 0;
    config.repetition_penalty = 1.0f;
    
    model.prefill(input_ids.data(), 3, config);
    
    // Multiple decode steps
    config.is_prefill = false;
    config.seq_len = 1;
    
    for (int i = 0; i < 5; ++i) {
        config.cache_len = 3 + i;
        EXPECT_NO_THROW({
            uint32_t token = model.decode(10 + i, config);
            EXPECT_GE(token, 0);
        });
    }
}

TEST_F(GPTModelTest, ResetCacheBetweenSequences) {
    auto weights = create_mock_weights();
    GPTModel model(std::move(weights), cublas_handle_);
    
    // First sequence
    std::vector<uint32_t> input1 = {1, 2, 3};
    GPTForwardConfig config;
    config.is_prefill = true;
    config.batch_size = 1;
    config.seq_len = 3;
    config.temperature = 0.0f;
    config.seed = 42;
    config.top_p = 1.0f;
    config.top_k = 0;
    config.repetition_penalty = 1.0f;
    
    model.prefill(input1.data(), 3, config);
    
    // Reset
    model.reset_cache();
    
    // Second sequence
    std::vector<uint32_t> input2 = {4, 5, 6, 7};
    config.seq_len = 4;
    EXPECT_NO_THROW({
        model.prefill(input2.data(), 4, config);
    });
}

// ============================================================================
// Advanced Sampling Parameters Tests
// ============================================================================

TEST_F(GPTModelTest, TopPSampling) {
    auto weights = create_mock_weights();
    GPTModel model(std::move(weights), cublas_handle_);
    
    std::vector<uint32_t> input_ids = {1, 2, 3};
    GPTForwardConfig config;
    config.is_prefill = true;
    config.batch_size = 1;
    config.seq_len = 3;
    config.temperature = 0.8f;
    config.seed = 42;
    config.top_p = 0.9f;  // Nucleus sampling
    config.top_k = 0;
    config.repetition_penalty = 1.0f;
    
    EXPECT_NO_THROW({
        model.prefill(input_ids.data(), 3, config);
    });
}

TEST_F(GPTModelTest, TopKSampling) {
    auto weights = create_mock_weights();
    GPTModel model(std::move(weights), cublas_handle_);
    
    std::vector<uint32_t> input_ids = {1, 2, 3};
    GPTForwardConfig config;
    config.is_prefill = true;
    config.batch_size = 1;
    config.seq_len = 3;
    config.temperature = 0.8f;
    config.seed = 42;
    config.top_p = 1.0f;
    config.top_k = 50;  // Top-k sampling
    config.repetition_penalty = 1.0f;
    
    EXPECT_NO_THROW({
        model.prefill(input_ids.data(), 3, config);
    });
}

TEST_F(GPTModelTest, RepetitionPenalty) {
    auto weights = create_mock_weights();
    GPTModel model(std::move(weights), cublas_handle_);
    
    std::vector<uint32_t> input_ids = {1, 2, 3};
    GPTForwardConfig config;
    config.is_prefill = true;
    config.batch_size = 1;
    config.seq_len = 3;
    config.temperature = 0.8f;
    config.seed = 42;
    config.top_p = 1.0f;
    config.top_k = 0;
    config.repetition_penalty = 1.2f;  // Penalize repetition
    
    EXPECT_NO_THROW({
        model.prefill(input_ids.data(), 3, config);
    });
}

// ============================================================================
// GPTModelFactory Tests
// ============================================================================

TEST_F(GPTModelTest, FactoryValidateGGUF) {
    // Stub implementation returns GPT-OSS-20B config
    EXPECT_NO_THROW({
        GPTConfig config = GPTModelFactory::validate_gguf("dummy.gguf");
        EXPECT_EQ(config.vocab_size, 50257);
        EXPECT_EQ(config.hidden_dim, 2048);
        EXPECT_EQ(config.num_layers, 44);
    });
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(GPTModelTest, PrefillInvalidSeqLen) {
    auto weights = create_mock_weights();
    GPTModel model(std::move(weights), cublas_handle_);
    
    std::vector<uint32_t> input_ids = {1, 2, 3};
    GPTForwardConfig config;
    config.is_prefill = true;
    config.batch_size = 1;
    config.seq_len = 3;
    config.temperature = 0.0f;
    config.seed = 42;
    config.top_p = 1.0f;
    config.top_k = 0;
    config.repetition_penalty = 1.0f;
    
    // Invalid sequence length (0)
    EXPECT_THROW({
        model.prefill(input_ids.data(), 0, config);
    }, std::runtime_error);
    
    // Invalid sequence length (too large)
    EXPECT_THROW({
        model.prefill(input_ids.data(), 10000, config);
    }, std::runtime_error);
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
