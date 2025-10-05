/**
 * GPT Weight Mapping Tests
 * 
 * Unit tests for GPT weight mapping, validation, and loading infrastructure.
 * Uses mock data (no actual model file required).
 * 
 * Story: GT-024, GT-025
 */

#include "../src/model/gpt_weights.h"
#include <gtest/gtest.h>
#include <vector>

using namespace worker::model;

// ============================================================================
// GPTConfig Tests
// ============================================================================

TEST(GPTConfig, ValidateGPTOSS20B) {
    GPTConfig config;
    config.vocab_size = 50257;
    config.hidden_dim = 2048;
    config.num_layers = 44;
    config.num_heads = 64;
    config.head_dim = 32;
    config.ffn_dim = 8192;
    config.max_seq_len = 2048;
    config.context_length = 8192;
    config.quant_kind = "Q4_K_M";
    
    EXPECT_TRUE(config.validate());
}

TEST(GPTConfig, ValidateGPT2Small) {
    GPTConfig config;
    config.vocab_size = 50257;
    config.hidden_dim = 768;
    config.num_layers = 12;
    config.num_heads = 12;
    config.head_dim = 64;
    config.ffn_dim = 3072;
    config.max_seq_len = 1024;
    config.context_length = 1024;
    config.quant_kind = "Q4_K_M";
    
    EXPECT_TRUE(config.validate());
}

TEST(GPTConfig, InvalidVocabSize) {
    GPTConfig config;
    config.vocab_size = 0;  // Invalid
    config.hidden_dim = 768;
    config.num_layers = 12;
    config.num_heads = 12;
    config.head_dim = 64;
    config.ffn_dim = 3072;
    config.max_seq_len = 1024;
    config.context_length = 1024;
    
    EXPECT_FALSE(config.validate());
}

TEST(GPTConfig, InvalidHeadDimension) {
    GPTConfig config;
    config.vocab_size = 50257;
    config.hidden_dim = 768;
    config.num_layers = 12;
    config.num_heads = 12;
    config.head_dim = 100;  // Should be 64 (768/12)
    config.ffn_dim = 3072;
    config.max_seq_len = 1024;
    config.context_length = 1024;
    
    EXPECT_FALSE(config.validate());
}

TEST(GPTConfig, HiddenDimNotDivisibleByHeads) {
    GPTConfig config;
    config.vocab_size = 50257;
    config.hidden_dim = 769;  // Not divisible by 12
    config.num_layers = 12;
    config.num_heads = 12;
    config.head_dim = 64;
    config.ffn_dim = 3072;
    config.max_seq_len = 1024;
    config.context_length = 1024;
    
    EXPECT_FALSE(config.validate());
}

// ============================================================================
// GPTWeightMapper Tests
// ============================================================================

TEST(GPTWeightMapper, GetTensorNameEmbeddings) {
    EXPECT_EQ(GPTWeightMapper::get_tensor_name("token_embd.weight"), 
              "token_embd.weight");
    EXPECT_EQ(GPTWeightMapper::get_tensor_name("position_embd.weight"), 
              "position_embd.weight");
}

TEST(GPTWeightMapper, GetTensorNameLayerSpecific) {
    EXPECT_EQ(GPTWeightMapper::get_tensor_name("attn_norm.weight", 0), 
              "blk.0.attn_norm.weight");
    EXPECT_EQ(GPTWeightMapper::get_tensor_name("attn_qkv.weight", 5), 
              "blk.5.attn_qkv.weight");
    EXPECT_EQ(GPTWeightMapper::get_tensor_name("ffn_down.bias", 43), 
              "blk.43.ffn_down.bias");
}

TEST(GPTWeightMapper, GetExpectedShapeTokenEmbeddings) {
    GPTConfig config;
    config.vocab_size = 50257;
    config.hidden_dim = 2048;
    config.num_layers = 44;
    config.num_heads = 64;
    config.head_dim = 32;
    config.ffn_dim = 8192;
    config.max_seq_len = 2048;
    
    auto shape = GPTWeightMapper::get_expected_shape("token_embd.weight", config);
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 50257);
    EXPECT_EQ(shape[1], 2048);
}

TEST(GPTWeightMapper, GetExpectedShapePositionEmbeddings) {
    GPTConfig config;
    config.vocab_size = 50257;
    config.hidden_dim = 2048;
    config.max_seq_len = 2048;
    
    auto shape = GPTWeightMapper::get_expected_shape("position_embd.weight", config);
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2048);
    EXPECT_EQ(shape[1], 2048);
}

TEST(GPTWeightMapper, GetExpectedShapeQKV) {
    GPTConfig config;
    config.hidden_dim = 2048;
    
    auto shape = GPTWeightMapper::get_expected_shape("blk.0.attn_qkv.weight", config);
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2048);
    EXPECT_EQ(shape[1], 6144);  // 3 * hidden_dim
}

TEST(GPTWeightMapper, GetExpectedShapeFFNUp) {
    GPTConfig config;
    config.hidden_dim = 2048;
    config.ffn_dim = 8192;
    
    auto shape = GPTWeightMapper::get_expected_shape("blk.0.ffn_up.weight", config);
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 2048);
    EXPECT_EQ(shape[1], 8192);
}

TEST(GPTWeightMapper, GetExpectedShapeFFNDown) {
    GPTConfig config;
    config.hidden_dim = 2048;
    config.ffn_dim = 8192;
    
    auto shape = GPTWeightMapper::get_expected_shape("blk.0.ffn_down.weight", config);
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 8192);
    EXPECT_EQ(shape[1], 2048);
}

TEST(GPTWeightMapper, GetExpectedShapeLayerNorm) {
    GPTConfig config;
    config.hidden_dim = 2048;
    
    auto shape = GPTWeightMapper::get_expected_shape("blk.0.attn_norm.weight", config);
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 2048);
    
    shape = GPTWeightMapper::get_expected_shape("output_norm.bias", config);
    ASSERT_EQ(shape.size(), 1);
    EXPECT_EQ(shape[0], 2048);
}

TEST(GPTWeightMapper, IsValidGPTTensor) {
    EXPECT_TRUE(GPTWeightMapper::is_valid_gpt_tensor("token_embd.weight"));
    EXPECT_TRUE(GPTWeightMapper::is_valid_gpt_tensor("position_embd.weight"));
    EXPECT_TRUE(GPTWeightMapper::is_valid_gpt_tensor("blk.0.attn_norm.weight"));
    EXPECT_TRUE(GPTWeightMapper::is_valid_gpt_tensor("blk.43.ffn_down.bias"));
    EXPECT_TRUE(GPTWeightMapper::is_valid_gpt_tensor("output_norm.weight"));
    EXPECT_TRUE(GPTWeightMapper::is_valid_gpt_tensor("output.weight"));
    
    EXPECT_FALSE(GPTWeightMapper::is_valid_gpt_tensor("invalid.tensor"));
    EXPECT_FALSE(GPTWeightMapper::is_valid_gpt_tensor("llama.weight"));
}

TEST(GPTWeightMapper, ParseLayerIndex) {
    EXPECT_EQ(GPTWeightMapper::parse_layer_index("blk.0.attn_norm.weight"), 0);
    EXPECT_EQ(GPTWeightMapper::parse_layer_index("blk.5.ffn_up.weight"), 5);
    EXPECT_EQ(GPTWeightMapper::parse_layer_index("blk.43.ffn_down.bias"), 43);
    
    EXPECT_EQ(GPTWeightMapper::parse_layer_index("token_embd.weight"), -1);
    EXPECT_EQ(GPTWeightMapper::parse_layer_index("output.weight"), -1);
    EXPECT_EQ(GPTWeightMapper::parse_layer_index("invalid"), -1);
}

// ============================================================================
// GPTWeightLoader Tests
// ============================================================================

TEST(GPTWeightLoader, CalculateVRAMUsageGPTOSS20B) {
    GPTConfig config;
    config.vocab_size = 50257;
    config.hidden_dim = 2048;
    config.num_layers = 44;
    config.num_heads = 64;
    config.head_dim = 32;
    config.ffn_dim = 8192;
    config.max_seq_len = 2048;
    
    size_t vram = GPTWeightLoader::calculate_vram_usage(config);
    
    // Should be reasonable for GPT-OSS-20B
    // Expected: ~12-16GB for model + KV cache + activations
    EXPECT_GT(vram, 10ULL * 1024 * 1024 * 1024);  // > 10GB
    EXPECT_LT(vram, 20ULL * 1024 * 1024 * 1024);  // < 20GB
}

TEST(GPTWeightLoader, CalculateVRAMUsageGPT2Small) {
    GPTConfig config;
    config.vocab_size = 50257;
    config.hidden_dim = 768;
    config.num_layers = 12;
    config.num_heads = 12;
    config.head_dim = 64;
    config.ffn_dim = 3072;
    config.max_seq_len = 1024;
    
    size_t vram = GPTWeightLoader::calculate_vram_usage(config);
    
    // Should be < 1GB for GPT-2 small
    EXPECT_LT(vram, 1ULL * 1024 * 1024 * 1024);
}

TEST(GPTWeightLoader, VRAMUsageScalesWithLayers) {
    GPTConfig config_12_layers;
    config_12_layers.vocab_size = 50257;
    config_12_layers.hidden_dim = 768;
    config_12_layers.num_layers = 12;
    config_12_layers.num_heads = 12;
    config_12_layers.head_dim = 64;
    config_12_layers.ffn_dim = 3072;
    config_12_layers.max_seq_len = 1024;
    
    GPTConfig config_24_layers = config_12_layers;
    config_24_layers.num_layers = 24;
    
    size_t vram_12 = GPTWeightLoader::calculate_vram_usage(config_12_layers);
    size_t vram_24 = GPTWeightLoader::calculate_vram_usage(config_24_layers);
    
    // 24 layers should use more VRAM than 12 layers
    EXPECT_GT(vram_24, vram_12);
    
    // But not exactly 2x (due to embeddings and other fixed costs)
    EXPECT_LT(vram_24, vram_12 * 2.5);
}

// ============================================================================
// Mock Tensor Validation Tests
// ============================================================================

TEST(GPTWeightLoader, ValidateTensorShapeSuccess) {
    GGUFTensorInfo tensor;
    tensor.name = "token_embd.weight";
    tensor.dimensions = {50257, 2048};
    
    std::vector<uint64_t> expected = {50257, 2048};
    
    // Should not throw
    EXPECT_NO_THROW(
        GPTWeightLoader::validate_tensor_shape(tensor, expected, "test")
    );
}

TEST(GPTWeightLoader, ValidateTensorShapeDimensionMismatch) {
    GGUFTensorInfo tensor;
    tensor.name = "token_embd.weight";
    tensor.dimensions = {50257, 2048};
    
    std::vector<uint64_t> expected = {50257, 2048, 1};  // Wrong number of dims
    
    EXPECT_THROW(
        GPTWeightLoader::validate_tensor_shape(tensor, expected, "test"),
        std::runtime_error
    );
}

TEST(GPTWeightLoader, ValidateTensorShapeValueMismatch) {
    GGUFTensorInfo tensor;
    tensor.name = "token_embd.weight";
    tensor.dimensions = {50257, 2048};
    
    std::vector<uint64_t> expected = {50257, 4096};  // Wrong hidden_dim
    
    EXPECT_THROW(
        GPTWeightLoader::validate_tensor_shape(tensor, expected, "test"),
        std::runtime_error
    );
}

// ============================================================================
// Integration Readiness Tests
// ============================================================================

TEST(GPTWeights, LayerWeightsConstruction) {
    // Test that layer weights can be constructed and destroyed
    auto layer = std::make_unique<GPTLayerWeights>();
    EXPECT_NE(layer, nullptr);
    EXPECT_EQ(layer->total_vram_bytes, 0);
}

TEST(GPTWeights, ModelWeightsConstruction) {
    // Test that model weights can be constructed and destroyed
    auto model = std::make_unique<GPTModelWeights>();
    EXPECT_NE(model, nullptr);
    EXPECT_EQ(model->total_vram_bytes, 0);
    EXPECT_EQ(model->layers.size(), 0);
}

TEST(GPTWeights, ModelWeightsWithLayers) {
    auto model = std::make_unique<GPTModelWeights>();
    
    // Add layers
    for (int i = 0; i < 44; ++i) {
        model->layers.push_back(std::make_unique<GPTLayerWeights>());
    }
    
    EXPECT_EQ(model->layers.size(), 44);
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
