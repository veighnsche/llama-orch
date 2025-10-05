/**
 * Unit Tests: GPT Config Parsing from GGUF
 * 
 * Tests parse_config_from_gguf() for multiple architectures.
 * Validates that config extraction is correct and dynamic.
 * 
 * Story: GT-051
 * Testing Team Requirements: Unit tests for config parsing
 */

#include "../src/model/gpt_weights.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

using namespace worker::model;

// Test fixture for GPT config parsing
class GPTConfigParsingTest : public ::testing::Test {
protected:
    // Path to test models (must exist for tests to pass)
    const std::string qwen_model_path = 
        "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";
};

/**
 * Test: Parse Qwen2.5-0.5B config
 * 
 * Validates that Qwen2 architecture is detected and config is extracted correctly.
 * This test OBSERVES the product's behavior - it does NOT pre-create any state.
 */
TEST_F(GPTConfigParsingTest, ParseQwen25Config) {
    // OBSERVE: Product parses the file
    GPTConfig config = GPTWeightLoader::parse_config_from_gguf(qwen_model_path);
    
    // VERIFY: Config matches Qwen2.5-0.5B specifications (from RESEARCH_RESULTS.md)
    EXPECT_EQ(config.vocab_size, 151643) << "Qwen2.5-0.5B has vocab_size=151643";
    EXPECT_EQ(config.hidden_dim, 896) << "Qwen2.5-0.5B has hidden_dim=896";
    EXPECT_EQ(config.num_layers, 24) << "Qwen2.5-0.5B has 24 layers";
    EXPECT_EQ(config.num_heads, 14) << "Qwen2.5-0.5B has 14 attention heads";
    EXPECT_EQ(config.head_dim, 64) << "head_dim should be 896/14=64";
    EXPECT_EQ(config.ffn_dim, 4864) << "Qwen2.5-0.5B has ffn_dim=4864";
    EXPECT_EQ(config.context_length, 32768) << "Qwen2.5-0.5B has context_length=32768";
    EXPECT_EQ(config.max_seq_len, 32768) << "max_seq_len should equal context_length";
    
    // VERIFY: Quantization detected correctly
    EXPECT_EQ(config.quant_kind, "Q4_K_M") << "Qwen model uses Q4_K_M quantization";
}

/**
 * Test: Config values are NOT hardcoded
 * 
 * Validates that the function actually reads the file, not returning hardcoded values.
 * The old stub returned vocab_size=50257 (GPT-2), which should NOT happen for Qwen.
 */
TEST_F(GPTConfigParsingTest, NotHardcoded) {
    GPTConfig config = GPTWeightLoader::parse_config_from_gguf(qwen_model_path);
    
    // VERIFY: NOT the old hardcoded GPT-OSS-20B values
    EXPECT_NE(config.vocab_size, 50257) << "Should NOT be hardcoded GPT-2 vocab_size";
    EXPECT_NE(config.hidden_dim, 2048) << "Should NOT be hardcoded GPT-OSS-20B hidden_dim";
    EXPECT_NE(config.num_layers, 44) << "Should NOT be hardcoded GPT-OSS-20B num_layers";
    EXPECT_NE(config.num_heads, 64) << "Should NOT be hardcoded GPT-OSS-20B num_heads";
}

/**
 * Test: Invalid file path
 * 
 * Validates that the function throws on nonexistent files.
 * This ensures error handling works correctly.
 */
TEST_F(GPTConfigParsingTest, InvalidFilePath) {
    // VERIFY: Product throws on invalid path
    EXPECT_THROW(
        GPTWeightLoader::parse_config_from_gguf("/nonexistent/file.gguf"),
        std::runtime_error
    ) << "Should throw on nonexistent file";
}

/**
 * Test: Unsupported architecture
 * 
 * Validates that the function throws on unsupported architectures.
 * This test would require a GGUF file with unsupported architecture.
 * 
 * NOTE: Skipped for now as we don't have such a test file.
 * TODO: Create test GGUF with unsupported architecture.
 */
TEST_F(GPTConfigParsingTest, DISABLED_UnsupportedArchitecture) {
    // TODO: Create test file with architecture="mistral" or similar
    // EXPECT_THROW(
    //     GPTWeightLoader::parse_config_from_gguf("test_unsupported.gguf"),
    //     std::runtime_error
    // );
}

/**
 * Test: Config validation
 * 
 * Validates that extracted config passes validation.
 */
TEST_F(GPTConfigParsingTest, ConfigValidation) {
    GPTConfig config = GPTWeightLoader::parse_config_from_gguf(qwen_model_path);
    
    // VERIFY: Config is valid
    EXPECT_TRUE(config.validate()) << "Extracted config should be valid";
    
    // VERIFY: All required fields are non-zero
    EXPECT_GT(config.vocab_size, 0) << "vocab_size must be > 0";
    EXPECT_GT(config.hidden_dim, 0) << "hidden_dim must be > 0";
    EXPECT_GT(config.num_layers, 0) << "num_layers must be > 0";
    EXPECT_GT(config.num_heads, 0) << "num_heads must be > 0";
    EXPECT_GT(config.head_dim, 0) << "head_dim must be > 0";
    EXPECT_GT(config.ffn_dim, 0) << "ffn_dim must be > 0";
    EXPECT_GT(config.context_length, 0) << "context_length must be > 0";
}

/**
 * Test: Head dimension calculation
 * 
 * Validates that head_dim is correctly calculated as hidden_dim / num_heads.
 */
TEST_F(GPTConfigParsingTest, HeadDimensionCalculation) {
    GPTConfig config = GPTWeightLoader::parse_config_from_gguf(qwen_model_path);
    
    // VERIFY: head_dim = hidden_dim / num_heads
    int expected_head_dim = config.hidden_dim / config.num_heads;
    EXPECT_EQ(config.head_dim, expected_head_dim) 
        << "head_dim should be hidden_dim / num_heads";
}

/**
 * Test: Quantization detection
 * 
 * Validates that quantization type is detected from tensor metadata.
 */
TEST_F(GPTConfigParsingTest, QuantizationDetection) {
    GPTConfig config = GPTWeightLoader::parse_config_from_gguf(qwen_model_path);
    
    // VERIFY: Quantization is detected (not "UNKNOWN")
    EXPECT_NE(config.quant_kind, "UNKNOWN") << "Should detect quantization type";
    EXPECT_FALSE(config.quant_kind.empty()) << "quant_kind should not be empty";
}

// ---
// Test artifacts verified by Testing Team 🔍
