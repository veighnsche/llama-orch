/**
 * Architecture Detection Tests
 * 
 * Unit tests for Llama variant detection.
 * 
 * Tests cover:
 * - Qwen detection
 * - Phi-3 detection
 * - Llama 2/3 detection
 * - Unknown variant handling
 * - Model name inference
 * 
 * Spec: M0-W-1212
 */

#include <gtest/gtest.h>
#include "model/arch_detect.h"

using namespace worker::model;
using namespace worker::gguf;

class ArchDetectTest : public ::testing::Test {
protected:
    // Helper: Create Qwen2.5-0.5B config
    LlamaConfig create_qwen_config() {
        LlamaConfig config;
        config.architecture = "llama";
        config.context_length = 32768;
        config.embedding_length = 896;
        config.block_count = 24;
        config.attention_head_count = 14;
        config.attention_head_count_kv = 2;  // GQA
        config.ffn_length = 4864;
        config.rope_dimension_count = 64;
        config.rope_freq_base = 1000000.0f;
        config.vocab_size = 151936;
        config.head_dim = 64;
        config.kv_head_dim = 448;
        return config;
    }
    
    // Helper: Create Phi-3 config
    LlamaConfig create_phi3_config() {
        LlamaConfig config;
        config.architecture = "llama";
        config.context_length = 4096;
        config.embedding_length = 3072;
        config.block_count = 32;
        config.attention_head_count = 32;
        config.attention_head_count_kv = 32;  // MHA
        config.ffn_length = 8192;
        config.rope_dimension_count = 96;
        config.rope_freq_base = 10000.0f;
        config.vocab_size = 32064;
        config.head_dim = 96;
        config.kv_head_dim = 96;
        return config;
    }
    
    // Helper: Create Llama-2-7B config
    LlamaConfig create_llama2_config() {
        LlamaConfig config;
        config.architecture = "llama";
        config.context_length = 4096;
        config.embedding_length = 4096;
        config.block_count = 32;
        config.attention_head_count = 32;
        config.attention_head_count_kv = 8;  // GQA
        config.ffn_length = 11008;
        config.rope_dimension_count = 128;
        config.rope_freq_base = 10000.0f;
        config.vocab_size = 32000;
        config.head_dim = 128;
        config.kv_head_dim = 512;
        return config;
    }
    
    // Helper: Create Llama-3-8B config
    LlamaConfig create_llama3_config() {
        LlamaConfig config;
        config.architecture = "llama";
        config.context_length = 8192;
        config.embedding_length = 4096;
        config.block_count = 32;
        config.attention_head_count = 32;
        config.attention_head_count_kv = 8;  // GQA
        config.ffn_length = 14336;
        config.rope_dimension_count = 128;
        config.rope_freq_base = 500000.0f;
        config.vocab_size = 128256;
        config.head_dim = 128;
        config.kv_head_dim = 512;
        return config;
    }
};

// Test: Detect Qwen2.5-0.5B
TEST_F(ArchDetectTest, DetectQwen) {
    auto config = create_qwen_config();
    
    ArchitectureInfo info = ArchitectureDetector::detect(config);
    
    EXPECT_EQ(info.architecture, "llama");
    EXPECT_EQ(info.variant, LlamaVariant::Qwen);
    EXPECT_TRUE(info.supports_gqa);
    EXPECT_FALSE(info.supports_mha);
    EXPECT_EQ(info.kv_heads, 2u);
    EXPECT_EQ(info.model_name, "Qwen2.5-0.5B");
}

// Test: Detect Phi-3-mini
TEST_F(ArchDetectTest, DetectPhi3) {
    auto config = create_phi3_config();
    
    ArchitectureInfo info = ArchitectureDetector::detect(config);
    
    EXPECT_EQ(info.architecture, "llama");
    EXPECT_EQ(info.variant, LlamaVariant::Phi3);
    EXPECT_FALSE(info.supports_gqa);
    EXPECT_TRUE(info.supports_mha);
    EXPECT_EQ(info.kv_heads, 32u);
    EXPECT_EQ(info.model_name, "Phi-3-mini");
}

// Test: Detect Llama-2-7B
TEST_F(ArchDetectTest, DetectLlama2) {
    auto config = create_llama2_config();
    
    ArchitectureInfo info = ArchitectureDetector::detect(config);
    
    EXPECT_EQ(info.architecture, "llama");
    EXPECT_EQ(info.variant, LlamaVariant::Llama2);
    EXPECT_TRUE(info.supports_gqa);
    EXPECT_FALSE(info.supports_mha);
    EXPECT_EQ(info.kv_heads, 8u);
    EXPECT_EQ(info.model_name, "Llama-2-7B");
}

// Test: Detect Llama-3-8B
TEST_F(ArchDetectTest, DetectLlama3) {
    auto config = create_llama3_config();
    
    ArchitectureInfo info = ArchitectureDetector::detect(config);
    
    EXPECT_EQ(info.architecture, "llama");
    EXPECT_EQ(info.variant, LlamaVariant::Llama3);
    EXPECT_TRUE(info.supports_gqa);
    EXPECT_FALSE(info.supports_mha);
    EXPECT_EQ(info.kv_heads, 8u);
    EXPECT_EQ(info.model_name, "Llama-3-8B");
}

// Test: Unknown variant handling
TEST_F(ArchDetectTest, DetectUnknownVariant) {
    auto config = create_qwen_config();
    
    // Modify to make it unrecognizable
    config.context_length = 16384;  // Non-standard context
    
    ArchitectureInfo info = ArchitectureDetector::detect(config);
    
    EXPECT_EQ(info.variant, LlamaVariant::Unknown);
    EXPECT_EQ(info.architecture, "llama");
    // Should still detect GQA capability
    EXPECT_TRUE(info.supports_gqa);
}

// Test: Qwen GQA configuration
TEST_F(ArchDetectTest, QwenGQAConfiguration) {
    auto config = create_qwen_config();
    ArchitectureInfo info = ArchitectureDetector::detect(config);
    
    // Qwen uses GQA with 2 KV heads
    EXPECT_TRUE(info.supports_gqa);
    EXPECT_FALSE(info.supports_mha);
    EXPECT_EQ(info.kv_heads, 2u);
}

// Test: Phi-3 MHA configuration
TEST_F(ArchDetectTest, Phi3MHAConfiguration) {
    auto config = create_phi3_config();
    ArchitectureInfo info = ArchitectureDetector::detect(config);
    
    // Phi-3 uses MHA (KV heads == attention heads)
    EXPECT_FALSE(info.supports_gqa);
    EXPECT_TRUE(info.supports_mha);
    EXPECT_EQ(info.kv_heads, 32u);
}

// Test: Model name inference for Qwen variants
TEST_F(ArchDetectTest, ModelNameInferenceQwenVariants) {
    auto config = create_qwen_config();
    
    // Qwen2.5-0.5B (896d)
    config.embedding_length = 896;
    auto info = ArchitectureDetector::detect(config);
    EXPECT_EQ(info.model_name, "Qwen2.5-0.5B");
    
    // Qwen2.5-1.5B (1536d)
    config.embedding_length = 1536;
    info = ArchitectureDetector::detect(config);
    EXPECT_EQ(info.model_name, "Qwen2.5-1.5B");
    
    // Qwen2.5-3B (2048d)
    config.embedding_length = 2048;
    info = ArchitectureDetector::detect(config);
    EXPECT_EQ(info.model_name, "Qwen2.5-3B");
}

// Test: Model name inference for Phi-3 variants
TEST_F(ArchDetectTest, ModelNameInferencePhi3Variants) {
    auto config = create_phi3_config();
    
    // Phi-3-mini (3072d)
    config.embedding_length = 3072;
    auto info = ArchitectureDetector::detect(config);
    EXPECT_EQ(info.model_name, "Phi-3-mini");
    
    // Phi-3-small (4096d)
    config.embedding_length = 4096;
    info = ArchitectureDetector::detect(config);
    EXPECT_EQ(info.model_name, "Phi-3-small");
    
    // Phi-3-medium (5120d)
    config.embedding_length = 5120;
    info = ArchitectureDetector::detect(config);
    EXPECT_EQ(info.model_name, "Phi-3-medium");
}

// Test: Model name inference for Llama 2 variants
TEST_F(ArchDetectTest, ModelNameInferenceLlama2Variants) {
    auto config = create_llama2_config();
    
    // Llama-2-7B (4096d)
    config.embedding_length = 4096;
    auto info = ArchitectureDetector::detect(config);
    EXPECT_EQ(info.model_name, "Llama-2-7B");
    
    // Llama-2-13B (5120d)
    config.embedding_length = 5120;
    info = ArchitectureDetector::detect(config);
    EXPECT_EQ(info.model_name, "Llama-2-13B");
    
    // Llama-2-70B (8192d)
    config.embedding_length = 8192;
    info = ArchitectureDetector::detect(config);
    EXPECT_EQ(info.model_name, "Llama-2-70B");
}

// ---
// Implemented by Llama-Beta 🦙
