/**
 * GGUF Llama Metadata Parser Tests
 * 
 * Unit tests for Llama-specific metadata extraction from GGUF files.
 * 
 * Tests cover:
 * - Qwen2.5-0.5B metadata parsing
 * - Phi-3 metadata parsing
 * - Required key validation
 * - Derived parameter calculation
 * - Error handling for missing/invalid metadata
 * 
 * Spec: M0-W-1211, M0-W-1212
 */

#include <gtest/gtest.h>
#include "gguf/llama_metadata.h"
#include "cuda_error.h"
#include <vector>

using namespace worker::gguf;

class LlamaMetadataTest : public ::testing::Test {
protected:
    // Helper: Create metadata vector
    std::vector<GGUFMetadata> create_metadata() {
        return std::vector<GGUFMetadata>();
    }
    
    // Helper: Add string metadata
    void add_string(std::vector<GGUFMetadata>& metadata,
                    const std::string& key,
                    const std::string& value) {
        GGUFMetadata kv;
        kv.key = key;
        kv.value_type = GGUFValueType::STRING;
        kv.string_value = value;
        metadata.push_back(kv);
    }
    
    // Helper: Add uint32 metadata
    void add_uint32(std::vector<GGUFMetadata>& metadata,
                    const std::string& key,
                    uint32_t value) {
        GGUFMetadata kv;
        kv.key = key;
        kv.value_type = GGUFValueType::UINT32;
        kv.uint_value = value;
        metadata.push_back(kv);
    }
    
    // Helper: Add float metadata
    void add_float(std::vector<GGUFMetadata>& metadata,
                   const std::string& key,
                   float value) {
        GGUFMetadata kv;
        kv.key = key;
        kv.value_type = GGUFValueType::FLOAT32;
        kv.float_value = value;
        metadata.push_back(kv);
    }
    
    // Helper: Add array metadata (simplified - just track length)
    void add_array(std::vector<GGUFMetadata>& metadata,
                   const std::string& key,
                   size_t length) {
        GGUFMetadata kv;
        kv.key = key;
        kv.value_type = GGUFValueType::ARRAY;
        kv.array_value.resize(length);
        metadata.push_back(kv);
    }
    
    // Create Qwen2.5-0.5B metadata
    std::vector<GGUFMetadata> create_qwen_metadata() {
        auto metadata = create_metadata();
        add_string(metadata, "general.architecture", "llama");
        add_uint32(metadata, "llama.context_length", 32768);
        add_uint32(metadata, "llama.embedding_length", 896);
        add_uint32(metadata, "llama.block_count", 24);
        add_uint32(metadata, "llama.attention.head_count", 14);
        add_uint32(metadata, "llama.attention.head_count_kv", 2);  // GQA
        add_uint32(metadata, "llama.feed_forward_length", 4864);
        add_uint32(metadata, "llama.rope.dimension_count", 64);
        add_float(metadata, "llama.rope.freq_base", 1000000.0f);  // Qwen uses 1M
        add_array(metadata, "tokenizer.ggml.tokens", 151936);  // Qwen vocab size
        return metadata;
    }
    
    // Create Phi-3 metadata
    std::vector<GGUFMetadata> create_phi3_metadata() {
        auto metadata = create_metadata();
        add_string(metadata, "general.architecture", "llama");
        add_uint32(metadata, "llama.context_length", 4096);
        add_uint32(metadata, "llama.embedding_length", 3072);
        add_uint32(metadata, "llama.block_count", 32);
        add_uint32(metadata, "llama.attention.head_count", 32);
        add_uint32(metadata, "llama.attention.head_count_kv", 32);  // MHA
        add_uint32(metadata, "llama.feed_forward_length", 8192);
        add_uint32(metadata, "llama.rope.dimension_count", 96);
        add_float(metadata, "llama.rope.freq_base", 10000.0f);
        add_array(metadata, "tokenizer.ggml.tokens", 32064);  // Phi-3 vocab size
        return metadata;
    }
};

// Test: Parse Qwen2.5-0.5B metadata
TEST_F(LlamaMetadataTest, ParseQwenMetadata) {
    auto metadata = create_qwen_metadata();
    
    LlamaConfig config = parse_llama_metadata(metadata);
    
    EXPECT_EQ(config.architecture, "llama");
    EXPECT_EQ(config.context_length, 32768u);
    EXPECT_EQ(config.embedding_length, 896u);
    EXPECT_EQ(config.block_count, 24u);
    EXPECT_EQ(config.attention_head_count, 14u);
    EXPECT_EQ(config.attention_head_count_kv, 2u);  // GQA
    EXPECT_EQ(config.ffn_length, 4864u);
    EXPECT_EQ(config.rope_dimension_count, 64u);
    EXPECT_FLOAT_EQ(config.rope_freq_base, 1000000.0f);
    EXPECT_EQ(config.vocab_size, 151936u);
    
    // Derived parameters
    EXPECT_EQ(config.head_dim, 64u);  // 896 / 14
    EXPECT_EQ(config.kv_head_dim, 448u);  // 896 / 2
}

// Test: Parse Phi-3 metadata
TEST_F(LlamaMetadataTest, ParsePhi3Metadata) {
    auto metadata = create_phi3_metadata();
    
    LlamaConfig config = parse_llama_metadata(metadata);
    
    EXPECT_EQ(config.architecture, "llama");
    EXPECT_EQ(config.context_length, 4096u);
    EXPECT_EQ(config.embedding_length, 3072u);
    EXPECT_EQ(config.block_count, 32u);
    EXPECT_EQ(config.attention_head_count, 32u);
    EXPECT_EQ(config.attention_head_count_kv, 32u);  // MHA
    EXPECT_EQ(config.ffn_length, 8192u);
    EXPECT_EQ(config.rope_dimension_count, 96u);
    EXPECT_FLOAT_EQ(config.rope_freq_base, 10000.0f);
    EXPECT_EQ(config.vocab_size, 32064u);
    
    // Derived parameters
    EXPECT_EQ(config.head_dim, 96u);  // 3072 / 32
    EXPECT_EQ(config.kv_head_dim, 96u);  // 3072 / 32 (MHA)
}

// Test: Missing required key
TEST_F(LlamaMetadataTest, ErrorOnMissingRequiredKey) {
    auto metadata = create_metadata();
    add_string(metadata, "general.architecture", "llama");
    // Missing all other required keys
    
    EXPECT_THROW({
        parse_llama_metadata(metadata);
    }, worker::CudaError);
}

// Test: Invalid architecture
TEST_F(LlamaMetadataTest, ErrorOnInvalidArchitecture) {
    auto metadata = create_qwen_metadata();
    // Change architecture to invalid value
    metadata[0].string_value = "gpt2";
    
    EXPECT_THROW({
        parse_llama_metadata(metadata);
    }, worker::CudaError);
}

// Test: Default rope_freq_base
TEST_F(LlamaMetadataTest, DefaultRopeFreqBase) {
    auto metadata = create_qwen_metadata();
    
    // Remove rope_freq_base (should default to 10000.0)
    metadata.erase(
        std::remove_if(metadata.begin(), metadata.end(),
                      [](const GGUFMetadata& kv) {
                          return kv.key == "llama.rope.freq_base";
                      }),
        metadata.end()
    );
    
    LlamaConfig config = parse_llama_metadata(metadata);
    EXPECT_FLOAT_EQ(config.rope_freq_base, 10000.0f);
}

// Test: Default rope_dimension_count
TEST_F(LlamaMetadataTest, DefaultRopeDimensionCount) {
    auto metadata = create_qwen_metadata();
    
    // Remove rope_dimension_count (should default to head_dim)
    metadata.erase(
        std::remove_if(metadata.begin(), metadata.end(),
                      [](const GGUFMetadata& kv) {
                          return kv.key == "llama.rope.dimension_count";
                      }),
        metadata.end()
    );
    
    LlamaConfig config = parse_llama_metadata(metadata);
    EXPECT_EQ(config.rope_dimension_count, 64u);  // 896 / 14 = 64
}

// Test: Derived parameter calculation
TEST_F(LlamaMetadataTest, DerivedParameterCalculation) {
    auto metadata = create_qwen_metadata();
    
    LlamaConfig config = parse_llama_metadata(metadata);
    
    // head_dim = embedding_length / attention_head_count
    EXPECT_EQ(config.head_dim, config.embedding_length / config.attention_head_count);
    
    // kv_head_dim = embedding_length / attention_head_count_kv
    EXPECT_EQ(config.kv_head_dim, config.embedding_length / config.attention_head_count_kv);
}

// Test: Zero attention head count
TEST_F(LlamaMetadataTest, ErrorOnZeroAttentionHeadCount) {
    auto metadata = create_qwen_metadata();
    
    // Set attention_head_count to 0
    for (auto& kv : metadata) {
        if (kv.key == "llama.attention.head_count") {
            kv.uint_value = 0;
        }
    }
    
    EXPECT_THROW({
        parse_llama_metadata(metadata);
    }, worker::CudaError);
}

// Test: Zero KV head count
TEST_F(LlamaMetadataTest, ErrorOnZeroKVHeadCount) {
    auto metadata = create_qwen_metadata();
    
    // Set attention_head_count_kv to 0
    for (auto& kv : metadata) {
        if (kv.key == "llama.attention.head_count_kv") {
            kv.uint_value = 0;
        }
    }
    
    EXPECT_THROW({
        parse_llama_metadata(metadata);
    }, worker::CudaError);
}

// Test: Embedding length not divisible by head count
TEST_F(LlamaMetadataTest, ErrorOnNonDivisibleEmbeddingLength) {
    auto metadata = create_qwen_metadata();
    
    // Set embedding_length to 897 (not divisible by 14)
    for (auto& kv : metadata) {
        if (kv.key == "llama.embedding_length") {
            kv.uint_value = 897;
        }
    }
    
    EXPECT_THROW({
        parse_llama_metadata(metadata);
    }, worker::CudaError);
}

// Test: Invalid GQA configuration (KV heads > attention heads)
TEST_F(LlamaMetadataTest, ErrorOnInvalidGQAConfiguration) {
    auto metadata = create_qwen_metadata();
    
    // Set KV heads > attention heads (invalid)
    for (auto& kv : metadata) {
        if (kv.key == "llama.attention.head_count_kv") {
            kv.uint_value = 20;  // > 14 attention heads
        }
    }
    
    EXPECT_THROW({
        parse_llama_metadata(metadata);
    }, worker::CudaError);
}

// Test: Helper function - find_metadata
TEST_F(LlamaMetadataTest, FindMetadataHelper) {
    auto metadata = create_qwen_metadata();
    
    const GGUFMetadata* found = find_metadata(metadata, "general.architecture");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->string_value, "llama");
    
    const GGUFMetadata* not_found = find_metadata(metadata, "nonexistent.key");
    EXPECT_EQ(not_found, nullptr);
}

// Test: Helper function - get_required_uint32
TEST_F(LlamaMetadataTest, GetRequiredUint32Helper) {
    auto metadata = create_qwen_metadata();
    
    uint32_t value = get_required_uint32(metadata, "llama.context_length");
    EXPECT_EQ(value, 32768u);
    
    EXPECT_THROW({
        get_required_uint32(metadata, "nonexistent.key");
    }, worker::CudaError);
}

// Test: Helper function - get_optional_uint32
TEST_F(LlamaMetadataTest, GetOptionalUint32Helper) {
    auto metadata = create_qwen_metadata();
    
    uint32_t value = get_optional_uint32(metadata, "llama.context_length", 999);
    EXPECT_EQ(value, 32768u);
    
    uint32_t default_value = get_optional_uint32(metadata, "nonexistent.key", 999);
    EXPECT_EQ(default_value, 999u);
}

// Test: Helper function - get_required_string
TEST_F(LlamaMetadataTest, GetRequiredStringHelper) {
    auto metadata = create_qwen_metadata();
    
    std::string value = get_required_string(metadata, "general.architecture");
    EXPECT_EQ(value, "llama");
    
    EXPECT_THROW({
        get_required_string(metadata, "nonexistent.key");
    }, worker::CudaError);
}

// Test: Helper function - get_required_float
TEST_F(LlamaMetadataTest, GetRequiredFloatHelper) {
    auto metadata = create_qwen_metadata();
    
    float value = get_required_float(metadata, "llama.rope.freq_base");
    EXPECT_FLOAT_EQ(value, 1000000.0f);
    
    EXPECT_THROW({
        get_required_float(metadata, "nonexistent.key");
    }, worker::CudaError);
}

// Test: Helper function - get_optional_float
TEST_F(LlamaMetadataTest, GetOptionalFloatHelper) {
    auto metadata = create_qwen_metadata();
    
    float value = get_optional_float(metadata, "llama.rope.freq_base", 999.0f);
    EXPECT_FLOAT_EQ(value, 1000000.0f);
    
    float default_value = get_optional_float(metadata, "nonexistent.key", 999.0f);
    EXPECT_FLOAT_EQ(default_value, 999.0f);
}

// Test: Qwen GQA configuration (2 KV heads)
TEST_F(LlamaMetadataTest, QwenGQAConfiguration) {
    auto metadata = create_qwen_metadata();
    LlamaConfig config = parse_llama_metadata(metadata);
    
    // Qwen uses GQA with 2 KV heads
    EXPECT_EQ(config.attention_head_count, 14u);
    EXPECT_EQ(config.attention_head_count_kv, 2u);
    EXPECT_LT(config.attention_head_count_kv, config.attention_head_count);
}

// Test: Phi-3 MHA configuration (32 KV heads = 32 attention heads)
TEST_F(LlamaMetadataTest, Phi3MHAConfiguration) {
    auto metadata = create_phi3_metadata();
    LlamaConfig config = parse_llama_metadata(metadata);
    
    // Phi-3 uses MHA (KV heads == attention heads)
    EXPECT_EQ(config.attention_head_count, 32u);
    EXPECT_EQ(config.attention_head_count_kv, 32u);
    EXPECT_EQ(config.attention_head_count_kv, config.attention_head_count);
}

// Test: Qwen vocab size (151936 tokens)
TEST_F(LlamaMetadataTest, QwenVocabSize) {
    auto metadata = create_qwen_metadata();
    LlamaConfig config = parse_llama_metadata(metadata);
    
    EXPECT_EQ(config.vocab_size, 151936u);
}

// Test: Phi-3 vocab size (32064 tokens)
TEST_F(LlamaMetadataTest, Phi3VocabSize) {
    auto metadata = create_phi3_metadata();
    LlamaConfig config = parse_llama_metadata(metadata);
    
    EXPECT_EQ(config.vocab_size, 32064u);
}

// ---
// Implemented by Llama-Beta 🦙
