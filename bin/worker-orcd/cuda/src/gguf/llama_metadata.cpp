/**
 * GGUF Llama Metadata Parser Implementation
 * 
 * Extracts Llama-specific configuration from GGUF metadata.
 * 
 * Spec: M0-W-1211, M0-W-1212
 */

#include "gguf/llama_metadata.h"
#include "../cuda_error.h"
#include <sstream>

namespace worker {
namespace gguf {

const GGUFMetadata* find_metadata(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key
) {
    for (const auto& kv : metadata) {
        if (kv.key == key) {
            return &kv;
        }
    }
    return nullptr;
}

uint32_t get_required_uint32(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key
) {
    const GGUFMetadata* kv = find_metadata(metadata, key);
    if (!kv) {
        throw CudaError::model_load_failed(
            "Required metadata key not found: " + key
        );
    }
    
    // Accept UINT32, UINT64, INT32, INT64
    switch (kv->value_type) {
        case GGUFValueType::UINT32:
        case GGUFValueType::UINT64:
            return static_cast<uint32_t>(kv->uint_value);
        case GGUFValueType::INT32:
        case GGUFValueType::INT64:
            if (kv->int_value < 0) {
                throw CudaError::model_load_failed(
                    "Metadata key '" + key + "' has negative value"
                );
            }
            return static_cast<uint32_t>(kv->int_value);
        default:
            throw CudaError::model_load_failed(
                "Metadata key '" + key + "' has wrong type (expected integer)"
            );
    }
}

uint32_t get_optional_uint32(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key,
    uint32_t default_value
) {
    const GGUFMetadata* kv = find_metadata(metadata, key);
    if (!kv) {
        return default_value;
    }
    
    switch (kv->value_type) {
        case GGUFValueType::UINT32:
        case GGUFValueType::UINT64:
            return static_cast<uint32_t>(kv->uint_value);
        case GGUFValueType::INT32:
        case GGUFValueType::INT64:
            if (kv->int_value < 0) {
                return default_value;
            }
            return static_cast<uint32_t>(kv->int_value);
        default:
            return default_value;
    }
}

float get_required_float(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key
) {
    const GGUFMetadata* kv = find_metadata(metadata, key);
    if (!kv) {
        throw CudaError::model_load_failed(
            "Required metadata key not found: " + key
        );
    }
    
    switch (kv->value_type) {
        case GGUFValueType::FLOAT32:
        case GGUFValueType::FLOAT64:
            return static_cast<float>(kv->float_value);
        default:
            throw CudaError::model_load_failed(
                "Metadata key '" + key + "' has wrong type (expected float)"
            );
    }
}

float get_optional_float(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key,
    float default_value
) {
    const GGUFMetadata* kv = find_metadata(metadata, key);
    if (!kv) {
        return default_value;
    }
    
    switch (kv->value_type) {
        case GGUFValueType::FLOAT32:
        case GGUFValueType::FLOAT64:
            return static_cast<float>(kv->float_value);
        default:
            return default_value;
    }
}

std::string get_required_string(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key
) {
    const GGUFMetadata* kv = find_metadata(metadata, key);
    if (!kv) {
        throw CudaError::model_load_failed(
            "Required metadata key not found: " + key
        );
    }
    
    if (kv->value_type != GGUFValueType::STRING) {
        throw CudaError::model_load_failed(
            "Metadata key '" + key + "' has wrong type (expected string)"
        );
    }
    
    return kv->string_value;
}

uint32_t get_array_length(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key
) {
    const GGUFMetadata* kv = find_metadata(metadata, key);
    if (!kv) {
        throw CudaError::model_load_failed(
            "Required metadata key not found: " + key
        );
    }
    
    if (kv->value_type != GGUFValueType::ARRAY) {
        throw CudaError::model_load_failed(
            "Metadata key '" + key + "' has wrong type (expected array)"
        );
    }
    
    // Array length is stored in array_value.size() or we need to parse it
    // For now, we'll need to track this during parsing
    // This is a simplified implementation - full array parsing needed
    return static_cast<uint32_t>(kv->array_value.size());
}

LlamaConfig parse_llama_metadata(const std::vector<GGUFMetadata>& metadata) {
    LlamaConfig config;
    
    // Validate architecture
    config.architecture = get_required_string(metadata, "general.architecture");
    if (config.architecture != "llama") {
        throw CudaError::model_load_failed(
            "Invalid architecture: '" + config.architecture + "' (expected 'llama')"
        );
    }
    
    // Extract required parameters
    config.context_length = get_required_uint32(metadata, "llama.context_length");
    config.embedding_length = get_required_uint32(metadata, "llama.embedding_length");
    config.block_count = get_required_uint32(metadata, "llama.block_count");
    config.attention_head_count = get_required_uint32(metadata, "llama.attention.head_count");
    config.attention_head_count_kv = get_required_uint32(metadata, "llama.attention.head_count_kv");
    config.ffn_length = get_required_uint32(metadata, "llama.feed_forward_length");
    
    // Extract optional RoPE parameters
    // Default rope_dimension_count to head_dim if not specified
    uint32_t default_rope_dims = config.embedding_length / config.attention_head_count;
    config.rope_dimension_count = get_optional_uint32(
        metadata,
        "llama.rope.dimension_count",
        default_rope_dims
    );
    
    // Default rope_freq_base to 10000.0 (standard for most models)
    config.rope_freq_base = get_optional_float(
        metadata,
        "llama.rope.freq_base",
        10000.0f
    );
    
    // Extract vocab size from tokenizer metadata
    // Try multiple possible keys (different GGUF versions use different keys)
    const GGUFMetadata* vocab_kv = find_metadata(metadata, "tokenizer.ggml.tokens");
    if (!vocab_kv) {
        vocab_kv = find_metadata(metadata, "tokenizer.ggml.token_type");
    }
    if (!vocab_kv) {
        // Fallback: try to get from model metadata
        vocab_kv = find_metadata(metadata, "llama.vocab_size");
    }
    
    if (vocab_kv) {
        if (vocab_kv->value_type == GGUFValueType::ARRAY) {
            config.vocab_size = static_cast<uint32_t>(vocab_kv->array_value.size());
        } else if (vocab_kv->value_type == GGUFValueType::UINT32 ||
                   vocab_kv->value_type == GGUFValueType::UINT64) {
            config.vocab_size = static_cast<uint32_t>(vocab_kv->uint_value);
        } else {
            throw CudaError::model_load_failed(
                "Cannot determine vocab_size from metadata"
            );
        }
    } else {
        throw CudaError::model_load_failed(
            "Required tokenizer metadata not found (tokenizer.ggml.tokens or llama.vocab_size)"
        );
    }
    
    // Calculate derived parameters
    if (config.attention_head_count == 0) {
        throw CudaError::model_load_failed(
            "Invalid attention_head_count: 0"
        );
    }
    if (config.attention_head_count_kv == 0) {
        throw CudaError::model_load_failed(
            "Invalid attention_head_count_kv: 0"
        );
    }
    
    config.head_dim = config.embedding_length / config.attention_head_count;
    config.kv_head_dim = config.embedding_length / config.attention_head_count_kv;
    
    // Validate derived parameters
    if (config.head_dim == 0) {
        throw CudaError::model_load_failed(
            "Invalid head_dim: embedding_length=" +
            std::to_string(config.embedding_length) +
            " attention_head_count=" +
            std::to_string(config.attention_head_count)
        );
    }
    if (config.kv_head_dim == 0) {
        throw CudaError::model_load_failed(
            "Invalid kv_head_dim: embedding_length=" +
            std::to_string(config.embedding_length) +
            " attention_head_count_kv=" +
            std::to_string(config.attention_head_count_kv)
        );
    }
    
    // Validate embedding_length is divisible by head counts
    if (config.embedding_length % config.attention_head_count != 0) {
        throw CudaError::model_load_failed(
            "embedding_length (" + std::to_string(config.embedding_length) +
            ") is not divisible by attention_head_count (" +
            std::to_string(config.attention_head_count) + ")"
        );
    }
    if (config.embedding_length % config.attention_head_count_kv != 0) {
        throw CudaError::model_load_failed(
            "embedding_length (" + std::to_string(config.embedding_length) +
            ") is not divisible by attention_head_count_kv (" +
            std::to_string(config.attention_head_count_kv) + ")"
        );
    }
    
    // Validate GQA configuration (KV heads <= attention heads)
    if (config.attention_head_count_kv > config.attention_head_count) {
        throw CudaError::model_load_failed(
            "Invalid GQA configuration: attention_head_count_kv (" +
            std::to_string(config.attention_head_count_kv) +
            ") > attention_head_count (" +
            std::to_string(config.attention_head_count) + ")"
        );
    }
    
    // Extract optional RoPE scaling parameters
    const GGUFMetadata* rope_scale_kv = find_metadata(metadata, "llama.rope.scale");
    if (rope_scale_kv && rope_scale_kv->value_type == GGUFValueType::FLOAT32) {
        config.rope_scale = rope_scale_kv->float_value;
    }
    
    const GGUFMetadata* rope_scaling_type_kv = find_metadata(metadata, "llama.rope.scaling.type");
    if (rope_scaling_type_kv && rope_scaling_type_kv->value_type == GGUFValueType::UINT32) {
        config.rope_scaling_type = static_cast<uint32_t>(rope_scaling_type_kv->uint_value);
    }
    
    return config;
}

} // namespace gguf
} // namespace worker

// ---
// Implemented by Llama-Beta 🦙
