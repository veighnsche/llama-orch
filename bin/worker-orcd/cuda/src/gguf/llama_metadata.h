/**
 * GGUF Llama Metadata Parser
 * 
 * Extracts Llama-specific configuration from GGUF metadata.
 * Supports Qwen2.5, Phi-3, and standard Llama 2/3 models.
 * 
 * Spec: M0-W-1211, M0-W-1212
 */

#ifndef WORKER_GGUF_LLAMA_METADATA_H
#define WORKER_GGUF_LLAMA_METADATA_H

#include "header_parser.h"
#include <cstdint>
#include <string>
#include <optional>

namespace worker {
namespace gguf {

/**
 * Llama model configuration extracted from GGUF metadata
 */
struct LlamaConfig {
    // Architecture
    std::string architecture;           // "llama"
    
    // Model dimensions
    uint32_t context_length;            // e.g., 32768 (Qwen), 4096 (Phi-3)
    uint32_t embedding_length;          // e.g., 896 (Qwen), 3072 (Phi-3)
    uint32_t block_count;               // e.g., 24 (Qwen), 32 (Phi-3)
    
    // Attention configuration
    uint32_t attention_head_count;      // e.g., 14 (Qwen), 32 (Phi-3)
    uint32_t attention_head_count_kv;   // e.g., 2 (Qwen GQA), 32 (Phi-3 MHA)
    
    // FFN configuration
    uint32_t ffn_length;                // e.g., 4864 (Qwen), 8192 (Phi-3)
    
    // RoPE configuration
    uint32_t rope_dimension_count;      // e.g., 64
    float rope_freq_base;               // e.g., 10000.0 or 1000000.0
    
    // Tokenizer
    uint32_t vocab_size;                // from tokenizer metadata
    
    // Derived parameters (calculated from above)
    uint32_t head_dim;                  // embedding_length / attention_head_count
    uint32_t kv_head_dim;               // embedding_length / attention_head_count_kv
    
    // Optional parameters
    std::optional<float> rope_scale;    // RoPE scaling factor
    std::optional<uint32_t> rope_scaling_type;  // RoPE scaling type
};

/**
 * Parse Llama configuration from GGUF metadata
 * 
 * Extracts all Llama-specific metadata keys and calculates derived parameters.
 * 
 * Required metadata keys:
 * - general.architecture (must be "llama")
 * - llama.context_length
 * - llama.embedding_length
 * - llama.block_count
 * - llama.attention.head_count
 * - llama.attention.head_count_kv
 * - llama.feed_forward_length
 * - tokenizer.ggml.tokens (for vocab_size)
 * 
 * Optional metadata keys:
 * - llama.rope.dimension_count (default: head_dim)
 * - llama.rope.freq_base (default: 10000.0)
 * - llama.rope.scale (optional)
 * - llama.rope.scaling.type (optional)
 * 
 * @param metadata Vector of GGUF metadata key-value pairs
 * @return Parsed Llama configuration
 * @throws CudaError if required keys missing or invalid
 */
LlamaConfig parse_llama_metadata(const std::vector<GGUFMetadata>& metadata);

/**
 * Get metadata value by key (helper function)
 * 
 * @param metadata Vector of metadata key-value pairs
 * @param key Metadata key to find
 * @return Pointer to metadata value, or nullptr if not found
 */
const GGUFMetadata* find_metadata(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key
);

/**
 * Get required uint32 metadata value
 * 
 * @param metadata Vector of metadata
 * @param key Metadata key
 * @return Value as uint32
 * @throws CudaError if key not found or wrong type
 */
uint32_t get_required_uint32(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key
);

/**
 * Get optional uint32 metadata value
 * 
 * @param metadata Vector of metadata
 * @param key Metadata key
 * @param default_value Default if key not found
 * @return Value as uint32
 */
uint32_t get_optional_uint32(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key,
    uint32_t default_value
);

/**
 * Get required float metadata value
 * 
 * @param metadata Vector of metadata
 * @param key Metadata key
 * @return Value as float
 * @throws CudaError if key not found or wrong type
 */
float get_required_float(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key
);

/**
 * Get optional float metadata value
 * 
 * @param metadata Vector of metadata
 * @param key Metadata key
 * @param default_value Default if key not found
 * @return Value as float
 */
float get_optional_float(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key,
    float default_value
);

/**
 * Get required string metadata value
 * 
 * @param metadata Vector of metadata
 * @param key Metadata key
 * @return Value as string
 * @throws CudaError if key not found or wrong type
 */
std::string get_required_string(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key
);

/**
 * Get array length from metadata
 * 
 * Used for vocab_size calculation from tokenizer.ggml.tokens array.
 * 
 * @param metadata Vector of metadata
 * @param key Metadata key
 * @return Array length
 * @throws CudaError if key not found or not an array
 */
uint32_t get_array_length(
    const std::vector<GGUFMetadata>& metadata,
    const std::string& key
);

} // namespace gguf
} // namespace worker

#endif // WORKER_GGUF_LLAMA_METADATA_H

// ---
// Implemented by Llama-Beta 🦙
