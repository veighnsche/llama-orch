/**
 * GPT Weight Mapping and Loading
 * 
 * Defines structures and functions for loading GPT model weights from GGUF files.
 * Supports Q4_K_M and MXFP4 quantization formats.
 * 
 * Spec: M0-W-1220, M0-W-1212
 * Story: GT-024, GT-025
 */

#ifndef WORKER_MODEL_GPT_WEIGHTS_H
#define WORKER_MODEL_GPT_WEIGHTS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <memory>

namespace worker {
namespace model {

/**
 * GPT model configuration
 */
struct GPTConfig {
    int vocab_size;        // 50257 for GPT-2/GPT-OSS
    int hidden_dim;        // 2048 for GPT-OSS-20B
    int num_layers;        // 44 for GPT-OSS-20B
    int num_heads;         // 64 for GPT-OSS-20B (MHA)
    int head_dim;          // hidden_dim / num_heads
    int ffn_dim;           // 8192 for GPT-OSS-20B
    int max_seq_len;       // 2048 for GPT-OSS-20B
    int context_length;    // 8192 for GPT-OSS-20B
    
    // Quantization
    std::string quant_kind;  // "Q4_K_M", "MXFP4", "Q4_0"
    
    // Validation
    bool validate() const;
};

/**
 * GPT layer weights (per transformer block)
 */
struct GPTLayerWeights {
    // Pre-attention LayerNorm
    half* attn_norm_weight;  // [hidden_dim]
    half* attn_norm_bias;    // [hidden_dim]
    
    // Multi-Head Attention
    void* attn_qkv_weight;   // [hidden_dim, 3*hidden_dim] (quantized)
    half* attn_qkv_bias;     // [3*hidden_dim]
    void* attn_out_weight;   // [hidden_dim, hidden_dim] (quantized)
    half* attn_out_bias;     // [hidden_dim]
    
    // Pre-FFN LayerNorm
    half* ffn_norm_weight;   // [hidden_dim]
    half* ffn_norm_bias;     // [hidden_dim]
    
    // Feed-Forward Network
    void* ffn_up_weight;     // [hidden_dim, ffn_dim] (quantized)
    half* ffn_up_bias;       // [ffn_dim]
    void* ffn_down_weight;   // [ffn_dim, hidden_dim] (quantized)
    half* ffn_down_bias;     // [hidden_dim]
    
    // Memory management
    size_t total_vram_bytes;
    
    GPTLayerWeights();
    ~GPTLayerWeights();
    
    // Non-copyable
    GPTLayerWeights(const GPTLayerWeights&) = delete;
    GPTLayerWeights& operator=(const GPTLayerWeights&) = delete;
};

/**
 * Complete GPT model weights
 */
struct GPTModelWeights {
    GPTConfig config;
    
    // Embeddings
    void* token_embeddings;      // [vocab_size, hidden_dim] (quantized)
    void* position_embeddings;   // [max_seq_len, hidden_dim] (quantized)
    
    // Transformer layers
    std::vector<std::unique_ptr<GPTLayerWeights>> layers;
    
    // Output head
    half* output_norm_weight;    // [hidden_dim]
    half* output_norm_bias;      // [hidden_dim]
    void* lm_head_weight;        // [hidden_dim, vocab_size] (quantized)
    
    // Memory tracking
    size_t total_vram_bytes;
    
    GPTModelWeights();
    ~GPTModelWeights();
    
    // Non-copyable
    GPTModelWeights(const GPTModelWeights&) = delete;
    GPTModelWeights& operator=(const GPTModelWeights&) = delete;
};

/**
 * GGUF tensor information
 */
struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t type;  // GGML type enum
    uint64_t offset;
    size_t size_bytes;
};

/**
 * GPT weight loader
 */
class GPTWeightLoader {
public:
    /**
     * Load GPT model from GGUF file
     * 
     * @param path Path to GGUF file
     * @return Loaded model weights
     * @throws std::runtime_error on failure
     */
    static std::unique_ptr<GPTModelWeights> load_from_gguf(const std::string& path);
    
    /**
     * Validate GGUF file for GPT model
     * 
     * @param path Path to GGUF file
     * @return Configuration if valid
     * @throws std::runtime_error if invalid
     */
    static GPTConfig validate_gguf(const std::string& path);
    
    /**
     * Calculate VRAM requirements
     * 
     * @param config Model configuration
     * @return Total VRAM bytes needed
     */
    static size_t calculate_vram_usage(const GPTConfig& config);
    
    /**
     * Validate tensor shape (public for testing)
     */
    static void validate_tensor_shape(
        const GGUFTensorInfo& tensor,
        const std::vector<uint64_t>& expected_shape,
        const std::string& context
    );
    
private:
    /**
     * Parse GGUF metadata to extract GPT config
     */
    static GPTConfig parse_config_from_gguf(const std::string& path);
    
    /**
     * Load embeddings from GGUF
     */
    static void load_embeddings(
        GPTModelWeights* model,
        const std::string& path,
        const std::vector<GGUFTensorInfo>& tensors
    );
    
    /**
     * Load transformer layer weights
     */
    static void load_layer(
        GPTLayerWeights* layer,
        int layer_idx,
        const std::string& path,
        const std::vector<GGUFTensorInfo>& tensors,
        const GPTConfig& config
    );
    
    /**
     * Load output head weights
     */
    static void load_output_head(
        GPTModelWeights* model,
        const std::string& path,
        const std::vector<GGUFTensorInfo>& tensors
    );
    
    /**
     * Allocate VRAM for tensor
     */
    static void* allocate_and_copy(
        const void* host_data,
        size_t size_bytes,
        const std::string& tensor_name
    );
    
    /**
     * Find tensor by name
     */
    static const GGUFTensorInfo* find_tensor(
        const std::vector<GGUFTensorInfo>& tensors,
        const std::string& name
    );
};

/**
 * GPT weight mapping helpers
 */
class GPTWeightMapper {
public:
    /**
     * Get expected tensor name for component
     */
    static std::string get_tensor_name(
        const std::string& component,
        int layer_idx = -1
    );
    
    /**
     * Get expected shape for tensor
     */
    static std::vector<uint64_t> get_expected_shape(
        const std::string& tensor_name,
        const GPTConfig& config
    );
    
    /**
     * Validate tensor name is valid GPT tensor
     */
    static bool is_valid_gpt_tensor(const std::string& name);
    
    /**
     * Parse layer index from tensor name
     * 
     * @return Layer index, or -1 if not a layer tensor
     */
    static int parse_layer_index(const std::string& name);
};

} // namespace model
} // namespace worker

#endif // WORKER_MODEL_GPT_WEIGHTS_H

// ---
// Crafted by GPT-Gamma 🤖
