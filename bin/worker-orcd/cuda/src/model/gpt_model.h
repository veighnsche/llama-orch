/**
 * GPT Model Forward Pass
 * 
 * Implements GPT forward pass for prefill and decode modes.
 * Integrates all GPT kernels and manages KV cache.
 * 
 * Spec: M0-W-1434
 * Story: GT-026
 */

#ifndef WORKER_MODEL_GPT_MODEL_H
#define WORKER_MODEL_GPT_MODEL_H

#include "gpt_weights.h"
#include "../kv_cache.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>

namespace worker {
namespace model {

/**
 * GPT forward pass configuration
 */
struct GPTForwardConfig {
    bool is_prefill;        // True for prefill, false for decode
    int batch_size;         // Always 1 for M0
    int seq_len;            // Sequence length
    int cache_len;          // KV cache length
    float temperature;      // Sampling temperature
    uint64_t seed;          // Random seed
    
    // Advanced sampling parameters (M0 Sprint 3)
    float top_p;            // Nucleus sampling (1.0 = disabled)
    int top_k;              // Top-k sampling (0 = disabled)
    float repetition_penalty;  // Repetition penalty (1.0 = disabled)
};

/**
 * GPT model instance
 */
class GPTModel {
public:
    /**
     * Create GPT model from loaded weights
     * 
     * @param weights Model weights (ownership transferred)
     * @param cublas_handle cuBLAS handle for GEMM operations
     */
    GPTModel(
        std::unique_ptr<GPTModelWeights> weights,
        cublasHandle_t cublas_handle
    );
    
    ~GPTModel();
    
    // Non-copyable
    GPTModel(const GPTModel&) = delete;
    GPTModel& operator=(const GPTModel&) = delete;
    
    /**
     * Prefill: Process full prompt
     * 
     * @param input_ids Token IDs [seq_len]
     * @param seq_len Sequence length
     * @param config Forward pass configuration
     * @return Sampled token ID
     */
    uint32_t prefill(
        const uint32_t* input_ids,
        int seq_len,
        const GPTForwardConfig& config
    );
    
    /**
     * Decode: Generate single token
     * 
     * @param input_id Previous token ID
     * @param config Forward pass configuration
     * @return Sampled token ID
     */
    uint32_t decode(
        uint32_t input_id,
        const GPTForwardConfig& config
    );
    
    /**
     * Reset KV cache
     */
    void reset_cache();
    
    /**
     * Get model configuration
     */
    const GPTConfig& config() const { return weights_->config; }
    
    /**
     * Get total VRAM usage
     */
    size_t vram_usage() const { return weights_->total_vram_bytes; }
    
private:
    std::unique_ptr<GPTModelWeights> weights_;
    cublasHandle_t cublas_handle_;
    cudaStream_t stream_;
    
    // KV cache
    std::unique_ptr<KVCache> kv_cache_;
    
    // Workspace buffers
    half* workspace_;
    size_t workspace_size_;
    
    // Current position in sequence
    int current_position_;
    
    /**
     * Allocate workspace buffers
     */
    void allocate_workspace();
    
    /**
     * Free workspace buffers
     */
    void free_workspace();
    
    /**
     * Execute single transformer layer
     */
    void execute_layer(
        int layer_idx,
        const half* input,
        half* output,
        bool is_prefill
    );
    
    /**
     * Apply embeddings (token + position)
     */
    void apply_embeddings(
        const uint32_t* token_ids,
        int seq_len,
        half* output
    );
    
    /**
     * Apply final LayerNorm and LM head
     */
    void apply_output_head(
        const half* input,
        half* logits
    );
    
    /**
     * Sample token from logits
     */
    uint32_t sample_token(
        const half* logits,
        const GPTForwardConfig& config
    );
};

/**
 * GPT model factory
 */
class GPTModelFactory {
public:
    /**
     * Load GPT model from GGUF file
     * 
     * @param path Path to GGUF file
     * @param cublas_handle cuBLAS handle
     * @return Loaded model instance
     */
    static std::unique_ptr<GPTModel> load_from_gguf(
        const std::string& path,
        cublasHandle_t cublas_handle
    );
    
    /**
     * Validate GGUF file before loading
     * 
     * @param path Path to GGUF file
     * @return Configuration if valid
     * @throws std::runtime_error if invalid
     */
    static GPTConfig validate_gguf(const std::string& path);
};

} // namespace model
} // namespace worker

#endif // WORKER_MODEL_GPT_MODEL_H

// ---
// Crafted by GPT-Gamma 🤖
