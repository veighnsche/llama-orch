/**
 * GPT Model Forward Pass Implementation
 * 
 * Implements GPT forward pass for prefill and decode modes.
 * Integrates all GPT kernels and manages KV cache.
 * 
 * Spec: M0-W-1434
 * Story: GT-026
 */

#include "gpt_model.h"
#include "gpt_transformer_layer.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

// External kernel declarations
extern "C" {
    void cuda_embedding_lookup(
        const void* embedding_table,
        const uint32_t* token_ids,
        half* output,
        int batch_size,
        int seq_len,
        int embedding_dim,
        int vocab_size,
        cudaStream_t stream
    );
    
    void cuda_positional_embedding_absolute(
        const half* pos_embeddings,
        half* output,
        int batch_size,
        int seq_len,
        int hidden_dim,
        int start_pos,
        cudaStream_t stream
    );
    
    void cuda_layernorm(
        half* output,
        const half* input,
        const half* gamma,
        const half* beta,
        int batch_size,
        int seq_len,
        int hidden_size,
        float epsilon,
        cudaStream_t stream
    );
    
    uint32_t cuda_sample_token(
        const half* logits,
        int vocab_size,
        float temperature,
        float top_p,
        int top_k,
        uint64_t seed,
        cudaStream_t stream
    );
}

namespace worker {
namespace model {

// ============================================================================
// GPTModel Implementation
// ============================================================================

GPTModel::GPTModel(
    std::unique_ptr<GPTModelWeights> weights,
    cublasHandle_t cublas_handle
)
    : weights_(std::move(weights))
    , cublas_handle_(cublas_handle)
    , stream_(nullptr)
    , workspace_(nullptr)
    , workspace_size_(0)
    , current_position_(0)
{
    // Create CUDA stream
    cudaStreamCreate(&stream_);
    
    // Create KV cache
    kv_cache_ = std::make_unique<KVCache>(
        weights_->config.num_layers,
        weights_->config.max_seq_len,
        weights_->config.hidden_dim,
        1  // batch_size = 1 for M0
    );
    
    // Allocate workspace
    allocate_workspace();
}

GPTModel::~GPTModel() {
    free_workspace();
    
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void GPTModel::allocate_workspace() {
    const GPTConfig& cfg = weights_->config;
    
    // Calculate workspace size
    // Need buffers for:
    // - Embeddings: max_seq_len * hidden_dim
    // - Layer outputs: max_seq_len * hidden_dim
    // - Logits: vocab_size
    // - Transformer layer workspace (per layer)
    
    size_t embedding_buffer = cfg.max_seq_len * cfg.hidden_dim * sizeof(half);
    size_t layer_buffer = cfg.max_seq_len * cfg.hidden_dim * sizeof(half);
    size_t logits_buffer = cfg.vocab_size * sizeof(half);
    
    // Transformer layer workspace
    GPTLayerConfig layer_config;
    layer_config.batch_size = 1;
    layer_config.seq_len = cfg.max_seq_len;
    layer_config.d_model = cfg.hidden_dim;
    layer_config.num_heads = cfg.num_heads;
    layer_config.head_dim = cfg.head_dim;
    layer_config.ffn_dim = cfg.ffn_dim;
    layer_config.epsilon = 1e-5f;
    
    size_t transformer_workspace = gpt_transformer_layer_workspace_size(&layer_config);
    
    workspace_size_ = embedding_buffer + layer_buffer + logits_buffer + transformer_workspace;
    
    // Allocate
    cudaError_t err = cudaMalloc(&workspace_, workspace_size_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate workspace: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    // Zero initialize
    cudaMemset(workspace_, 0, workspace_size_);
}

void GPTModel::free_workspace() {
    if (workspace_) {
        cudaFree(workspace_);
        workspace_ = nullptr;
    }
}

uint32_t GPTModel::prefill(
    const uint32_t* input_ids,
    int seq_len,
    const GPTForwardConfig& config
) {
    if (seq_len <= 0 || seq_len > weights_->config.max_seq_len) {
        throw std::runtime_error("Invalid sequence length");
    }
    
    // Reset cache
    reset_cache();
    current_position_ = 0;
    
    // Allocate buffers from workspace
    half* embeddings = workspace_;
    half* layer_input = embeddings + (seq_len * weights_->config.hidden_dim);
    half* layer_output = layer_input + (seq_len * weights_->config.hidden_dim);
    half* logits = layer_output + (seq_len * weights_->config.hidden_dim);
    
    // Apply embeddings (token + position)
    apply_embeddings(input_ids, seq_len, embeddings);
    
    // Copy embeddings to layer input
    cudaMemcpy(layer_input, embeddings, 
               seq_len * weights_->config.hidden_dim * sizeof(half),
               cudaMemcpyDeviceToDevice);
    
    // Execute transformer layers
    for (int i = 0; i < weights_->config.num_layers; ++i) {
        execute_layer(i, layer_input, layer_output, true);
        
        // Swap buffers
        half* tmp = layer_input;
        layer_input = layer_output;
        layer_output = tmp;
    }
    
    // Apply output head (final LayerNorm + LM head)
    // Use last token for next token prediction
    half* last_token_hidden = layer_input + ((seq_len - 1) * weights_->config.hidden_dim);
    apply_output_head(last_token_hidden, logits);
    
    // Sample token
    uint32_t token = sample_token(logits, config);
    
    // Update position
    current_position_ = seq_len;
    
    return token;
}

uint32_t GPTModel::decode(
    uint32_t input_id,
    const GPTForwardConfig& config
) {
    // Allocate buffers from workspace
    half* embeddings = workspace_;
    half* layer_input = embeddings + weights_->config.hidden_dim;
    half* layer_output = layer_input + weights_->config.hidden_dim;
    half* logits = layer_output + weights_->config.hidden_dim;
    
    // Apply embeddings for single token
    apply_embeddings(&input_id, 1, embeddings);
    
    // Copy to layer input
    cudaMemcpy(layer_input, embeddings,
               weights_->config.hidden_dim * sizeof(half),
               cudaMemcpyDeviceToDevice);
    
    // Execute transformer layers (decode mode with KV cache)
    for (int i = 0; i < weights_->config.num_layers; ++i) {
        execute_layer(i, layer_input, layer_output, false);
        
        // Swap buffers
        half* tmp = layer_input;
        layer_input = layer_output;
        layer_output = tmp;
    }
    
    // Apply output head
    apply_output_head(layer_input, logits);
    
    // Sample token
    uint32_t token = sample_token(logits, config);
    
    // Update position
    current_position_++;
    
    return token;
}

void GPTModel::reset_cache() {
    if (kv_cache_) {
        kv_cache_->reset();
    }
    current_position_ = 0;
}

void GPTModel::execute_layer(
    int layer_idx,
    const half* input,
    half* output,
    bool is_prefill
) {
    // TODO: Implement actual layer execution
    // For now, just copy input to output (stub)
    
    const GPTConfig& cfg = weights_->config;
    int seq_len = is_prefill ? current_position_ : 1;
    
    cudaMemcpy(output, input,
               seq_len * cfg.hidden_dim * sizeof(half),
               cudaMemcpyDeviceToDevice);
    
    // In real implementation:
    // 1. Pre-attention LayerNorm
    // 2. Multi-Head Attention (with KV cache for decode)
    // 3. Residual connection
    // 4. Pre-FFN LayerNorm
    // 5. Feed-Forward Network
    // 6. Residual connection
}

void GPTModel::apply_embeddings(
    const uint32_t* token_ids,
    int seq_len,
    half* output
) {
    const GPTConfig& cfg = weights_->config;
    
    // Token embeddings
    cuda_embedding_lookup(
        weights_->token_embeddings,
        token_ids,
        output,
        1,  // batch_size
        seq_len,
        cfg.hidden_dim,
        cfg.vocab_size,
        stream_
    );
    
    // Add positional embeddings
    cuda_positional_embedding_absolute(
        static_cast<const half*>(weights_->position_embeddings),
        output,
        1,  // batch_size
        seq_len,
        cfg.hidden_dim,
        current_position_,
        stream_
    );
    
    cudaStreamSynchronize(stream_);
}

void GPTModel::apply_output_head(
    const half* input,
    half* logits
) {
    const GPTConfig& cfg = weights_->config;
    
    // Allocate temp buffer for normalized output
    half* normalized = workspace_ + (cfg.max_seq_len * cfg.hidden_dim * 2);
    
    // Apply final LayerNorm
    cuda_layernorm(
        normalized,
        input,
        weights_->output_norm_weight,
        weights_->output_norm_bias,
        1,  // batch_size
        1,  // seq_len (single token)
        cfg.hidden_dim,
        1e-5f,
        stream_
    );
    
    // TODO: Apply LM head (GEMM: normalized @ lm_head_weight)
    // For now, just copy (stub)
    cudaMemcpy(logits, normalized,
               cfg.hidden_dim * sizeof(half),
               cudaMemcpyDeviceToDevice);
    
    cudaStreamSynchronize(stream_);
}

uint32_t GPTModel::sample_token(
    const half* logits,
    const GPTForwardConfig& config
) {
    // Sample token using CUDA sampling kernel
    return cuda_sample_token(
        logits,
        weights_->config.vocab_size,
        config.temperature,
        config.top_p,
        config.top_k,
        config.seed,
        stream_
    );
}

// ============================================================================
// GPTModelFactory Implementation
// ============================================================================

std::unique_ptr<GPTModel> GPTModelFactory::load_from_gguf(
    const std::string& path,
    cublasHandle_t cublas_handle
) {
    // Load weights
    auto weights = GPTWeightLoader::load_from_gguf(path);
    
    // Create model
    return std::make_unique<GPTModel>(std::move(weights), cublas_handle);
}

GPTConfig GPTModelFactory::validate_gguf(const std::string& path) {
    return GPTWeightLoader::validate_gguf(path);
}

} // namespace model
} // namespace worker

// ---
// Crafted by GPT-Gamma 🤖
