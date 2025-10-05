#ifndef WORKER_TRANSFORMER_QWEN_TRANSFORMER_H
#define WORKER_TRANSFORMER_QWEN_TRANSFORMER_H

#include "../model/qwen_weight_loader.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace worker {
namespace transformer {

struct TransformerConfig {
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t ffn_dim;
    uint32_t context_length;
    float rope_freq_base;
};

struct KVCache {
    void* k_cache;  // [num_layers, max_seq_len, hidden_dim]
    void* v_cache;  // [num_layers, max_seq_len, hidden_dim]
    uint32_t* seq_lens;  // [batch_size] - current sequence length per batch
    uint32_t max_seq_len;
    uint32_t num_layers;
    uint32_t hidden_dim;
};

class QwenTransformer {
public:
    QwenTransformer(
        const model::QwenModel* model,
        const TransformerConfig& config
    );
    
    ~QwenTransformer();
    
    // Forward pass for a single token
    void forward(
        const uint32_t* token_ids,  // [batch_size]
        uint32_t batch_size,
        float* output_logits        // [batch_size, vocab_size]
    );
    
    // Reset KV cache
    void reset_cache();
    
private:
    const model::QwenModel* model_;
    TransformerConfig config_;
    KVCache kv_cache_;
    
    // Intermediate buffers
    void* hidden_states_;      // [batch_size, hidden_dim]
    void* residual_;           // [batch_size, hidden_dim]
    void* attn_output_;        // [batch_size, hidden_dim]
    void* ffn_output_;         // [batch_size, hidden_dim]
    void* normed_;             // [batch_size, hidden_dim]
    
    // QKV projection buffers
    void* q_proj_;             // [batch_size, num_heads * head_dim]
    void* k_proj_;             // [batch_size, num_kv_heads * head_dim]
    void* v_proj_;             // [batch_size, num_kv_heads * head_dim]
    
    // cuBLAS handle
    cublasHandle_t cublas_handle_;
    
    // Layer-specific forward
    void forward_layer(
        uint32_t layer_idx,
        void* input,
        void* output,
        uint32_t batch_size,
        uint32_t pos
    );
    
    // Embedding lookup
    void embed_tokens(
        const uint32_t* token_ids,
        uint32_t batch_size,
        void* output
    );
    
    // LM head projection
    void project_to_vocab(
        const void* hidden_states,
        uint32_t batch_size,
        float* logits
    );
};

} // namespace transformer
} // namespace worker

#endif
