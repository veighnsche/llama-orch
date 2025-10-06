#ifndef WORKER_TRANSFORMER_QWEN_TRANSFORMER_H
#define WORKER_TRANSFORMER_QWEN_TRANSFORMER_H

#include "../model/qwen_weight_loader.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace worker {
namespace transformer {

// ============================================================================
// [TEAM_HOTEL] CRITICAL UNDERSTANDING: vocab_size vs padded_vocab_size
// ============================================================================
//
// The output.weight (lm_head) tensor in GGUF has dimensions [896, 151936]:
//   - dimensions[0] = 896 = hidden_dim (input to matrix multiply)
//   - dimensions[1] = 151936 = padded_vocab_size (output, includes padding)
//
// The tokenizer metadata has vocab_size = 151643 (logical valid tokens)
//
// THREE CRITICAL VALUES:
//   1. vocab_size = 151643 (logical) - Use for argmax to skip 293 padding tokens
//   2. padded_vocab_size = 151936 (physical) - Use for cuBLAS stride and buffer size
//   3. hidden_dim = 896 - Use for input dimension
//
// USAGE:
//   - cuBLAS: m=padded_vocab_size, lda=padded_vocab_size, ldc=padded_vocab_size
//   - Buffer allocation: padded_vocab_size * sizeof(float)
//   - Argmax: Only scan first vocab_size positions (skip padding)
//
struct TransformerConfig {
    uint32_t vocab_size;         // Logical vocab size (actual tokens, e.g., 151643)
    uint32_t padded_vocab_size;  // Physical storage size (padded, e.g., 151936)
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
