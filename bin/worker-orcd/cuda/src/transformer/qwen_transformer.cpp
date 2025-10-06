#include "qwen_transformer.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cstring>

// Forward declarations of CUDA kernels
extern "C" {
    void cuda_rmsnorm_forward(
        const void* input,
        const void* weight,
        void* output,
        uint32_t batch_size,
        uint32_t hidden_dim,
        float eps,
        cudaStream_t stream
    );
    
    void cuda_rope_forward(
        void* q,
        void* k,
        uint32_t batch_size,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t pos,
        float rope_freq_base,
        cudaStream_t stream
    );
    
    void cuda_gqa_attention_forward(
        const void* q,
        const void* k,
        const void* v,
        const void* k_cache,
        const void* v_cache,
        void* output,
        uint32_t batch_size,
        uint32_t num_q_heads,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t seq_len,
        uint32_t cache_len,
        uint32_t max_seq_len,
        cudaStream_t stream
    );
    
    void cuda_swiglu_forward(
        const void* input,
        const void* gate_weight,
        const void* up_weight,
        const void* down_weight,
        void* output,
        uint32_t batch_size,
        uint32_t hidden_dim,
        uint32_t ffn_dim,
        cudaStream_t stream
    );
    
    void cuda_residual_add(
        const void* input,
        const void* residual,
        void* output,
        uint32_t batch_size,
        uint32_t hidden_dim,
        cudaStream_t stream
    );
    
    void cuda_bias_add(
        void* output,
        const void* input,
        const void* bias,
        uint32_t batch_size,
        uint32_t dim,
        cudaStream_t stream
    );
    
    void cuda_embedding_lookup(
        const uint32_t* token_ids,
        const void* embedding_table,
        void* output,
        uint32_t batch_size,
        uint32_t vocab_size,
        uint32_t hidden_dim,
        cudaStream_t stream
    );
}

namespace worker {
namespace transformer {

QwenTransformer::QwenTransformer(
    const model::QwenModel* model,
    const TransformerConfig& config
) : model_(model), config_(config) {
    
    // Allocate KV cache
    // Layout: [num_layers, batch=1, num_kv_heads, context_length, head_dim]
    size_t kv_cache_size = config.num_layers * 1 * config.num_kv_heads * config.context_length * config.head_dim * sizeof(half);
    cudaMalloc(&kv_cache_.k_cache, kv_cache_size);
    cudaMalloc(&kv_cache_.v_cache, kv_cache_size);
    cudaMalloc(&kv_cache_.seq_lens, sizeof(uint32_t));
    
    kv_cache_.max_seq_len = config.context_length;
    kv_cache_.num_layers = config.num_layers;
    kv_cache_.hidden_dim = config.hidden_dim;
    
    // Initialize seq_len to 0
    uint32_t zero = 0;
    cudaMemcpy(kv_cache_.seq_lens, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Allocate intermediate buffers (for batch_size=1)
    size_t hidden_size = config.hidden_dim * sizeof(half);
    cudaMalloc(&hidden_states_, hidden_size);
    cudaMalloc(&residual_, hidden_size);
    cudaMalloc(&attn_output_, hidden_size);
    cudaMalloc(&ffn_output_, hidden_size);
    cudaMalloc(&normed_, hidden_size);
    
    // Allocate QKV projection buffers
    size_t q_size = config.num_heads * config.head_dim * sizeof(half);
    size_t kv_size = config.num_kv_heads * config.head_dim * sizeof(half);
    cudaMalloc(&q_proj_, q_size);
    cudaMalloc(&k_proj_, kv_size);
    cudaMalloc(&v_proj_, kv_size);
    
    // Create cuBLAS handle
    cublasCreate(&cublas_handle_);
    cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH);
    
    fprintf(stderr, "✅ QwenTransformer initialized\n");
    fprintf(stderr, "   Layers: %u, Hidden: %u, Heads: %u, KV Heads: %u\n",
            config.num_layers, config.hidden_dim, config.num_heads, config.num_kv_heads);
}

QwenTransformer::~QwenTransformer() {
    cudaFree(kv_cache_.k_cache);
    cudaFree(kv_cache_.v_cache);
    cudaFree(kv_cache_.seq_lens);
    cudaFree(hidden_states_);
    cudaFree(residual_);
    cudaFree(attn_output_);
    cudaFree(ffn_output_);
    cudaFree(normed_);
    cudaFree(q_proj_);
    cudaFree(k_proj_);
    cudaFree(v_proj_);
    cublasDestroy(cublas_handle_);
}

void QwenTransformer::reset_cache() {
    uint32_t zero = 0;
    cudaMemcpy(kv_cache_.seq_lens, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
}

void QwenTransformer::embed_tokens(
    const uint32_t* token_ids,
    uint32_t batch_size,
    void* output
) {
    if (!model_->weights.token_embd) {
        fprintf(stderr, "❌ token_embd is NULL!\n");
        cudaMemset(output, 0, batch_size * config_.hidden_dim * sizeof(half));
        return;
    }
    
    // Debug: Log parameters being passed
    static int call_count = 0;
    if (call_count == 0) {
        fprintf(stderr, "🔍 [C++] Embedding lookup parameters:\n");
        fprintf(stderr, "   batch_size = %u\n", batch_size);
        fprintf(stderr, "   vocab_size = %u\n", config_.vocab_size);
        fprintf(stderr, "   hidden_dim = %u\n", config_.hidden_dim);
        fprintf(stderr, "   embedding_table = %p\n", model_->weights.token_embd);
        fprintf(stderr, "   output = %p\n", output);
    }
    
    cuda_embedding_lookup(
        token_ids,
        model_->weights.token_embd,
        output,
        batch_size,
        config_.vocab_size,
        config_.hidden_dim,
        nullptr  // default stream
    );
    
    // Debug: Check first few embedding values
    if (call_count < 2) {
        half host_emb[10];
        cudaMemcpy(host_emb, output, 10 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "First 10 embedding values: ");
        for (int i = 0; i < 10; i++) {
            fprintf(stderr, "%.2f ", __half2float(host_emb[i]));
        }
        fprintf(stderr, "\n");
        
        // Also check the embedding table itself
        half host_table[10];
        cudaMemcpy(host_table, model_->weights.token_embd, 10 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "First 10 values from embedding table: ");
        for (int i = 0; i < 10; i++) {
            fprintf(stderr, "%.2f ", __half2float(host_table[i]));
        }
        fprintf(stderr, "\n");
        
        call_count++;
    }
}

void QwenTransformer::forward_layer(
    uint32_t layer_idx,
    void* input,
    void* output,
    uint32_t batch_size,
    uint32_t pos
) {
    auto& layer = model_->weights.layers[layer_idx];
    // Cast input/output for later use if needed
    
    // 1. Attention RMSNorm
    cuda_rmsnorm_forward(
        input,
        layer.attn_norm,
        normed_,
        batch_size,
        config_.hidden_dim,
        1e-6f,
        nullptr
    );
    
    // 2. Q, K, V projections with biases
    const half* normed_half = reinterpret_cast<const half*>(normed_);
    half* q_half = reinterpret_cast<half*>(q_proj_);
    half* k_half = reinterpret_cast<half*>(k_proj_);
    half* v_half = reinterpret_cast<half*>(v_proj_);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Q projection: q = normed @ q_weight^T + q_bias
    uint32_t q_dim = config_.num_heads * config_.head_dim;
    cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        q_dim, batch_size, config_.hidden_dim,
        &alpha,
        layer.attn_q_weight, CUDA_R_16F, config_.hidden_dim,
        normed_half, CUDA_R_16F, config_.hidden_dim,
        &beta,
        q_half, CUDA_R_16F, q_dim,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    // Add Q bias
    cuda_bias_add(q_proj_, q_proj_, layer.attn_q_bias, batch_size, q_dim, nullptr);
    
    // K projection: k = normed @ k_weight^T + k_bias
    uint32_t kv_dim = config_.num_kv_heads * config_.head_dim;
    cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        kv_dim, batch_size, config_.hidden_dim,
        &alpha,
        layer.attn_k_weight, CUDA_R_16F, config_.hidden_dim,
        normed_half, CUDA_R_16F, config_.hidden_dim,
        &beta,
        k_half, CUDA_R_16F, kv_dim,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    // Add K bias
    cuda_bias_add(k_proj_, k_proj_, layer.attn_k_bias, batch_size, kv_dim, nullptr);
    
    // V projection: v = normed @ v_weight^T + v_bias
    cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        kv_dim, batch_size, config_.hidden_dim,
        &alpha,
        layer.attn_v_weight, CUDA_R_16F, config_.hidden_dim,
        normed_half, CUDA_R_16F, config_.hidden_dim,
        &beta,
        v_half, CUDA_R_16F, kv_dim,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    // Add V bias
    cuda_bias_add(v_proj_, v_proj_, layer.attn_v_bias, batch_size, kv_dim, nullptr);
    
    // 3. Apply RoPE to Q and K
    cuda_rope_forward(
        q_proj_,
        k_proj_,
        batch_size,
        config_.num_heads,
        config_.head_dim,
        pos,
        1000000.0f,  // RoPE freq_base for Qwen2.5
        nullptr
    );
    
    // 4. GQA Attention with KV cache
    // Calculate layer-specific cache offset
    // Layout: [layer, batch=1, kv_head, pos, d]
    size_t layer_cache_offset = layer_idx * 1 * config_.num_kv_heads * config_.context_length * config_.head_dim;
    half* layer_k_cache = reinterpret_cast<half*>(kv_cache_.k_cache) + layer_cache_offset;
    half* layer_v_cache = reinterpret_cast<half*>(kv_cache_.v_cache) + layer_cache_offset;
    
    cuda_gqa_attention_forward(
        q_proj_,
        k_proj_,
        v_proj_,
        layer_k_cache,
        layer_v_cache,
        attn_output_,
        batch_size,
        config_.num_heads,
        config_.num_kv_heads,
        config_.head_dim,
        1,    // seq_len = 1 for single token
        pos,  // cache_len = current position
        config_.context_length,  // max_seq_len
        nullptr
    );
    
    // 5. Attention output projection
    // CRITICAL: Use separate buffer for output to avoid in-place corruption
    half* attn_out_half = reinterpret_cast<half*>(attn_output_);
    half* ffn_out_half = reinterpret_cast<half*>(ffn_output_);  // Reuse ffn_output_ as temp buffer
    cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        config_.hidden_dim, batch_size, config_.hidden_dim,
        &alpha,
        layer.attn_output, CUDA_R_16F, config_.hidden_dim,
        attn_out_half, CUDA_R_16F, config_.hidden_dim,
        &beta,
        ffn_out_half, CUDA_R_16F, config_.hidden_dim,  // Write to separate buffer
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    // Copy result back
    cudaMemcpy(attn_output_, ffn_output_, config_.hidden_dim * sizeof(half), cudaMemcpyDeviceToDevice);
    
    // 6. Residual connection
    cuda_residual_add(
        input,
        attn_output_,
        residual_,
        batch_size,
        config_.hidden_dim,
        nullptr
    );
    
    // 7. FFN RMSNorm
    cuda_rmsnorm_forward(
        residual_,
        layer.ffn_norm,
        normed_,
        batch_size,
        config_.hidden_dim,
        1e-6f,
        nullptr
    );
    
    // 8. SwiGLU FFN
    cuda_swiglu_forward(
        normed_,
        layer.ffn_gate,
        layer.ffn_up,
        layer.ffn_down,
        ffn_output_,
        batch_size,
        config_.hidden_dim,
        config_.ffn_dim,
        nullptr
    );
    
    // 9. Final residual
    cuda_residual_add(
        residual_,
        ffn_output_,
        output,
        batch_size,
        config_.hidden_dim,
        nullptr
    );
}

void QwenTransformer::project_to_vocab(
    const void* hidden_states,
    uint32_t batch_size,
    float* logits
) {
    // LM head projection: logits = hidden @ lm_head^T
    // hidden: [batch, hidden_dim] (FP16)
    // lm_head: [vocab_size, hidden_dim] (FP16)
    // logits: [batch, vocab_size] (FP32)
    
    if (!model_->weights.lm_head) {
        fprintf(stderr, "❌ lm_head is NULL!\n");
        // Fill with zeros to avoid NaN
        cudaMemset(logits, 0, batch_size * config_.vocab_size * sizeof(float));
        return;
    }
    
    const half* hidden_half = reinterpret_cast<const half*>(hidden_states);
    const half* lm_head_half = reinterpret_cast<const half*>(model_->weights.lm_head);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Use FP32 output for logits (better numerical stability for sampling)
    cublasStatus_t status = cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        config_.vocab_size, batch_size, config_.hidden_dim,
        &alpha,
        lm_head_half, CUDA_R_16F, config_.hidden_dim,
        hidden_half, CUDA_R_16F, config_.hidden_dim,
        &beta,
        logits, CUDA_R_32F, config_.vocab_size,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "❌ cuBLAS GEMM failed with status: %d\n", status);
    }
}

void QwenTransformer::forward(
    const uint32_t* token_ids,
    uint32_t batch_size,
    float* output_logits
) {
    // Get current position
    uint32_t pos;
    cudaMemcpy(&pos, kv_cache_.seq_lens, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // 1. Embed tokens
    embed_tokens(token_ids, batch_size, hidden_states_);
    
    // 2. Process through all layers
    void* layer_input = hidden_states_;
    void* layer_output = residual_;
    
    for (uint32_t i = 0; i < config_.num_layers; i++) {
        forward_layer(i, layer_input, layer_output, batch_size, pos);
        
        // Swap buffers
        void* temp = layer_input;
        layer_input = layer_output;
        layer_output = temp;
    }
    
    // 3. Final RMSNorm
    cuda_rmsnorm_forward(
        layer_input,
        model_->weights.output_norm,
        normed_,
        batch_size,
        config_.hidden_dim,
        1e-6f,
        nullptr
    );
    
    // 4. Project to vocabulary
    project_to_vocab(normed_, batch_size, output_logits);
    
    // 5. Update position
    pos++;
    cudaMemcpy(kv_cache_.seq_lens, &pos, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
}

} // namespace transformer
} // namespace worker
