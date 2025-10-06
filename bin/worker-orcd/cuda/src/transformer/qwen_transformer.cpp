#include "qwen_transformer.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cstring>

// Debug: Track number of calls per layer for verbose logging
static int layer_call_count[256] = {0};

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

    void cuda_rope_forward_ex(
        void* q,
        void* k,
        uint32_t batch_size,
        uint32_t num_heads,
        uint32_t num_kv_heads,
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
    
    // Allocate intermediate buffers
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
    fprintf(stderr, "   Vocab: %u, Layers: %u, Hidden: %u, Heads: %u, KV Heads: %u\n",
            config.vocab_size, config.num_layers, config.hidden_dim, config.num_heads, config.num_kv_heads);
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
    cuda_embedding_lookup(
        token_ids,
        model_->weights.token_embd,
        output,
        batch_size,
        config_.vocab_size,
        config_.hidden_dim,
        nullptr
    );
}

void QwenTransformer::forward_layer(
    uint32_t layer_idx,
    void* input,
    void* output,
    uint32_t batch_size,
    uint32_t pos
) {
    auto& layer = model_->weights.layers[layer_idx];
    
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
    
    // 2. Q, K, V projections
    const half* normed_half = reinterpret_cast<const half*>(normed_);
    half* q_half = reinterpret_cast<half*>(q_proj_);
    half* k_half = reinterpret_cast<half*>(k_proj_);
    half* v_half = reinterpret_cast<half*>(v_proj_);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    uint32_t q_dim = config_.num_heads * config_.head_dim;
    cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, q_dim, batch_size, config_.hidden_dim, &alpha, layer.attn_q_weight, CUDA_R_16F, q_dim, normed_half, CUDA_R_16F, config_.hidden_dim, &beta, q_half, CUDA_R_16F, q_dim, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    uint32_t kv_dim = config_.num_kv_heads * config_.head_dim;
    cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, kv_dim, batch_size, config_.hidden_dim, &alpha, layer.attn_k_weight, CUDA_R_16F, kv_dim, normed_half, CUDA_R_16F, config_.hidden_dim, &beta, k_half, CUDA_R_16F, kv_dim, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, kv_dim, batch_size, config_.hidden_dim, &alpha, layer.attn_v_weight, CUDA_R_16F, kv_dim, normed_half, CUDA_R_16F, config_.hidden_dim, &beta, v_half, CUDA_R_16F, kv_dim, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    // 3. Apply RoPE
    cuda_rope_forward_ex(q_proj_, k_proj_, batch_size, config_.num_heads, config_.num_kv_heads, config_.head_dim, pos, config_.rope_freq_base, nullptr);
    
    // 4. GQA Attention
    size_t layer_cache_offset = layer_idx * 1 * config_.num_kv_heads * config_.context_length * config_.head_dim;
    half* layer_k_cache = reinterpret_cast<half*>(kv_cache_.k_cache) + layer_cache_offset;
    half* layer_v_cache = reinterpret_cast<half*>(kv_cache_.v_cache) + layer_cache_offset;
    
    cuda_gqa_attention_forward(q_proj_, k_proj_, v_proj_, layer_k_cache, layer_v_cache, attn_output_, batch_size, config_.num_heads, config_.num_kv_heads, config_.head_dim, 1, pos, config_.context_length, nullptr);
    
    // 5. Attention output projection
    half* attn_out_half = reinterpret_cast<half*>(attn_output_);
    half* ffn_out_half = reinterpret_cast<half*>(ffn_output_);
    cublasGemmEx(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, config_.hidden_dim, batch_size, q_dim, &alpha, layer.attn_output, CUDA_R_16F, config_.hidden_dim, attn_out_half, CUDA_R_16F, q_dim, &beta, ffn_out_half, CUDA_R_16F, config_.hidden_dim, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaMemcpy(attn_output_, ffn_output_, config_.hidden_dim * sizeof(half), cudaMemcpyDeviceToDevice);
    
    // 6. Residual connection
    cuda_residual_add(input, attn_output_, residual_, batch_size, config_.hidden_dim, nullptr);
    
    // 7. FFN RMSNorm
    cuda_rmsnorm_forward(residual_, layer.ffn_norm, normed_, batch_size, config_.hidden_dim, 1e-6f, nullptr);
    
    // 8. SwiGLU FFN
    cuda_swiglu_forward(normed_, layer.ffn_gate, layer.ffn_up, layer.ffn_down, ffn_output_, batch_size, config_.hidden_dim, config_.ffn_dim, nullptr);
    
    // 9. Final residual
    cuda_residual_add(residual_, ffn_output_, output, batch_size, config_.hidden_dim, nullptr);
}

void QwenTransformer::project_to_vocab(
    const void* hidden_states,
    uint32_t batch_size,
    float* logits
) {
    if (!model_->weights.lm_head) {
        fprintf(stderr, "❌ lm_head is NULL!\n");
        cudaMemset(logits, 0, batch_size * config_.vocab_size * sizeof(float));
        return;
    }
    
    const half* hidden_half = reinterpret_cast<const half*>(hidden_states);
    const half* lm_head_half = reinterpret_cast<const half*>(model_->weights.lm_head);
    
    static bool first_call = true;
    if (first_call) {
        // --- RIGOROUS DEBUGGING (2025-10-06) ---
        int m = config_.vocab_size;
        int n = batch_size;
        int k = config_.hidden_dim;

        fprintf(stderr, "\n--- cublasGemmEx PARAMETERS ---\n");
        fprintf(stderr, "  m: %d, n: %d, k: %d\n", m, n, k);

        // Verify lm_head memory layout (assuming row-major [vocab_size, hidden_dim])
        half h_lm_head_sample[4];
        cudaMemcpy(h_lm_head_sample, lm_head_half, sizeof(half), cudaMemcpyDeviceToHost); // val at [0][0]
        cudaMemcpy(h_lm_head_sample + 1, lm_head_half + 1, sizeof(half), cudaMemcpyDeviceToHost); // val at [0][1]
        cudaMemcpy(h_lm_head_sample + 2, lm_head_half + k, sizeof(half), cudaMemcpyDeviceToHost); // val at [1][0]
        cudaMemcpy(h_lm_head_sample + 3, lm_head_half + k + 1, sizeof(half), cudaMemcpyDeviceToHost); // val at [1][1]

        fprintf(stderr, "\n--- lm_head MEMORY LAYOUT ---\n");
        fprintf(stderr, "  Assuming row-major [vocab_size, hidden_dim] = [%d, %d]\n", m, k);
        fprintf(stderr, "  lm_head[0][0]: %.4f\n", __half2float(h_lm_head_sample[0]));
        fprintf(stderr, "  lm_head[0][1]: %.4f\n", __half2float(h_lm_head_sample[1]));
        fprintf(stderr, "  lm_head[1][0]: %.4f\n", __half2float(h_lm_head_sample[2]));
        fprintf(stderr, "  lm_head[1][1]: %.4f\n", __half2float(h_lm_head_sample[3]));
        fprintf(stderr, "------------------------------\n\n");
        // --- END DEBUGGING ---
    }
    
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        config_.vocab_size,
        batch_size,
        config_.hidden_dim,
        &alpha,
        lm_head_half, CUDA_R_16F, config_.vocab_size,
        hidden_half, CUDA_R_16F, config_.hidden_dim,
        &beta,
        logits, CUDA_R_32F, config_.vocab_size,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "❌ cuBLAS GEMM failed with status: %d\n", status);
        return;
    }
    
    if (first_call) {
        first_call = false;
    }
}

void QwenTransformer::forward(
    const uint32_t* token_ids,
    uint32_t batch_size,
    float* output_logits
) {
    uint32_t pos;
    cudaMemcpy(&pos, kv_cache_.seq_lens, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    embed_tokens(token_ids, batch_size, hidden_states_);
    
    void* layer_input = hidden_states_;
    void* layer_output = residual_;
    
    for (uint32_t i = 0; i < config_.num_layers; i++) {
        forward_layer(i, layer_input, layer_output, batch_size, pos);
        
        void* temp = layer_input;
        layer_input = layer_output;
        layer_output = temp;
    }
    
    cuda_rmsnorm_forward(
        layer_input,
        model_->weights.output_norm,
        normed_,
        batch_size,
        config_.hidden_dim,
        1e-6f,
        nullptr
    );
    
    project_to_vocab(normed_, batch_size, output_logits);
    
    pos++;
    cudaMemcpy(kv_cache_.seq_lens, &pos, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
}

} // namespace transformer
} // namespace worker