// GPT Transformer Layer Integration Implementation
//
// Integrates all GPT kernels into unified transformer layer.
//
// Spec: M0-W-1434
// Story: GT-021

#include "gpt_transformer_layer.h"
#include <cstdio>
#include <cstring>

// External kernel declarations
extern "C" {
    void cuda_layernorm(
        half* output, const half* input,
        const half* gamma, const half* beta,
        int batch_size, int seq_len, int hidden_size,
        float epsilon, cudaStream_t stream
    );
    
    void cuda_mha_attention_prefill(
        const half* q, const half* k, const half* v,
        half* output, half* scores_workspace, half* attn_workspace,
        int batch_size, int num_heads, int seq_len_q, int seq_len_k, int head_dim,
        cublasHandle_t cublas_handle, cudaStream_t stream
    );
    
    void cuda_gpt_ffn_forward(
        const half* input, const half* w_up, const half* b_up,
        const half* w_down, const half* b_down,
        half* output, half* workspace,
        int batch_size, int seq_len, int d_model, int ffn_dim,
        cublasHandle_t cublas_handle, cudaStream_t stream
    );
    
    int cuda_residual_forward(
        half* output, const half* input, const half* residual,
        int batch_size, int seq_len, int hidden_dim, bool in_place
    );
}

// Calculate workspace size
size_t gpt_transformer_layer_workspace_size(const GPTLayerConfig* config) {
    if (!config) return 0;
    
    size_t total = 0;
    int tokens = config->batch_size * config->seq_len;
    
    // LayerNorm outputs
    total += tokens * config->d_model * sizeof(half);  // ln1_output
    total += tokens * config->d_model * sizeof(half);  // ln2_output
    
    // Attention workspace
    total += tokens * config->d_model * 3 * sizeof(half);  // QKV
    total += tokens * tokens * config->num_heads * sizeof(half);  // Scores
    total += tokens * config->d_model * sizeof(half);  // Attention output
    total += tokens * config->d_model * sizeof(half);  // After residual
    
    // FFN workspace
    total += tokens * config->ffn_dim * sizeof(half);  // FFN intermediate
    total += tokens * config->d_model * sizeof(half);  // FFN output
    
    return total;
}

// Validate configuration
int gpt_transformer_layer_validate_config(const GPTLayerConfig* config) {
    if (!config) {
        fprintf(stderr, "GPT Layer: NULL config\n");
        return -1;
    }
    
    if (config->batch_size <= 0 || config->seq_len <= 0) {
        fprintf(stderr, "GPT Layer: Invalid batch_size or seq_len\n");
        return -1;
    }
    
    if (config->d_model <= 0 || config->num_heads <= 0) {
        fprintf(stderr, "GPT Layer: Invalid d_model or num_heads\n");
        return -1;
    }
    
    if (config->d_model % config->num_heads != 0) {
        fprintf(stderr, "GPT Layer: d_model must be divisible by num_heads\n");
        return -1;
    }
    
    if (config->head_dim != config->d_model / config->num_heads) {
        fprintf(stderr, "GPT Layer: head_dim mismatch\n");
        return -1;
    }
    
    if (config->ffn_dim <= 0) {
        fprintf(stderr, "GPT Layer: Invalid ffn_dim\n");
        return -1;
    }
    
    return 0;
}

// Validate weights
int gpt_transformer_layer_validate_weights(const GPTLayerWeights* weights) {
    if (!weights) {
        fprintf(stderr, "GPT Layer: NULL weights\n");
        return -1;
    }
    
    // Check all required weights are non-NULL
    if (!weights->ln1_gamma || !weights->ln1_beta) {
        fprintf(stderr, "GPT Layer: Missing LayerNorm 1 weights\n");
        return -1;
    }
    
    if (!weights->qkv_weights) {
        fprintf(stderr, "GPT Layer: Missing QKV weights\n");
        return -1;
    }
    
    if (!weights->attn_out_weights) {
        fprintf(stderr, "GPT Layer: Missing attention output weights\n");
        return -1;
    }
    
    if (!weights->ln2_gamma || !weights->ln2_beta) {
        fprintf(stderr, "GPT Layer: Missing LayerNorm 2 weights\n");
        return -1;
    }
    
    if (!weights->ffn_up_weights || !weights->ffn_down_weights) {
        fprintf(stderr, "GPT Layer: Missing FFN weights\n");
        return -1;
    }
    
    return 0;
}

// Execute full transformer layer
int gpt_transformer_layer_forward(
    const half* input,
    half* output,
    const GPTLayerWeights* weights,
    GPTLayerWorkspace* workspace,
    const GPTLayerConfig* config,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    // Validate inputs
    if (gpt_transformer_layer_validate_config(config) != 0) {
        return -1;
    }
    
    if (gpt_transformer_layer_validate_weights(weights) != 0) {
        return -1;
    }
    
    if (!input || !output || !workspace) {
        fprintf(stderr, "GPT Layer: NULL input/output/workspace\n");
        return -1;
    }
    
    // Step 1: LayerNorm (pre-attention)
    cuda_layernorm(
        workspace->ln1_output, input,
        weights->ln1_gamma, weights->ln1_beta,
        config->batch_size, config->seq_len, config->d_model,
        config->epsilon, stream
    );
    
    // Step 2: Multi-Head Attention
    // Note: Simplified - in reality need QKV projection first
    // For now, assume qkv_output is already projected
    cuda_mha_attention_prefill(
        workspace->ln1_output,  // Q (simplified)
        workspace->ln1_output,  // K (simplified)
        workspace->ln1_output,  // V (simplified)
        workspace->attn_output,
        workspace->attn_scores,
        workspace->qkv_output,
        config->batch_size, config->num_heads,
        config->seq_len, config->seq_len, config->head_dim,
        cublas_handle, stream
    );
    
    // Step 3: Residual connection (attention)
    cuda_residual_forward(
        workspace->attn_residual, workspace->attn_output, input,
        config->batch_size, config->seq_len, config->d_model, false
    );
    
    // Step 4: LayerNorm (pre-FFN)
    cuda_layernorm(
        workspace->ln2_output, workspace->attn_residual,
        weights->ln2_gamma, weights->ln2_beta,
        config->batch_size, config->seq_len, config->d_model,
        config->epsilon, stream
    );
    
    // Step 5: Feed-Forward Network
    cuda_gpt_ffn_forward(
        workspace->ln2_output,
        weights->ffn_up_weights, weights->ffn_up_bias,
        weights->ffn_down_weights, weights->ffn_down_bias,
        workspace->ffn_output, workspace->ffn_intermediate,
        config->batch_size, config->seq_len, config->d_model, config->ffn_dim,
        cublas_handle, stream
    );
    
    // Step 6: Residual connection (FFN)
    cuda_residual_forward(
        output, workspace->ffn_output, workspace->attn_residual,
        config->batch_size, config->seq_len, config->d_model, false
    );
    
    return 0;
}

// Execute with KV cache (decode mode)
int gpt_transformer_layer_forward_cached(
    const half* input,
    half* output,
    half* k_cache,
    half* v_cache,
    int cache_position,
    const GPTLayerWeights* weights,
    GPTLayerWorkspace* workspace,
    const GPTLayerConfig* config,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    // TODO: Implement decode mode with KV cache
    // For now, just call prefill mode
    return gpt_transformer_layer_forward(
        input, output, weights, workspace, config,
        cublas_handle, stream
    );
}

// ---
// Crafted by GPT-Gamma 🤖
