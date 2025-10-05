// GPT Transformer Layer Integration
//
// Unified interface for GPT transformer layer execution.
// Integrates: LayerNorm, MHA, FFN, GELU, Residual
//
// Spec: M0-W-1434
// Story: GT-021

#ifndef GPT_TRANSFORMER_LAYER_H
#define GPT_TRANSFORMER_LAYER_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPT Transformer Layer Configuration
typedef struct {
    int batch_size;
    int seq_len;
    int d_model;
    int num_heads;
    int head_dim;
    int ffn_dim;
    float epsilon;  // LayerNorm epsilon
} GPTLayerConfig;

// GPT Transformer Layer Weights
typedef struct {
    // LayerNorm 1 (pre-attention)
    const half* ln1_gamma;
    const half* ln1_beta;
    
    // Multi-Head Attention
    const half* qkv_weights;  // Combined Q/K/V projection
    const half* qkv_bias;
    const half* attn_out_weights;
    const half* attn_out_bias;
    
    // LayerNorm 2 (pre-FFN)
    const half* ln2_gamma;
    const half* ln2_beta;
    
    // Feed-Forward Network
    const half* ffn_up_weights;
    const half* ffn_up_bias;
    const half* ffn_down_weights;
    const half* ffn_down_bias;
} GPTLayerWeights;

// GPT Transformer Layer Workspace
typedef struct {
    half* ln1_output;      // After first LayerNorm
    half* qkv_output;      // After QKV projection
    half* attn_scores;     // Attention scores workspace
    half* attn_output;     // After attention
    half* attn_residual;   // After attention + residual
    half* ln2_output;      // After second LayerNorm
    half* ffn_intermediate;// FFN intermediate activations
    half* ffn_output;      // After FFN
    size_t total_size;     // Total workspace size in bytes
} GPTLayerWorkspace;

// Calculate workspace size needed for transformer layer
size_t gpt_transformer_layer_workspace_size(const GPTLayerConfig* config);

// Execute full GPT transformer layer
// output = FFN(LayerNorm(input + Attention(LayerNorm(input))))
int gpt_transformer_layer_forward(
    const half* input,
    half* output,
    const GPTLayerWeights* weights,
    GPTLayerWorkspace* workspace,
    const GPTLayerConfig* config,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
);

// Execute GPT transformer layer with KV cache (decode mode)
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
);

// Validate layer configuration
int gpt_transformer_layer_validate_config(const GPTLayerConfig* config);

// Validate layer weights
int gpt_transformer_layer_validate_weights(const GPTLayerWeights* weights);

#ifdef __cplusplus
}
#endif

#endif // GPT_TRANSFORMER_LAYER_H

// ---
// Crafted by GPT-Gamma 🤖
