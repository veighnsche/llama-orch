// GPT Inference Adapter
//
// Implements InferenceAdapter interface for GPT architecture.
// Orchestrates GPT-specific inference pipeline (LayerNorm, MHA, GELU FFN).
//
// Story: GT-039
// Spec: M0-W-1213, M0-W-1214

#pragma once

#include "../model/gpt_model.h"
#include "../../include/worker_types.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <vector>

// Forward declarations
struct InferenceState;
struct InferenceAdapter;

// GPT Inference Adapter
class GPTInferenceAdapter {
public:
    GPTInferenceAdapter(const GPTConfig& config);
    ~GPTInferenceAdapter();
    
    // Model loading
    void load_weights(const std::string& model_path);
    void load_weights_mxfp4(const std::string& model_path);
    
    // Inference pipeline
    void prefill(const std::vector<int>& tokens, InferenceState& state);
    int decode_next_token(InferenceState& state, float temperature, uint64_t seed);
    
    // State management
    void allocate_state(InferenceState& state, int max_seq_len);
    void free_state(InferenceState& state);
    
    // Utilities
    size_t get_vram_usage() const;
    const GPTConfig& get_config() const { return config_; }
    
private:
    // Forward pass
    void forward(
        const int* tokens,
        int num_tokens,
        float* logits,
        InferenceState& state
    );
    
    // Transformer layers
    void transformer_layer(
        const float* input,
        float* output,
        int layer_idx,
        int seq_len,
        InferenceState& state
    );
    
    // Components
    void embedding_layer(const int* tokens, int num_tokens, float* output);
    void layernorm(const float* input, float* output, int size, const float* weight, const float* bias);
    void attention_layer(const float* input, float* output, int layer_idx, int seq_len, InferenceState& state);
    void ffn_layer(const float* input, float* output, int layer_idx);
    void lm_head(const float* input, float* logits);
    
    // Sampling
    int sample_token(const float* logits, float temperature, uint64_t seed);
    
    GPTConfig config_;
    GPTModel* model_;
    cublasHandle_t cublas_;
    cudaStream_t stream_;
    
    // Device buffers
    float* d_hidden_states_;
    float* d_residual_;
    float* d_ln_output_;
    
    bool weights_loaded_;
};

// C interface for FFI
extern "C" {
    GPTInferenceAdapter* gpt_adapter_create(const GPTConfig* config);
    void gpt_adapter_destroy(GPTInferenceAdapter* adapter);
    void gpt_adapter_load_weights(GPTInferenceAdapter* adapter, const char* model_path);
    void gpt_adapter_load_weights_mxfp4(GPTInferenceAdapter* adapter, const char* model_path);
    void gpt_adapter_prefill(GPTInferenceAdapter* adapter, const int* tokens, int num_tokens, InferenceState* state);
    int gpt_adapter_decode(GPTInferenceAdapter* adapter, InferenceState* state, float temperature, uint64_t seed);
    void gpt_adapter_allocate_state(GPTInferenceAdapter* adapter, InferenceState* state, int max_seq_len);
    void gpt_adapter_free_state(GPTInferenceAdapter* adapter, InferenceState* state);
    size_t gpt_adapter_vram_usage(const GPTInferenceAdapter* adapter);
}

// ---
// Crafted by GPT-Gamma 🤖
