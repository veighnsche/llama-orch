// GPT Inference Adapter Implementation
//
// Story: GT-039
// Spec: M0-W-1213, M0-W-1214

#include "gpt_adapter.h"
#include "../model/gpt_model.h"
#include "../../kernels/layernorm.cu"
#include "../../kernels/mha_attention.cu"
#include "../../kernels/gpt_ffn.cu"
#include "../../kernels/mxfp4_gemm.cu"
#include "../../kernels/mxfp4_embedding.cu"
#include "../../kernels/mxfp4_attention.cu"
#include "../../kernels/mxfp4_ffn.cu"
#include "../../kernels/mxfp4_lm_head.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstring>
#include <random>

// External CUDA kernels
extern "C" {
    void cuda_layernorm(
        const half* input,
        half* output,
        const half* weight,
        const half* bias,
        int batch_size,
        int hidden_dim,
        float eps,
        cudaStream_t stream
    );
    
    void cuda_mha_attention(
        const half* input,
        const half* wq,
        const half* wk,
        const half* wv,
        const half* wo,
        half* output,
        int batch_size,
        int seq_len,
        int hidden_dim,
        int num_heads,
        cudaStream_t stream
    );
    
    void cuda_gpt_ffn(
        const half* input,
        const half* w_up,
        const half* w_down,
        const half* b_up,
        const half* b_down,
        half* output,
        int batch_size,
        int seq_len,
        int hidden_dim,
        int ffn_dim,
        cudaStream_t stream
    );
}

GPTInferenceAdapter::GPTInferenceAdapter(const GPTConfig& config)
    : config_(config)
    , model_(nullptr)
    , cublas_(nullptr)
    , stream_(nullptr)
    , d_hidden_states_(nullptr)
    , d_residual_(nullptr)
    , d_ln_output_(nullptr)
    , weights_loaded_(false)
{
    // Create cuBLAS handle
    cublasCreate(&cublas_);
    
    // Create CUDA stream
    cudaStreamCreate(&stream_);
    
    // Allocate working buffers
    int max_batch_seq = config_.max_batch_size * config_.max_seq_len;
    cudaMalloc(&d_hidden_states_, max_batch_seq * config_.hidden_dim * sizeof(half));
    cudaMalloc(&d_residual_, max_batch_seq * config_.hidden_dim * sizeof(half));
    cudaMalloc(&d_ln_output_, max_batch_seq * config_.hidden_dim * sizeof(half));
}

GPTInferenceAdapter::~GPTInferenceAdapter() {
    if (model_) {
        delete model_;
    }
    
    if (cublas_) {
        cublasDestroy(cublas_);
    }
    
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    
    if (d_hidden_states_) cudaFree(d_hidden_states_);
    if (d_residual_) cudaFree(d_residual_);
    if (d_ln_output_) cudaFree(d_ln_output_);
}

void GPTInferenceAdapter::load_weights(const std::string& model_path) {
    model_ = new GPTModel(config_);
    model_->load_from_gguf(model_path);
    weights_loaded_ = true;
}

void GPTInferenceAdapter::load_weights_mxfp4(const std::string& model_path) {
    model_ = new GPTModel(config_);
    model_->load_from_gguf_mxfp4(model_path);
    weights_loaded_ = true;
}

void GPTInferenceAdapter::allocate_state(InferenceState& state, int max_seq_len) {
    state.max_seq_len = max_seq_len;
    state.current_pos = 0;
    
    // Allocate KV cache
    int kv_cache_size = config_.num_layers * 2 * max_seq_len * config_.hidden_dim;
    cudaMalloc(&state.kv_cache, kv_cache_size * sizeof(half));
    cudaMemset(state.kv_cache, 0, kv_cache_size * sizeof(half));
}

void GPTInferenceAdapter::free_state(InferenceState& state) {
    if (state.kv_cache) {
        cudaFree(state.kv_cache);
        state.kv_cache = nullptr;
    }
}

void GPTInferenceAdapter::prefill(const std::vector<int>& tokens, InferenceState& state) {
    int num_tokens = tokens.size();
    
    // Copy tokens to device
    int* d_tokens;
    cudaMalloc(&d_tokens, num_tokens * sizeof(int));
    cudaMemcpy(d_tokens, tokens.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);
    
    // Allocate logits buffer
    float* d_logits;
    cudaMalloc(&d_logits, config_.vocab_size * sizeof(float));
    
    // Forward pass
    forward(d_tokens, num_tokens, d_logits, state);
    
    // Update position
    state.current_pos = num_tokens;
    
    // Cleanup
    cudaFree(d_tokens);
    cudaFree(d_logits);
}

int GPTInferenceAdapter::decode_next_token(InferenceState& state, float temperature, uint64_t seed) {
    // Use last token from state
    int last_token = state.last_token;
    
    // Forward pass with single token
    int* d_token;
    cudaMalloc(&d_token, sizeof(int));
    cudaMemcpy(d_token, &last_token, sizeof(int), cudaMemcpyHostToDevice);
    
    float* d_logits;
    cudaMalloc(&d_logits, config_.vocab_size * sizeof(float));
    
    forward(d_token, 1, d_logits, state);
    
    // Copy logits to host
    float* h_logits = new float[config_.vocab_size];
    cudaMemcpy(h_logits, d_logits, config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Sample next token
    int next_token = sample_token(h_logits, temperature, seed);
    
    // Update state
    state.current_pos++;
    state.last_token = next_token;
    
    // Cleanup
    delete[] h_logits;
    cudaFree(d_token);
    cudaFree(d_logits);
    
    return next_token;
}

void GPTInferenceAdapter::forward(
    const int* tokens,
    int num_tokens,
    float* logits,
    InferenceState& state
) {
    // Embedding layer
    half* embeddings;
    cudaMalloc(&embeddings, num_tokens * config_.hidden_dim * sizeof(half));
    embedding_layer(tokens, num_tokens, (float*)embeddings);
    
    // Transformer layers
    half* layer_input = embeddings;
    half* layer_output = d_hidden_states_;
    
    for (int layer = 0; layer < config_.num_layers; layer++) {
        transformer_layer((float*)layer_input, (float*)layer_output, layer, num_tokens, state);
        
        // Swap buffers
        half* tmp = layer_input;
        layer_input = layer_output;
        layer_output = tmp;
    }
    
    // Final LayerNorm
    cuda_layernorm(
        layer_input,
        d_ln_output_,
        model_->ln_f_weight,
        model_->ln_f_bias,
        num_tokens,
        config_.hidden_dim,
        1e-5f,
        stream_
    );
    
    // LM head
    lm_head((float*)d_ln_output_, logits);
    
    cudaFree(embeddings);
}

void GPTInferenceAdapter::transformer_layer(
    const float* input,
    float* output,
    int layer_idx,
    int seq_len,
    InferenceState& state
) {
    half* h_input = (half*)input;
    half* h_output = (half*)output;
    
    // Pre-attention LayerNorm
    cuda_layernorm(
        h_input,
        d_ln_output_,
        model_->layers[layer_idx].ln_1_weight,
        model_->layers[layer_idx].ln_1_bias,
        seq_len,
        config_.hidden_dim,
        1e-5f,
        stream_
    );
    
    // Multi-head attention
    half* attn_output;
    cudaMalloc(&attn_output, seq_len * config_.hidden_dim * sizeof(half));
    
    attention_layer((float*)d_ln_output_, (float*)attn_output, layer_idx, seq_len, state);
    
    // Residual connection
    for (int i = 0; i < seq_len * config_.hidden_dim; i++) {
        d_residual_[i] = __hadd(h_input[i], attn_output[i]);
    }
    
    // Pre-FFN LayerNorm
    cuda_layernorm(
        d_residual_,
        d_ln_output_,
        model_->layers[layer_idx].ln_2_weight,
        model_->layers[layer_idx].ln_2_bias,
        seq_len,
        config_.hidden_dim,
        1e-5f,
        stream_
    );
    
    // FFN
    half* ffn_output;
    cudaMalloc(&ffn_output, seq_len * config_.hidden_dim * sizeof(half));
    
    ffn_layer((float*)d_ln_output_, (float*)ffn_output, layer_idx);
    
    // Residual connection
    for (int i = 0; i < seq_len * config_.hidden_dim; i++) {
        h_output[i] = __hadd(d_residual_[i], ffn_output[i]);
    }
    
    cudaFree(attn_output);
    cudaFree(ffn_output);
}

void GPTInferenceAdapter::embedding_layer(const int* tokens, int num_tokens, float* output) {
    if (model_->use_mxfp4) {
        mxfp4_embedding_lookup(
            (half*)output,
            model_->token_embeddings_mxfp4,
            tokens,
            num_tokens,
            config_.hidden_dim,
            config_.vocab_size,
            stream_
        );
    } else {
        // FP16 embedding lookup (simplified)
        // Would use standard embedding kernel
    }
}

void GPTInferenceAdapter::attention_layer(
    const float* input,
    float* output,
    int layer_idx,
    int seq_len,
    InferenceState& state
) {
    if (model_->use_mxfp4) {
        mxfp4_qkv_projection(
            (const half*)input,
            model_->layers[layer_idx].attn_wq_mxfp4,
            model_->layers[layer_idx].attn_wk_mxfp4,
            model_->layers[layer_idx].attn_wv_mxfp4,
            (half*)output,  // Q
            (half*)output,  // K (simplified)
            (half*)output,  // V (simplified)
            1,  // batch_size
            seq_len,
            config_.hidden_dim,
            cublas_,
            stream_
        );
    } else {
        cuda_mha_attention(
            (const half*)input,
            model_->layers[layer_idx].attn_wq,
            model_->layers[layer_idx].attn_wk,
            model_->layers[layer_idx].attn_wv,
            model_->layers[layer_idx].attn_wo,
            (half*)output,
            1,  // batch_size
            seq_len,
            config_.hidden_dim,
            config_.num_heads,
            stream_
        );
    }
}

void GPTInferenceAdapter::ffn_layer(const float* input, float* output, int layer_idx) {
    if (model_->use_mxfp4) {
        mxfp4_ffn_forward(
            (const half*)input,
            model_->layers[layer_idx].ffn_w_up_mxfp4,
            model_->layers[layer_idx].ffn_w_down_mxfp4,
            (half*)output,
            1,  // batch_size
            1,  // seq_len (decode)
            config_.hidden_dim,
            config_.ffn_dim,
            cublas_,
            stream_
        );
    } else {
        cuda_gpt_ffn(
            (const half*)input,
            model_->layers[layer_idx].ffn_w_up,
            model_->layers[layer_idx].ffn_w_down,
            model_->layers[layer_idx].ffn_b_up,
            model_->layers[layer_idx].ffn_b_down,
            (half*)output,
            1,  // batch_size
            1,  // seq_len
            config_.hidden_dim,
            config_.ffn_dim,
            stream_
        );
    }
}

void GPTInferenceAdapter::lm_head(const float* input, float* logits) {
    if (model_->use_mxfp4) {
        mxfp4_lm_head_forward(
            (const half*)input,
            model_->lm_head_mxfp4,
            (half*)logits,
            1,  // batch_size
            1,  // seq_len
            config_.hidden_dim,
            config_.vocab_size,
            cublas_,
            stream_
        );
    } else {
        // FP16 LM head (simplified)
        // Would use standard GEMM
    }
}

int GPTInferenceAdapter::sample_token(const float* logits, float temperature, uint64_t seed) {
    if (temperature == 0.0f) {
        // Greedy sampling (argmax)
        // [TEAM_HOTEL] CRITICAL: Only scan vocab_size (151643) positions, not padded_vocab_size!
        //   The logits buffer has 151936 positions, but the last 293 are padding values.
        //   Scanning them would potentially pick garbage tokens from the padding region.
        //   This is CORRECT - we use config_.vocab_size (logical size) for argmax.
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < config_.vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    } else {
        // Temperature sampling
        std::mt19937 rng(seed);
        
        // Apply temperature
        float* scaled_logits = new float[config_.vocab_size];
        for (int i = 0; i < config_.vocab_size; i++) {
            scaled_logits[i] = logits[i] / temperature;
        }
        
        // Softmax
        float max_logit = scaled_logits[0];
        for (int i = 1; i < config_.vocab_size; i++) {
            max_logit = std::max(max_logit, scaled_logits[i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i < config_.vocab_size; i++) {
            scaled_logits[i] = expf(scaled_logits[i] - max_logit);
            sum += scaled_logits[i];
        }
        
        for (int i = 0; i < config_.vocab_size; i++) {
            scaled_logits[i] /= sum;
        }
        
        // Sample
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        float cumsum = 0.0f;
        
        for (int i = 0; i < config_.vocab_size; i++) {
            cumsum += scaled_logits[i];
            if (r < cumsum) {
                delete[] scaled_logits;
                return i;
            }
        }
        
        delete[] scaled_logits;
        return config_.vocab_size - 1;
    }
}

size_t GPTInferenceAdapter::get_vram_usage() const {
    if (!model_) return 0;
    return model_->get_vram_usage();
}

// C interface implementation
extern "C" {
    GPTInferenceAdapter* gpt_adapter_create(const GPTConfig* config) {
        return new GPTInferenceAdapter(*config);
    }
    
    void gpt_adapter_destroy(GPTInferenceAdapter* adapter) {
        delete adapter;
    }
    
    void gpt_adapter_load_weights(GPTInferenceAdapter* adapter, const char* model_path) {
        adapter->load_weights(std::string(model_path));
    }
    
    void gpt_adapter_load_weights_mxfp4(GPTInferenceAdapter* adapter, const char* model_path) {
        adapter->load_weights_mxfp4(std::string(model_path));
    }
    
    void gpt_adapter_prefill(GPTInferenceAdapter* adapter, const int* tokens, int num_tokens, InferenceState* state) {
        std::vector<int> token_vec(tokens, tokens + num_tokens);
        adapter->prefill(token_vec, *state);
    }
    
    int gpt_adapter_decode(GPTInferenceAdapter* adapter, InferenceState* state, float temperature, uint64_t seed) {
        return adapter->decode_next_token(*state, temperature, seed);
    }
    
    void gpt_adapter_allocate_state(GPTInferenceAdapter* adapter, InferenceState* state, int max_seq_len) {
        adapter->allocate_state(*state, max_seq_len);
    }
    
    void gpt_adapter_free_state(GPTInferenceAdapter* adapter, InferenceState* state) {
        adapter->free_state(*state);
    }
    
    size_t gpt_adapter_vram_usage(const GPTInferenceAdapter* adapter) {
        return adapter->get_vram_usage();
    }
}

// ---
// Crafted by GPT-Gamma 🤖
