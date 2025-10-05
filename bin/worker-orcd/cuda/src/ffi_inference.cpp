// ffi_inference.cpp — FFI Interface for Inference
//
// Provides C interface for running inference with Qwen transformer
//
// Spec: M0-W-1032

#include "transformer/qwen_transformer.h"
#include "model/qwen_weight_loader.h"
#include "model_impl.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>

// External sampling function
extern "C" int cuda_sample_token(
    float* logits,
    uint32_t vocab_size,
    float temperature,
    uint32_t top_k,
    float top_p,
    uint64_t seed
);

extern "C" {

struct InferenceContext {
    worker::transformer::QwenTransformer* transformer;
    worker::model::QwenModel* model;
    float* logits_buffer;  // Device memory for logits
};

/**
 * Initialize inference context
 * 
 * @param model Loaded model with weights
 * @param vocab_size Vocabulary size
 * @param hidden_dim Hidden dimension
 * @param num_layers Number of layers
 * @param num_heads Number of attention heads
 * @param num_kv_heads Number of KV heads
 * @param head_dim Head dimension
 * @param ffn_dim FFN intermediate dimension
 * @param context_length Maximum context length
 * @param error Output error code
 * @return Inference context or nullptr on error
 */
InferenceContext* cuda_inference_init(
    void* model_ptr,
    uint32_t vocab_size,
    uint32_t hidden_dim,
    uint32_t num_layers,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t ffn_dim,
    uint32_t context_length,
    int* error
) {
    try {
        // model_ptr is actually a ModelImpl* (stub), not QwenModel*
        // We need to load the real weights from the GGUF file
        auto* model_impl = reinterpret_cast<worker::ModelImpl*>(model_ptr);
        const char* model_path = model_impl->model_path().c_str();
        
        fprintf(stderr, "🔧 Loading real weights from GGUF: %s\n", model_path);
        
        // Create Qwen config
        worker::model::QwenConfig qwen_config;
        qwen_config.vocab_size = vocab_size;
        qwen_config.hidden_dim = hidden_dim;
        qwen_config.num_layers = num_layers;
        qwen_config.num_heads = num_heads;
        qwen_config.num_kv_heads = num_kv_heads;
        qwen_config.context_length = context_length;
        
        // Load real weights from GGUF
        worker::model::QwenModel* qwen_model = worker::model::QwenWeightLoader::load(
            model_path,
            qwen_config
        );
        
        fprintf(stderr, "✅ Weights loaded: %.2f MB\n", 
                qwen_model->vram_usage / 1024.0 / 1024.0);
        
        // Create transformer config
        worker::transformer::TransformerConfig config;
        config.vocab_size = vocab_size;
        config.hidden_dim = hidden_dim;
        config.num_layers = num_layers;
        config.num_heads = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.head_dim = head_dim;
        config.ffn_dim = ffn_dim;
        config.context_length = context_length;
        config.rope_freq_base = 1000000.0f;  // Qwen2.5 specific
        
        // Create transformer
        auto* transformer = new worker::transformer::QwenTransformer(qwen_model, config);
        
        // Allocate logits buffer
        float* logits;
        cudaMalloc(&logits, vocab_size * sizeof(float));
        
        // Create context
        auto* ctx = new InferenceContext();
        ctx->transformer = transformer;
        ctx->model = qwen_model;
        ctx->logits_buffer = logits;
        
        fprintf(stderr, "✅ Inference context initialized\n");
        fprintf(stderr, "   Vocab: %u, Hidden: %u, Layers: %u\n",
                vocab_size, hidden_dim, num_layers);
        
        *error = 0;
        return ctx;
    } catch (const std::exception& e) {
        fprintf(stderr, "❌ Inference init failed: %s\n", e.what());
        *error = -1;
        return nullptr;
    }
}

/**
 * Generate next token
 * 
 * @param ctx Inference context
 * @param token_id Current token ID
 * @param temperature Sampling temperature
 * @param top_k Top-k filtering
 * @param top_p Top-p (nucleus) sampling
 * @param seed Random seed
 * @param error Output error code
 * @return Next token ID
 */
uint32_t cuda_inference_generate_token(
    InferenceContext* ctx,
    uint32_t token_id,
    float temperature,
    uint32_t top_k,
    float top_p,
    uint64_t seed,
    int* error
) {
    try {
        if (!ctx || !ctx->transformer) {
            *error = -1;
            return 0;
        }
        
        // Copy token_id to device memory
        uint32_t* d_token_id;
        cudaMalloc(&d_token_id, sizeof(uint32_t));
        cudaMemcpy(d_token_id, &token_id, sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        // Debug: Check token ID
        uint32_t host_token;
        cudaMemcpy(&host_token, d_token_id, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (host_token >= ctx->model->config.vocab_size) {
            fprintf(stderr, "❌ Invalid token ID: %u (vocab_size: %u)\n", host_token, ctx->model->config.vocab_size);
            cudaFree(d_token_id);
            *error = -1;
            return 0;
        }
        
        // Run transformer forward pass
        ctx->transformer->forward(d_token_id, 1, ctx->logits_buffer);
        
        // Free device memory
        cudaFree(d_token_id);
        
        // Debug: Check first few logits
        static int token_idx = 0;
        float host_logits[10];
        cudaMemcpy(host_logits, ctx->logits_buffer, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        if (token_idx < 3) {
            fprintf(stderr, "First 10 logits: ");
            for (int i = 0; i < 10; i++) {
                fprintf(stderr, "%.2f ", host_logits[i]);
            }
            fprintf(stderr, "\n");
        }
        token_idx++;
        
        // Sample next token
        int next_token = cuda_sample_token(
            ctx->logits_buffer,
            ctx->model->config.vocab_size,
            temperature,
            top_k,
            top_p,
            seed
        );
        
        fprintf(stderr, "Sampled token: %d\n", next_token);
        
        *error = 0;
        return static_cast<uint32_t>(next_token);
    } catch (const std::exception& e) {
        fprintf(stderr, "❌ Token generation failed: %s\n", e.what());
        *error = -1;
        return 0;
    }
}

/**
 * Reset KV cache
 * 
 * @param ctx Inference context
 */
void cuda_inference_reset(InferenceContext* ctx) {
    if (ctx && ctx->transformer) {
        ctx->transformer->reset_cache();
    }
}

/**
 * Free inference context
 * 
 * @param ctx Inference context
 */
void cuda_inference_context_free(InferenceContext* ctx) {
    if (ctx) {
        if (ctx->logits_buffer) {
            cudaFree(ctx->logits_buffer);
        }
        if (ctx->transformer) {
            delete ctx->transformer;
        }
        delete ctx;
    }
}

} // extern "C"

// ---
// Crafted by GPT-Gamma 🤖
