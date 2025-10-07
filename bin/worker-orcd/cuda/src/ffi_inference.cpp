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
#include <vector>
#include <cmath>

// [TEAM PICASSO 2025-10-07T15:47Z] Numeric parity logging
#include "orch_log.hpp"

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
    uint32_t padded_vocab_size,
    uint32_t hidden_dim,
    uint32_t num_layers,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t ffn_dim,
    uint32_t context_length,
    float rope_freq_base,
    int* error
) {
    try {
        // NEW: model_ptr is now a CudaModel* (ModelImpl*) with weights already loaded by Rust!
        auto* model_impl = reinterpret_cast<worker::ModelImpl*>(model_ptr);
        auto* qwen_model = model_impl->get_qwen_model();
        
        if (!qwen_model) {
            fprintf(stderr, "❌ QwenModel is null!\n");
            *error = -1;
            return nullptr;
        }
        
        fprintf(stderr, "🎉 [C++] Using pre-loaded model from Rust (VRAM: %.2f MB)\n", 
                qwen_model->vram_usage / 1024.0 / 1024.0);
        
        // Create transformer config
        worker::transformer::TransformerConfig config;
        config.vocab_size = vocab_size;
        config.padded_vocab_size = padded_vocab_size;
        config.hidden_dim = hidden_dim;
        config.num_layers = num_layers;
        config.num_heads = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.head_dim = head_dim;
        config.ffn_dim = ffn_dim;
        config.context_length = context_length;
        config.rope_freq_base = rope_freq_base;
        
        // Create transformer
        auto* transformer = new worker::transformer::QwenTransformer(qwen_model, config);
        
        // ============================================================================
        // [TEAM_HOTEL] CRITICAL: Allocate buffer for PADDED vocab size! (2025-10-06 20:12 UTC)
        // ============================================================================
        //
        // THOUGHT: cuBLAS will compute logits for ALL padded_vocab_size (151936) positions,
        //   not just vocab_size (151643). So buffer must be large enough for padded size!
        //
        // WRONG (Team GEMMA DELTA's code):
        //   cudaMalloc(&logits, vocab_size * sizeof(float));  // Only 151643 floats
        //
        // CORRECT:
        //   cudaMalloc(&logits, padded_vocab_size * sizeof(float));  // Full 151936 floats
        //
        // WHY: cuBLAS writes to positions 0..151935 (all padded positions)
        //   If buffer is only 151643 floats, positions 151643..151935 overflow!
        //   This causes memory corruption and undefined behavior.
        //
        // NOTE: After cuBLAS, argmax will only scan first vocab_size (151643) positions
        //   to avoid the 293 padding values. But buffer must hold all 151936!
        //
        float* logits;
        cudaMalloc(&logits, padded_vocab_size * sizeof(float));
        
        // Initialize buffer to -INFINITY on host, then copy to device
        std::vector<float> init_logits(padded_vocab_size, -INFINITY);
        cudaMemcpy(logits, init_logits.data(), padded_vocab_size * sizeof(float), cudaMemcpyHostToDevice);
        
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
        // TEAM FREE [Review]
        // Category: Memory management
        // Hypothesis: cudaMalloc/cudaFree per token (line 168-186) causes allocator thrashing; 1000 tokens = 2000 malloc/free calls.
        // Evidence: No error check on cudaMalloc; if allocation fails mid-generation, nullptr deref in forward() → crash.
        // Risk: Performance degradation (malloc overhead); potential crash if VRAM fragmented.
        // Confidence: High
        // Next step: Pre-allocate persistent d_token_id buffer in InferenceContext; reuse across generate_token calls.
        cudaMalloc(&d_token_id, sizeof(uint32_t));
        cudaMemcpy(d_token_id, &token_id, sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        // Debug: Check token ID
        uint32_t host_token;
        cudaMemcpy(&host_token, d_token_id, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        // TEAM FREE [Review]
        // Category: Concurrency
        // Hypothesis: D2H memcpy for debug check (line 174) forces GPU-CPU sync every token; blocks async execution pipeline.
        // Evidence: cudaMemcpy without stream parameter → implicit sync; 1000 tokens = 1000 forced syncs.
        // Risk: 10-50% throughput loss vs async validation; latency spike per token.
        // Confidence: High
        // Next step: Remove debug check in production or use cudaMemcpyAsync + separate validation stream.
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
        
        fprintf(stderr, "[TEAM CHAIR] Checkpoint G: After forward, before logits debug\n");
        fflush(stderr);
        
        // Debug: Check first few logits
        // [TEAM CHAIR] 2025-10-07T02:52Z - Disabled to fix crash
        static int token_idx = 0;
        if (false) {
            float host_logits[10];
            cudaMemcpy(host_logits, ctx->logits_buffer, 10 * sizeof(float), cudaMemcpyDeviceToHost);
            if (token_idx < 3) {
                fprintf(stderr, "First 10 logits: ");
                for (int i = 0; i < 10; i++) {
                    fprintf(stderr, "%.2f ", host_logits[i]);
                }
                fprintf(stderr, "\n");
            }
        }
        token_idx++;
        
        fprintf(stderr, "[TEAM CHAIR] Checkpoint H: Before sampling\n");
        fflush(stderr);
        
        // [TEAM AEGIS] 2025-10-07T23:27Z
        // SUSPECT: Temperature showing 0.00 in prefill logs, thought this was bug
        // PLAN: Added top-10 logits instrumentation to debug
        // FALSE_LEAD: Prefill uses temp=0.0 by design. Instrumentation only captured prefill, not generation.
        // LESSON: Need to log generation tokens (after prefill complete), not just first 3 tokens total.
        
        // Sample next token
        // NOTE: ctx->transformer->config_.vocab_size is now the ACTUAL vocab from output.weight
        //       (e.g., 151643 for Qwen2.5-0.5B), not the padded tokenizer vocab (151936)
        //
        // 🕵️ [TEAM_LOVE] INVESTIGATION TRAIL (2025-10-06 18:33-18:40 UTC)
        // ✅ VERIFIED CORRECT: Sampling logic is correct
        //    - logits_buffer is passed correctly to cuda_sample_token() ✅
        //    - vocab_size is correct (from config, not padded) ✅
        //    - Sampling parameters (temperature, top_k, top_p) are passed correctly ✅
        //
        // ❌ FALSE LEAD: I suspected logits_buffer might not be updated between tokens
        //    But line 162 shows forward() is called BEFORE sampling, which updates logits_buffer.
        //    The buffer is correctly updated for each token.
        //
        // ✅ VERIFIED: This function correctly:
        //    1. Receives token_id from Rust
        //    2. Calls transformer->forward() to compute logits
        //    3. Calls cuda_sample_token() to sample from logits
        //    4. Returns sampled token_id back to Rust
        //
        // The bug is NOT in this FFI layer!
        
        // [TEAM PICASSO 2025-10-07T19:45Z] Parity logging: Copy logits to HOST first!
        // BUG FIX: ctx->logits_buffer is DEVICE memory, can't read directly from CPU
        // MUST use cudaMemcpy to copy to host buffer first
        #ifdef ORCH_LOGGING
        static int generation_token_idx = 0;
        {
            // Copy first 10 logits to host memory
            float host_logits[10];
            cudaMemcpy(host_logits, ctx->logits_buffer, 10 * sizeof(float), cudaMemcpyDeviceToHost);
            ORCH_LOG_LOGITS(host_logits, 10, generation_token_idx);
            generation_token_idx++;
        }
        #endif
        
        int next_token = cuda_sample_token(
            ctx->logits_buffer,
            ctx->model->config.vocab_size,  // Use actual vocab from config
            temperature,
            top_k,
            top_p,
            seed
        );
        
        // Only log first few tokens to reduce noise
        // (Rust side will show summary at end)
        
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
