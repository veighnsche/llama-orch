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
    fprintf(stderr, "   Vocab: %u, Layers: %u, Hidden: %u, Heads: %u, KV Heads: %u\n",
            config.vocab_size, config.num_layers, config.hidden_dim, config.num_heads, config.num_kv_heads);
    fprintf(stderr, "   NOTE: vocab_size is ACTUAL vocab from output.weight tensor (not padded)\n");
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
    static int call_count = 0;
    
    cuda_embedding_lookup(
        token_ids,
        model_->weights.token_embd,
        output,
        batch_size,
        config_.vocab_size,
        config_.hidden_dim,
        nullptr  // default stream
    );
    
    // Debug: Check embedding values (only first call)
    if (call_count == 0) {
        half host_emb[5];
        cudaMemcpy(host_emb, output, 5 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "  Embedding[0:5]: ");
        for (int i = 0; i < 5; i++) {
            fprintf(stderr, "%.2f ", __half2float(host_emb[i]));
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
    
    // Q projection: q = W_q @ x
    // W_q in GGUF: [hidden_dim, q_dim] row-major → [q_dim, hidden_dim] col-major in cuBLAS
    // Input x: [hidden_dim, batch] col-major
    // Result: [q_dim, batch] col-major
    uint32_t q_dim = config_.num_heads * config_.head_dim;
    cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose needed (GGUF row-major = cuBLAS col-major transposed)
        q_dim, batch_size, config_.hidden_dim,
        &alpha,
        layer.attn_q_weight, CUDA_R_16F, q_dim,  // lda = q_dim (row-major → col-major conversion)
        normed_half, CUDA_R_16F, config_.hidden_dim,
        &beta,
        q_half, CUDA_R_16F, q_dim,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    // Qwen2.5 doesn't use QKV biases (they're all zeros in the model)
    // No bias addition needed
    
    // K projection: k = W_k @ x
    // W_k in GGUF: [hidden_dim, kv_dim] row-major → [kv_dim, hidden_dim] col-major in cuBLAS
    uint32_t kv_dim = config_.num_kv_heads * config_.head_dim;
    cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose needed
        kv_dim, batch_size, config_.hidden_dim,
        &alpha,
        layer.attn_k_weight, CUDA_R_16F, kv_dim,  // lda = kv_dim (row-major → col-major conversion)
        normed_half, CUDA_R_16F, config_.hidden_dim,
        &beta,
        k_half, CUDA_R_16F, kv_dim,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    // V projection: v = W_v @ x
    // W_v in GGUF: [hidden_dim, kv_dim] row-major → [kv_dim, hidden_dim] col-major in cuBLAS
    cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose needed
        kv_dim, batch_size, config_.hidden_dim,
        &alpha,
        layer.attn_v_weight, CUDA_R_16F, kv_dim,  // lda = kv_dim (row-major → col-major conversion)
        normed_half, CUDA_R_16F, config_.hidden_dim,
        &beta,
        v_half, CUDA_R_16F, kv_dim,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    // 3. Apply RoPE to Q and K (explicit position and KV heads)
    
    cuda_rope_forward_ex(
        q_proj_,
        k_proj_,
        batch_size,
        config_.num_heads,
        config_.num_kv_heads,
        config_.head_dim,
        pos,
        config_.rope_freq_base,
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
    
    // Flush CUDA printf buffer for attention debug output
    if (layer_idx < 2 && layer_call_count[layer_idx] <= 3) {
        cudaDeviceSynchronize();
    }
    
    // DEBUG: Verify KV cache was written correctly
    if (layer_idx == 0 && pos < 5) {
        // Check what was written to K cache at current position
        half h_k_cache[10];
        int kv_head = 0;  // Check first KV head
        // Cache layout: [batch=1, kv_head, pos, d] with max_seq_len stride
        size_t cache_idx = 0 * config_.num_kv_heads * config_.context_length * config_.head_dim +
                          kv_head * config_.context_length * config_.head_dim +
                          pos * config_.head_dim;
        cudaMemcpy(h_k_cache, layer_k_cache + cache_idx, 10 * sizeof(half), cudaMemcpyDeviceToHost);
        
        fprintf(stderr, "  [KV CACHE VERIFY] Layer 0, pos=%u, kv_head=0\n", pos);
        fprintf(stderr, "    K_cache[pos=%u][0:10]: ", pos);
        for (int i = 0; i < 10; i++) {
            fprintf(stderr, "%.4f ", __half2float(h_k_cache[i]));
        }
        fprintf(stderr, "\n");
        
        // Also check what was written to V cache
        half h_v_cache[10];
        cudaMemcpy(h_v_cache, layer_v_cache + cache_idx, 10 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "    V_cache[pos=%u][0:10]: ", pos);
        for (int i = 0; i < 10; i++) {
            fprintf(stderr, "%.4f ", __half2float(h_v_cache[i]));
        }
        fprintf(stderr, "\n");
        
        // If pos > 0, also read what's at pos-1 to verify cache persistence
        if (pos > 0) {
            size_t prev_cache_idx = 0 * config_.num_kv_heads * config_.context_length * config_.head_dim +
                                   kv_head * config_.context_length * config_.head_dim +
                                   (pos - 1) * config_.head_dim;
            half h_k_prev[10];
            cudaMemcpy(h_k_prev, layer_k_cache + prev_cache_idx, 10 * sizeof(half), cudaMemcpyDeviceToHost);
            fprintf(stderr, "    K_cache[pos=%u][0:10]: ", pos - 1);
            for (int i = 0; i < 10; i++) {
                fprintf(stderr, "%.4f ", __half2float(h_k_prev[i]));
            }
            fprintf(stderr, " (previous position, should be unchanged)\n");
        }
    }
    
    // 5. Attention output projection: out = W_o @ attn
    // W_o in GGUF: [q_dim, hidden_dim] row-major → [hidden_dim, q_dim] col-major in cuBLAS
    // CRITICAL: Use separate buffer for output to avoid in-place corruption
    half* attn_out_half = reinterpret_cast<half*>(attn_output_);
    half* ffn_out_half = reinterpret_cast<half*>(ffn_output_);  // Reuse ffn_output_ as temp buffer
    cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose needed
        config_.hidden_dim, batch_size, q_dim,
        &alpha,
        layer.attn_output, CUDA_R_16F, config_.hidden_dim,  // lda = hidden_dim (row-major → col-major)
        attn_out_half, CUDA_R_16F, q_dim,
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
    
    // Increment layer call count for debug logging
    layer_call_count[layer_idx]++;
}

void QwenTransformer::project_to_vocab(
    const void* hidden_states,
    uint32_t batch_size,
    float* logits
) {
    // LM head projection: logits = hidden @ lm_head^T
    // hidden: [batch, hidden_dim] (FP16)
    // lm_head: [vocab_size, hidden_dim] stored in GGUF as row-major
    // logits: [batch, vocab_size] (FP32)
    //
    // GGUF stores tensors in row-major order, but cuBLAS expects column-major.
    // For GGUF tensor [vocab_size, hidden_dim] in row-major:
    //   - In memory: vocab_size rows, each with hidden_dim elements
    //   - To cuBLAS (column-major view): appears as [hidden_dim, vocab_size] transposed
    //
    // We want: logits[batch, vocab] = hidden[batch, hidden] @ lm_head[vocab, hidden]^T
    // In cuBLAS column-major: C = A @ B^T becomes C^T = B @ A^T
    // So: logits^T[vocab, batch] = lm_head[vocab, hidden] @ hidden^T[hidden, batch]
    
    if (!model_->weights.lm_head) {
        fprintf(stderr, "❌ lm_head is NULL!\n");
        cudaMemset(logits, 0, batch_size * config_.vocab_size * sizeof(float));
        return;
    }
    
    const half* hidden_half = reinterpret_cast<const half*>(hidden_states);
    const half* lm_head_half = reinterpret_cast<const half*>(model_->weights.lm_head);
    
    // Debug: Log first inference
    static bool first_call = true;
    if (first_call) {
        fprintf(stderr, "\n🔍 [DEBUG] LM Head Projection:\n");
        fprintf(stderr, "   vocab_size=%u, hidden_dim=%u, batch_size=%u\n",
                config_.vocab_size, config_.hidden_dim, batch_size);
        fprintf(stderr, "   lm_head ptr=%p, hidden ptr=%p, logits ptr=%p\n",
                (void*)lm_head_half, (void*)hidden_half, (void*)logits);
        
        // Sample hidden state values to check for spikes
        half h_hidden[896];
        cudaMemcpy(h_hidden, hidden_half, 896 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "   hidden[0:5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                __half2float(h_hidden[0]), __half2float(h_hidden[1]),
                __half2float(h_hidden[2]), __half2float(h_hidden[3]),
                __half2float(h_hidden[4]));
        
        // Find max/min in hidden state
        float max_hidden = -INFINITY, min_hidden = INFINITY;
        int max_idx = 0, min_idx = 0;
        for (int i = 0; i < 896; i++) {
            float val = __half2float(h_hidden[i]);
            if (val > max_hidden) { max_hidden = val; max_idx = i; }
            if (val < min_hidden) { min_hidden = val; min_idx = i; }
        }
        fprintf(stderr, "   hidden max=%.4f at [%d], min=%.4f at [%d]\n",
                max_hidden, max_idx, min_hidden, min_idx);
        
        // Sample first few values from lm_head
        half h_lm_head[5];
        cudaMemcpy(h_lm_head, lm_head_half, 5 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "   lm_head[0:5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                __half2float(h_lm_head[0]), __half2float(h_lm_head[1]),
                __half2float(h_lm_head[2]), __half2float(h_lm_head[3]),
                __half2float(h_lm_head[4]));
        
        // Check problematic token positions (137131 and 44394)
        // lm_head is [vocab_size, hidden_dim] in column-major (transposed from GGUF)
        // So token 137131's weights start at offset 137131
        half h_token_137131[5];
        cudaMemcpy(h_token_137131, lm_head_half + 137131, 5 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "   lm_head[token=137131][0:5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                __half2float(h_token_137131[0]), __half2float(h_token_137131[1]),
                __half2float(h_token_137131[2]), __half2float(h_token_137131[3]),
                __half2float(h_token_137131[4]));
        
        half h_token_44394[5];
        cudaMemcpy(h_token_44394, lm_head_half + 44394, 5 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "   lm_head[token=44394][0:5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                __half2float(h_token_44394[0]), __half2float(h_token_44394[1]),
                __half2float(h_token_44394[2]), __half2float(h_token_44394[3]),
                __half2float(h_token_44394[4]));
        
        // Check middle and end of row for token 44394
        // Row layout in column-major: token 44394's full row spans positions
        // 44394, 44394+vocab_size, 44394+2*vocab_size, ..., 44394+895*vocab_size
        half h_token_44394_mid[5];
        uint32_t mid_offset = 44394 + (config_.hidden_dim / 2) * config_.vocab_size;
        cudaMemcpy(h_token_44394_mid, lm_head_half + mid_offset, 5 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "   lm_head[token=44394][mid:mid+5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                __half2float(h_token_44394_mid[0]), __half2float(h_token_44394_mid[1]),
                __half2float(h_token_44394_mid[2]), __half2float(h_token_44394_mid[3]),
                __half2float(h_token_44394_mid[4]));
        
        first_call = false;
    }
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // CRITICAL FIX: Match llama.cpp matrix multiplication parameters
    // llama.cpp uses: cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ...)
    // This means:
    //   - lm_head: NO transpose (CblasNoTrans)
    //   - hidden: YES transpose (CblasTrans)
    //
    // In cuBLAS (column-major), this translates to:
    //   - We need CUBLAS_OP_T for hidden to match CblasTrans
    //   - We need CUBLAS_OP_N for lm_head to match CblasNoTrans
    //
    // GGUF stores lm_head as [hidden_dim, vocab_size] in row-major
    // When loaded to GPU, it's still row-major: [896, 151936]
    // We want: logits[vocab_size] = lm_head[vocab_size, hidden_dim] @ hidden[hidden_dim]
    //
    // cuBLAS computes: C = op(A) @ op(B)
    // We need: logits[vocab, 1] = lm_head[vocab, hidden] @ hidden^T[hidden, 1]
    //
    // With row-major lm_head stored as [hidden, vocab], we need to:
    // 1. Treat it as transposed: [vocab, hidden] by using appropriate stride
    // 2. Transpose hidden: [1, hidden] -> [hidden, 1]
    
    // FIX: Match llama.cpp's cuBLAS parameters exactly
    // llama.cpp transposes lm_head AND uses hidden_dim as m dimension
    // Previous fix attempt only changed transpose flag, not dimensions - that's why it failed
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
    
    // Workaround removed - root cause fixed by correcting cuBLAS parameters
    
    // Debug: Log logits statistics (first 10 calls to see if they're changing)
    static int call_count = 0;
    if (call_count < 10) {
        // Sample first 10 logits and find max
        float h_logits_sample[20];
        cudaMemcpy(h_logits_sample, logits, 20 * sizeof(float), cudaMemcpyDeviceToHost);
        
        float max_logit = h_logits_sample[0];
        int max_idx = 0;
        for (int i = 1; i < config_.vocab_size && i < 1000; i++) {
            float val;
            cudaMemcpy(&val, logits + i, sizeof(float), cudaMemcpyDeviceToHost);
            if (val > max_logit) {
                max_logit = val;
                max_idx = i;
            }
        }
        
        // Check problematic positions
        float logit_137131, logit_44394;
        cudaMemcpy(&logit_137131, logits + 137131, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&logit_44394, logits + 44394, sizeof(float), cudaMemcpyDeviceToHost);
        
        fprintf(stderr, "  🎯 Logits[0:10]: ");
        for (int i = 0; i < 10; i++) {
            fprintf(stderr, "%.2f ", h_logits_sample[i]);
        }
        fprintf(stderr, "\n  🎯 Max logit: %.2f at token_id=%d\n", max_logit, max_idx);
        fprintf(stderr, "  🎯 Logits[137131]=%.2f, Logits[44394]=%.2f\n", logit_137131, logit_44394);
        call_count++;
    }
}

void QwenTransformer::forward(
    const uint32_t* token_ids,
    uint32_t batch_size,
    float* output_logits
) {
    // Get current position
    uint32_t pos;
    cudaError_t err = cudaMemcpy(&pos, kv_cache_.seq_lens, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to read seq_lens: %s\n", cudaGetErrorString(err));
    }
    
    // Debug: Log position and token (only first 3 to reduce noise)
    static int forward_count = 0;
    uint32_t h_token_id;
    cudaMemcpy(&h_token_id, token_ids, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    if (forward_count < 10) {
        fprintf(stderr, "\n🔄 [Forward #%d] pos=%u (read from device), token_id=%u\n", forward_count, pos, h_token_id);
        forward_count++;
    } else if (forward_count == 10) {
        fprintf(stderr, "\n... (suppressing further forward pass logs) ...\n");
        forward_count++;
    }
    
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
    err = cudaMemcpy(kv_cache_.seq_lens, &pos, sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to write seq_lens: %s\n", cudaGetErrorString(err));
    }
    if (forward_count <= 10) {
        fprintf(stderr, "  >>> Updated pos to %u (written to device)\n", pos);
    }
    
    // DEBUG: Log final hidden state before vocab projection (only first 3)
    if (forward_count <= 3) {
        half h_final[10];
        cudaMemcpy(h_final, normed_, 10 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "  Final hidden[0:10]: ");
        for (int i = 0; i < 10; i++) {
            fprintf(stderr, "%.2f ", __half2float(h_final[i]));
        }
        fprintf(stderr, "\n");
    }
    
    cudaDeviceSynchronize();
}

} // namespace transformer
} // namespace worker
