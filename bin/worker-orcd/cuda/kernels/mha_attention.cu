// Multi-Head Attention (MHA) kernel for GPT architecture
//
// Implements standard MHA where each head has separate K/V projections.
// This differs from Llama's GQA where multiple query heads share K/V.
//
// MHA: num_heads = num_kv_heads (all heads independent)
// GQA: num_kv_heads < num_heads (grouped queries)
//
// Spec: M0-W-1215
// Story: GT-017, GT-018

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>

// Softmax kernel for attention scores
__global__ void softmax_kernel(
    half* output,
    const half* input,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k
) {
    // Each block processes one (batch, head, query_pos)
    int batch_idx = blockIdx.x / (num_heads * seq_len_q);
    int head_idx = (blockIdx.x / seq_len_q) % num_heads;
    int q_idx = blockIdx.x % seq_len_q;
    
    if (batch_idx >= batch_size) return;
    
    // Offset for this attention row
    int offset = ((batch_idx * num_heads + head_idx) * seq_len_q + q_idx) * seq_len_k;
    const half* scores = input + offset;
    half* probs = output + offset;
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    
    // Step 1: Find max for numerical stability
    float thread_max = -INFINITY;
    for (int k = threadIdx.x; k < seq_len_k; k += blockDim.x) {
        float val = __half2float(scores[k]);
        thread_max = fmaxf(thread_max, val);
    }
    shared_mem[threadIdx.x] = thread_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] = fmaxf(shared_mem[threadIdx.x], 
                                            shared_mem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_mem[0];
    __syncthreads();
    
    // Step 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int k = threadIdx.x; k < seq_len_k; k += blockDim.x) {
        float val = expf(__half2float(scores[k]) - max_val);
        probs[k] = __float2half(val);
        thread_sum += val;
    }
    shared_mem[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Reduce to find total sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum = shared_mem[0];
    __syncthreads();
    
    // Step 3: Normalize
    float inv_sum = 1.0f / sum;
    for (int k = threadIdx.x; k < seq_len_k; k += blockDim.x) {
        probs[k] = __float2half(__half2float(probs[k]) * inv_sum);
    }
}

// MHA prefill: Compute attention for full sequence
//
// Q: [batch, num_heads, seq_len_q, head_dim]
// K: [batch, num_heads, seq_len_k, head_dim]
// V: [batch, num_heads, seq_len_v, head_dim]
// Output: [batch, num_heads, seq_len_q, head_dim]
//
// Algorithm:
// 1. scores = Q @ K^T / sqrt(head_dim)
// 2. attn = softmax(scores)
// 3. output = attn @ V
extern "C" void cuda_mha_attention_prefill(
    const half* q,              // Query [batch, num_heads, seq_len_q, head_dim]
    const half* k,              // Key [batch, num_heads, seq_len_k, head_dim]
    const half* v,              // Value [batch, num_heads, seq_len_v, head_dim]
    half* output,               // Output [batch, num_heads, seq_len_q, head_dim]
    half* scores_workspace,     // Workspace for scores [batch, num_heads, seq_len_q, seq_len_k]
    half* attn_workspace,       // Workspace for attention [batch, num_heads, seq_len_q, seq_len_k]
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    cublasSetStream(cublas_handle, stream);
    
    float scale = 1.0f / sqrtf((float)head_dim);
    half h_scale = __float2half(scale);
    half h_zero = __float2half(0.0f);
    
    // For each (batch, head) pair, compute attention
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            int offset_q = (b * num_heads + h) * seq_len_q * head_dim;
            int offset_k = (b * num_heads + h) * seq_len_k * head_dim;
            int offset_v = (b * num_heads + h) * seq_len_k * head_dim;
            int offset_scores = (b * num_heads + h) * seq_len_q * seq_len_k;
            int offset_out = (b * num_heads + h) * seq_len_q * head_dim;
            
            // Step 1: scores = Q @ K^T * scale
            // Q: [seq_len_q, head_dim], K^T: [head_dim, seq_len_k]
            // scores: [seq_len_q, seq_len_k]
            cublasGemmEx(
                cublas_handle,
                CUBLAS_OP_T,    // K is transposed
                CUBLAS_OP_N,    // Q is not transposed
                seq_len_k,      // Rows of K^T
                seq_len_q,      // Cols of Q
                head_dim,       // Cols of K^T, rows of Q
                &h_scale,
                k + offset_k, CUDA_R_16F, head_dim,
                q + offset_q, CUDA_R_16F, head_dim,
                &h_zero,
                scores_workspace + offset_scores, CUDA_R_16F, seq_len_k,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT
            );
            
            // Step 2: attn = softmax(scores)
            int num_blocks = seq_len_q;
            int threads_per_block = 256;
            size_t shared_mem_size = threads_per_block * sizeof(float);
            
            softmax_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
                attn_workspace + offset_scores,
                scores_workspace + offset_scores,
                1, 1, seq_len_q, seq_len_k
            );
            
            // Step 3: output = attn @ V
            // attn: [seq_len_q, seq_len_k], V: [seq_len_k, head_dim]
            // output: [seq_len_q, head_dim]
            half h_one = __float2half(1.0f);
            cublasGemmEx(
                cublas_handle,
                CUBLAS_OP_N,    // V is not transposed
                CUBLAS_OP_N,    // attn is not transposed
                head_dim,       // Rows of V
                seq_len_q,      // Cols of attn
                seq_len_k,      // Cols of V, rows of attn
                &h_one,
                v + offset_v, CUDA_R_16F, head_dim,
                attn_workspace + offset_scores, CUDA_R_16F, seq_len_k,
                &h_zero,
                output + offset_out, CUDA_R_16F, head_dim,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT
            );
        }
    }
}

// MHA decode: Compute attention for single new token
//
// Used during autoregressive generation where we process one token at a time.
// KV cache stores all previous keys and values.
extern "C" void cuda_mha_attention_decode(
    const half* q,              // Query [batch, num_heads, 1, head_dim]
    const half* k_cache,        // Key cache [batch, num_heads, max_seq_len, head_dim]
    const half* v_cache,        // Value cache [batch, num_heads, max_seq_len, head_dim]
    half* output,               // Output [batch, num_heads, 1, head_dim]
    half* scores_workspace,     // Workspace for scores [batch, num_heads, 1, current_len]
    half* attn_workspace,       // Workspace for attention [batch, num_heads, 1, current_len]
    int batch_size,
    int num_heads,
    int current_len,            // Current sequence length (including new token)
    int head_dim,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    cublasSetStream(cublas_handle, stream);
    
    float scale = 1.0f / sqrtf((float)head_dim);
    half h_scale = __float2half(scale);
    half h_zero = __float2half(0.0f);
    half h_one = __float2half(1.0f);
    
    // For each (batch, head) pair
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            int offset_q = (b * num_heads + h) * head_dim;
            int offset_k = (b * num_heads + h) * current_len * head_dim;
            int offset_v = (b * num_heads + h) * current_len * head_dim;
            int offset_scores = (b * num_heads + h) * current_len;
            int offset_out = (b * num_heads + h) * head_dim;
            
            // Step 1: scores = q @ K_cache^T * scale
            // q: [1, head_dim], K_cache^T: [head_dim, current_len]
            // scores: [1, current_len]
            cublasGemmEx(
                cublas_handle,
                CUBLAS_OP_T,    // K is transposed
                CUBLAS_OP_N,    // q is not transposed
                current_len,    // Rows of K^T
                1,              // Cols of q
                head_dim,       // Cols of K^T, rows of q
                &h_scale,
                k_cache + offset_k, CUDA_R_16F, head_dim,
                q + offset_q, CUDA_R_16F, head_dim,
                &h_zero,
                scores_workspace + offset_scores, CUDA_R_16F, current_len,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT
            );
            
            // Step 2: attn = softmax(scores)
            softmax_kernel<<<1, 256, 256 * sizeof(float), stream>>>(
                attn_workspace + offset_scores,
                scores_workspace + offset_scores,
                1, 1, 1, current_len
            );
            
            // Step 3: output = attn @ V_cache
            // attn: [1, current_len], V_cache: [current_len, head_dim]
            // output: [1, head_dim]
            cublasGemmEx(
                cublas_handle,
                CUBLAS_OP_N,    // V is not transposed
                CUBLAS_OP_N,    // attn is not transposed
                head_dim,       // Rows of V
                1,              // Cols of attn
                current_len,    // Cols of V, rows of attn
                &h_one,
                v_cache + offset_v, CUDA_R_16F, head_dim,
                attn_workspace + offset_scores, CUDA_R_16F, current_len,
                &h_zero,
                output + offset_out, CUDA_R_16F, head_dim,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT
            );
        }
    }
}

// Calculate workspace size for MHA attention
extern "C" size_t cuda_mha_workspace_size(
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k
) {
    // Need space for scores and attention weights
    size_t scores_size = batch_size * num_heads * seq_len_q * seq_len_k * sizeof(half);
    size_t attn_size = batch_size * num_heads * seq_len_q * seq_len_k * sizeof(half);
    return scores_size + attn_size;
}

// ---
// Crafted by GPT-Gamma 🤖
