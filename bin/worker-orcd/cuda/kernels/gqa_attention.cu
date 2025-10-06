// gqa_attention.cu — Grouped Query Attention - LT-015, LT-016
//
// Implements GQA for Llama models (prefill and decode phases).
// Supports variable Q and KV head counts for memory efficiency.
//
// Spec: M0-W-1214, M0-W-1430

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>

/**
 * GQA Attention Decode Kernel - Single Token Generation
 * 
 * Computes attention for single token using KV cache.
 * Each block processes one query head for one batch item.
 * 
 * Algorithm:
 *   1. Load query vector for this head
 *   2. Compute attention scores with all cached K vectors
 *   3. Apply softmax
 *   4. Compute weighted sum of V vectors
 */
__global__ void gqa_attention_decode_kernel_impl(
    half* output,
    const half* q,
    const half* k_current,
    const half* v_current,
    half* kv_cache_k,
    half* kv_cache_v,
    int batch_size,
    int cache_len,
    int max_seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    int batch = blockIdx.y;
    int q_head = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch >= batch_size || q_head >= num_q_heads) {
        return;
    }
    
    // Determine which KV head this Q head uses (GQA grouping)
    int kv_head = q_head / (num_q_heads / num_kv_heads);
    
    // Shared memory for attention scores and reduction
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    float* max_val = &shared_mem[cache_len + 1];
    float* sum_exp = &max_val[1];
    
    // Load query vector into shared memory (not registers, since all threads need access)
    __shared__ float q_shared[64];  // Assuming head_dim <= 64
    
    // Each thread loads its portion
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int q_idx = batch * num_q_heads * head_dim + q_head * head_dim + d;
        q_shared[d] = __half2float(q[q_idx]);
    }
    __syncthreads();
    
    // DEBUG: Print Q values and magnitude for first head on first few tokens
    if (tid == 0 && batch == 0 && q_head == 0 && cache_len < 5) {
        // Compute Q magnitude
        float q_mag_sq = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            q_mag_sq += q_shared[d] * q_shared[d];
        }
        float q_mag = sqrtf(q_mag_sq);
        
        printf("\n[ATTENTION DEBUG] cache_len=%d, q_head=%d, kv_head=%d\n", cache_len, q_head, kv_head);
        printf("  Q[0:5]: %.4f, %.4f, %.4f, %.4f, %.4f\n",
               q_shared[0], q_shared[1], q_shared[2], q_shared[3], q_shared[4]);
        printf("  Q magnitude: %.4f (norm of 64-dim vector)\n", q_mag);
    }
    
    // Compute attention scores for all positions (including current)
    for (int pos = tid; pos <= cache_len; pos += blockDim.x) {
        float score = 0.0f;
        
        if (pos < cache_len) {
            // Score with cached K
            // Cache layout: [batch, kv_head, pos, d] with max_seq_len stride
            for (int d = 0; d < head_dim; d++) {
                int k_cache_idx = batch * num_kv_heads * max_seq_len * head_dim +
                                  kv_head * max_seq_len * head_dim +
                                  pos * head_dim + d;
                float k_val = __half2float(kv_cache_k[k_cache_idx]);
                score += q_shared[d] * k_val;
            }
        } else {
            // Score with current K
            for (int d = 0; d < head_dim; d++) {
                int k_idx = batch * num_kv_heads * head_dim + kv_head * head_dim + d;
                float k_val = __half2float(k_current[k_idx]);
                score += q_shared[d] * k_val;
            }
            // DEBUG: Print current K values and magnitude
            if (tid == 0 && batch == 0 && q_head == 0 && cache_len < 5) {
                int k_idx = batch * num_kv_heads * head_dim + kv_head * head_dim;
                printf("  K_current[0:5]: %.4f, %.4f, %.4f, %.4f, %.4f\n",
                       __half2float(k_current[k_idx]), __half2float(k_current[k_idx+1]),
                       __half2float(k_current[k_idx+2]), __half2float(k_current[k_idx+3]),
                       __half2float(k_current[k_idx+4]));
                
                // Compute K magnitude
                float k_mag_sq = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float k_val = __half2float(k_current[k_idx + d]);
                    k_mag_sq += k_val * k_val;
                }
                printf("  K magnitude: %.4f\n", sqrtf(k_mag_sq));
                printf("  Unscaled Q·K: %.4f (before scale=%.4f)\n", score, scale);
            }
        }
        
        scores[pos] = score * scale;
    }
    __syncthreads();
    
    // Find max score for numerical stability
    if (tid == 0) {
        float max_score = -1e9f;
        for (int i = 0; i <= cache_len; i++) {
            max_score = fmaxf(max_score, scores[i]);
        }
        max_val[0] = max_score;
        
        // DEBUG: Print raw attention scores (AFTER scaling)
        if (batch == 0 && q_head == 0 && cache_len < 5) {
            printf("  DEBUG: cache_len=%d, should have %d scores\n", cache_len, cache_len + 1);
            printf("  Scaled scores (after scale): ");
            for (int i = 0; i <= cache_len && i < 8; i++) {
                printf("[%d]=%.4f ", i, scores[i]);
            }
            printf("\n  Max scaled score: %.4f\n", max_score);
        }
    }
    __syncthreads();
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int pos = tid; pos <= cache_len; pos += blockDim.x) {
        float exp_score = expf(scores[pos] - max_val[0]);
        scores[pos] = exp_score;
        local_sum += exp_score;
    }
    
    // Reduce sum across threads
    __shared__ float partial_sums[256];
    partial_sums[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        sum_exp[0] = partial_sums[0];
        
        // DEBUG: Verify softmax sum
        if (batch == 0 && q_head == 0 && cache_len < 5) {
            printf("  Softmax sum: %.6f (should be ~1.0)\n", sum_exp[0]);
        }
    }
    __syncthreads();
    
    // Normalize to get attention weights
    for (int pos = tid; pos <= cache_len; pos += blockDim.x) {
        scores[pos] /= sum_exp[0];
    }
    __syncthreads();
    
    // DEBUG: Print normalized attention weights
    if (tid == 0 && batch == 0 && q_head == 0 && cache_len < 5) {
        printf("  Attention weights (should have %d): ", cache_len + 1);
        float weight_sum = 0.0f;
        for (int i = 0; i <= cache_len && i < 8; i++) {
            printf("[%d]=%.4f ", i, scores[i]);
            weight_sum += scores[i];
        }
        printf("\n  Weight sum: %.6f (should be ~1.0)\n", weight_sum);
    }
    
    // Compute weighted sum of V vectors
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        
        for (int pos = 0; pos < cache_len; pos++) {
            // Cache layout: [batch, kv_head, pos, d] with max_seq_len stride
            int v_cache_idx = batch * num_kv_heads * max_seq_len * head_dim +
                              kv_head * max_seq_len * head_dim +
                              pos * head_dim + d;
            float v_val = __half2float(kv_cache_v[v_cache_idx]);
            out_val += scores[pos] * v_val;
        }
        
        // Add current V
        int v_idx = batch * num_kv_heads * head_dim + kv_head * head_dim + d;
        float v_val = __half2float(v_current[v_idx]);
        out_val += scores[cache_len] * v_val;
        
        // DEBUG: Print V values and output
        if (d < 5 && batch == 0 && q_head == 0 && cache_len < 5) {
            printf("  V_current[%d]: %.4f, out_val[%d]: %.4f\n", d, v_val, d, out_val);
        }
        
        int out_idx = batch * num_q_heads * head_dim + q_head * head_dim + d;
        output[out_idx] = __float2half(out_val);
        
        // Write current K, V to cache at position cache_len (only once per KV head)
        // Use one q_head per KV group to perform the write
        if (kv_cache_k != nullptr && (q_head % (num_q_heads / num_kv_heads) == 0)) {
            int k_idx = batch * num_kv_heads * head_dim + kv_head * head_dim + d;
            // Cache layout: [batch, kv_head, pos, d] with max_seq_len stride
            int cache_write_idx = batch * num_kv_heads * max_seq_len * head_dim +
                                  kv_head * max_seq_len * head_dim +
                                  cache_len * head_dim + d;
            kv_cache_k[cache_write_idx] = k_current[k_idx];
            kv_cache_v[cache_write_idx] = v_current[v_idx];
        }
    }
}

/**
 * GQA Attention Prefill Kernel (Simplified for single token)
 * 
 * For seq_len=1, this is essentially the same as decode but without cache.
 */
__global__ void gqa_attention_prefill_kernel(
    half* output,
    const half* q,
    const half* k,
    const half* v,
    half* kv_cache_k,
    half* kv_cache_v,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    int batch = blockIdx.y;
    int q_head = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch >= batch_size || q_head >= num_q_heads) {
        return;
    }
    
    // For seq_len=1, just compute self-attention
    if (seq_len != 1) {
        // Multi-token prefill not implemented yet
        return;
    }
    
    int kv_head = q_head / (num_q_heads / num_kv_heads);
    
    // For single token: attention score is just 1.0 (only attending to itself)
    // Output = V (after proper projection)
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int v_idx = batch * num_kv_heads * head_dim + kv_head * head_dim + d;
        int k_idx = batch * num_kv_heads * head_dim + kv_head * head_dim + d;
        int out_idx = batch * num_q_heads * head_dim + q_head * head_dim + d;
        output[out_idx] = v[v_idx];
        
        // Write K, V to cache at position 0 (only once per KV head)
        // Cache layout: [batch, kv_head, pos, d] with max_seq_len stride
        // NOTE: This function doesn't receive max_seq_len parameter, so we can't write to cache correctly
        // The prefill kernel should not be used - always use decode kernel even for first token
        // TODO: Remove this prefill kernel or fix the API to pass max_seq_len
    }
}

/**
 * Legacy wrapper - redirects to new implementation
 */
__global__ void gqa_attention_decode_kernel(
    half* output,
    const half* q,
    const half* k_current,
    const half* v_current,
    half* kv_cache_k,
    half* kv_cache_v,
    int batch_size,
    int cache_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    // This is now handled by gqa_attention_decode_kernel_impl
    // This wrapper exists for compatibility
}

extern "C" {

/**
 * GQA Attention Prefill
 * 
 * @param output Output tensor [batch, seq_len, num_q_heads * head_dim]
 * @param q Query tensor [batch, seq_len, num_q_heads, head_dim]
 * @param k Key tensor [batch, seq_len, num_kv_heads, head_dim]
 * @param v Value tensor [batch, seq_len, num_kv_heads, head_dim]
 * @param kv_cache_k KV cache for keys
 * @param kv_cache_v KV cache for values
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param num_q_heads Number of query heads
 * @param num_kv_heads Number of KV heads
 * @param head_dim Dimension per head
 * @param scale Attention scale (1/sqrt(head_dim))
 * @return 0 on success, error code on failure
 */
int cuda_gqa_attention_prefill(
    half* output,
    const half* q,
    const half* k,
    const half* v,
    half* kv_cache_k,
    half* kv_cache_v,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    // Validate dimensions
    if (batch_size <= 0 || seq_len <= 0 || num_q_heads <= 0 ||
        num_kv_heads <= 0 || head_dim <= 0) {
        fprintf(stderr, "GQA Prefill: Invalid dimensions\n");
        return -1;
    }
    
    if (num_q_heads % num_kv_heads != 0) {
        fprintf(stderr, "GQA Prefill: num_q_heads must be divisible by num_kv_heads\n");
        return -1;
    }
    
    // Launch kernel
    dim3 grid(num_q_heads, batch_size);
    dim3 block(256);  // Use 256 threads for better occupancy
    
    gqa_attention_prefill_kernel<<<grid, block>>>(
        output, q, k, v, kv_cache_k, kv_cache_v,
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, scale
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GQA Prefill kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

/**
 * GQA Attention Decode
 * 
 * @param output Output tensor [batch, 1, num_q_heads * head_dim]
 * @param q Query tensor [batch, 1, num_q_heads, head_dim]
 * @param k_current Current key [batch, 1, num_kv_heads, head_dim]
 * @param v_current Current value [batch, 1, num_kv_heads, head_dim]
 * @param kv_cache_k KV cache for keys
 * @param kv_cache_v KV cache for values
 * @param batch_size Batch size
 * @param cache_len Current cache length
 * @param num_q_heads Number of query heads
 * @param num_kv_heads Number of KV heads
 * @param head_dim Dimension per head
 * @param scale Attention scale
 * @return 0 on success, error code on failure
 */
int cuda_gqa_attention_decode(
    half* output,
    const half* q,
    const half* k_current,
    const half* v_current,
    half* kv_cache_k,
    half* kv_cache_v,
    int batch_size,
    int cache_len,
    int max_seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    // Validate dimensions
    if (batch_size <= 0 || cache_len < 0 || num_q_heads <= 0 ||
        num_kv_heads <= 0 || head_dim <= 0) {
        fprintf(stderr, "GQA Decode: Invalid dimensions\n");
        return -1;
    }
    
    if (num_q_heads % num_kv_heads != 0) {
        fprintf(stderr, "GQA Decode: num_q_heads must be divisible by num_kv_heads\n");
        return -1;
    }
    
    // Launch kernel with shared memory for scores
    dim3 grid(num_q_heads, batch_size);
    dim3 block(256);
    size_t shared_mem_size = (cache_len + 1 + 2) * sizeof(float) + 256 * sizeof(float);
    
    gqa_attention_decode_kernel_impl<<<grid, block, shared_mem_size>>>(
        output, q, k_current, v_current, kv_cache_k, kv_cache_v,
        batch_size, cache_len, max_seq_len, num_q_heads, num_kv_heads, head_dim, scale
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GQA Decode kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

/**
 * Unified GQA attention wrapper for transformer
 * Automatically chooses between prefill and decode based on cache_len
 */
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
) {
    const half* q_half = reinterpret_cast<const half*>(q);
    const half* k_half = reinterpret_cast<const half*>(k);
    const half* v_half = reinterpret_cast<const half*>(v);
    half* k_cache_half = const_cast<half*>(reinterpret_cast<const half*>(k_cache));
    half* v_cache_half = const_cast<half*>(reinterpret_cast<const half*>(v_cache));
    half* output_half = reinterpret_cast<half*>(output);
    
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // Always use decode kernel (it handles cache_len=0 correctly)
    // The prefill kernel has a bug where it doesn't use max_seq_len for cache indexing
    cuda_gqa_attention_decode(
        output_half,
        q_half,
        k_half,
        v_half,
        k_cache_half,
        v_cache_half,
        batch_size,
        cache_len,
        max_seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale
    );
}

} // extern "C"
