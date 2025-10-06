// kv_cache.cu — KV Cache Management - LT-017
//
// Implements KV cache for efficient autoregressive generation.
// Spec: M0-W-1214, CUDA-5341
//

#include "kv_cache.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace worker {
namespace kernels {

/**
 * Update KV cache with new keys/values.
 * 
 * Copies new keys and values into cache at specified position.
 * Used during both prefill (multiple tokens) and decode (single token).
 * 
 * Grid: (num_tokens, num_heads)
 * Block: (head_dim)
 * 
 * @param cache_keys Cache keys [max_context, num_heads, head_dim]
 * @param cache_values Cache values [max_context, num_heads, head_dim]
 * @param new_keys New keys [num_tokens, num_heads, head_dim]
 * @param new_values New values [num_tokens, num_heads, head_dim]
 * @param position Starting position in cache
 * @param num_tokens Number of tokens to update
 * @param num_heads Number of KV heads
 * @param head_dim Head dimension
 */
__global__ void update_kv_cache(
    half* cache_keys,
    half* cache_values,
    const half* new_keys,
    const half* new_values,
    int position,
    int num_tokens,
    int num_heads,
    int head_dim
) {
    // Each thread handles one element
    int token_idx = blockIdx.x;  // Which token (0 to num_tokens-1)
    int head_idx = blockIdx.y;   // Which head (0 to num_heads-1)
    int dim_idx = threadIdx.x;   // Which dimension (0 to head_dim-1)
    
    // Bounds check
    if (token_idx >= num_tokens || head_idx >= num_heads || dim_idx >= head_dim) {
        return;
    }
    
    // Calculate cache position for this token
    int cache_pos = position + token_idx;
    
    // Source index in new_keys/new_values
    // Layout: [num_tokens, num_heads, head_dim]
    int src_idx = (token_idx * num_heads + head_idx) * head_dim + dim_idx;
    
    // Destination index in cache
    // Layout: [max_context, num_heads, head_dim]
    int dst_idx = (cache_pos * num_heads + head_idx) * head_dim + dim_idx;
    
    // Copy to cache
    cache_keys[dst_idx] = new_keys[src_idx];
    cache_values[dst_idx] = new_values[src_idx];
}

/**
 * Launch KV cache update kernel.
 * 
 * @param cache_keys Device pointer to cache keys
 * @param cache_values Device pointer to cache values
 * @param new_keys Device pointer to new keys
 * @param new_values Device pointer to new values
 * @param position Starting position in cache
 * @param num_tokens Number of tokens to update
 * @param num_heads Number of KV heads
 * @param head_dim Head dimension
 * @param stream CUDA stream (default: 0)
 */
void launch_update_kv_cache(
    half* cache_keys,
    half* cache_values,
    const half* new_keys,
    const half* new_values,
    int position,
    int num_tokens,
    int num_heads,
    int head_dim,
    cudaStream_t stream
) {
    // Grid: (num_tokens, num_heads)
    // Block: (head_dim)
    dim3 grid(num_tokens, num_heads);
    dim3 block(head_dim);
    
    update_kv_cache<<<grid, block, 0, stream>>>(
        cache_keys,
        cache_values,
        new_keys,
        new_values,
        position,
        num_tokens,
        num_heads,
        head_dim
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "KV cache update kernel launch failed: %s\n", 
                cudaGetErrorString(err));
    }
}

} // namespace kernels
} // namespace worker

// ---
// Built by Foundation-Alpha 🏗️
