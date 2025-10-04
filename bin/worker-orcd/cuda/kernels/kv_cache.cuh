/**
 * KV Cache Update Kernel Interface
 * 
 * Spec: M0-W-1421, CUDA-5341
 */

#ifndef WORKER_KV_CACHE_CUH
#define WORKER_KV_CACHE_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace worker {
namespace kernels {

/**
 * Launch KV cache update kernel.
 * 
 * Updates cache with new keys/values at specified position.
 * Supports both prefill (multiple tokens) and decode (single token).
 * 
 * @param cache_keys Device pointer to cache keys [max_context, num_heads, head_dim]
 * @param cache_values Device pointer to cache values [max_context, num_heads, head_dim]
 * @param new_keys Device pointer to new keys [num_tokens, num_heads, head_dim]
 * @param new_values Device pointer to new values [num_tokens, num_heads, head_dim]
 * @param position Starting position in cache (0-based)
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
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace worker

#endif // WORKER_KV_CACHE_CUH

// ---
// Built by Foundation-Alpha 🏗️
