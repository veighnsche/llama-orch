/**
 * KV Cache Allocation
 * 
 * Provides KV (Key-Value) cache allocation for attention mechanism.
 * Stores computed keys and values to avoid recomputation during
 * autoregressive generation.
 * 
 * Spec: M0-W-1421, CUDA-5340
 */

#ifndef WORKER_KV_CACHE_H
#define WORKER_KV_CACHE_H

#include <memory>
#include <cuda_fp16.h>
#include "device_memory.h"
#include "vram_tracker.h"

namespace worker {

/**
 * Configuration for KV cache allocation.
 * 
 * Defines the shape and size of the KV cache based on model architecture.
 */
struct KVCacheConfig {
    int num_layers;          ///< Number of transformer layers
    int max_context_length;  ///< Maximum sequence length (context window)
    int num_kv_heads;        ///< Number of KV heads (for GQA)
    int head_dim;            ///< Dimension per attention head
};

/**
 * KV Cache for attention mechanism.
 * 
 * Allocates and manages device memory for storing keys and values
 * during autoregressive generation. Memory layout per layer:
 * 
 * [Keys: max_context_length × num_kv_heads × head_dim]
 * [Values: max_context_length × num_kv_heads × head_dim]
 * 
 * Features:
 * - RAII: Automatic cleanup when destroyed
 * - Zero-initialized to prevent garbage data
 * - Per-layer pointer access
 * - VRAM tracking integration
 * - Supports GQA (Grouped Query Attention)
 * 
 * Memory requirements:
 * Total bytes = 2 × num_layers × max_context_length × num_kv_heads × head_dim × sizeof(half)
 * 
 * Example (Qwen2.5-0.5B with 2048 context):
 * - 24 layers, 2048 tokens, 2 KV heads, 64 head_dim
 * - Size: 2 × 24 × 2048 × 2 × 64 × 2 = ~25 MB
 * 
 * Example (Llama-3-8B with 4096 context):
 * - 32 layers, 4096 tokens, 8 KV heads, 128 head_dim
 * - Size: 2 × 32 × 4096 × 8 × 128 × 2 = ~536 MB
 * 
 * Thread safety: Not thread-safe. Each inference should have its own KVCache.
 */
class KVCache {
public:
    /**
     * Allocate KV cache for inference.
     * 
     * Allocates device memory for keys and values across all layers.
     * Memory is zero-initialized to prevent garbage data in attention.
     * 
     * @param config Cache configuration (layers, context, heads, dims)
     * @param tracker Optional VRAM tracker for usage monitoring
     * @throws std::invalid_argument if config parameters invalid
     * @throws std::runtime_error if allocation fails (OOM)
     */
    explicit KVCache(const KVCacheConfig& config, VramTracker* tracker = nullptr);
    
    /**
     * Get device pointer to keys for specific layer.
     * 
     * Returns pointer to keys tensor with shape:
     * [max_context_length, num_kv_heads, head_dim]
     * 
     * @param layer Layer index (0-based, must be < num_layers)
     * @return Device pointer to keys (half precision)
     * @throws std::out_of_range if layer index invalid
     */
    half* keys(int layer);
    
    /**
     * Get device pointer to values for specific layer.
     * 
     * Returns pointer to values tensor with shape:
     * [max_context_length, num_kv_heads, head_dim]
     * 
     * @param layer Layer index (0-based, must be < num_layers)
     * @return Device pointer to values (half precision)
     * @throws std::out_of_range if layer index invalid
     */
    half* values(int layer);
    
    /**
     * Get total cache size in bytes.
     * 
     * @return Total allocated size in bytes
     */
    size_t size_bytes() const { return cache_->size(); }
    
    /**
     * Get configuration.
     * 
     * @return Cache configuration
     */
    const KVCacheConfig& config() const { return config_; }
    
    /**
     * Calculate required cache size for given configuration.
     * 
     * Formula: 2 × num_layers × max_context_length × num_kv_heads × head_dim × sizeof(half)
     * 
     * @param config Cache configuration
     * @return Required size in bytes
     */
    static size_t calculate_size(const KVCacheConfig& config);
    
    /**
     * Get current position in cache.
     * 
     * Position indicates how many tokens have been stored in the cache.
     * Range: [0, max_context_length]
     * 
     * @return Current position (0-based)
     */
    int position() const { return position_; }
    
    /**
     * Check if cache is full.
     * 
     * Cache is full when position >= max_context_length.
     * 
     * @return true if cache is full, false otherwise
     */
    bool is_full() const { return position_ >= config_.max_context_length; }
    
    /**
     * Get remaining capacity.
     * 
     * Returns number of tokens that can still be stored in cache.
     * 
     * @return Remaining capacity (0 if full)
     */
    int remaining_capacity() const { 
        return config_.max_context_length - position_; 
    }
    
    /**
     * Advance position after updating cache.
     * 
     * Call this after writing new keys/values to cache.
     * 
     * @param num_tokens Number of tokens to advance (default: 1)
     * @throws std::runtime_error if position would exceed max_context_length
     */
    void advance_position(int num_tokens = 1);
    
    /**
     * Reset cache for new inference.
     * 
     * Resets position to 0 and zero-initializes cache memory.
     * Use this when starting a new inference with the same cache object.
     */
    void reset();
    
private:
    KVCacheConfig config_;                      ///< Cache configuration
    std::unique_ptr<DeviceMemory> cache_;       ///< Device memory (RAII)
    size_t layer_stride_;                       ///< Bytes per layer (keys + values)
    int position_ = 0;                          ///< Current position in cache
};

} // namespace worker

#endif // WORKER_KV_CACHE_H

// ---
// Built by Foundation-Alpha 🏗️
