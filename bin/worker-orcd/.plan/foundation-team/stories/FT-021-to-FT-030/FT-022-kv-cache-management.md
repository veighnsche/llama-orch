# FT-022: KV Cache Management

**Team**: Foundation-Alpha  
**Sprint**: Sprint 4 - Integration + Gate 1  
**Size**: M (2 days)  
**Days**: 41 - 42  
**Spec Ref**: M0-W-1421, CUDA-5341

---

## Story Description

Implement KV cache update logic for autoregressive generation. Each token generation updates the cache with new keys/values at the current position.

---

## Acceptance Criteria

- [ ] Cache update function writes new K/V at current position
- [ ] Position tracking: starts at 0, increments each token
- [ ] Bounds checking: position < max_context_length
- [ ] Unit tests validate cache updates
- [ ] Integration tests validate multi-token generation
- [ ] Error handling for cache overflow (position >= max_context_length)
- [ ] Cache state preserved across token generation steps
- [ ] Support for both prefill (multiple tokens) and decode (single token)

---

## Dependencies

### Upstream (Blocks This Story)
- FT-021: KV cache allocation (Expected completion: Day 40)

### Downstream (This Story Blocks)
- FT-023: Integration test framework needs cache management
- Llama/GPT attention kernels need cache management

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/include/kv_cache.h` - Add management functions
- `bin/worker-orcd/cuda/src/kv_cache.cpp` - Implement management
- `bin/worker-orcd/cuda/kernels/kv_cache.cu` - Cache update kernels
- `bin/worker-orcd/cuda/tests/kv_cache_test.cpp` - Add management tests

### Key Interfaces
```cpp
// kv_cache.h (additions)
namespace worker {

class KVCache {
public:
    // ... existing methods ...
    
    /**
     * Get current position in cache.
     */
    int position() const { return position_; }
    
    /**
     * Check if cache is full.
     */
    bool is_full() const { return position_ >= config_.max_context_length; }
    
    /**
     * Get remaining capacity.
     */
    int remaining_capacity() const { 
        return config_.max_context_length - position_; 
    }
    
    /**
     * Advance position (called after updating cache).
     */
    void advance_position(int num_tokens = 1);
    
    /**
     * Reset cache (for new inference).
     */
    void reset();
    
private:
    int position_ = 0;  // Current position in cache
};

} // namespace worker

// kv_cache.cu
#ifndef WORKER_KV_CACHE_CU
#define WORKER_KV_CACHE_CU

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace worker {
namespace kernels {

/**
 * Update KV cache with new keys/values.
 * 
 * @param cache_keys Device pointer to cache keys [max_context, num_heads, head_dim]
 * @param cache_values Device pointer to cache values [max_context, num_heads, head_dim]
 * @param new_keys Device pointer to new keys [num_tokens, num_heads, head_dim]
 * @param new_values Device pointer to new values [num_tokens, num_heads, head_dim]
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
);

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

// Implementation
namespace worker {
namespace kernels {

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
    int token_idx = blockIdx.x;  // Which token
    int head_idx = blockIdx.y;   // Which head
    int dim_idx = threadIdx.x;   // Which dimension
    
    if (token_idx >= num_tokens || head_idx >= num_heads || dim_idx >= head_dim) {
        return;
    }
    
    // Calculate indices
    int cache_pos = position + token_idx;
    
    // Source index in new_keys/new_values
    int src_idx = (token_idx * num_heads + head_idx) * head_dim + dim_idx;
    
    // Destination index in cache
    int dst_idx = (cache_pos * num_heads + head_idx) * head_dim + dim_idx;
    
    // Copy to cache
    cache_keys[dst_idx] = new_keys[src_idx];
    cache_values[dst_idx] = new_values[src_idx];
}

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
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "KV cache update kernel launch failed: %s\n", 
                cudaGetErrorString(err));
    }
}

} // namespace kernels
} // namespace worker

// kv_cache.cpp (additions)
namespace worker {

void KVCache::advance_position(int num_tokens) {
    position_ += num_tokens;
    
    if (position_ > config_.max_context_length) {
        throw CudaError::out_of_memory(
            "KV cache overflow: position " + std::to_string(position_) + 
            " exceeds max context length " + std::to_string(config_.max_context_length)
        );
    }
}

void KVCache::reset() {
    position_ = 0;
    
    // Zero out cache
    cudaMemset(cache_->get(), 0, cache_->size());
}

} // namespace worker

// Integration with forward pass
namespace worker {

void InferenceResult::run_forward_pass() {
    // ... embedding lookup ...
    
    // For each layer
    for (int layer = 0; layer < model_.metadata().num_layers; ++layer) {
        // Compute Q, K, V projections
        // ... (using GEMM) ...
        
        // Update KV cache with new K, V
        kernels::launch_update_kv_cache(
            kv_cache_->keys(layer),
            kv_cache_->values(layer),
            new_keys,
            new_values,
            kv_cache_->position(),
            1,  // num_tokens (1 for decode, >1 for prefill)
            model_.metadata().num_kv_heads,
            model_.metadata().head_dim,
            stream_
        );
        
        // Attention using full cache (0 to position+1)
        // ... (using attention kernel) ...
    }
    
    // Advance cache position
    kv_cache_->advance_position(1);
}

} // namespace worker

// Unit tests (additions)
TEST(KVCacheManagementTest, PositionTracking) {
    KVCacheConfig config{
        .num_layers = 1,
        .max_context_length = 10,
        .hidden_dim = 64,
        .num_kv_heads = 2,
        .head_dim = 32,
    };
    
    KVCache cache(config);
    
    EXPECT_EQ(cache.position(), 0);
    EXPECT_FALSE(cache.is_full());
    EXPECT_EQ(cache.remaining_capacity(), 10);
    
    cache.advance_position(3);
    EXPECT_EQ(cache.position(), 3);
    EXPECT_EQ(cache.remaining_capacity(), 7);
    
    cache.advance_position(7);
    EXPECT_EQ(cache.position(), 10);
    EXPECT_TRUE(cache.is_full());
    EXPECT_EQ(cache.remaining_capacity(), 0);
}

TEST(KVCacheManagementTest, Overflow) {
    KVCacheConfig config{
        .num_layers = 1,
        .max_context_length = 5,
        .hidden_dim = 64,
        .num_kv_heads = 2,
        .head_dim = 32,
    };
    
    KVCache cache(config);
    
    cache.advance_position(5);
    EXPECT_TRUE(cache.is_full());
    
    // Advancing beyond capacity should throw
    EXPECT_THROW(cache.advance_position(1), CudaError);
}

TEST(KVCacheManagementTest, Reset) {
    KVCacheConfig config{
        .num_layers = 1,
        .max_context_length = 10,
        .hidden_dim = 64,
        .num_kv_heads = 2,
        .head_dim = 32,
    };
    
    KVCache cache(config);
    
    cache.advance_position(5);
    EXPECT_EQ(cache.position(), 5);
    
    cache.reset();
    EXPECT_EQ(cache.position(), 0);
    EXPECT_FALSE(cache.is_full());
}

TEST(KVCacheManagementTest, UpdateKernel) {
    KVCacheConfig config{
        .num_layers = 1,
        .max_context_length = 10,
        .hidden_dim = 64,
        .num_kv_heads = 2,
        .head_dim = 32,
    };
    
    KVCache cache(config);
    
    // Allocate new keys/values
    int num_tokens = 3;
    size_t kv_size = num_tokens * config.num_kv_heads * config.head_dim;
    
    std::vector<half> h_new_keys(kv_size, __float2half(1.0f));
    std::vector<half> h_new_values(kv_size, __float2half(2.0f));
    
    half *d_new_keys, *d_new_values;
    cudaMalloc(&d_new_keys, kv_size * sizeof(half));
    cudaMalloc(&d_new_values, kv_size * sizeof(half));
    
    cudaMemcpy(d_new_keys, h_new_keys.data(), kv_size * sizeof(half), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_values, h_new_values.data(), kv_size * sizeof(half), 
               cudaMemcpyHostToDevice);
    
    // Update cache
    kernels::launch_update_kv_cache(
        cache.keys(0),
        cache.values(0),
        d_new_keys,
        d_new_values,
        cache.position(),
        num_tokens,
        config.num_kv_heads,
        config.head_dim
    );
    cudaDeviceSynchronize();
    
    // Verify cache was updated
    size_t cache_size = config.max_context_length * config.num_kv_heads * config.head_dim;
    std::vector<half> h_cache_keys(cache_size);
    cudaMemcpy(h_cache_keys.data(), cache.keys(0), 
               cache_size * sizeof(half), cudaMemcpyDeviceToHost);
    
    // First 3 tokens should be 1.0
    for (int i = 0; i < num_tokens * config.num_kv_heads * config.head_dim; ++i) {
        EXPECT_NEAR(__half2float(h_cache_keys[i]), 1.0f, 0.01f);
    }
    
    // Rest should be 0.0 (zero-initialized)
    for (size_t i = num_tokens * config.num_kv_heads * config.head_dim; i < cache_size; ++i) {
        EXPECT_NEAR(__half2float(h_cache_keys[i]), 0.0f, 0.01f);
    }
    
    cudaFree(d_new_keys);
    cudaFree(d_new_values);
}
```

### Implementation Notes
- Position tracking: starts at 0, increments after each token
- Bounds checking prevents cache overflow
- Reset function for reusing cache (future optimization)
- Update kernel uses 2D grid: (num_tokens, num_heads)
- Supports both prefill (multiple tokens) and decode (single token)
- Cache overflow throws descriptive error with position info

---

## Testing Strategy

### Unit Tests
- Test position tracking and advancement
- Test overflow detection and error
- Test reset functionality
- Test update kernel with multiple tokens
- Test cache state preservation

### Integration Tests
- Test multi-token generation updates cache correctly
- Test prefill vs decode cache updates
- Test cache overflow during long generation

### Manual Verification
1. Run unit tests: `./build/tests/kv_cache_test`
2. Test with real inference generating multiple tokens
3. Verify cache position advances correctly

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (5+ tests)
- [ ] Integration tests passing (3+ tests)
- [ ] Documentation updated (cache management docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §9.3 Inference Execution (M0-W-1421, CUDA-5341)
- Related Stories: FT-021 (cache allocation), FT-023 (integration tests)

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
