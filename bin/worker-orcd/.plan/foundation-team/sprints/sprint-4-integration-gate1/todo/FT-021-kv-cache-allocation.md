# FT-021: KV Cache Allocation

**Team**: Foundation-Alpha  
**Sprint**: Sprint 4 - Integration + Gate 1  
**Size**: M (2 days)  
**Days**: 39 - 40  
**Spec Ref**: M0-W-1421, CUDA-5340

---

## Story Description

Implement KV (Key-Value) cache allocation for attention mechanism. This stores computed keys and values to avoid recomputation during autoregressive generation.

---

## Acceptance Criteria

- [ ] KV cache allocated per inference request (not per model)
- [ ] Size calculated based on: 2 * num_layers * context_length * hidden_dim * sizeof(half)
- [ ] Zero-initialized to prevent garbage data
- [ ] Integrated with VramTracker for usage monitoring
- [ ] Unit tests validate allocation size calculation
- [ ] Integration tests validate cache lifecycle (allocate → use → free)
- [ ] Error handling for OOM during cache allocation
- [ ] Cache freed automatically when inference completes

---

## Dependencies

### Upstream (Blocks This Story)
- FT-013: Device memory RAII (Expected completion: Day 26)
- FT-020: Seeded RNG (Expected completion: Day 36)

### Downstream (This Story Blocks)
- FT-022: KV cache management needs allocation
- Llama/GPT teams need KV cache for attention

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/include/kv_cache.h` - KV cache interface
- `bin/worker-orcd/cuda/src/kv_cache.cpp` - KV cache implementation
- `bin/worker-orcd/cuda/tests/kv_cache_test.cpp` - Unit tests

### Key Interfaces
```cpp
// kv_cache.h
#ifndef WORKER_KV_CACHE_H
#define WORKER_KV_CACHE_H

#include <memory>
#include "device_memory.h"
#include "vram_tracker.h"

namespace worker {

struct KVCacheConfig {
    int num_layers;
    int max_context_length;
    int hidden_dim;
    int num_kv_heads;  // For GQA (Grouped Query Attention)
    int head_dim;
};

class KVCache {
public:
    /**
     * Allocate KV cache for inference.
     * 
     * Layout per layer: [Keys, Values]
     * Keys: [max_context_length, num_kv_heads, head_dim]
     * Values: [max_context_length, num_kv_heads, head_dim]
     * 
     * @param config Cache configuration
     * @param tracker VRAM tracker for usage monitoring
     * @throws CudaError if allocation fails
     */
    explicit KVCache(const KVCacheConfig& config, VramTracker* tracker = nullptr);
    
    /**
     * Get device pointer to keys for specific layer.
     * 
     * @param layer Layer index (0-based)
     * @return Device pointer to keys [max_context_length, num_kv_heads, head_dim]
     */
    half* keys(int layer);
    
    /**
     * Get device pointer to values for specific layer.
     * 
     * @param layer Layer index (0-based)
     * @return Device pointer to values [max_context_length, num_kv_heads, head_dim]
     */
    half* values(int layer);
    
    /**
     * Get total cache size in bytes.
     */
    size_t size_bytes() const { return cache_->size(); }
    
    /**
     * Get configuration.
     */
    const KVCacheConfig& config() const { return config_; }
    
    /**
     * Calculate required cache size.
     */
    static size_t calculate_size(const KVCacheConfig& config);
    
private:
    KVCacheConfig config_;
    std::unique_ptr<DeviceMemory> cache_;
    size_t layer_stride_;  // Bytes per layer (keys + values)
};

} // namespace worker

#endif // WORKER_KV_CACHE_H

// kv_cache.cpp
#include "kv_cache.h"

namespace worker {

size_t KVCache::calculate_size(const KVCacheConfig& config) {
    // Per layer: 2 (K and V) * context_length * num_kv_heads * head_dim * sizeof(half)
    size_t elements_per_layer = 2 * 
                                config.max_context_length * 
                                config.num_kv_heads * 
                                config.head_dim;
    
    size_t total_elements = elements_per_layer * config.num_layers;
    size_t total_bytes = total_elements * sizeof(half);
    
    return total_bytes;
}

KVCache::KVCache(const KVCacheConfig& config, VramTracker* tracker)
    : config_(config) {
    
    size_t total_size = calculate_size(config);
    
    // Allocate cache with zero-initialization
    cache_ = std::make_unique<DeviceMemory>(
        total_size,
        tracker,
        VramPurpose::KVCache,
        true  // zero_init
    );
    
    // Calculate stride per layer
    size_t elements_per_layer = 2 * 
                                config.max_context_length * 
                                config.num_kv_heads * 
                                config.head_dim;
    layer_stride_ = elements_per_layer * sizeof(half);
}

half* KVCache::keys(int layer) {
    if (layer < 0 || layer >= config_.num_layers) {
        throw CudaError::invalid_parameter(
            "Layer index out of range: " + std::to_string(layer)
        );
    }
    
    // Keys are at the start of each layer's allocation
    char* base = static_cast<char*>(cache_->get());
    return reinterpret_cast<half*>(base + layer * layer_stride_);
}

half* KVCache::values(int layer) {
    if (layer < 0 || layer >= config_.num_layers) {
        throw CudaError::invalid_parameter(
            "Layer index out of range: " + std::to_string(layer)
        );
    }
    
    // Values are after keys in each layer's allocation
    size_t keys_size = config_.max_context_length * 
                       config_.num_kv_heads * 
                       config_.head_dim * 
                       sizeof(half);
    
    char* base = static_cast<char*>(cache_->get());
    return reinterpret_cast<half*>(base + layer * layer_stride_ + keys_size);
}

} // namespace worker

// Integration with InferenceResult
namespace worker {

class InferenceResult {
public:
    InferenceResult(
        const Model& model,
        const std::string& prompt,
        const InferenceConfig& config
    ) : model_(model),
        config_(config),
        rng_(config.seed) {
        
        // Allocate KV cache
        KVCacheConfig kv_config{
            .num_layers = model.metadata().num_layers,
            .max_context_length = config.max_tokens,
            .hidden_dim = model.metadata().embedding_length,
            .num_kv_heads = model.metadata().num_kv_heads,
            .head_dim = model.metadata().head_dim,
        };
        
        try {
            kv_cache_ = std::make_unique<KVCache>(
                kv_config,
                &model.context().vram_tracker()
            );
        } catch (const CudaError& e) {
            // OOM during KV cache allocation
            if (e.code() == CUDA_ERROR_OUT_OF_MEMORY) {
                throw CudaError::out_of_memory(
                    "Failed to allocate KV cache (" + 
                    std::to_string(kv_config.max_context_length) + 
                    " tokens): " + e.what()
                );
            }
            throw;
        }
    }
    
private:
    const Model& model_;
    InferenceConfig config_;
    RNG rng_;
    std::unique_ptr<KVCache> kv_cache_;
    // ... other members
};

} // namespace worker

// Unit tests
// cuda/tests/kv_cache_test.cpp
#include <gtest/gtest.h>
#include "kv_cache.h"

using namespace worker;

TEST(KVCacheTest, SizeCalculation) {
    KVCacheConfig config{
        .num_layers = 12,
        .max_context_length = 2048,
        .hidden_dim = 768,
        .num_kv_heads = 12,
        .head_dim = 64,
    };
    
    size_t expected_size = 2 * 12 * 2048 * 12 * 64 * sizeof(half);
    size_t actual_size = KVCache::calculate_size(config);
    
    EXPECT_EQ(actual_size, expected_size);
}

TEST(KVCacheTest, Allocation) {
    KVCacheConfig config{
        .num_layers = 2,
        .max_context_length = 128,
        .hidden_dim = 256,
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    KVCache cache(config);
    
    EXPECT_GT(cache.size_bytes(), 0);
    EXPECT_EQ(cache.config().num_layers, 2);
}

TEST(KVCacheTest, LayerPointers) {
    KVCacheConfig config{
        .num_layers = 2,
        .max_context_length = 128,
        .hidden_dim = 256,
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    KVCache cache(config);
    
    // Get pointers for each layer
    half* keys0 = cache.keys(0);
    half* values0 = cache.values(0);
    half* keys1 = cache.keys(1);
    half* values1 = cache.values(1);
    
    EXPECT_NE(keys0, nullptr);
    EXPECT_NE(values0, nullptr);
    EXPECT_NE(keys1, nullptr);
    EXPECT_NE(values1, nullptr);
    
    // Pointers should be different
    EXPECT_NE(keys0, values0);
    EXPECT_NE(keys0, keys1);
    EXPECT_NE(values0, values1);
}

TEST(KVCacheTest, InvalidLayerIndex) {
    KVCacheConfig config{
        .num_layers = 2,
        .max_context_length = 128,
        .hidden_dim = 256,
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    KVCache cache(config);
    
    // Negative index
    EXPECT_THROW(cache.keys(-1), CudaError);
    
    // Out of range index
    EXPECT_THROW(cache.keys(2), CudaError);
}

TEST(KVCacheTest, ZeroInitialization) {
    KVCacheConfig config{
        .num_layers = 1,
        .max_context_length = 16,
        .hidden_dim = 64,
        .num_kv_heads = 2,
        .head_dim = 32,
    };
    
    KVCache cache(config);
    
    // Copy keys to host and verify zeros
    size_t keys_size = config.max_context_length * 
                       config.num_kv_heads * 
                       config.head_dim;
    std::vector<half> h_keys(keys_size);
    
    cudaMemcpy(h_keys.data(), cache.keys(0), 
               keys_size * sizeof(half), 
               cudaMemcpyDeviceToHost);
    
    // All values should be zero
    for (const half& val : h_keys) {
        EXPECT_EQ(__half2float(val), 0.0f);
    }
}

TEST(KVCacheTest, LargeCache) {
    // Test with realistic large model dimensions
    KVCacheConfig config{
        .num_layers = 32,
        .max_context_length = 4096,
        .hidden_dim = 4096,
        .num_kv_heads = 32,
        .head_dim = 128,
    };
    
    size_t expected_size = KVCache::calculate_size(config);
    
    // Should be ~2GB for this config
    EXPECT_GT(expected_size, 1024 * 1024 * 1024);  // > 1GB
    
    // Note: Actual allocation test may fail on GPUs with <4GB VRAM
    // This test just validates size calculation
}

TEST(KVCacheTest, VramTracking) {
    VramTracker tracker;
    
    KVCacheConfig config{
        .num_layers = 2,
        .max_context_length = 128,
        .hidden_dim = 256,
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    size_t initial_usage = tracker.total_usage();
    
    {
        KVCache cache(config, &tracker);
        
        size_t usage_with_cache = tracker.total_usage();
        EXPECT_GT(usage_with_cache, initial_usage);
        EXPECT_EQ(usage_with_cache - initial_usage, cache.size_bytes());
    }
    
    // After cache destroyed, usage should return to initial
    size_t final_usage = tracker.total_usage();
    EXPECT_EQ(final_usage, initial_usage);
}
```

### Implementation Notes
- KV cache layout: per layer [Keys, Values] contiguous in memory
- Zero-initialized to prevent garbage data in attention
- Allocated per inference request (not shared across requests)
- Automatically freed when InferenceResult destroyed (RAII)
- Size scales with context length (longer context = more VRAM)
- GQA support: num_kv_heads may differ from num_query_heads
- Error handling for OOM includes context length in message

---

## Testing Strategy

### Unit Tests
- Test size calculation with various configurations
- Test allocation succeeds
- Test layer pointers are valid and distinct
- Test invalid layer index throws error
- Test zero-initialization
- Test large cache size calculation
- Test VRAM tracking integration

### Integration Tests
- Test cache lifecycle with real inference
- Test OOM handling when cache too large
- Test cache freed after inference completes

### Manual Verification
1. Run unit tests: `./build/tests/kv_cache_test`
2. Monitor VRAM: `nvidia-smi dmon -s m`
3. Verify cache allocation/deallocation

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (7+ tests)
- [ ] Integration tests passing (3+ tests)
- [ ] Documentation updated (KVCache class docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §9.3 Inference Execution (M0-W-1421, CUDA-5340)
- Related Stories: FT-013 (device memory), FT-022 (KV cache management)

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **KV cache allocated**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_VRAM_RESIDENCY,
       action: ACTION_VRAM_ALLOCATE,
       target: "kv-cache".to_string(),
       device: Some(format!("GPU{}", device_id)),
       human: format!("Allocated {} MB KV cache (max_tokens={}, layers={})", bytes / 1024 / 1024, max_tokens, num_layers),
       ..Default::default()
   });
   ```

2. **KV cache freed**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_VRAM_RESIDENCY,
       action: ACTION_VRAM_DEALLOCATE,
       target: "kv-cache".to_string(),
       device: Some(format!("GPU{}", device_id)),
       human: format!("Freed {} MB KV cache", bytes / 1024 / 1024),
       ..Default::default()
   });
   ```

3. **KV cache OOM**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_VRAM_RESIDENCY,
       action: ACTION_VRAM_ALLOCATE,
       target: "kv-cache".to_string(),
       error_kind: Some("vram_oom".to_string()),
       human: format!("KV cache allocation failed: requested {} MB, only {} MB available", requested / 1024 / 1024, available / 1024 / 1024),
       ..Default::default()
   });
   ```

**Why this matters**: KV cache is the largest VRAM consumer in inference. Narration helps track allocation sizes and diagnose OOM issues.

---
*Narration guidance added by Narration-Core Team 🎀*
