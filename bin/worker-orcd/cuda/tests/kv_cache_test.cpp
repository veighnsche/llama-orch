/**
 * KV Cache Unit Tests
 * 
 * Tests KV cache allocation, lifecycle, and VRAM tracking.
 * 
 * Spec: M0-W-1421, CUDA-5340
 */

#include <gtest/gtest.h>
#include "../include/kv_cache.h"
#include "../include/vram_tracker.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

using namespace worker;

// ============================================================================
// Size Calculation Tests
// ============================================================================

TEST(KVCacheTest, SizeCalculation_SmallModel) {
    // Qwen2.5-0.5B-like config (simplified)
    KVCacheConfig config{
        .num_layers = 24,
        .max_context_length = 2048,
        .num_kv_heads = 2,
        .head_dim = 64,
    };
    
    // Expected: 2 * 24 * 2048 * 2 * 64 * 2 bytes
    size_t expected_size = 2 * 24 * 2048 * 2 * 64 * sizeof(half);
    size_t actual_size = KVCache::calculate_size(config);
    
    EXPECT_EQ(actual_size, expected_size);
    
    // Should be ~25 MB (24 MB exactly = 25165824 bytes)
    EXPECT_GE(actual_size, 24 * 1024 * 1024);
    EXPECT_LT(actual_size, 26 * 1024 * 1024);
}

TEST(KVCacheTest, SizeCalculation_MediumModel) {
    // Llama-3-8B-like config
    KVCacheConfig config{
        .num_layers = 32,
        .max_context_length = 4096,
        .num_kv_heads = 8,
        .head_dim = 128,
    };
    
    size_t expected_size = 2 * 32 * 4096 * 8 * 128 * sizeof(half);
    size_t actual_size = KVCache::calculate_size(config);
    
    EXPECT_EQ(actual_size, expected_size);
    
    // Should be ~536 MB
    EXPECT_GT(actual_size, 500 * 1024 * 1024);
    EXPECT_LT(actual_size, 600 * 1024 * 1024);
}

TEST(KVCacheTest, SizeCalculation_LargeContext) {
    // Long context config
    KVCacheConfig config{
        .num_layers = 32,
        .max_context_length = 32768,  // 32K context
        .num_kv_heads = 8,
        .head_dim = 128,
    };
    
    size_t actual_size = KVCache::calculate_size(config);
    
    // Should be ~4.3 GB
    EXPECT_GT(actual_size, 4000ULL * 1024 * 1024);
    EXPECT_LT(actual_size, 5000ULL * 1024 * 1024);
}

TEST(KVCacheTest, SizeCalculation_InvalidConfig) {
    // Negative num_layers
    KVCacheConfig config1{
        .num_layers = -1,
        .max_context_length = 2048,
        .num_kv_heads = 2,
        .head_dim = 64,
    };
    EXPECT_THROW(KVCache::calculate_size(config1), std::invalid_argument);
    
    // Zero max_context_length
    KVCacheConfig config2{
        .num_layers = 24,
        .max_context_length = 0,
        .num_kv_heads = 2,
        .head_dim = 64,
    };
    EXPECT_THROW(KVCache::calculate_size(config2), std::invalid_argument);
    
    // Negative num_kv_heads
    KVCacheConfig config3{
        .num_layers = 24,
        .max_context_length = 2048,
        .num_kv_heads = -1,
        .head_dim = 64,
    };
    EXPECT_THROW(KVCache::calculate_size(config3), std::invalid_argument);
    
    // Zero head_dim
    KVCacheConfig config4{
        .num_layers = 24,
        .max_context_length = 2048,
        .num_kv_heads = 2,
        .head_dim = 0,
    };
    EXPECT_THROW(KVCache::calculate_size(config4), std::invalid_argument);
}

// ============================================================================
// Allocation Tests
// ============================================================================

TEST(KVCacheTest, Allocation_Basic) {
    KVCacheConfig config{
        .num_layers = 2,
        .max_context_length = 128,
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    KVCache cache(config);
    
    EXPECT_GT(cache.size_bytes(), 0);
    EXPECT_EQ(cache.config().num_layers, 2);
    EXPECT_EQ(cache.config().max_context_length, 128);
    EXPECT_EQ(cache.config().num_kv_heads, 4);
    EXPECT_EQ(cache.config().head_dim, 64);
}

TEST(KVCacheTest, Allocation_WithTracker) {
    VramTracker tracker;
    
    KVCacheConfig config{
        .num_layers = 2,
        .max_context_length = 128,
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

// ============================================================================
// Layer Pointer Tests
// ============================================================================

TEST(KVCacheTest, LayerPointers_Valid) {
    KVCacheConfig config{
        .num_layers = 3,
        .max_context_length = 128,
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    KVCache cache(config);
    
    // Get pointers for each layer
    half* keys0 = cache.keys(0);
    half* values0 = cache.values(0);
    half* keys1 = cache.keys(1);
    half* values1 = cache.values(1);
    half* keys2 = cache.keys(2);
    half* values2 = cache.values(2);
    
    // All pointers should be non-null
    EXPECT_NE(keys0, nullptr);
    EXPECT_NE(values0, nullptr);
    EXPECT_NE(keys1, nullptr);
    EXPECT_NE(values1, nullptr);
    EXPECT_NE(keys2, nullptr);
    EXPECT_NE(values2, nullptr);
    
    // Pointers should be different
    EXPECT_NE(keys0, values0);
    EXPECT_NE(keys0, keys1);
    EXPECT_NE(keys1, keys2);
    EXPECT_NE(values0, values1);
    EXPECT_NE(values1, values2);
}

TEST(KVCacheTest, LayerPointers_InvalidIndex) {
    KVCacheConfig config{
        .num_layers = 2,
        .max_context_length = 128,
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    KVCache cache(config);
    
    // Negative index
    EXPECT_THROW(cache.keys(-1), std::out_of_range);
    EXPECT_THROW(cache.values(-1), std::out_of_range);
    
    // Out of range index
    EXPECT_THROW(cache.keys(2), std::out_of_range);
    EXPECT_THROW(cache.values(2), std::out_of_range);
    
    // Way out of range
    EXPECT_THROW(cache.keys(100), std::out_of_range);
    EXPECT_THROW(cache.values(100), std::out_of_range);
}

TEST(KVCacheTest, LayerPointers_Spacing) {
    KVCacheConfig config{
        .num_layers = 2,
        .max_context_length = 128,
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    KVCache cache(config);
    
    half* keys0 = cache.keys(0);
    half* values0 = cache.values(0);
    half* keys1 = cache.keys(1);
    
    // Values should be after keys within same layer
    size_t keys_size = config.max_context_length * config.num_kv_heads * config.head_dim;
    EXPECT_EQ(reinterpret_cast<char*>(values0) - reinterpret_cast<char*>(keys0),
              keys_size * sizeof(half));
    
    // Next layer keys should be after previous layer values
    size_t layer_stride = 2 * keys_size * sizeof(half);
    EXPECT_EQ(reinterpret_cast<char*>(keys1) - reinterpret_cast<char*>(keys0),
              layer_stride);
}

// ============================================================================
// Zero Initialization Tests
// ============================================================================

TEST(KVCacheTest, ZeroInitialization_Keys) {
    KVCacheConfig config{
        .num_layers = 1,
        .max_context_length = 16,
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

TEST(KVCacheTest, ZeroInitialization_Values) {
    KVCacheConfig config{
        .num_layers = 1,
        .max_context_length = 16,
        .num_kv_heads = 2,
        .head_dim = 32,
    };
    
    KVCache cache(config);
    
    // Copy values to host and verify zeros
    size_t values_size = config.max_context_length * 
                         config.num_kv_heads * 
                         config.head_dim;
    std::vector<half> h_values(values_size);
    
    cudaMemcpy(h_values.data(), cache.values(0), 
               values_size * sizeof(half), 
               cudaMemcpyDeviceToHost);
    
    // All values should be zero
    for (const half& val : h_values) {
        EXPECT_EQ(__half2float(val), 0.0f);
    }
}

TEST(KVCacheTest, ZeroInitialization_AllLayers) {
    KVCacheConfig config{
        .num_layers = 3,
        .max_context_length = 8,
        .num_kv_heads = 2,
        .head_dim = 16,
    };
    
    KVCache cache(config);
    
    size_t elements_per_tensor = config.max_context_length * 
                                 config.num_kv_heads * 
                                 config.head_dim;
    std::vector<half> h_data(elements_per_tensor);
    
    // Check all layers
    for (int layer = 0; layer < config.num_layers; ++layer) {
        // Check keys
        cudaMemcpy(h_data.data(), cache.keys(layer), 
                   elements_per_tensor * sizeof(half), 
                   cudaMemcpyDeviceToHost);
        for (const half& val : h_data) {
            EXPECT_EQ(__half2float(val), 0.0f) 
                << "Non-zero key in layer " << layer;
        }
        
        // Check values
        cudaMemcpy(h_data.data(), cache.values(layer), 
                   elements_per_tensor * sizeof(half), 
                   cudaMemcpyDeviceToHost);
        for (const half& val : h_data) {
            EXPECT_EQ(__half2float(val), 0.0f) 
                << "Non-zero value in layer " << layer;
        }
    }
}

// ============================================================================
// VRAM Tracking Tests
// ============================================================================

TEST(KVCacheTest, VramTracking_Allocation) {
    VramTracker tracker;
    
    KVCacheConfig config{
        .num_layers = 2,
        .max_context_length = 128,
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    size_t initial_usage = tracker.total_usage();
    size_t expected_size = KVCache::calculate_size(config);
    
    KVCache cache(config, &tracker);
    
    size_t usage_with_cache = tracker.total_usage();
    EXPECT_EQ(usage_with_cache - initial_usage, expected_size);
    EXPECT_EQ(usage_with_cache - initial_usage, cache.size_bytes());
}

TEST(KVCacheTest, VramTracking_Deallocation) {
    VramTracker tracker;
    
    KVCacheConfig config{
        .num_layers = 2,
        .max_context_length = 128,
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    size_t initial_usage = tracker.total_usage();
    
    {
        KVCache cache(config, &tracker);
        EXPECT_GT(tracker.total_usage(), initial_usage);
    }
    
    // After cache destroyed, usage should return to initial
    size_t final_usage = tracker.total_usage();
    EXPECT_EQ(final_usage, initial_usage);
}

TEST(KVCacheTest, VramTracking_MultipleAllocations) {
    VramTracker tracker;
    
    KVCacheConfig config1{
        .num_layers = 2,
        .max_context_length = 128,
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    KVCacheConfig config2{
        .num_layers = 3,
        .max_context_length = 256,
        .num_kv_heads = 8,
        .head_dim = 128,
    };
    
    size_t initial_usage = tracker.total_usage();
    
    KVCache cache1(config1, &tracker);
    size_t usage_after_cache1 = tracker.total_usage();
    EXPECT_EQ(usage_after_cache1 - initial_usage, cache1.size_bytes());
    
    KVCache cache2(config2, &tracker);
    size_t usage_after_cache2 = tracker.total_usage();
    EXPECT_EQ(usage_after_cache2 - initial_usage, 
              cache1.size_bytes() + cache2.size_bytes());
}

// ============================================================================
// Realistic Model Tests
// ============================================================================

TEST(KVCacheTest, RealisticModel_Qwen2_5_0_5B) {
    // Qwen2.5-0.5B with 2K context
    KVCacheConfig config{
        .num_layers = 24,
        .max_context_length = 2048,
        .num_kv_heads = 2,
        .head_dim = 64,
    };
    
    size_t size = KVCache::calculate_size(config);
    
    // Should be ~25 MB (24 MB exactly = 25165824 bytes)
    EXPECT_GE(size, 24 * 1024 * 1024);
    EXPECT_LT(size, 26 * 1024 * 1024);
    
    // Allocation should succeed on any modern GPU
    EXPECT_NO_THROW({
        KVCache cache(config);
        EXPECT_GT(cache.size_bytes(), 0);
    });
}

TEST(KVCacheTest, RealisticModel_Llama3_8B) {
    // Llama-3-8B with 4K context
    KVCacheConfig config{
        .num_layers = 32,
        .max_context_length = 4096,
        .num_kv_heads = 8,
        .head_dim = 128,
    };
    
    size_t size = KVCache::calculate_size(config);
    
    // Should be ~536 MB
    EXPECT_GT(size, 500 * 1024 * 1024);
    EXPECT_LT(size, 600 * 1024 * 1024);
    
    // Note: Actual allocation may fail on GPUs with <2GB VRAM
    // This test just validates size calculation
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(KVCacheTest, EdgeCase_SingleLayer) {
    KVCacheConfig config{
        .num_layers = 1,
        .max_context_length = 128,
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    KVCache cache(config);
    
    EXPECT_NO_THROW({
        half* keys = cache.keys(0);
        half* values = cache.values(0);
        EXPECT_NE(keys, nullptr);
        EXPECT_NE(values, nullptr);
    });
    
    EXPECT_THROW(cache.keys(1), std::out_of_range);
}

TEST(KVCacheTest, EdgeCase_SingleHead) {
    KVCacheConfig config{
        .num_layers = 2,
        .max_context_length = 128,
        .num_kv_heads = 1,  // Single head (MHA, not GQA)
        .head_dim = 64,
    };
    
    EXPECT_NO_THROW({
        KVCache cache(config);
        EXPECT_GT(cache.size_bytes(), 0);
    });
}

TEST(KVCacheTest, EdgeCase_SmallContext) {
    KVCacheConfig config{
        .num_layers = 2,
        .max_context_length = 1,  // Minimal context
        .num_kv_heads = 4,
        .head_dim = 64,
    };
    
    EXPECT_NO_THROW({
        KVCache cache(config);
        EXPECT_GT(cache.size_bytes(), 0);
    });
}

// ---
// Built by Foundation-Alpha 🏗️
