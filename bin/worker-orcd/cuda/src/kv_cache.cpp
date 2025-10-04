/**
 * KV Cache Allocation Implementation
 * 
 * Spec: M0-W-1421, CUDA-5340
 */

#include "../include/kv_cache.h"
#include "cuda_error.h"
#include <stdexcept>
#include <sstream>

namespace worker {

size_t KVCache::calculate_size(const KVCacheConfig& config) {
    // Validate configuration
    if (config.num_layers <= 0) {
        throw std::invalid_argument("num_layers must be positive");
    }
    if (config.max_context_length <= 0) {
        throw std::invalid_argument("max_context_length must be positive");
    }
    if (config.num_kv_heads <= 0) {
        throw std::invalid_argument("num_kv_heads must be positive");
    }
    if (config.head_dim <= 0) {
        throw std::invalid_argument("head_dim must be positive");
    }
    
    // Calculate elements per layer
    // Keys: max_context_length × num_kv_heads × head_dim
    // Values: max_context_length × num_kv_heads × head_dim
    // Total: 2 × max_context_length × num_kv_heads × head_dim
    size_t elements_per_layer = static_cast<size_t>(2) * 
                                static_cast<size_t>(config.max_context_length) * 
                                static_cast<size_t>(config.num_kv_heads) * 
                                static_cast<size_t>(config.head_dim);
    
    // Total elements across all layers
    size_t total_elements = elements_per_layer * static_cast<size_t>(config.num_layers);
    
    // Convert to bytes (half precision = 2 bytes)
    size_t total_bytes = total_elements * sizeof(half);
    
    return total_bytes;
}

KVCache::KVCache(const KVCacheConfig& config, VramTracker* tracker)
    : config_(config) {
    
    // Calculate total size
    size_t total_size = calculate_size(config);
    
    // Build description for VRAM tracker
    std::ostringstream desc;
    desc << "KV cache: " << config.num_layers << " layers, "
         << config.max_context_length << " tokens, "
         << config.num_kv_heads << " heads, "
         << config.head_dim << " dim";
    
    try {
        // Allocate cache with zero-initialization
        // Zero-init is critical to prevent garbage data in attention
        cache_ = std::make_unique<DeviceMemory>(
            total_size,
            tracker,
            VramPurpose::KVCache,
            true  // zero_init
        );
    } catch (const std::exception& e) {
        // Re-throw with more context
        std::ostringstream err;
        err << "Failed to allocate KV cache (" 
            << (total_size / (1024 * 1024)) << " MB, "
            << config.max_context_length << " tokens, "
            << config.num_layers << " layers): " << e.what();
        throw std::runtime_error(err.str());
    }
    
    // Calculate stride per layer (keys + values)
    size_t elements_per_layer = static_cast<size_t>(2) * 
                                static_cast<size_t>(config.max_context_length) * 
                                static_cast<size_t>(config.num_kv_heads) * 
                                static_cast<size_t>(config.head_dim);
    layer_stride_ = elements_per_layer * sizeof(half);
}

half* KVCache::keys(int layer) {
    if (layer < 0 || layer >= config_.num_layers) {
        std::ostringstream err;
        err << "Layer index out of range: " << layer 
            << " (must be 0-" << (config_.num_layers - 1) << ")";
        throw std::out_of_range(err.str());
    }
    
    // Keys are at the start of each layer's allocation
    char* base = static_cast<char*>(cache_->get());
    return reinterpret_cast<half*>(base + layer * layer_stride_);
}

half* KVCache::values(int layer) {
    if (layer < 0 || layer >= config_.num_layers) {
        std::ostringstream err;
        err << "Layer index out of range: " << layer 
            << " (must be 0-" << (config_.num_layers - 1) << ")";
        throw std::out_of_range(err.str());
    }
    
    // Values are after keys in each layer's allocation
    // Keys size: max_context_length × num_kv_heads × head_dim × sizeof(half)
    size_t keys_size = static_cast<size_t>(config_.max_context_length) * 
                       static_cast<size_t>(config_.num_kv_heads) * 
                       static_cast<size_t>(config_.head_dim) * 
                       sizeof(half);
    
    char* base = static_cast<char*>(cache_->get());
    return reinterpret_cast<half*>(base + layer * layer_stride_ + keys_size);
}

} // namespace worker

// ---
// Built by Foundation-Alpha 🏗️
