#include "model/qwen_weight_loader.h"
#include "model_impl.h"
#include "../include/worker_ffi.h"
#include <cstdio>
#include <cuda_runtime.h>

extern "C" {

// ============================================================================
// CUDA Memory Management (for Rust weight loading)
// ============================================================================

void* cuda_malloc_device(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ cudaMalloc failed: %s (size=%zu bytes)\n", 
                cudaGetErrorString(err), size);
        return nullptr;
    }
    
    return ptr;
}

int cuda_memcpy_host_to_device(void* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0) {
        return 1; // Error
    }
    
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "❌ cudaMemcpy H2D failed: %s (size=%zu bytes)\n",
                cudaGetErrorString(err), size);
        return 1; // Error
    }
    
    return 0; // Success
}

void cuda_free_memory(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

// ============================================================================
// Model Loading from Pre-allocated GPU Pointers (Rust → C++)
// ============================================================================

/// Opaque handle to GPU pointer map
struct GpuPointerMap {
    std::map<std::string, void*> pointers;
    uint64_t total_vram_bytes;
};

GpuPointerMap* cuda_create_pointer_map(uint64_t total_vram_bytes) {
    auto map = new GpuPointerMap();
    map->total_vram_bytes = total_vram_bytes;
    return map;
}

void cuda_pointer_map_insert(
    GpuPointerMap* map,
    const char* name,
    void* gpu_ptr
) {
    if (map && name && gpu_ptr) {
        map->pointers[name] = gpu_ptr;
    }
}

CudaModel* cuda_load_model_from_pointers(
    void* ctx,
    GpuPointerMap* pointer_map,
    uint32_t vocab_size,
    uint32_t hidden_dim,
    uint32_t num_layers,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t context_length,
    int* error
) {
    try {
        if (!pointer_map) {
            *error = -1;
            return nullptr;
        }
        
        worker::model::QwenConfig config;
        config.vocab_size = vocab_size;
        config.hidden_dim = hidden_dim;
        config.num_layers = num_layers;
        config.num_heads = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.context_length = context_length;
        
        auto qwen_model = worker::model::QwenWeightLoader::load_from_gpu_pointers(
            pointer_map->pointers,
            config,
            pointer_map->total_vram_bytes
        );
        
        // Create ModelImpl with the loaded Qwen model
        auto model_impl = new worker::ModelImpl();
        model_impl->set_qwen_model(qwen_model);
        model_impl->set_vram_bytes(pointer_map->total_vram_bytes);
        
        *error = 0;
        return reinterpret_cast<CudaModel*>(model_impl);
    } catch (const std::exception& e) {
        fprintf(stderr, "❌ Model load from pointers failed: %s\n", e.what());
        *error = -1;
        return nullptr;
    }
}

void cuda_free_pointer_map(GpuPointerMap* map) {
    if (map) {
        delete map;
    }
}

} // extern "C"
