/**
 * Model Implementation - Simplified stub for FFI
 * 
 * This is a minimal implementation to make FFI work.
 * GGUF parsing is now done in Rust.
 * 
 * Spec: M0-W-1211
 */

#ifndef WORKER_MODEL_IMPL_H
#define WORKER_MODEL_IMPL_H

#include "context.h"
#include <memory>
#include <string>

namespace worker {

/**
 * Loaded model with VRAM allocation (stub)
 */
class ModelImpl {
public:
    /**
     * Load model from GGUF file (stub - just stores path)
     * 
     * @param ctx CUDA context
     * @param model_path Path to .gguf file
     * @throws CudaError on failure
     */
    ModelImpl(Context& ctx, const char* model_path);
    
    ~ModelImpl();
    
    /**
     * Get VRAM bytes used by model weights (stub)
     */
    uint64_t vram_bytes() const { return vram_bytes_; }
    
    /**
     * Get model path
     */
    const std::string& model_path() const { return model_path_; }
    
private:
    std::string model_path_;
    uint64_t vram_bytes_;
};

} // namespace worker

#endif // WORKER_MODEL_IMPL_H

// ---
// Built by Foundation-Alpha 🏗️
