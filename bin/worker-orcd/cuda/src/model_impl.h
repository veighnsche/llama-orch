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
#include "model/qwen_weight_loader.h"
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
    
    /**
     * Default constructor for Rust weight loading
     */
    ModelImpl() : vram_bytes_(0), qwen_model_(nullptr) {}
    
    ~ModelImpl();
    
    /**
     * Get VRAM bytes used by model weights (stub)
     */
    uint64_t vram_bytes() const { return vram_bytes_; }
    
    /**
     * Get model path
     */
    const std::string& model_path() const { return model_path_; }
    
    /**
     * Set Qwen model (for Rust weight loading)
     */
    void set_qwen_model(model::QwenModel* model) { qwen_model_ = model; }
    
    /**
     * Get Qwen model
     */
    model::QwenModel* get_qwen_model() const { return qwen_model_; }
    
    /**
     * Set VRAM bytes (for Rust weight loading)
     */
    void set_vram_bytes(uint64_t bytes) { vram_bytes_ = bytes; }
    
private:
    std::string model_path_;
    uint64_t vram_bytes_;
    model::QwenModel* qwen_model_;
};

} // namespace worker

#endif // WORKER_MODEL_IMPL_H

// ---
// Built by Foundation-Alpha 🏗️
