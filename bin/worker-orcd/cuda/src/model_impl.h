/**
 * Model Implementation - Wires GGUF parser to model loading
 * 
 * This class integrates:
 * - GGUF header parsing (already implemented)
 * - GPT model loading (already implemented)
 * - VRAM tracking
 * 
 * Spec: M0-W-1211
 */

#ifndef WORKER_MODEL_IMPL_H
#define WORKER_MODEL_IMPL_H

#include "context.h"
// GGUF parsing now done in Rust
#include "model/gpt_model.h"
#include <memory>
#include <string>

namespace worker {

// Forward declarations
namespace model {
    class GPTModel;
}

namespace io {
    class MmapFile;
}


/**
 * Loaded model with VRAM allocation
 */
class ModelImpl {
public:
    /**
     * Load model from GGUF file
     * 
     * @param ctx CUDA context
     * @param model_path Path to .gguf file
     * @throws CudaError on failure
     */
    ModelImpl(Context& ctx, const char* model_path);
    
    ~ModelImpl();
    
    /**
     * Get VRAM bytes used by model weights
     */
    uint64_t vram_bytes() const { return vram_bytes_; }
    
    /**
     * Get GGUF header
     */
    const gguf::GGUFHeader& header() const { return header_; }
    
    /**
     * Get GPT model (if architecture is GPT)
     */
    model::GPTModel* gpt_model() { return gpt_model_.get(); }
    
private:
    gguf::GGUFHeader header_;
    std::unique_ptr<model::GPTModel> gpt_model_;
    uint64_t vram_bytes_;
    std::string architecture_;
    
    // Keep mmap alive for the lifetime of the model
    std::unique_ptr<io::MmapFile> mmap_;
};

} // namespace worker

#endif // WORKER_MODEL_IMPL_H

// ---
// Built by Foundation-Alpha 🏗️
