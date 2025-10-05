/**
 * Model Implementation - Simplified stub
 * 
 * GGUF parsing is now done in Rust.
 * This is just a minimal stub to make FFI work.
 */

#include "model_impl.h"
#include "cuda_error.h"
#include <sys/stat.h>

namespace worker {

ModelImpl::ModelImpl(Context& ctx, const char* model_path) 
    : model_path_(model_path ? model_path : ""),
      vram_bytes_(0) {
    
    if (!model_path) {
        throw CudaError::invalid_parameter("model_path is NULL");
    }
    
    // Check file exists
    struct stat st;
    if (stat(model_path, &st) != 0) {
        throw CudaError::model_load_failed("File not found");
    }
    
    // Estimate VRAM usage from file size
    vram_bytes_ = static_cast<uint64_t>(st.st_size);
    
    // Add overhead for KV cache and workspace (estimate 20%)
    vram_bytes_ = static_cast<uint64_t>(vram_bytes_ * 1.2);
    
    fprintf(stderr, "Model stub loaded: %s, VRAM estimate: %zu MB\n", 
            model_path, vram_bytes_ / 1024 / 1024);
}

ModelImpl::~ModelImpl() {
    // Cleanup handled automatically
}

} // namespace worker

// ---
// Built by Foundation-Alpha 🏗️
