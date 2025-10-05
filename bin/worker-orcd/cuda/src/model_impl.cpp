/**
 * Model Implementation
 * 
 * Wires together existing components:
 * - GGUF parser (already done)
 * - GPT model (already done)
 * - Memory mapping (already done)
 */

#include "model_impl.h"
#include "io/mmap_file.h"
#include "gguf/header_parser.h"
#include "gguf/llama_metadata.h"
#include "model/gpt_model.h"
#include "model/gpt_weights.h"
#include "cuda_error.h"
#include <cstring>

using namespace worker::model;
using namespace worker::io;

namespace worker {

ModelImpl::ModelImpl(Context& ctx, const char* model_path) 
    : vram_bytes_(0) {
    
    if (!model_path) {
        throw CudaError::invalid_parameter("model_path is NULL");
    }
    
    // 1. Memory-map the GGUF file (using existing mmap code)
    mmap_ = std::make_unique<MmapFile>(MmapFile::open(model_path));
    const void* file_data = mmap_->data();
    size_t file_size = mmap_->size();
    
    fprintf(stderr, "DEBUG: Opened GGUF file: %s, size: %zu bytes\n", model_path, file_size);
    fprintf(stderr, "DEBUG: First 4 bytes: %02x %02x %02x %02x\n",
            ((const uint8_t*)file_data)[0],
            ((const uint8_t*)file_data)[1],
            ((const uint8_t*)file_data)[2],
            ((const uint8_t*)file_data)[3]);
    
    // 2. Parse GGUF header (using existing parser - ALREADY IMPLEMENTED!)
    header_ = gguf::parse_gguf_header(file_data, file_size);
    
    // 3. Extract architecture from metadata
    architecture_ = "unknown";
    for (const auto& kv : header_.metadata) {
        if (kv.key == "general.architecture" && kv.value_type == gguf::GGUFValueType::STRING) {
            architecture_ = kv.string_value;
            break;
        }
    }
    
    // 4. Calculate VRAM usage from header
    // For now, use file size as estimate (includes all tensors)
    vram_bytes_ = file_size;
    
    // Add overhead for KV cache and workspace (estimate 20%)
    vram_bytes_ = static_cast<uint64_t>(vram_bytes_ * 1.2);
    
    fprintf(stderr, "DEBUG: Model loaded, VRAM estimate: %zu MB\n", vram_bytes_ / 1024 / 1024);
    
    // TODO: Actually load model weights to VRAM
    // For M0: We just need to prove the pipeline works
    // Full weight loading will be implemented in Phase 3
    
    // mmap_ is now stored as a member variable to keep it alive
}

ModelImpl::~ModelImpl() {
    // Cleanup handled by unique_ptr
}

} // namespace worker

// ---
// Built by Foundation-Alpha 🏗️
