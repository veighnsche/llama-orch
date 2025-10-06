/**
 * FFI Implementation - Main Entry Points
 * 
 * Implements the C API defined in worker_ffi.h.
 * All functions use exception-to-error-code pattern.
 * 
 * Spec: M0-W-1052, CUDA-4030
 */

#include "../include/worker_ffi.h"
#include "../include/health.h"
#include "context.h"
#include "model_impl.h"
#include "inference_impl.h"
#include "cuda_error.h"
#include "transformer/qwen_transformer.h"
#include <cuda_runtime.h>
#include <memory>
#include <cstdio>

using namespace worker;

// ============================================================================
// Error Messages
// ============================================================================

extern "C" const char* cuda_error_message(int error_code) {
    switch (error_code) {
        case CUDA_SUCCESS:
            return "Success";
        case CUDA_ERROR_INVALID_DEVICE:
            return "Invalid GPU device ID";
        case CUDA_ERROR_OUT_OF_MEMORY:
            return "Insufficient VRAM";
        case CUDA_ERROR_MODEL_LOAD_FAILED:
            return "Model loading failed";
        case CUDA_ERROR_INFERENCE_FAILED:
            return "Inference execution failed";
        case CUDA_ERROR_INVALID_PARAMETER:
            return "Invalid function parameter";
        case CUDA_ERROR_KERNEL_LAUNCH_FAILED:
            return "CUDA kernel launch failed";
        case CUDA_ERROR_VRAM_RESIDENCY_FAILED:
            return "VRAM residency check failed";
        case CUDA_ERROR_DEVICE_NOT_FOUND:
            return "No CUDA devices found";
        case CUDA_ERROR_UNKNOWN:
            return "Unknown error";
        default:
            return "Unrecognized error code";
    }
}

// ============================================================================
// Context Management (Stub Implementation)
// ============================================================================

extern "C" CudaContext* cuda_init(int gpu_device, int* error_code) {
    try {
        auto ctx = std::make_unique<Context>(gpu_device);
        *error_code = CUDA_SUCCESS;
        return reinterpret_cast<CudaContext*>(ctx.release());
    } catch (const CudaError& e) {
        *error_code = e.code();
        return nullptr;
    } catch (const std::exception& e) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    }
}

extern "C" void cuda_destroy(CudaContext* ctx) {
    if (ctx == nullptr) {
        return;
    }
    
    try {
        auto* context = reinterpret_cast<Context*>(ctx);
        delete context;
    } catch (...) {
        // Suppress exceptions in destructor
    }
}

extern "C" int cuda_get_device_count(void) {
    try {
        return Context::device_count();
    } catch (...) {
        return 0;
    }
}

// ============================================================================
// Model Loading (Stub Implementation)
// ============================================================================

extern "C" CudaModel* cuda_load_model(
    CudaContext* ctx,
    const char* model_path,
    uint64_t* vram_bytes_used,
    int* error_code
) {
    try {
        if (ctx == nullptr || model_path == nullptr || vram_bytes_used == nullptr) {
            throw CudaError::invalid_parameter("NULL pointer provided");
        }
        
        // NOTE: For now, just create a stub ModelImpl
        // The real weights will be loaded when inference context is created
        // This is because we need model config from Rust to load weights properly
        auto* context = reinterpret_cast<Context*>(ctx);
        auto model = std::make_unique<ModelImpl>(*context, model_path);
        *vram_bytes_used = model->vram_bytes();
        *error_code = CUDA_SUCCESS;
        return reinterpret_cast<CudaModel*>(model.release());
    } catch (const CudaError& e) {
        fprintf(stderr, "CUDA Error in cuda_load_model: %s (code: %d)\n", e.what(), e.code());
        *error_code = e.code();
        return nullptr;
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception in cuda_load_model: %s\n", e.what());
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    } catch (...) {
        fprintf(stderr, "Unknown exception in cuda_load_model\n");
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    }
}

extern "C" void cuda_unload_model(CudaModel* model) {
    if (model == nullptr) {
        return;
    }
    
    try {
        // IMPLEMENTED: Clean up model
        auto* m = reinterpret_cast<ModelImpl*>(model);
        delete m;
    } catch (...) {
        // Suppress exceptions in destructor
    }
}

extern "C" uint64_t cuda_model_get_vram_usage(CudaModel* model) {
    if (model == nullptr) {
        return 0;
    }
    
    try {
        // IMPLEMENTED: Return actual VRAM usage
        auto* m = reinterpret_cast<ModelImpl*>(model);
        return m->vram_bytes();
    } catch (...) {
        return 0;
    }
}

// ============================================================================
// Inference Execution (Stub Implementation)
// ============================================================================

extern "C" InferenceResult* cuda_inference_start(
    CudaModel* model,
    const char* prompt,
    int max_tokens,
    float temperature,
    uint64_t seed,
    int* error_code
) {
    try {
        if (model == nullptr || prompt == nullptr) {
            throw CudaError::invalid_parameter("NULL pointer provided");
        }
        
        // IMPLEMENTED: Create inference session
        auto* m = reinterpret_cast<ModelImpl*>(model);
        auto inference = std::make_unique<InferenceImpl>(*m, prompt, max_tokens, temperature, seed);
        *error_code = CUDA_SUCCESS;
        return reinterpret_cast<InferenceResult*>(inference.release());
    } catch (const CudaError& e) {
        *error_code = e.code();
        return nullptr;
    } catch (const std::exception& e) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    }
}

extern "C" bool cuda_inference_next_token(
    InferenceResult* result,
    char* token_out,
    int token_buffer_size,
    int* token_index,
    int* error_code
) {
    try {
        if (result == nullptr || token_out == nullptr || error_code == nullptr) {
            *error_code = CUDA_ERROR_INVALID_PARAMETER;
            return false;
        }
        
        // IMPLEMENTED: Generate next token
        auto* inference = reinterpret_cast<InferenceImpl*>(result);
        bool has_token = inference->next_token(token_out, token_buffer_size, token_index);
        *error_code = CUDA_SUCCESS;
        return has_token;
    } catch (const CudaError& e) {
        *error_code = e.code();
        return false;
    } catch (const std::exception& e) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return false;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return false;
    }
}

extern "C" void cuda_inference_free(InferenceResult* result) {
    if (result == nullptr) {
        return;
    }
    
    try {
        // IMPLEMENTED: Clean up inference
        auto* inference = reinterpret_cast<InferenceImpl*>(result);
        delete inference;
    } catch (...) {
        // Suppress exceptions in destructor
    }
}

// ============================================================================
// Health & Monitoring (Stub Implementation)
// ============================================================================

extern "C" bool cuda_check_vram_residency(CudaModel* model, int* error_code) {
    try {
        if (model == nullptr) {
            throw CudaError::invalid_parameter("NULL model pointer");
        }
        
        if (error_code == nullptr) {
            return false;
        }
        
        // TODO: Once Model class is implemented, use:
        // auto* m = reinterpret_cast<Model*>(model);
        // bool resident = Health::check_vram_residency(m->vram_tracker());
        // *error_code = resident ? CUDA_SUCCESS : CUDA_ERROR_VRAM_RESIDENCY_FAILED;
        // return resident;
        
        // Stub: Return true (assume resident) until Model class is implemented
        *error_code = CUDA_SUCCESS;
        return true;
    } catch (const CudaError& e) {
        *error_code = e.code();
        return false;
    } catch (const std::exception& e) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return false;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return false;
    }
}

extern "C" uint64_t cuda_get_vram_usage(CudaModel* model) {
    if (model == nullptr) {
        return 0;
    }
    
    try {
        // TODO: Implement VRAM usage query
        // auto* m = reinterpret_cast<Model*>(model);
        // return m->vram_bytes();
        
        // Stub: Return 0 for now
        return 0;
    } catch (...) {
        return 0;
    }
}

extern "C" uint64_t cuda_get_process_vram_usage(CudaContext* ctx) {
    if (ctx == nullptr) {
        return 0;
    }
    
    try {
        auto* context = reinterpret_cast<Context*>(ctx);
        
        // Query CUDA for memory info
        size_t free_bytes, total_bytes;
        cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
        
        if (err != cudaSuccess) {
            return 0;
        }
        
        // Return used VRAM (total - free)
        return total_bytes - free_bytes;
    } catch (...) {
        return 0;
    }
}

extern "C" bool cuda_check_device_health(CudaContext* ctx, int* error_code) {
    try {
        if (ctx == nullptr) {
            throw CudaError::invalid_parameter("NULL context pointer");
        }
        
        if (error_code == nullptr) {
            return false;
        }
        
        auto* context = reinterpret_cast<Context*>(ctx);
        
        // Check if device is responsive by querying device properties
        // This will fail if device has errors or is not responsive
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            *error_code = CUDA_ERROR_INVALID_DEVICE;
            return false;
        }
        
        // Try to query memory info as a health check
        size_t free_bytes, total_bytes;
        err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess) {
            *error_code = CUDA_ERROR_INVALID_DEVICE;
            return false;
        }
        
        *error_code = CUDA_SUCCESS;
        return true;
    } catch (const CudaError& e) {
        *error_code = e.code();
        return false;
    } catch (const std::exception& e) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return false;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return false;
    }
}

// ---
// Built by Foundation-Alpha 🏗️

