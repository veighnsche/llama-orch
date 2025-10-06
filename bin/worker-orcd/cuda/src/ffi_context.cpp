// FFI for CUDA context management
// Minimal implementation - no stubs

#include "context.h"
#include "cuda_error.h"
#include "model_impl.h"
#include <cstdio>

using namespace worker;

extern "C" {

// Opaque handle for Rust
typedef void CudaContext;

const char* cuda_error_message(int error_code) {
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

CudaContext* cuda_init(int gpu_device, int* error_code) {
    try {
        auto ctx = std::make_unique<Context>(gpu_device);
        *error_code = CUDA_SUCCESS;
        return reinterpret_cast<CudaContext*>(ctx.release());
    } catch (const CudaError& e) {
        *error_code = e.code();
        return nullptr;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    }
}

void cuda_destroy(CudaContext* ctx) {
    if (ctx) {
        try {
            delete reinterpret_cast<Context*>(ctx);
        } catch (...) {
            // Suppress exceptions in destructor
        }
    }
}

int cuda_get_device_count() {
    try {
        return Context::device_count();
    } catch (...) {
        return 0;
    }
}

// Stub for cuda_unload_model - actual model cleanup is handled in Rust weight loader
void cuda_unload_model(void* model) {
    // Model is just a ModelImpl wrapper - cleanup happens in Rust
    if (model) {
        delete reinterpret_cast<worker::ModelImpl*>(model);
    }
}

} // extern "C"
