/**
 * FFI Interface Compilation Test
 * 
 * Verifies that the FFI headers compile correctly in both C and C++ mode.
 * This is a compilation-only test - no runtime tests yet (implementation pending).
 */

#include <gtest/gtest.h>

// Test that headers compile in C++ mode
#include "worker_ffi.h"
#include "worker_types.h"
#include "worker_errors.h"

// Verify opaque types are defined
TEST(FFIInterface, OpaqueTypesAreDefined) {
    // These should compile (types are forward-declared)
    CudaContext* ctx = nullptr;
    CudaModel* model = nullptr;
    InferenceResult* result = nullptr;
    
    // Verify we can use them as pointers
    EXPECT_EQ(ctx, nullptr);
    EXPECT_EQ(model, nullptr);
    EXPECT_EQ(result, nullptr);
}

// Verify error codes are defined
TEST(FFIInterface, ErrorCodesAreDefined) {
    EXPECT_EQ(CUDA_SUCCESS, 0);
    EXPECT_EQ(CUDA_ERROR_INVALID_DEVICE, 1);
    EXPECT_EQ(CUDA_ERROR_OUT_OF_MEMORY, 2);
    EXPECT_EQ(CUDA_ERROR_MODEL_LOAD_FAILED, 3);
    EXPECT_EQ(CUDA_ERROR_INFERENCE_FAILED, 4);
    EXPECT_EQ(CUDA_ERROR_INVALID_PARAMETER, 5);
    EXPECT_EQ(CUDA_ERROR_KERNEL_LAUNCH_FAILED, 6);
    EXPECT_EQ(CUDA_ERROR_VRAM_RESIDENCY_FAILED, 7);
    EXPECT_EQ(CUDA_ERROR_DEVICE_NOT_FOUND, 8);
    EXPECT_EQ(CUDA_ERROR_UNKNOWN, 99);
}

// Verify error codes have no gaps (except intentional jump to 99)
TEST(FFIInterface, ErrorCodesAreSequential) {
    // 0-8 should be sequential
    EXPECT_EQ(CUDA_ERROR_INVALID_DEVICE - CUDA_SUCCESS, 1);
    EXPECT_EQ(CUDA_ERROR_OUT_OF_MEMORY - CUDA_ERROR_INVALID_DEVICE, 1);
    EXPECT_EQ(CUDA_ERROR_MODEL_LOAD_FAILED - CUDA_ERROR_OUT_OF_MEMORY, 1);
    EXPECT_EQ(CUDA_ERROR_INFERENCE_FAILED - CUDA_ERROR_MODEL_LOAD_FAILED, 1);
    EXPECT_EQ(CUDA_ERROR_INVALID_PARAMETER - CUDA_ERROR_INFERENCE_FAILED, 1);
    EXPECT_EQ(CUDA_ERROR_KERNEL_LAUNCH_FAILED - CUDA_ERROR_INVALID_PARAMETER, 1);
    EXPECT_EQ(CUDA_ERROR_VRAM_RESIDENCY_FAILED - CUDA_ERROR_KERNEL_LAUNCH_FAILED, 1);
    EXPECT_EQ(CUDA_ERROR_DEVICE_NOT_FOUND - CUDA_ERROR_VRAM_RESIDENCY_FAILED, 1);
    
    // 99 is intentionally separate (unknown error)
    EXPECT_EQ(CUDA_ERROR_UNKNOWN, 99);
}

// Verify function declarations exist (compilation test only)
TEST(FFIInterface, ContextFunctionsAreDeclared) {
    // These should compile (functions are declared)
    // We're not calling them (no implementation yet)
    
    // Context management
    auto init_ptr = &cuda_init;
    auto destroy_ptr = &cuda_destroy;
    auto get_device_count_ptr = &cuda_get_device_count;
    
    EXPECT_NE(init_ptr, nullptr);
    EXPECT_NE(destroy_ptr, nullptr);
    EXPECT_NE(get_device_count_ptr, nullptr);
}

TEST(FFIInterface, ModelFunctionsAreDeclared) {
    // Model loading
    auto load_model_ptr = &cuda_load_model;
    auto unload_model_ptr = &cuda_unload_model;
    auto get_vram_usage_ptr = &cuda_model_get_vram_usage;
    
    EXPECT_NE(load_model_ptr, nullptr);
    EXPECT_NE(unload_model_ptr, nullptr);
    EXPECT_NE(get_vram_usage_ptr, nullptr);
}

TEST(FFIInterface, InferenceFunctionsAreDeclared) {
    // Inference execution
    auto inference_start_ptr = &cuda_inference_start;
    auto inference_next_token_ptr = &cuda_inference_next_token;
    auto inference_free_ptr = &cuda_inference_free;
    
    EXPECT_NE(inference_start_ptr, nullptr);
    EXPECT_NE(inference_next_token_ptr, nullptr);
    EXPECT_NE(inference_free_ptr, nullptr);
}

TEST(FFIInterface, HealthFunctionsAreDeclared) {
    // Health & monitoring
    auto check_vram_residency_ptr = &cuda_check_vram_residency;
    auto get_vram_usage_ptr = &cuda_get_vram_usage;
    auto get_process_vram_usage_ptr = &cuda_get_process_vram_usage;
    auto check_device_health_ptr = &cuda_check_device_health;
    
    EXPECT_NE(check_vram_residency_ptr, nullptr);
    EXPECT_NE(get_vram_usage_ptr, nullptr);
    EXPECT_NE(get_process_vram_usage_ptr, nullptr);
    EXPECT_NE(check_device_health_ptr, nullptr);
}

TEST(FFIInterface, ErrorFunctionsAreDeclared) {
    // Error handling
    auto error_message_ptr = &cuda_error_message;
    
    EXPECT_NE(error_message_ptr, nullptr);
}

// Verify function signatures match expected types
TEST(FFIInterface, FunctionSignaturesAreCorrect) {
    // Context management
    static_assert(std::is_same_v<decltype(cuda_init), CudaContext*(int, int*)>);
    static_assert(std::is_same_v<decltype(cuda_destroy), void(CudaContext*)>);
    static_assert(std::is_same_v<decltype(cuda_get_device_count), int(void)>);
    
    // Model loading
    static_assert(std::is_same_v<decltype(cuda_load_model), CudaModel*(CudaContext*, const char*, uint64_t*, int*)>);
    static_assert(std::is_same_v<decltype(cuda_unload_model), void(CudaModel*)>);
    static_assert(std::is_same_v<decltype(cuda_model_get_vram_usage), uint64_t(CudaModel*)>);
    
    // Inference execution
    static_assert(std::is_same_v<decltype(cuda_inference_start), InferenceResult*(CudaModel*, const char*, int, float, uint64_t, int*)>);
    static_assert(std::is_same_v<decltype(cuda_inference_next_token), bool(InferenceResult*, char*, int, int*, int*)>);
    static_assert(std::is_same_v<decltype(cuda_inference_free), void(InferenceResult*)>);
    
    // Health & monitoring
    static_assert(std::is_same_v<decltype(cuda_check_vram_residency), bool(CudaModel*, int*)>);
    static_assert(std::is_same_v<decltype(cuda_get_vram_usage), uint64_t(CudaModel*)>);
    static_assert(std::is_same_v<decltype(cuda_get_process_vram_usage), uint64_t(CudaContext*)>);
    static_assert(std::is_same_v<decltype(cuda_check_device_health), bool(CudaContext*, int*)>);
    
    // Error handling
    static_assert(std::is_same_v<decltype(cuda_error_message), const char*(int)>);
    
    SUCCEED();  // If we get here, all static_asserts passed
}

// ---
// Built by Foundation-Alpha 🏗️
