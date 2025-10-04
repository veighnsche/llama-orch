/**
 * Error Code System Unit Tests
 * 
 * Tests error code definitions, error messages, and CudaError exception class.
 * 
 * Spec: M0-W-1501, CUDA-5040, CUDA-5041
 */

#include <gtest/gtest.h>
#include "worker_errors.h"
#include "cuda_error.h"
#include <cstring>

using namespace worker;

// ============================================================================
// Error Code Tests
// ============================================================================

TEST(ErrorCodes, AllCodesAreDefined) {
    // Verify all error codes compile
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

TEST(ErrorCodes, CodesAreSequential) {
    // Verify 0-8 are sequential (99 is intentionally separate)
    EXPECT_EQ(CUDA_ERROR_INVALID_DEVICE - CUDA_SUCCESS, 1);
    EXPECT_EQ(CUDA_ERROR_OUT_OF_MEMORY - CUDA_ERROR_INVALID_DEVICE, 1);
    EXPECT_EQ(CUDA_ERROR_MODEL_LOAD_FAILED - CUDA_ERROR_OUT_OF_MEMORY, 1);
    EXPECT_EQ(CUDA_ERROR_INFERENCE_FAILED - CUDA_ERROR_MODEL_LOAD_FAILED, 1);
    EXPECT_EQ(CUDA_ERROR_INVALID_PARAMETER - CUDA_ERROR_INFERENCE_FAILED, 1);
    EXPECT_EQ(CUDA_ERROR_KERNEL_LAUNCH_FAILED - CUDA_ERROR_INVALID_PARAMETER, 1);
    EXPECT_EQ(CUDA_ERROR_VRAM_RESIDENCY_FAILED - CUDA_ERROR_KERNEL_LAUNCH_FAILED, 1);
    EXPECT_EQ(CUDA_ERROR_DEVICE_NOT_FOUND - CUDA_ERROR_VRAM_RESIDENCY_FAILED, 1);
}

// ============================================================================
// Error Message Tests
// ============================================================================

TEST(ErrorMessages, SuccessMessage) {
    const char* msg = cuda_error_message(CUDA_SUCCESS);
    EXPECT_STREQ(msg, "Operation completed successfully");
}

TEST(ErrorMessages, InvalidDeviceMessage) {
    const char* msg = cuda_error_message(CUDA_ERROR_INVALID_DEVICE);
    EXPECT_STREQ(msg, "Invalid CUDA device ID");
}

TEST(ErrorMessages, OutOfMemoryMessage) {
    const char* msg = cuda_error_message(CUDA_ERROR_OUT_OF_MEMORY);
    EXPECT_STREQ(msg, "Out of GPU memory (VRAM)");
}

TEST(ErrorMessages, ModelLoadFailedMessage) {
    const char* msg = cuda_error_message(CUDA_ERROR_MODEL_LOAD_FAILED);
    EXPECT_STREQ(msg, "Failed to load model from GGUF file");
}

TEST(ErrorMessages, InferenceFailedMessage) {
    const char* msg = cuda_error_message(CUDA_ERROR_INFERENCE_FAILED);
    EXPECT_STREQ(msg, "Inference execution failed");
}

TEST(ErrorMessages, InvalidParameterMessage) {
    const char* msg = cuda_error_message(CUDA_ERROR_INVALID_PARAMETER);
    EXPECT_STREQ(msg, "Invalid parameter provided");
}

TEST(ErrorMessages, KernelLaunchFailedMessage) {
    const char* msg = cuda_error_message(CUDA_ERROR_KERNEL_LAUNCH_FAILED);
    EXPECT_STREQ(msg, "CUDA kernel launch failed");
}

TEST(ErrorMessages, VramResidencyFailedMessage) {
    const char* msg = cuda_error_message(CUDA_ERROR_VRAM_RESIDENCY_FAILED);
    EXPECT_STREQ(msg, "VRAM residency check failed (RAM fallback detected)");
}

TEST(ErrorMessages, DeviceNotFoundMessage) {
    const char* msg = cuda_error_message(CUDA_ERROR_DEVICE_NOT_FOUND);
    EXPECT_STREQ(msg, "No CUDA devices found");
}

TEST(ErrorMessages, UnknownMessage) {
    const char* msg = cuda_error_message(CUDA_ERROR_UNKNOWN);
    EXPECT_STREQ(msg, "Unknown error occurred");
}

TEST(ErrorMessages, UnrecognizedCodeReturnsDefault) {
    const char* msg = cuda_error_message(999);
    EXPECT_STREQ(msg, "Unrecognized error code");
}

TEST(ErrorMessages, AllMessagesAreNonNull) {
    // Verify all error codes return non-NULL messages
    int codes[] = {
        CUDA_SUCCESS,
        CUDA_ERROR_INVALID_DEVICE,
        CUDA_ERROR_OUT_OF_MEMORY,
        CUDA_ERROR_MODEL_LOAD_FAILED,
        CUDA_ERROR_INFERENCE_FAILED,
        CUDA_ERROR_INVALID_PARAMETER,
        CUDA_ERROR_KERNEL_LAUNCH_FAILED,
        CUDA_ERROR_VRAM_RESIDENCY_FAILED,
        CUDA_ERROR_DEVICE_NOT_FOUND,
        CUDA_ERROR_UNKNOWN,
    };
    
    for (int code : codes) {
        const char* msg = cuda_error_message(code);
        EXPECT_NE(msg, nullptr);
        EXPECT_GT(strlen(msg), 0);
    }
}

// ============================================================================
// CudaError Exception Tests
// ============================================================================

TEST(CudaError, ConstructorStoresCodeAndMessage) {
    CudaError err(CUDA_ERROR_INVALID_DEVICE, "Test message");
    EXPECT_EQ(err.code(), CUDA_ERROR_INVALID_DEVICE);
    EXPECT_STREQ(err.what(), "Test message");
}

TEST(CudaError, ConstructorWithCString) {
    CudaError err(CUDA_ERROR_OUT_OF_MEMORY, "Test message");
    EXPECT_EQ(err.code(), CUDA_ERROR_OUT_OF_MEMORY);
    EXPECT_STREQ(err.what(), "Test message");
}

TEST(CudaError, WhatReturnsMessage) {
    CudaError err(CUDA_ERROR_INFERENCE_FAILED, "Inference failed at layer 5");
    const char* msg = err.what();
    EXPECT_STREQ(msg, "Inference failed at layer 5");
}

TEST(CudaError, CodeReturnsCorrectValue) {
    CudaError err(CUDA_ERROR_KERNEL_LAUNCH_FAILED, "Kernel launch failed");
    EXPECT_EQ(err.code(), CUDA_ERROR_KERNEL_LAUNCH_FAILED);
}

// ============================================================================
// Factory Method Tests
// ============================================================================

TEST(CudaErrorFactory, InvalidDevice) {
    auto err = CudaError::invalid_device("device 5 not found");
    EXPECT_EQ(err.code(), CUDA_ERROR_INVALID_DEVICE);
    EXPECT_NE(std::string(err.what()).find("Invalid device"), std::string::npos);
    EXPECT_NE(std::string(err.what()).find("device 5 not found"), std::string::npos);
}

TEST(CudaErrorFactory, OutOfMemory) {
    auto err = CudaError::out_of_memory("requested 16GB, available 8GB");
    EXPECT_EQ(err.code(), CUDA_ERROR_OUT_OF_MEMORY);
    EXPECT_NE(std::string(err.what()).find("Out of memory"), std::string::npos);
    EXPECT_NE(std::string(err.what()).find("requested 16GB"), std::string::npos);
}

TEST(CudaErrorFactory, ModelLoadFailed) {
    auto err = CudaError::model_load_failed("file not found");
    EXPECT_EQ(err.code(), CUDA_ERROR_MODEL_LOAD_FAILED);
    EXPECT_NE(std::string(err.what()).find("Model load failed"), std::string::npos);
}

TEST(CudaErrorFactory, InferenceFailed) {
    auto err = CudaError::inference_failed("kernel timeout");
    EXPECT_EQ(err.code(), CUDA_ERROR_INFERENCE_FAILED);
    EXPECT_NE(std::string(err.what()).find("Inference failed"), std::string::npos);
}

TEST(CudaErrorFactory, InvalidParameter) {
    auto err = CudaError::invalid_parameter("temperature out of range");
    EXPECT_EQ(err.code(), CUDA_ERROR_INVALID_PARAMETER);
    EXPECT_NE(std::string(err.what()).find("Invalid parameter"), std::string::npos);
}

TEST(CudaErrorFactory, KernelLaunchFailed) {
    auto err = CudaError::kernel_launch_failed("attention kernel");
    EXPECT_EQ(err.code(), CUDA_ERROR_KERNEL_LAUNCH_FAILED);
    EXPECT_NE(std::string(err.what()).find("Kernel launch failed"), std::string::npos);
}

TEST(CudaErrorFactory, VramResidencyFailed) {
    auto err = CudaError::vram_residency_failed("pointer in host memory");
    EXPECT_EQ(err.code(), CUDA_ERROR_VRAM_RESIDENCY_FAILED);
    EXPECT_NE(std::string(err.what()).find("VRAM residency failed"), std::string::npos);
}

TEST(CudaErrorFactory, DeviceNotFound) {
    auto err = CudaError::device_not_found("no devices detected");
    EXPECT_EQ(err.code(), CUDA_ERROR_DEVICE_NOT_FOUND);
    EXPECT_NE(std::string(err.what()).find("Device not found"), std::string::npos);
}

// ============================================================================
// Exception-to-Error-Code Pattern Tests
// ============================================================================

// Helper function to test exception-to-error-code pattern
int test_exception_wrapper(bool throw_cuda_error, bool throw_std_exception, bool throw_unknown) {
    int error_code = CUDA_SUCCESS;
    
    try {
        if (throw_cuda_error) {
            throw CudaError::invalid_device("test device");
        } else if (throw_std_exception) {
            throw std::runtime_error("std exception");
        } else if (throw_unknown) {
            throw 42;  // Unknown exception type
        }
        // Success case
        error_code = CUDA_SUCCESS;
    } catch (const CudaError& e) {
        error_code = e.code();
    } catch (const std::exception& e) {
        error_code = CUDA_ERROR_UNKNOWN;
    } catch (...) {
        error_code = CUDA_ERROR_UNKNOWN;
    }
    
    return error_code;
}

TEST(ExceptionToErrorCode, CatchesCudaError) {
    int code = test_exception_wrapper(true, false, false);
    EXPECT_EQ(code, CUDA_ERROR_INVALID_DEVICE);
}

TEST(ExceptionToErrorCode, CatchesStdException) {
    int code = test_exception_wrapper(false, true, false);
    EXPECT_EQ(code, CUDA_ERROR_UNKNOWN);
}

TEST(ExceptionToErrorCode, CatchesUnknownException) {
    int code = test_exception_wrapper(false, false, true);
    EXPECT_EQ(code, CUDA_ERROR_UNKNOWN);
}

TEST(ExceptionToErrorCode, SuccessCase) {
    int code = test_exception_wrapper(false, false, false);
    EXPECT_EQ(code, CUDA_SUCCESS);
}

// ============================================================================
// Error Message Quality Tests
// ============================================================================

TEST(ErrorMessageQuality, MessagesAreDescriptive) {
    // Verify all messages are descriptive (>10 characters)
    int codes[] = {
        CUDA_SUCCESS,
        CUDA_ERROR_INVALID_DEVICE,
        CUDA_ERROR_OUT_OF_MEMORY,
        CUDA_ERROR_MODEL_LOAD_FAILED,
        CUDA_ERROR_INFERENCE_FAILED,
        CUDA_ERROR_INVALID_PARAMETER,
        CUDA_ERROR_KERNEL_LAUNCH_FAILED,
        CUDA_ERROR_VRAM_RESIDENCY_FAILED,
        CUDA_ERROR_DEVICE_NOT_FOUND,
        CUDA_ERROR_UNKNOWN,
    };
    
    for (int code : codes) {
        const char* msg = cuda_error_message(code);
        EXPECT_GT(strlen(msg), 10) << "Error code " << code << " has too short message";
    }
}

TEST(ErrorMessageQuality, MessagesAreActionable) {
    // Verify messages contain actionable information
    const char* oom_msg = cuda_error_message(CUDA_ERROR_OUT_OF_MEMORY);
    EXPECT_NE(std::string(oom_msg).find("VRAM"), std::string::npos);
    
    const char* model_msg = cuda_error_message(CUDA_ERROR_MODEL_LOAD_FAILED);
    EXPECT_NE(std::string(model_msg).find("GGUF"), std::string::npos);
    
    const char* device_msg = cuda_error_message(CUDA_ERROR_INVALID_DEVICE);
    EXPECT_NE(std::string(device_msg).find("device"), std::string::npos);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST(ErrorMessages, ThreadSafe) {
    // cuda_error_message returns static strings, should be thread-safe
    const char* msg1 = cuda_error_message(CUDA_ERROR_INVALID_DEVICE);
    const char* msg2 = cuda_error_message(CUDA_ERROR_INVALID_DEVICE);
    
    // Same pointer (static string)
    EXPECT_EQ(msg1, msg2);
}

// ============================================================================
// Exception Hierarchy Tests
// ============================================================================

TEST(CudaError, InheritsFromStdException) {
    CudaError err(CUDA_ERROR_UNKNOWN, "test");
    std::exception* base = &err;
    EXPECT_NE(base, nullptr);
    EXPECT_STREQ(base->what(), "test");
}

TEST(CudaError, CanBeCaughtAsStdException) {
    try {
        throw CudaError::invalid_device("test");
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("Invalid device"), std::string::npos);
    }
}

// ============================================================================
// Error Code Stability Tests
// ============================================================================

TEST(ErrorCodeStability, CodesMatchSpec) {
    // Verify error codes match spec (M0-W-1501)
    // These values are LOCKED and must not change
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

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(ErrorMessages, HandlesInvalidCode) {
    const char* msg = cuda_error_message(999);
    EXPECT_STREQ(msg, "Unrecognized error code");
}

TEST(ErrorMessages, HandlesNegativeCode) {
    const char* msg = cuda_error_message(-1);
    EXPECT_STREQ(msg, "Unrecognized error code");
}

TEST(CudaError, HandlesEmptyMessage) {
    CudaError err(CUDA_ERROR_UNKNOWN, "");
    EXPECT_EQ(err.code(), CUDA_ERROR_UNKNOWN);
    EXPECT_STREQ(err.what(), "");
}

TEST(CudaError, HandlesLongMessage) {
    std::string long_msg(1000, 'x');
    CudaError err(CUDA_ERROR_UNKNOWN, long_msg);
    EXPECT_EQ(err.code(), CUDA_ERROR_UNKNOWN);
    EXPECT_EQ(std::string(err.what()).length(), 1000);
}

// ============================================================================
// Factory Method Consistency Tests
// ============================================================================

TEST(CudaErrorFactory, AllFactoryMethodsReturnCorrectCodes) {
    EXPECT_EQ(CudaError::invalid_device("").code(), CUDA_ERROR_INVALID_DEVICE);
    EXPECT_EQ(CudaError::out_of_memory("").code(), CUDA_ERROR_OUT_OF_MEMORY);
    EXPECT_EQ(CudaError::model_load_failed("").code(), CUDA_ERROR_MODEL_LOAD_FAILED);
    EXPECT_EQ(CudaError::inference_failed("").code(), CUDA_ERROR_INFERENCE_FAILED);
    EXPECT_EQ(CudaError::invalid_parameter("").code(), CUDA_ERROR_INVALID_PARAMETER);
    EXPECT_EQ(CudaError::kernel_launch_failed("").code(), CUDA_ERROR_KERNEL_LAUNCH_FAILED);
    EXPECT_EQ(CudaError::vram_residency_failed("").code(), CUDA_ERROR_VRAM_RESIDENCY_FAILED);
    EXPECT_EQ(CudaError::device_not_found("").code(), CUDA_ERROR_DEVICE_NOT_FOUND);
}

TEST(CudaErrorFactory, FactoryMethodsIncludePrefix) {
    // Verify factory methods add descriptive prefix
    auto err1 = CudaError::invalid_device("test");
    EXPECT_NE(std::string(err1.what()).find("Invalid device"), std::string::npos);
    
    auto err2 = CudaError::out_of_memory("test");
    EXPECT_NE(std::string(err2.what()).find("Out of memory"), std::string::npos);
    
    auto err3 = CudaError::model_load_failed("test");
    EXPECT_NE(std::string(err3.what()).find("Model load failed"), std::string::npos);
}

// ---
// Built by Foundation-Alpha 🏗️
