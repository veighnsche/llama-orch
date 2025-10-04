/**
 * FFI Integration Tests (C++ Side)
 * 
 * Tests the C++ implementation of FFI boundary functions.
 * Validates that C API works correctly before Rust calls it.
 * 
 * Spec: M0-W-1810, M0-W-1811
 */

#include <gtest/gtest.h>
#include "context.h"
#include "vram_tracker.h"
#include "cuda_error.h"
#include "../include/worker_ffi.h"
#include <cuda_runtime.h>

using namespace worker;

class FFIIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure at least one CUDA device available
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices found";
        }
    }
};

// ============================================================================
// Context Initialization Tests
// ============================================================================

TEST_F(FFIIntegrationTest, ContextInitialization) {
    Context ctx(0);
    EXPECT_EQ(ctx.device(), 0);
    EXPECT_GT(ctx.total_vram(), 0);
}

TEST_F(FFIIntegrationTest, ContextInvalidDevice) {
    EXPECT_THROW({
        Context ctx(999);
    }, CudaError);
}

TEST_F(FFIIntegrationTest, ContextNegativeDevice) {
    EXPECT_THROW({
        Context ctx(-1);
    }, CudaError);
}

TEST_F(FFIIntegrationTest, DeviceCount) {
    int count = Context::device_count();
    EXPECT_GT(count, 0) << "Should have at least one CUDA device";
}

// ============================================================================
// FFI Function Tests
// ============================================================================

TEST_F(FFIIntegrationTest, FFI_cuda_init_valid_device) {
    int error_code = -1;
    CudaContext* ctx = cuda_init(0, &error_code);
    
    ASSERT_NE(ctx, nullptr) << "Context should not be null";
    EXPECT_EQ(error_code, CUDA_SUCCESS);
    
    // Cleanup
    cuda_destroy(ctx);
}

TEST_F(FFIIntegrationTest, FFI_cuda_init_invalid_device) {
    int error_code = -1;
    CudaContext* ctx = cuda_init(999, &error_code);
    
    EXPECT_EQ(ctx, nullptr) << "Context should be null for invalid device";
    EXPECT_EQ(error_code, CUDA_ERROR_INVALID_DEVICE);
}

TEST_F(FFIIntegrationTest, FFI_cuda_get_device_count) {
    int count = cuda_get_device_count();
    EXPECT_GT(count, 0) << "Should have at least one CUDA device";
}

TEST_F(FFIIntegrationTest, FFI_cuda_destroy_null_safe) {
    // Should not crash with null pointer
    cuda_destroy(nullptr);
    // If we reach here, test passed
    SUCCEED();
}

// ============================================================================
// VRAM Tracker Integration Tests
// ============================================================================

TEST_F(FFIIntegrationTest, VramTrackerIntegration) {
    Context ctx(0);
    auto& tracker = ctx.vram_tracker();
    
    EXPECT_EQ(tracker.total_usage(), 0);
    EXPECT_EQ(tracker.allocation_count(), 0);
    
    // Simulate allocation
    void* test_ptr = reinterpret_cast<void*>(0x1000);
    tracker.record_allocation(test_ptr, 1024, VramPurpose::ModelWeights, "test");
    
    EXPECT_EQ(tracker.total_usage(), 1024);
    EXPECT_EQ(tracker.allocation_count(), 1);
    
    tracker.record_deallocation(test_ptr);
    EXPECT_EQ(tracker.total_usage(), 0);
}

TEST_F(FFIIntegrationTest, VramTrackerAccessibleViaContext) {
    Context ctx(0);
    
    // Verify tracker is accessible
    const auto& const_tracker = ctx.vram_tracker();
    EXPECT_EQ(const_tracker.total_usage(), 0);
    
    auto& mutable_tracker = ctx.vram_tracker();
    EXPECT_EQ(mutable_tracker.allocation_count(), 0);
}

// ============================================================================
// Error Code Conversion Tests
// ============================================================================

TEST_F(FFIIntegrationTest, ErrorCodeConversion) {
    try {
        throw CudaError::invalid_device("test device");
    } catch (const CudaError& e) {
        EXPECT_EQ(e.code(), CUDA_ERROR_INVALID_DEVICE);
        EXPECT_STREQ(e.what(), "Invalid device: test device");
    }
}

TEST_F(FFIIntegrationTest, ErrorCodeOutOfMemory) {
    try {
        throw CudaError::out_of_memory("test allocation");
    } catch (const CudaError& e) {
        EXPECT_EQ(e.code(), CUDA_ERROR_OUT_OF_MEMORY);
        std::string msg = e.what();
        EXPECT_NE(msg.find("memory"), std::string::npos);
    }
}

// ============================================================================
// Context Cleanup Tests
// ============================================================================

TEST_F(FFIIntegrationTest, ContextCleanup) {
    size_t initial_free;
    {
        Context ctx(0);
        initial_free = ctx.free_vram();
    }
    // Context destroyed here
    
    Context ctx2(0);
    size_t final_free = ctx2.free_vram();
    
    // Free VRAM should be approximately the same
    size_t diff = (final_free > initial_free) ? 
                  (final_free - initial_free) : 
                  (initial_free - final_free);
    
    EXPECT_LT(diff, 1024 * 1024) << "VRAM leak detected: " << diff << " bytes";
}

TEST_F(FFIIntegrationTest, FFI_ContextCleanup) {
    int error_code;
    
    size_t initial_free;
    {
        CudaContext* ctx = cuda_init(0, &error_code);
        ASSERT_NE(ctx, nullptr);
        ASSERT_EQ(error_code, CUDA_SUCCESS);
        
        initial_free = cuda_get_process_vram_usage(ctx);
        
        cuda_destroy(ctx);
    }
    
    // Create new context and check VRAM
    CudaContext* ctx2 = cuda_init(0, &error_code);
    ASSERT_NE(ctx2, nullptr);
    
    size_t final_free = cuda_get_process_vram_usage(ctx2);
    
    size_t diff = (final_free > initial_free) ? 
                  (final_free - initial_free) : 
                  (initial_free - final_free);
    
    EXPECT_LT(diff, 1024 * 1024) << "VRAM leak detected: " << diff << " bytes";
    
    cuda_destroy(ctx2);
}

// ============================================================================
// Device Properties Tests
// ============================================================================

TEST_F(FFIIntegrationTest, DeviceProperties) {
    Context ctx(0);
    
    EXPECT_NE(ctx.device_name(), nullptr);
    EXPECT_GT(strlen(ctx.device_name()), 0);
    
    EXPECT_GE(ctx.compute_capability(), 75) << "Compute capability should be >= 7.5 (Turing)";
    
    EXPECT_GT(ctx.total_vram(), 0);
    EXPECT_LE(ctx.free_vram(), ctx.total_vram());
}

TEST_F(FFIIntegrationTest, FFI_ProcessVramUsage) {
    int error_code;
    CudaContext* ctx = cuda_init(0, &error_code);
    ASSERT_NE(ctx, nullptr);
    
    uint64_t vram_usage = cuda_get_process_vram_usage(ctx);
    EXPECT_GE(vram_usage, 0);
    
    cuda_destroy(ctx);
}

TEST_F(FFIIntegrationTest, FFI_DeviceHealth) {
    int error_code;
    CudaContext* ctx = cuda_init(0, &error_code);
    ASSERT_NE(ctx, nullptr);
    
    bool healthy = cuda_check_device_health(ctx, &error_code);
    EXPECT_EQ(error_code, CUDA_SUCCESS);
    EXPECT_TRUE(healthy);
    
    cuda_destroy(ctx);
}

// ============================================================================
// Multiple Context Tests
// ============================================================================

TEST_F(FFIIntegrationTest, MultipleContextsSequential) {
    for (int i = 0; i < 3; ++i) {
        Context ctx(0);
        EXPECT_EQ(ctx.device(), 0);
        EXPECT_GT(ctx.total_vram(), 0);
        // Context destroyed at end of iteration
    }
}

TEST_F(FFIIntegrationTest, FFI_MultipleContextsSequential) {
    for (int i = 0; i < 3; ++i) {
        int error_code;
        CudaContext* ctx = cuda_init(0, &error_code);
        ASSERT_NE(ctx, nullptr) << "Failed to create context " << i;
        EXPECT_EQ(error_code, CUDA_SUCCESS);
        
        cuda_destroy(ctx);
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(FFIIntegrationTest, RapidContextCreationDestruction) {
    for (int i = 0; i < 10; ++i) {
        int error_code;
        CudaContext* ctx = cuda_init(0, &error_code);
        ASSERT_NE(ctx, nullptr) << "Failed at iteration " << i;
        cuda_destroy(ctx);
    }
}

TEST_F(FFIIntegrationTest, ContextOutlivesMultipleOperations) {
    int error_code;
    CudaContext* ctx = cuda_init(0, &error_code);
    ASSERT_NE(ctx, nullptr);
    
    for (int i = 0; i < 5; ++i) {
        uint64_t vram = cuda_get_process_vram_usage(ctx);
        EXPECT_GE(vram, 0) << "VRAM query failed at iteration " << i;
        
        bool healthy = cuda_check_device_health(ctx, &error_code);
        EXPECT_EQ(error_code, CUDA_SUCCESS) << "Health check failed at iteration " << i;
        EXPECT_TRUE(healthy) << "Device unhealthy at iteration " << i;
    }
    
    cuda_destroy(ctx);
}

// ============================================================================
// Test Metadata
// ============================================================================

TEST_F(FFIIntegrationTest, TestSuiteMetadata) {
    // Document test suite
    std::cout << "FFI Integration Test Suite (C++)" << std::endl;
    std::cout << "=================================" << std::endl;
    std::cout << "Spec: M0-W-1810, M0-W-1811" << std::endl;
    std::cout << "Tests: FFI boundary, context lifecycle, error propagation" << std::endl;
    std::cout << "CUDA devices: " << Context::device_count() << std::endl;
}

// ---
// Built by Foundation-Alpha 🏗️
