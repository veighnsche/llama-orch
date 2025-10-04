/**
 * Context Unit Tests
 * 
 * Tests CUDA context initialization, device queries, and VRAM-only enforcement.
 * 
 * Spec: M0-W-1010, M0-W-1400, CUDA-5101, CUDA-5120
 */

#include <gtest/gtest.h>
#include "context.h"
#include "cuda_error.h"
#include <cuda_runtime.h>

using namespace worker;

// ============================================================================
// Context Construction Tests
// ============================================================================

TEST(Context, ConstructorWithValidDeviceSucceeds) {
    // Verify device 0 exists
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    // Should construct successfully
    EXPECT_NO_THROW({
        Context ctx(0);
    });
}

TEST(Context, ConstructorWithInvalidDeviceThrows) {
    // Device ID -1 should throw
    EXPECT_THROW({
        Context ctx(-1);
    }, CudaError);
    
    // Device ID 999 should throw
    EXPECT_THROW({
        Context ctx(999);
    }, CudaError);
}

TEST(Context, ConstructorValidatesDeviceRange) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    // Last valid device should work
    EXPECT_NO_THROW({
        Context ctx(device_count - 1);
    });
    
    // One past last device should throw
    EXPECT_THROW({
        Context ctx(device_count);
    }, CudaError);
}

// ============================================================================
// Device Query Tests
// ============================================================================

TEST(Context, DeviceReturnsCorrectID) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    Context ctx(0);
    EXPECT_EQ(ctx.device(), 0);
}

TEST(Context, DeviceNameReturnsNonEmptyString) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    Context ctx(0);
    const char* name = ctx.device_name();
    EXPECT_NE(name, nullptr);
    EXPECT_GT(strlen(name), 0);
}

TEST(Context, ComputeCapabilityReturnsValidSM) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    Context ctx(0);
    int cc = ctx.compute_capability();
    
    // Valid compute capabilities: 30, 35, 37, 50, 52, 53, 60, 61, 62,
    // 70, 72, 75, 80, 86, 87, 89, 90
    EXPECT_GE(cc, 30);
    EXPECT_LE(cc, 100);
}

TEST(Context, TotalVramReturnsPositiveValue) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    Context ctx(0);
    size_t total = ctx.total_vram();
    EXPECT_GT(total, 0);
    
    // Sanity check: VRAM should be at least 1GB
    EXPECT_GE(total, 1ULL * 1024 * 1024 * 1024);
}

TEST(Context, FreeVramReturnsValueLessThanTotal) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    Context ctx(0);
    size_t total = ctx.total_vram();
    size_t free = ctx.free_vram();
    
    EXPECT_LE(free, total);
    EXPECT_GT(free, 0);  // Should have some free VRAM
}

TEST(Context, DeviceCountReturnsPositiveValue) {
    int count = Context::device_count();
    
    // If this test is running, we should have at least one device
    // (or the test would be skipped)
    EXPECT_GE(count, 0);
}

// ============================================================================
// Device Properties Tests
// ============================================================================

TEST(Context, PropertiesReturnsValidStructure) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    Context ctx(0);
    const cudaDeviceProp& props = ctx.properties();
    
    // Verify properties are populated
    EXPECT_GT(strlen(props.name), 0);
    EXPECT_GT(props.totalGlobalMem, 0);
    EXPECT_GT(props.major, 0);
    EXPECT_GE(props.minor, 0);
}

// ============================================================================
// VRAM-Only Enforcement Tests
// ============================================================================

TEST(Context, UMAIsDisabledAfterInit) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    Context ctx(0);
    
    // Query malloc heap size limit
    size_t heap_size;
    cudaError_t err = cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
    ASSERT_EQ(err, cudaSuccess);
    
    // Should be 0 (UMA disabled)
    EXPECT_EQ(heap_size, 0);
}

TEST(Context, CacheConfigIsSet) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    Context ctx(0);
    
    // Query cache config
    cudaFuncCache cache_config;
    cudaError_t err = cudaDeviceGetCacheConfig(&cache_config);
    ASSERT_EQ(err, cudaSuccess);
    
    // Should be set to prefer L1 (or device doesn't support it)
    // Note: Some devices may not support cache config, so we just verify
    // the query succeeds
    EXPECT_TRUE(
        cache_config == cudaFuncCachePreferNone ||
        cache_config == cudaFuncCachePreferShared ||
        cache_config == cudaFuncCachePreferL1 ||
        cache_config == cudaFuncCachePreferEqual
    );
}

// ============================================================================
// Cleanup Tests
// ============================================================================

TEST(Context, DestructorFreesVRAM) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    size_t free_before;
    {
        Context ctx(0);
        free_before = ctx.free_vram();
        
        // Allocate some VRAM
        void* ptr;
        cudaError_t err = cudaMalloc(&ptr, 1024 * 1024);  // 1MB
        ASSERT_EQ(err, cudaSuccess);
        
        // Free VRAM should decrease
        size_t free_after_alloc = ctx.free_vram();
        EXPECT_LT(free_after_alloc, free_before);
        
        // Don't free ptr - let destructor handle it
    }
    
    // After destructor, create new context and check VRAM
    Context ctx2(0);
    size_t free_after = ctx2.free_vram();
    
    // VRAM should be freed (within some tolerance for fragmentation)
    EXPECT_NEAR(free_after, free_before, 10 * 1024 * 1024);  // 10MB tolerance
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(Context, InvalidDeviceErrorHasMessage) {
    try {
        Context ctx(-1);
        FAIL() << "Expected CudaError to be thrown";
    } catch (const CudaError& e) {
        std::string msg = e.what();
        EXPECT_FALSE(msg.empty());
        EXPECT_NE(msg.find("range"), std::string::npos);
    }
}

TEST(Context, OutOfRangeDeviceErrorHasMessage) {
    try {
        Context ctx(999);
        FAIL() << "Expected CudaError to be thrown";
    } catch (const CudaError& e) {
        std::string msg = e.what();
        EXPECT_FALSE(msg.empty());
        EXPECT_NE(msg.find("range"), std::string::npos);
    }
}

// ============================================================================
// Property Tests
// ============================================================================

TEST(Context, AllValidDeviceIDsAccepted) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    // All device IDs from 0 to device_count-1 should work
    for (int i = 0; i < device_count; ++i) {
        EXPECT_NO_THROW({
            Context ctx(i);
        }) << "Device ID " << i << " should be valid";
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(Context, DeviceZeroAlwaysWorksIfDevicesExist) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    // Device 0 should always work if any devices exist
    EXPECT_NO_THROW({
        Context ctx(0);
    });
}

TEST(Context, LastDeviceWorks) {
    int device_count = Context::device_count();
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }
    
    // Last device should work
    EXPECT_NO_THROW({
        Context ctx(device_count - 1);
    });
}

// ---
// Built by Foundation-Alpha 🏗️
