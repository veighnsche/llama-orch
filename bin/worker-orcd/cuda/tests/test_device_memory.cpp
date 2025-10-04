/**
 * Device Memory RAII Wrapper Unit Tests
 * 
 * Tests RAII semantics, move operations, and VramTracker integration.
 * 
 * Spec: M0-W-1220, CUDA-5222
 */

#include "device_memory.h"
#include "vram_tracker.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>

using namespace worker;

class DeviceMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA device 0
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
        cudaSetDevice(0);
    }
};

// ============================================================================
// Basic Allocation Tests
// ============================================================================

TEST_F(DeviceMemoryTest, AllocatesMemorySuccessfully) {
    DeviceMemory mem(1024);
    
    EXPECT_NE(mem.get(), nullptr);
    EXPECT_EQ(mem.size(), 1024);
    EXPECT_TRUE(mem.is_allocated());
}

TEST_F(DeviceMemoryTest, FreesMemoryInDestructor) {
    VramTracker tracker;
    
    {
        DeviceMemory mem(1024 * 1024, &tracker, VramPurpose::ModelWeights);
        EXPECT_EQ(tracker.total_usage(), 1024 * 1024);
    }  // Destructor called here
    
    // Memory should be freed and tracker updated
    EXPECT_EQ(tracker.total_usage(), 0);
    EXPECT_EQ(tracker.allocation_count(), 0);
}

TEST_F(DeviceMemoryTest, AllocationWithZeroBytesThrows) {
    EXPECT_THROW({
        DeviceMemory mem(0);
    }, CudaError);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(DeviceMemoryTest, MoveConstructorTransfersOwnership) {
    VramTracker tracker;
    
    DeviceMemory mem1(1024, &tracker, VramPurpose::ModelWeights);
    void* original_ptr = mem1.get();
    
    EXPECT_EQ(tracker.allocation_count(), 1);
    
    // Move construct
    DeviceMemory mem2(std::move(mem1));
    
    EXPECT_EQ(mem2.get(), original_ptr);
    EXPECT_EQ(mem2.size(), 1024);
    EXPECT_EQ(mem1.get(), nullptr);
    EXPECT_EQ(mem1.size(), 0);
    
    // Tracker should still show 1 allocation
    EXPECT_EQ(tracker.allocation_count(), 1);
}

TEST_F(DeviceMemoryTest, MoveAssignmentTransfersOwnership) {
    VramTracker tracker;
    
    DeviceMemory mem1(1024, &tracker, VramPurpose::ModelWeights);
    DeviceMemory mem2(512, &tracker, VramPurpose::KVCache);
    
    void* mem1_ptr = mem1.get();
    
    EXPECT_EQ(tracker.allocation_count(), 2);
    EXPECT_EQ(tracker.total_usage(), 1024 + 512);
    
    // Move assign (mem2's old memory should be freed)
    mem2 = std::move(mem1);
    
    EXPECT_EQ(mem2.get(), mem1_ptr);
    EXPECT_EQ(mem2.size(), 1024);
    EXPECT_EQ(mem1.get(), nullptr);
    
    // Tracker should show 1 allocation (mem2's old memory freed)
    EXPECT_EQ(tracker.allocation_count(), 1);
    EXPECT_EQ(tracker.total_usage(), 1024);
}

TEST_F(DeviceMemoryTest, MoveAssignmentToSelfIsNoOp) {
    VramTracker tracker;
    
    DeviceMemory mem(1024, &tracker, VramPurpose::ModelWeights);
    void* original_ptr = mem.get();
    
    // Move to self
    mem = std::move(mem);
    
    EXPECT_EQ(mem.get(), original_ptr);
    EXPECT_EQ(mem.size(), 1024);
    EXPECT_EQ(tracker.allocation_count(), 1);
}

// ============================================================================
// Aligned Allocation Tests
// ============================================================================

TEST_F(DeviceMemoryTest, AlignedAllocationReturnsAlignedPointer) {
    auto mem = DeviceMemory::aligned(1024, 256);
    
    uintptr_t addr = reinterpret_cast<uintptr_t>(mem->get());
    EXPECT_EQ(addr % 256, 0) << "Pointer not aligned to 256 bytes";
}

TEST_F(DeviceMemoryTest, AlignedAllocationWithVariousAlignments) {
    std::vector<size_t> alignments = {64, 128, 256, 512, 1024};
    
    for (size_t alignment : alignments) {
        auto mem = DeviceMemory::aligned(1024, alignment);
        
        uintptr_t addr = reinterpret_cast<uintptr_t>(mem->get());
        EXPECT_EQ(addr % alignment, 0) 
            << "Pointer not aligned to " << alignment << " bytes";
    }
}

TEST_F(DeviceMemoryTest, AlignedAllocationWithNonPowerOf2Throws) {
    EXPECT_THROW({
        auto mem = DeviceMemory::aligned(1024, 100);  // Not power of 2
    }, CudaError);
}

TEST_F(DeviceMemoryTest, AlignedAllocationWithZeroAlignmentThrows) {
    EXPECT_THROW({
        auto mem = DeviceMemory::aligned(1024, 0);
    }, CudaError);
}

// ============================================================================
// Zero Initialization Tests
// ============================================================================

TEST_F(DeviceMemoryTest, ZeroInitializationSetsMemoryToZero) {
    const size_t size = 1024;
    DeviceMemory mem(size, nullptr, VramPurpose::Unknown, true);
    
    // Copy to host and verify all zeros
    std::vector<uint8_t> host_buf(size);
    mem.copy_to_host(host_buf.data(), size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(host_buf[i], 0) << "Byte " << i << " not zero";
    }
}

TEST_F(DeviceMemoryTest, ZeroMethodSetsMemoryToZero) {
    const size_t size = 1024;
    DeviceMemory mem(size);
    
    // Fill with non-zero pattern
    std::vector<uint8_t> pattern(size, 0xFF);
    mem.copy_from_host(pattern.data(), size);
    
    // Zero the memory
    mem.zero();
    
    // Verify all zeros
    std::vector<uint8_t> host_buf(size);
    mem.copy_to_host(host_buf.data(), size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(host_buf[i], 0) << "Byte " << i << " not zero after zero()";
    }
}

// ============================================================================
// Host-Device Copy Tests
// ============================================================================

TEST_F(DeviceMemoryTest, CopyFromHostWorks) {
    const size_t size = 1024;
    DeviceMemory mem(size);
    
    // Create pattern on host
    std::vector<uint8_t> host_pattern(size);
    for (size_t i = 0; i < size; ++i) {
        host_pattern[i] = static_cast<uint8_t>(i % 256);
    }
    
    // Copy to device
    mem.copy_from_host(host_pattern.data(), size);
    
    // Copy back and verify
    std::vector<uint8_t> host_result(size);
    mem.copy_to_host(host_result.data(), size);
    
    EXPECT_EQ(host_result, host_pattern);
}

TEST_F(DeviceMemoryTest, CopyToHostWorks) {
    const size_t size = 1024;
    DeviceMemory mem(size);
    
    // Create pattern on host
    std::vector<uint8_t> host_pattern(size, 0xAB);
    mem.copy_from_host(host_pattern.data(), size);
    
    // Copy to host
    std::vector<uint8_t> host_result(size);
    mem.copy_to_host(host_result.data(), size);
    
    EXPECT_EQ(host_result, host_pattern);
}

TEST_F(DeviceMemoryTest, CopyFromHostWithOversizeThrows) {
    DeviceMemory mem(1024);
    
    std::vector<uint8_t> large_buffer(2048);
    
    EXPECT_THROW({
        mem.copy_from_host(large_buffer.data(), 2048);
    }, CudaError);
}

TEST_F(DeviceMemoryTest, CopyToHostWithOversizeThrows) {
    DeviceMemory mem(1024);
    
    std::vector<uint8_t> large_buffer(2048);
    
    EXPECT_THROW({
        mem.copy_to_host(large_buffer.data(), 2048);
    }, CudaError);
}

// ============================================================================
// Release Ownership Tests
// ============================================================================

TEST_F(DeviceMemoryTest, ReleaseTransfersOwnership) {
    VramTracker tracker;
    
    void* ptr;
    {
        DeviceMemory mem(1024, &tracker, VramPurpose::ModelWeights);
        EXPECT_EQ(tracker.allocation_count(), 1);
        
        ptr = mem.release();
        
        EXPECT_NE(ptr, nullptr);
        EXPECT_EQ(mem.get(), nullptr);
        EXPECT_EQ(mem.size(), 0);
        
        // Tracker should show deallocation (ownership transferred)
        EXPECT_EQ(tracker.allocation_count(), 0);
    }
    
    // Manually free (caller's responsibility after release)
    cudaFree(ptr);
}

// ============================================================================
// VramTracker Integration Tests
// ============================================================================

TEST_F(DeviceMemoryTest, IntegratesWithVramTracker) {
    VramTracker tracker;
    
    {
        DeviceMemory mem(1024 * 1024, &tracker, VramPurpose::ModelWeights, false);
        
        EXPECT_EQ(tracker.total_usage(), 1024 * 1024);
        EXPECT_EQ(tracker.allocation_count(), 1);
        EXPECT_EQ(tracker.usage_by_purpose(VramPurpose::ModelWeights), 1024 * 1024);
    }
    
    // After destructor
    EXPECT_EQ(tracker.total_usage(), 0);
    EXPECT_EQ(tracker.allocation_count(), 0);
}

TEST_F(DeviceMemoryTest, WorksWithoutTracker) {
    // Should work fine without tracker
    DeviceMemory mem(1024, nullptr, VramPurpose::Unknown);
    
    EXPECT_NE(mem.get(), nullptr);
    EXPECT_EQ(mem.size(), 1024);
}

// ============================================================================
// Exception Safety Tests
// ============================================================================

TEST_F(DeviceMemoryTest, NoLeaksWhenMultipleAllocations) {
    VramTracker tracker;
    
    // Allocate multiple objects
    for (int i = 0; i < 10; ++i) {
        DeviceMemory mem(1024, &tracker, VramPurpose::IntermediateBuffers);
        EXPECT_EQ(tracker.allocation_count(), 1);
        // Destructor called at end of iteration
    }
    
    // All should be freed
    EXPECT_EQ(tracker.total_usage(), 0);
    EXPECT_EQ(tracker.allocation_count(), 0);
}

TEST_F(DeviceMemoryTest, ExceptionSafetyOnAllocationFailure) {
    VramTracker tracker;
    
    // Allocate some memory first
    DeviceMemory mem1(1024 * 1024, &tracker, VramPurpose::ModelWeights);
    EXPECT_EQ(tracker.allocation_count(), 1);
    
    // Try to allocate huge amount (likely to fail)
    try {
        DeviceMemory mem2(1024ULL * 1024 * 1024 * 1024, &tracker, VramPurpose::KVCache);
        // If allocation succeeds (unlikely), that's fine
    } catch (const CudaError&) {
        // Expected: OOM
    }
    
    // First allocation should still be tracked
    EXPECT_EQ(tracker.allocation_count(), 1);
    EXPECT_EQ(tracker.total_usage(), 1024 * 1024);
}

// ============================================================================
// Type-Safe Pointer Tests
// ============================================================================

TEST_F(DeviceMemoryTest, GetAsReturnsTypedPointer) {
    DeviceMemory mem(sizeof(float) * 100);
    
    float* typed_ptr = mem.get_as<float>();
    EXPECT_NE(typed_ptr, nullptr);
    EXPECT_EQ(typed_ptr, static_cast<float*>(mem.get()));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(DeviceMemoryTest, LargeAllocation) {
    // Allocate 100 MB
    const size_t size = 100 * 1024 * 1024;
    
    try {
        DeviceMemory mem(size);
        EXPECT_NE(mem.get(), nullptr);
        EXPECT_EQ(mem.size(), size);
    } catch (const CudaError& e) {
        // If allocation fails due to insufficient VRAM, that's acceptable
        GTEST_SKIP() << "Insufficient VRAM for 100MB allocation: " << e.what();
    }
}

TEST_F(DeviceMemoryTest, SmallAllocation) {
    // Allocate 1 byte
    DeviceMemory mem(1);
    
    EXPECT_NE(mem.get(), nullptr);
    EXPECT_EQ(mem.size(), 1);
}

TEST_F(DeviceMemoryTest, MultipleSequentialAllocations) {
    VramTracker tracker;
    
    for (int i = 0; i < 5; ++i) {
        DeviceMemory mem(1024 * (i + 1), &tracker, VramPurpose::IntermediateBuffers);
        EXPECT_EQ(tracker.allocation_count(), 1);
        // Destructor at end of iteration
    }
    
    EXPECT_EQ(tracker.total_usage(), 0);
}

// ============================================================================
// Integration with VramTracker Tests
// ============================================================================

TEST_F(DeviceMemoryTest, TrackerRecordsCorrectPurpose) {
    VramTracker tracker;
    
    {
        DeviceMemory mem1(1024, &tracker, VramPurpose::ModelWeights);
        DeviceMemory mem2(2048, &tracker, VramPurpose::KVCache);
        DeviceMemory mem3(512, &tracker, VramPurpose::IntermediateBuffers);
        
        EXPECT_EQ(tracker.usage_by_purpose(VramPurpose::ModelWeights), 1024);
        EXPECT_EQ(tracker.usage_by_purpose(VramPurpose::KVCache), 2048);
        EXPECT_EQ(tracker.usage_by_purpose(VramPurpose::IntermediateBuffers), 512);
    }
    
    EXPECT_EQ(tracker.total_usage(), 0);
}

// ============================================================================
// Aligned Allocation Comprehensive Tests
// ============================================================================

TEST_F(DeviceMemoryTest, AlignedAllocationRoundsUpSize) {
    auto mem = DeviceMemory::aligned(1000, 256);
    
    // Size should be rounded up to 1024 (next multiple of 256)
    EXPECT_GE(mem->size(), 1000);
    EXPECT_EQ(mem->size() % 256, 0);
}

TEST_F(DeviceMemoryTest, AlignedAllocationWithTrackerIntegration) {
    VramTracker tracker;
    
    {
        auto mem = DeviceMemory::aligned(1024, 256, &tracker, VramPurpose::ModelWeights);
        
        EXPECT_EQ(tracker.allocation_count(), 1);
        EXPECT_GE(tracker.total_usage(), 1024);
        
        uintptr_t addr = reinterpret_cast<uintptr_t>(mem->get());
        EXPECT_EQ(addr % 256, 0);
    }
    
    EXPECT_EQ(tracker.total_usage(), 0);
}

TEST_F(DeviceMemoryTest, AlignedAllocationWithZeroInit) {
    const size_t size = 1024;
    auto mem = DeviceMemory::aligned(size, 256, nullptr, VramPurpose::Unknown, true);
    
    // Verify alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(mem->get());
    EXPECT_EQ(addr % 256, 0);
    
    // Verify zero-init
    std::vector<uint8_t> host_buf(size);
    mem->copy_to_host(host_buf.data(), size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(host_buf[i], 0) << "Byte " << i << " not zero";
    }
}

// ============================================================================
// Copy Operations Tests
// ============================================================================

TEST_F(DeviceMemoryTest, PartialCopyFromHost) {
    DeviceMemory mem(1024);
    
    std::vector<uint8_t> pattern(512, 0xCD);
    mem.copy_from_host(pattern.data(), 512);  // Copy only 512 bytes
    
    std::vector<uint8_t> result(512);
    mem.copy_to_host(result.data(), 512);
    
    EXPECT_EQ(result, pattern);
}

TEST_F(DeviceMemoryTest, PartialCopyToHost) {
    DeviceMemory mem(1024);
    
    std::vector<uint8_t> pattern(1024, 0xEF);
    mem.copy_from_host(pattern.data(), 1024);
    
    // Copy only first 512 bytes
    std::vector<uint8_t> result(512);
    mem.copy_to_host(result.data(), 512);
    
    for (size_t i = 0; i < 512; ++i) {
        EXPECT_EQ(result[i], 0xEF);
    }
}

// ============================================================================
// RAII Stress Tests
// ============================================================================

TEST_F(DeviceMemoryTest, RapidAllocationDeallocation) {
    VramTracker tracker;
    
    for (int i = 0; i < 100; ++i) {
        DeviceMemory mem(1024, &tracker, VramPurpose::IntermediateBuffers);
        EXPECT_EQ(tracker.allocation_count(), 1);
    }
    
    EXPECT_EQ(tracker.total_usage(), 0);
}

TEST_F(DeviceMemoryTest, NestedAllocations) {
    VramTracker tracker;
    
    {
        DeviceMemory mem1(1024, &tracker, VramPurpose::ModelWeights);
        EXPECT_EQ(tracker.allocation_count(), 1);
        
        {
            DeviceMemory mem2(2048, &tracker, VramPurpose::KVCache);
            EXPECT_EQ(tracker.allocation_count(), 2);
            
            {
                DeviceMemory mem3(512, &tracker, VramPurpose::IntermediateBuffers);
                EXPECT_EQ(tracker.allocation_count(), 3);
            }
            
            EXPECT_EQ(tracker.allocation_count(), 2);
        }
        
        EXPECT_EQ(tracker.allocation_count(), 1);
    }
    
    EXPECT_EQ(tracker.allocation_count(), 0);
}

// ---
// Built by Foundation-Alpha 🏗️
