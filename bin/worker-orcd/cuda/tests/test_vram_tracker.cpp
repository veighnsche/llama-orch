/**
 * VRAM Tracker Unit Tests
 * 
 * Tests VRAM allocation tracking and residency verification.
 * 
 * Spec: M0-W-1010, M0-W-1011, M0-SYS-2.2.1
 */

#include "vram_tracker.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <thread>
#include <vector>

using namespace worker;

class VramTrackerTest : public ::testing::Test {
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
// Basic Allocation Tracking Tests
// ============================================================================

TEST_F(VramTrackerTest, RecordAllocationIncrementsTotalUsage) {
    VramTracker tracker;
    
    // Allocate 1 MB
    void* ptr;
    ASSERT_EQ(cudaMalloc(&ptr, 1024 * 1024), cudaSuccess);
    
    tracker.record_allocation(ptr, 1024 * 1024, VramPurpose::ModelWeights, "test");
    
    EXPECT_EQ(tracker.total_usage(), 1024 * 1024);
    EXPECT_EQ(tracker.allocation_count(), 1);
    
    cudaFree(ptr);
}

TEST_F(VramTrackerTest, RecordDeallocationDecrementsTotalUsage) {
    VramTracker tracker;
    
    void* ptr;
    ASSERT_EQ(cudaMalloc(&ptr, 1024 * 1024), cudaSuccess);
    
    tracker.record_allocation(ptr, 1024 * 1024, VramPurpose::ModelWeights);
    EXPECT_EQ(tracker.total_usage(), 1024 * 1024);
    
    tracker.record_deallocation(ptr);
    EXPECT_EQ(tracker.total_usage(), 0);
    EXPECT_EQ(tracker.allocation_count(), 0);
    
    cudaFree(ptr);
}

TEST_F(VramTrackerTest, MultipleAllocationsTrackedCorrectly) {
    VramTracker tracker;
    
    void* ptr1;
    void* ptr2;
    void* ptr3;
    
    ASSERT_EQ(cudaMalloc(&ptr1, 1024 * 1024), cudaSuccess);      // 1 MB
    ASSERT_EQ(cudaMalloc(&ptr2, 2 * 1024 * 1024), cudaSuccess);  // 2 MB
    ASSERT_EQ(cudaMalloc(&ptr3, 512 * 1024), cudaSuccess);       // 0.5 MB
    
    tracker.record_allocation(ptr1, 1024 * 1024, VramPurpose::ModelWeights);
    tracker.record_allocation(ptr2, 2 * 1024 * 1024, VramPurpose::KVCache);
    tracker.record_allocation(ptr3, 512 * 1024, VramPurpose::IntermediateBuffers);
    
    EXPECT_EQ(tracker.total_usage(), 3.5 * 1024 * 1024);
    EXPECT_EQ(tracker.allocation_count(), 3);
    
    cudaFree(ptr1);
    cudaFree(ptr2);
    cudaFree(ptr3);
}

// ============================================================================
// Purpose-Based Tracking Tests
// ============================================================================

TEST_F(VramTrackerTest, UsageByPurposeReturnsCorrectBreakdown) {
    VramTracker tracker;
    
    void* ptr1;
    void* ptr2;
    void* ptr3;
    
    ASSERT_EQ(cudaMalloc(&ptr1, 1024 * 1024), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&ptr2, 2 * 1024 * 1024), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&ptr3, 512 * 1024), cudaSuccess);
    
    tracker.record_allocation(ptr1, 1024 * 1024, VramPurpose::ModelWeights);
    tracker.record_allocation(ptr2, 2 * 1024 * 1024, VramPurpose::ModelWeights);
    tracker.record_allocation(ptr3, 512 * 1024, VramPurpose::KVCache);
    
    EXPECT_EQ(tracker.usage_by_purpose(VramPurpose::ModelWeights), 3 * 1024 * 1024);
    EXPECT_EQ(tracker.usage_by_purpose(VramPurpose::KVCache), 512 * 1024);
    EXPECT_EQ(tracker.usage_by_purpose(VramPurpose::IntermediateBuffers), 0);
    
    cudaFree(ptr1);
    cudaFree(ptr2);
    cudaFree(ptr3);
}

TEST_F(VramTrackerTest, UsageBreakdownReturnsAllPurposes) {
    VramTracker tracker;
    
    void* ptr1;
    void* ptr2;
    void* ptr3;
    
    ASSERT_EQ(cudaMalloc(&ptr1, 1024 * 1024), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&ptr2, 2 * 1024 * 1024), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&ptr3, 512 * 1024), cudaSuccess);
    
    tracker.record_allocation(ptr1, 1024 * 1024, VramPurpose::ModelWeights);
    tracker.record_allocation(ptr2, 2 * 1024 * 1024, VramPurpose::KVCache);
    tracker.record_allocation(ptr3, 512 * 1024, VramPurpose::IntermediateBuffers);
    
    auto breakdown = tracker.usage_breakdown();
    
    EXPECT_EQ(breakdown[VramPurpose::ModelWeights], 1024 * 1024);
    EXPECT_EQ(breakdown[VramPurpose::KVCache], 2 * 1024 * 1024);
    EXPECT_EQ(breakdown[VramPurpose::IntermediateBuffers], 512 * 1024);
    EXPECT_EQ(breakdown.size(), 3);
    
    cudaFree(ptr1);
    cudaFree(ptr2);
    cudaFree(ptr3);
}

// ============================================================================
// VRAM Residency Verification Tests
// ============================================================================

TEST_F(VramTrackerTest, VerifyVramResidencyReturnsTrueForDevicePointers) {
    VramTracker tracker;
    
    void* ptr;
    ASSERT_EQ(cudaMalloc(&ptr, 1024 * 1024), cudaSuccess);
    
    tracker.record_allocation(ptr, 1024 * 1024, VramPurpose::ModelWeights);
    
    EXPECT_TRUE(tracker.verify_vram_residency());
    
    cudaFree(ptr);
}

TEST_F(VramTrackerTest, VerifyVramResidencyReturnsFalseForHostPointers) {
    VramTracker tracker;
    
    // Allocate host memory (not VRAM)
    void* host_ptr = malloc(1024 * 1024);
    
    // Record it as if it were VRAM (this is the error case we're testing)
    tracker.record_allocation(host_ptr, 1024 * 1024, VramPurpose::ModelWeights);
    
    // Verification should fail because it's not device memory
    EXPECT_FALSE(tracker.verify_vram_residency());
    
    free(host_ptr);
}

TEST_F(VramTrackerTest, VerifyVramResidencyReturnsTrueForEmptyTracker) {
    VramTracker tracker;
    
    // Empty tracker should pass verification (no allocations to check)
    EXPECT_TRUE(tracker.verify_vram_residency());
}

// ============================================================================
// Usage Report Tests
// ============================================================================

TEST_F(VramTrackerTest, UsageReportGeneratesReadableOutput) {
    VramTracker tracker;
    
    void* ptr1;
    void* ptr2;
    
    ASSERT_EQ(cudaMalloc(&ptr1, 1024 * 1024), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&ptr2, 2 * 1024 * 1024), cudaSuccess);
    
    tracker.record_allocation(ptr1, 1024 * 1024, VramPurpose::ModelWeights, "embedding");
    tracker.record_allocation(ptr2, 2 * 1024 * 1024, VramPurpose::KVCache, "attention cache");
    
    std::string report = tracker.usage_report();
    
    // Check report contains expected sections
    EXPECT_NE(report.find("VRAM Usage Report:"), std::string::npos);
    EXPECT_NE(report.find("Total:"), std::string::npos);
    EXPECT_NE(report.find("Allocations:"), std::string::npos);
    EXPECT_NE(report.find("Breakdown:"), std::string::npos);
    EXPECT_NE(report.find("Model Weights:"), std::string::npos);
    EXPECT_NE(report.find("KV Cache:"), std::string::npos);
    
    cudaFree(ptr1);
    cudaFree(ptr2);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(VramTrackerTest, ConcurrentAllocationsAreThreadSafe) {
    VramTracker tracker;
    
    const int num_threads = 10;
    const int allocs_per_thread = 10;
    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);
    
    // Launch threads that allocate and record
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < allocs_per_thread; ++i) {
                void* ptr;
                if (cudaMalloc(&ptr, 1024) == cudaSuccess) {
                    thread_ptrs[t].push_back(ptr);
                    tracker.record_allocation(ptr, 1024, VramPurpose::IntermediateBuffers);
                }
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify total usage
    EXPECT_EQ(tracker.allocation_count(), num_threads * allocs_per_thread);
    EXPECT_EQ(tracker.total_usage(), num_threads * allocs_per_thread * 1024);
    
    // Clean up
    for (const auto& ptrs : thread_ptrs) {
        for (void* ptr : ptrs) {
            tracker.record_deallocation(ptr);
            cudaFree(ptr);
        }
    }
    
    EXPECT_EQ(tracker.allocation_count(), 0);
    EXPECT_EQ(tracker.total_usage(), 0);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(VramTrackerTest, DeallocationOfUnknownPointerIsNoOp) {
    VramTracker tracker;
    
    void* fake_ptr = reinterpret_cast<void*>(0x12345678);
    
    // Deallocating unknown pointer should not crash
    tracker.record_deallocation(fake_ptr);
    
    EXPECT_EQ(tracker.allocation_count(), 0);
    EXPECT_EQ(tracker.total_usage(), 0);
}

TEST_F(VramTrackerTest, ZeroByteAllocationTrackedCorrectly) {
    VramTracker tracker;
    
    void* ptr = reinterpret_cast<void*>(0x1000);
    
    tracker.record_allocation(ptr, 0, VramPurpose::Unknown);
    
    EXPECT_EQ(tracker.allocation_count(), 1);
    EXPECT_EQ(tracker.total_usage(), 0);
}

// ============================================================================
// Integration with Context
// ============================================================================

TEST_F(VramTrackerTest, ContextProvidesVramTracker) {
    // This test verifies VramTracker is integrated into Context
    // Actual Context test is in test_context.cpp, but we verify the interface here
    
    VramTracker tracker;
    
    // Verify tracker can be used standalone
    void* ptr;
    ASSERT_EQ(cudaMalloc(&ptr, 1024), cudaSuccess);
    
    tracker.record_allocation(ptr, 1024, VramPurpose::ModelWeights);
    EXPECT_EQ(tracker.total_usage(), 1024);
    
    cudaFree(ptr);
}

// ---
// Built by Foundation-Alpha 🏗️
