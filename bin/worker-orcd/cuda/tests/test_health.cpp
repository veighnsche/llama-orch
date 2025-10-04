/**
 * Health Monitoring Unit Tests
 * 
 * Tests VRAM residency verification and health monitoring.
 * 
 * Spec: M0-W-1012, CUDA-5421
 * Story: FT-014
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/health.h"
#include "../include/vram_tracker.h"
#include "../include/device_memory.h"

using namespace worker;

// ============================================================================
// Test Fixture
// ============================================================================

class HealthTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA device
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
        cudaSetDevice(0);
    }
};

// ============================================================================
// check_pointer_residency Tests
// ============================================================================

/**
 * Test: check_pointer_residency with device pointer returns true
 * 
 * Spec: M0-W-1012 (VRAM Residency Verification)
 * Critical: Core residency check must work
 */
TEST_F(HealthTest, CheckPointerResidencyDevicePointerReturnsTrue) {
    // Allocate device memory
    void* device_ptr = nullptr;
    cudaError_t err = cudaMalloc(&device_ptr, 1024);
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate device memory";
    ASSERT_NE(device_ptr, nullptr);
    
    // Check residency - should return true
    bool resident = Health::check_pointer_residency(device_ptr);
    EXPECT_TRUE(resident) << "Device pointer should be VRAM-resident";
    
    // Verify pointer attributes directly
    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, device_ptr);
    ASSERT_EQ(err, cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice) << "Pointer should be device memory";
    EXPECT_EQ(attrs.hostPointer, nullptr) << "No host pointer should exist";
    
    // Cleanup
    cudaFree(device_ptr);
}

/**
 * Test: check_pointer_residency with host pointer returns false
 * 
 * Spec: M0-W-1012 (VRAM Residency Verification)
 * Critical: Must detect VRAM-only policy violations
 */
TEST_F(HealthTest, CheckPointerResidencyHostPointerReturnsFalse) {
    // Allocate pinned host memory
    void* host_ptr = nullptr;
    cudaError_t err = cudaMallocHost(&host_ptr, 1024);
    ASSERT_EQ(err, cudaSuccess) << "Failed to allocate host memory";
    ASSERT_NE(host_ptr, nullptr);
    
    // Check residency - should return false (host memory, not device)
    bool resident = Health::check_pointer_residency(host_ptr);
    EXPECT_FALSE(resident) << "Host pointer should NOT be VRAM-resident";
    
    // Cleanup
    cudaFreeHost(host_ptr);
}

/**
 * Test: check_pointer_residency with nullptr returns false
 * 
 * Critical: Defensive programming
 */
TEST_F(HealthTest, CheckPointerResidencyNullptrReturnsFalse) {
    bool resident = Health::check_pointer_residency(nullptr);
    EXPECT_FALSE(resident) << "nullptr should not be considered resident";
}

/**
 * Test: check_pointer_residency with managed memory returns false
 * 
 * Spec: M0-W-1012 (VRAM Residency Verification)
 * Critical: Must detect UMA violations
 */
TEST_F(HealthTest, CheckPointerResidencyManagedMemoryReturnsFalse) {
    // Allocate managed memory (simulates UMA)
    void* managed_ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&managed_ptr, 1024);
    
    if (err != cudaSuccess) {
        // Managed memory not supported on this device - skip test
        GTEST_SKIP() << "Managed memory not supported on this device";
    }
    
    ASSERT_NE(managed_ptr, nullptr);
    
    // Check residency - should return false (managed memory has host pointer)
    bool resident = Health::check_pointer_residency(managed_ptr);
    EXPECT_FALSE(resident) << "Managed memory should NOT be VRAM-only resident";
    
    // Verify pointer attributes
    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, managed_ptr);
    ASSERT_EQ(err, cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeManaged) << "Pointer should be managed memory";
    EXPECT_NE(attrs.hostPointer, nullptr) << "Managed memory has host pointer";
    
    // Cleanup
    cudaFree(managed_ptr);
}

// ============================================================================
// get_process_vram_usage Tests
// ============================================================================

/**
 * Test: get_process_vram_usage returns positive value
 * 
 * Critical: VRAM usage query must work
 */
TEST_F(HealthTest, GetProcessVramUsageReturnsPositiveValue) {
    // Allocate some VRAM
    void* device_ptr = nullptr;
    size_t alloc_size = 10 * 1024 * 1024;  // 10 MB
    cudaError_t err = cudaMalloc(&device_ptr, alloc_size);
    ASSERT_EQ(err, cudaSuccess);
    
    // Query process VRAM usage
    uint64_t vram_used = Health::get_process_vram_usage();
    EXPECT_GT(vram_used, 0) << "Process VRAM usage should be positive";
    EXPECT_GE(vram_used, alloc_size) << "Process VRAM should include our allocation";
    
    // Cleanup
    cudaFree(device_ptr);
}

/**
 * Test: get_process_vram_usage increases with allocations
 * 
 * Critical: VRAM usage tracking must be accurate
 */
TEST_F(HealthTest, GetProcessVramUsageIncreasesWithAllocations) {
    // Get baseline usage
    uint64_t baseline = Health::get_process_vram_usage();
    
    // Allocate VRAM
    void* device_ptr = nullptr;
    size_t alloc_size = 5 * 1024 * 1024;  // 5 MB
    cudaError_t err = cudaMalloc(&device_ptr, alloc_size);
    ASSERT_EQ(err, cudaSuccess);
    
    // Query usage after allocation
    uint64_t after_alloc = Health::get_process_vram_usage();
    EXPECT_GT(after_alloc, baseline) << "VRAM usage should increase after allocation";
    
    // Cleanup
    cudaFree(device_ptr);
}

// ============================================================================
// check_vram_residency (VramTracker) Tests
// ============================================================================

/**
 * Test: check_vram_residency with device allocations returns true
 * 
 * Spec: M0-W-1012 (VRAM Residency Verification)
 * Critical: Real-world VRAM validation
 */
TEST_F(HealthTest, CheckVramResidencyDeviceAllocationsReturnsTrue) {
    VramTracker tracker;
    
    // Allocate device memory via DeviceMemory RAII
    DeviceMemory mem1(1024 * 1024, &tracker, VramPurpose::ModelWeights);
    DeviceMemory mem2(2 * 1024 * 1024, &tracker, VramPurpose::KVCache);
    
    // Check residency - should return true
    bool resident = Health::check_vram_residency(tracker);
    EXPECT_TRUE(resident) << "All device allocations should be VRAM-resident";
    
    // Verify tracker state
    EXPECT_EQ(tracker.allocation_count(), 2);
    EXPECT_GT(tracker.total_usage(), 0);
}

/**
 * Test: check_vram_residency with empty tracker returns true
 * 
 * Edge case: Empty tracker should pass residency check
 */
TEST_F(HealthTest, CheckVramResidencyEmptyTrackerReturnsTrue) {
    VramTracker tracker;
    
    // Check residency with no allocations - should return true
    bool resident = Health::check_vram_residency(tracker);
    EXPECT_TRUE(resident) << "Empty tracker should pass residency check";
}

/**
 * Test: check_vram_residency detects managed memory violation
 * 
 * Spec: M0-W-1012 (VRAM Residency Verification)
 * Critical: Must detect UMA violations
 */
TEST_F(HealthTest, CheckVramResidencyDetectsManagedMemoryViolation) {
    VramTracker tracker;
    
    // Allocate managed memory (simulates UMA violation)
    void* managed_ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&managed_ptr, 1024);
    
    if (err != cudaSuccess) {
        GTEST_SKIP() << "Managed memory not supported on this device";
    }
    
    // Record in tracker
    tracker.record_allocation(managed_ptr, 1024, VramPurpose::Unknown);
    
    // Check residency - should return false
    bool resident = Health::check_vram_residency(tracker);
    EXPECT_FALSE(resident) << "Managed memory should fail residency check";
    
    // Cleanup
    tracker.record_deallocation(managed_ptr);
    cudaFree(managed_ptr);
}

// ============================================================================
// residency_report Tests
// ============================================================================

/**
 * Test: residency_report generates readable output
 * 
 * Critical: Health endpoint needs human-readable reports
 */
TEST_F(HealthTest, ResidencyReportGeneratesReadableOutput) {
    VramTracker tracker;
    
    // Allocate some VRAM
    DeviceMemory mem1(1024 * 1024, &tracker, VramPurpose::ModelWeights);
    DeviceMemory mem2(512 * 1024, &tracker, VramPurpose::KVCache);
    
    // Generate report
    std::string report = Health::residency_report(tracker);
    
    // Verify report contains expected sections
    EXPECT_NE(report.find("VRAM Residency Report"), std::string::npos);
    EXPECT_NE(report.find("Status:"), std::string::npos);
    EXPECT_NE(report.find("Process VRAM Usage:"), std::string::npos);
    EXPECT_NE(report.find("Tracked VRAM Usage:"), std::string::npos);
    EXPECT_NE(report.find("Allocations:"), std::string::npos);
    
    // Should show RESIDENT status
    EXPECT_NE(report.find("RESIDENT"), std::string::npos);
    EXPECT_EQ(report.find("VIOLATION"), std::string::npos);
    EXPECT_EQ(report.find("WARNING"), std::string::npos);
}

/**
 * Test: residency_report shows warning on violation
 * 
 * Critical: Violations must be clearly indicated
 */
TEST_F(HealthTest, ResidencyReportShowsWarningOnViolation) {
    VramTracker tracker;
    
    // Allocate managed memory (violation)
    void* managed_ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&managed_ptr, 1024);
    
    if (err != cudaSuccess) {
        GTEST_SKIP() << "Managed memory not supported on this device";
    }
    
    tracker.record_allocation(managed_ptr, 1024, VramPurpose::Unknown);
    
    // Generate report
    std::string report = Health::residency_report(tracker);
    
    // Should show VIOLATION status and WARNING
    EXPECT_NE(report.find("VIOLATION"), std::string::npos);
    EXPECT_NE(report.find("WARNING"), std::string::npos);
    EXPECT_NE(report.find("RAM fallback or UMA detected"), std::string::npos);
    
    // Cleanup
    tracker.record_deallocation(managed_ptr);
    cudaFree(managed_ptr);
}

/**
 * Test: residency_report with empty tracker
 * 
 * Edge case: Report should work with no allocations
 */
TEST_F(HealthTest, ResidencyReportEmptyTracker) {
    VramTracker tracker;
    
    // Generate report with no allocations
    std::string report = Health::residency_report(tracker);
    
    // Should show RESIDENT status (no violations)
    EXPECT_NE(report.find("RESIDENT"), std::string::npos);
    EXPECT_NE(report.find("Allocations: 0"), std::string::npos);
}

// ============================================================================
// Integration Tests
// ============================================================================

/**
 * Test: Health check workflow with multiple allocations
 * 
 * Integration test: Full health check workflow
 */
TEST_F(HealthTest, HealthCheckWorkflowMultipleAllocations) {
    VramTracker tracker;
    
    // Allocate multiple buffers
    DeviceMemory weights(10 * 1024 * 1024, &tracker, VramPurpose::ModelWeights);
    DeviceMemory kv_cache(5 * 1024 * 1024, &tracker, VramPurpose::KVCache);
    DeviceMemory intermediate(2 * 1024 * 1024, &tracker, VramPurpose::IntermediateBuffers);
    
    // Verify all allocations
    EXPECT_EQ(tracker.allocation_count(), 3);
    EXPECT_GE(tracker.total_usage(), 17 * 1024 * 1024);  // >= not > (10+5+2 = exactly 17MB)
    
    // Check residency
    EXPECT_TRUE(Health::check_vram_residency(tracker));
    
    // Generate report
    std::string report = Health::residency_report(tracker);
    EXPECT_NE(report.find("RESIDENT"), std::string::npos);
    EXPECT_NE(report.find("Allocations: 3"), std::string::npos);
    
    // Verify process VRAM includes our allocations
    uint64_t process_vram = Health::get_process_vram_usage();
    EXPECT_GE(process_vram, tracker.total_usage());
}

// ---
// Built by Foundation-Alpha 🏗️
