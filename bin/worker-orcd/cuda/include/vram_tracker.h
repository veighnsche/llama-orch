/**
 * VRAM Allocation Tracker
 * 
 * Tracks VRAM allocations and verifies VRAM-only residency.
 * 
 * Spec: M0-W-1010, M0-W-1011, M0-SYS-2.2.1
 */

#ifndef WORKER_VRAM_TRACKER_H
#define WORKER_VRAM_TRACKER_H

#include <cstddef>
#include <string>
#include <unordered_map>
#include <mutex>

namespace worker {

/**
 * Purpose of VRAM allocation.
 * 
 * Used to categorize allocations for reporting and debugging.
 */
enum class VramPurpose {
    ModelWeights,         ///< Model weight tensors
    KVCache,              ///< Key-Value cache for attention
    IntermediateBuffers,  ///< Temporary buffers for inference
    Unknown,              ///< Uncategorized allocation
};

/**
 * Record of a single VRAM allocation.
 */
struct VramAllocation {
    void* ptr;                  ///< Device pointer
    size_t bytes;               ///< Allocation size in bytes
    VramPurpose purpose;        ///< Purpose of allocation
    std::string description;    ///< Human-readable description
};

/**
 * VRAM allocation tracker.
 * 
 * Tracks all VRAM allocations and provides:
 * - Total usage reporting
 * - Usage breakdown by purpose
 * - VRAM residency verification
 * - Human-readable usage reports
 * 
 * Thread safety: All methods are thread-safe via internal mutex.
 * 
 * Example:
 * ```cpp
 * VramTracker tracker;
 * 
 * // Record allocation
 * void* ptr;
 * cudaMalloc(&ptr, 1024 * 1024);  // 1 MB
 * tracker.record_allocation(ptr, 1024 * 1024, VramPurpose::ModelWeights, "embedding layer");
 * 
 * // Query usage
 * std::cout << "Total VRAM: " << tracker.total_usage() << " bytes" << std::endl;
 * std::cout << tracker.usage_report() << std::endl;
 * 
 * // Verify residency
 * if (!tracker.verify_vram_residency()) {
 *     std::cerr << "ERROR: Host pointer detected!" << std::endl;
 * }
 * 
 * // Record deallocation
 * cudaFree(ptr);
 * tracker.record_deallocation(ptr);
 * ```
 */
class VramTracker {
public:
    VramTracker() = default;
    
    // Non-copyable, movable
    VramTracker(const VramTracker&) = delete;
    VramTracker& operator=(const VramTracker&) = delete;
    VramTracker(VramTracker&&) = default;
    VramTracker& operator=(VramTracker&&) = default;
    
    /**
     * Record VRAM allocation.
     * 
     * @param ptr Device pointer returned by cudaMalloc
     * @param bytes Allocation size in bytes
     * @param purpose Purpose of allocation
     * @param description Optional human-readable description
     */
    void record_allocation(
        void* ptr,
        size_t bytes,
        VramPurpose purpose,
        const std::string& description = ""
    );
    
    /**
     * Record VRAM deallocation.
     * 
     * @param ptr Device pointer to be freed
     */
    void record_deallocation(void* ptr);
    
    /**
     * Get total VRAM usage in bytes.
     * 
     * @return Sum of all tracked allocations
     */
    size_t total_usage() const;
    
    /**
     * Get VRAM usage by purpose.
     * 
     * @param purpose Purpose to query
     * @return Sum of allocations with specified purpose
     */
    size_t usage_by_purpose(VramPurpose purpose) const;
    
    /**
     * Get allocation count.
     * 
     * @return Number of tracked allocations
     */
    size_t allocation_count() const;
    
    /**
     * Get detailed usage breakdown.
     * 
     * @return Map of purpose to bytes used
     */
    std::unordered_map<VramPurpose, size_t> usage_breakdown() const;
    
    /**
     * Verify all tracked pointers are device memory.
     * 
     * Uses cudaPointerGetAttributes to verify:
     * - Pointer type is cudaMemoryTypeDevice
     * - No host pointer exists (no UMA)
     * 
     * @return true if all pointers are VRAM, false if any host pointers found
     */
    bool verify_vram_residency() const;
    
    /**
     * Get human-readable usage report.
     * 
     * Format:
     * ```
     * VRAM Usage Report:
     *   Total: 1024.00 MB
     *   Allocations: 42
     *   Breakdown:
     *     Model Weights: 512.00 MB
     *     KV Cache: 256.00 MB
     *     Intermediate Buffers: 256.00 MB
     * ```
     * 
     * @return Formatted usage report
     */
    std::string usage_report() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<void*, VramAllocation> allocations_;
};

} // namespace worker

#endif // WORKER_VRAM_TRACKER_H

// ---
// Built by Foundation-Alpha 🏗️
