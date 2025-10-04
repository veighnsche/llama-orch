/**
 * CUDA Health Monitoring
 * 
 * Provides health checks and VRAM residency verification for worker-orcd.
 * 
 * Key Features:
 * - VRAM residency verification (detect RAM fallback)
 * - Process-wide VRAM usage queries
 * - Device health checks
 * - Human-readable health reports
 * 
 * Spec: M0-W-1012, CUDA-5421
 */

#ifndef WORKER_HEALTH_H
#define WORKER_HEALTH_H

#include <cuda_runtime.h>
#include <string>
#include "vram_tracker.h"

namespace worker {

/**
 * Health monitoring utilities.
 * 
 * All methods are static and thread-safe (read-only operations).
 * 
 * Example:
 * ```cpp
 * VramTracker tracker;
 * // ... allocate VRAM ...
 * 
 * // Check residency
 * if (!Health::check_vram_residency(tracker)) {
 *     std::cerr << "ERROR: RAM fallback detected!" << std::endl;
 * }
 * 
 * // Get process VRAM usage
 * uint64_t vram_used = Health::get_process_vram_usage();
 * std::cout << "Process VRAM: " << (vram_used / 1024.0 / 1024.0) << " MB" << std::endl;
 * 
 * // Generate report
 * std::cout << Health::residency_report(tracker) << std::endl;
 * ```
 */
class Health {
public:
    /**
     * Check VRAM residency for all tracked allocations.
     * 
     * Verifies that all allocations in the tracker are device memory
     * with no host pointers (no UMA/RAM fallback).
     * 
     * @param tracker VRAM tracker with allocations to check
     * @return true if all allocations are VRAM-resident, false otherwise
     * 
     * Thread safety: Safe to call concurrently (read-only)
     * 
     * Spec: M0-W-1012 (VRAM Residency Verification)
     */
    static bool check_vram_residency(const VramTracker& tracker);
    
    /**
     * Check VRAM residency for specific pointer.
     * 
     * Uses cudaPointerGetAttributes to verify:
     * - Pointer type is cudaMemoryTypeDevice (not managed/host)
     * - No host pointer exists (hostPointer == nullptr)
     * 
     * @param ptr Device pointer to check (may be nullptr)
     * @return true if pointer is VRAM-resident, false otherwise
     * 
     * Thread safety: Safe to call concurrently (read-only)
     * 
     * Spec: M0-W-1012 (VRAM Residency Verification)
     */
    static bool check_pointer_residency(const void* ptr);
    
    /**
     * Get process-wide VRAM usage.
     * 
     * Uses cudaMemGetInfo to query total VRAM allocated by this process.
     * 
     * @return VRAM bytes used by this process, or 0 on error
     * 
     * Thread safety: Safe to call concurrently (read-only)
     * 
     * Spec: M0-W-1012 (Process VRAM Usage Query)
     */
    static uint64_t get_process_vram_usage();
    
    /**
     * Get detailed residency report.
     * 
     * Generates human-readable report with:
     * - Residency status (RESIDENT or VIOLATION)
     * - Process VRAM usage
     * - Tracked VRAM usage
     * - Allocation count
     * - Warning if residency violation detected
     * 
     * Format:
     * ```
     * VRAM Residency Report:
     *   Status: RESIDENT
     *   Process VRAM Usage: 1024.00 MB
     *   Tracked VRAM Usage: 1024.00 MB
     *   Allocations: 42
     * ```
     * 
     * @param tracker VRAM tracker
     * @return Human-readable report
     * 
     * Thread safety: Safe to call concurrently (read-only)
     */
    static std::string residency_report(const VramTracker& tracker);
};

} // namespace worker

#endif // WORKER_HEALTH_H

// ---
// Built by Foundation-Alpha 🏗️
