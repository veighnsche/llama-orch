/**
 * Chunked Host-to-Device Transfer
 * 
 * Implements chunked cudaMemcpy for efficient large tensor transfers.
 * Prevents memory spikes and enables progress tracking.
 * 
 * Spec: M0-W-1222
 */

#ifndef WORKER_IO_CHUNKED_TRANSFER_H
#define WORKER_IO_CHUNKED_TRANSFER_H

#include <cstddef>
#include <functional>
#include <cuda_runtime.h>

namespace worker {
namespace io {

// Default chunk size: 256MB
constexpr size_t DEFAULT_CHUNK_SIZE = 256 * 1024 * 1024;

// Progress threshold: emit progress every 10%
constexpr float PROGRESS_EMIT_THRESHOLD = 0.1f;

/**
 * Transfer configuration
 */
struct TransferConfig {
    size_t chunk_size = DEFAULT_CHUNK_SIZE;  // 256MB default
    bool enable_progress = true;
    cudaStream_t stream = nullptr;  // nullptr = default stream
};

/**
 * Progress callback signature
 * 
 * @param bytes_transferred Bytes transferred so far
 * @param total_bytes Total bytes to transfer
 */
using ProgressCallback = std::function<void(size_t bytes_transferred, size_t total_bytes)>;

/**
 * Chunked transfer utilities
 */
class ChunkedTransfer {
public:
    /**
     * Transfer data from host to device in chunks
     * 
     * Performs chunked cudaMemcpy to avoid memory spikes during large transfers.
     * 
     * @param device_ptr Destination pointer in VRAM (must be valid)
     * @param host_ptr Source pointer in host memory (must be valid)
     * @param total_size Total bytes to transfer
     * @param config Transfer configuration
     * @throws CudaError on invalid pointers or cudaMemcpy failure
     */
    static void h2d_chunked(
        void* device_ptr,
        const void* host_ptr,
        size_t total_size,
        const TransferConfig& config = TransferConfig()
    );
    
    /**
     * Transfer data with progress callback
     * 
     * Same as h2d_chunked but invokes callback for progress tracking.
     * Callback is invoked after each chunk and at completion.
     * 
     * @param device_ptr Destination pointer in VRAM
     * @param host_ptr Source pointer in host memory
     * @param total_size Total bytes to transfer
     * @param config Transfer configuration
     * @param progress_callback Callback for progress updates
     * @throws CudaError on invalid pointers or cudaMemcpy failure
     */
    static void h2d_with_progress(
        void* device_ptr,
        const void* host_ptr,
        size_t total_size,
        const TransferConfig& config,
        ProgressCallback progress_callback
    );
    
    /**
     * Calculate optimal chunk size based on transfer size
     * 
     * For small transfers (<256MB), use single chunk.
     * For large transfers, use configured chunk size.
     * 
     * @param total_size Total bytes to transfer
     * @param config Transfer configuration
     * @return Optimal chunk size
     */
    static size_t calculate_chunk_size(
        size_t total_size,
        const TransferConfig& config
    );
    
    /**
     * Validate transfer parameters
     * 
     * Checks:
     * - Pointers are not null
     * - Total size is reasonable
     * - Chunk size is reasonable
     * 
     * @param device_ptr Destination pointer
     * @param host_ptr Source pointer
     * @param total_size Total bytes
     * @param config Transfer configuration
     * @throws CudaError if validation fails
     */
    static void validate_transfer_params(
        const void* device_ptr,
        const void* host_ptr,
        size_t total_size,
        const TransferConfig& config
    );
};

} // namespace io
} // namespace worker

#endif // WORKER_IO_CHUNKED_TRANSFER_H

// ---
// Implemented by Llama-Beta 🦙
