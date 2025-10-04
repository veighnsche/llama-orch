/**
 * Chunked Host-to-Device Transfer Implementation
 * 
 * Implements efficient chunked cudaMemcpy for large tensor transfers.
 * 
 * Spec: M0-W-1222
 */

#include "io/chunked_transfer.h"
#include "../cuda_error.h"
#include <algorithm>
#include <cstring>

namespace worker {
namespace io {

void ChunkedTransfer::validate_transfer_params(
    const void* device_ptr,
    const void* host_ptr,
    size_t total_size,
    const TransferConfig& config
) {
    if (!device_ptr) {
        throw CudaError::invalid_parameter("device_ptr is NULL");
    }
    
    if (!host_ptr) {
        throw CudaError::invalid_parameter("host_ptr is NULL");
    }
    
    if (total_size == 0) {
        throw CudaError::invalid_parameter("total_size is 0");
    }
    
    if (config.chunk_size == 0) {
        throw CudaError::invalid_parameter("chunk_size is 0");
    }
    
    // Sanity check: chunk size should be reasonable (1KB - 1GB)
    if (config.chunk_size < 1024) {
        throw CudaError::invalid_parameter(
            "chunk_size too small: " + std::to_string(config.chunk_size) +
            " (minimum 1KB)"
        );
    }
    
    if (config.chunk_size > 1024 * 1024 * 1024) {
        throw CudaError::invalid_parameter(
            "chunk_size too large: " + std::to_string(config.chunk_size) +
            " (maximum 1GB)"
        );
    }
}

size_t ChunkedTransfer::calculate_chunk_size(
    size_t total_size,
    const TransferConfig& config
) {
    // For small transfers, use single chunk
    if (total_size <= config.chunk_size) {
        return total_size;
    }
    
    return config.chunk_size;
}

void ChunkedTransfer::h2d_chunked(
    void* device_ptr,
    const void* host_ptr,
    size_t total_size,
    const TransferConfig& config
) {
    // Validate parameters
    validate_transfer_params(device_ptr, host_ptr, total_size, config);
    
    // Transfer in chunks
    size_t offset = 0;
    while (offset < total_size) {
        // Calculate chunk size for this iteration
        size_t chunk = std::min(config.chunk_size, total_size - offset);
        
        // Perform transfer
        cudaError_t err = cudaMemcpy(
            static_cast<char*>(device_ptr) + offset,
            static_cast<const char*>(host_ptr) + offset,
            chunk,
            cudaMemcpyHostToDevice
        );
        
        if (err != cudaSuccess) {
            throw CudaError::kernel_launch_failed(
                "cudaMemcpy failed at offset " + std::to_string(offset) +
                " (chunk size: " + std::to_string(chunk) + "): " +
                cudaGetErrorString(err)
            );
        }
        
        offset += chunk;
    }
    
    // Synchronize if using default stream
    if (config.stream == nullptr) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw CudaError::kernel_launch_failed(
                "cudaDeviceSynchronize failed after transfer: " +
                std::string(cudaGetErrorString(err))
            );
        }
    }
}

void ChunkedTransfer::h2d_with_progress(
    void* device_ptr,
    const void* host_ptr,
    size_t total_size,
    const TransferConfig& config,
    ProgressCallback progress_callback
) {
    // Validate parameters
    validate_transfer_params(device_ptr, host_ptr, total_size, config);
    
    if (!progress_callback) {
        // No callback, use regular chunked transfer
        h2d_chunked(device_ptr, host_ptr, total_size, config);
        return;
    }
    
    // Track progress
    size_t offset = 0;
    size_t last_progress_offset = 0;
    float progress_threshold = total_size * PROGRESS_EMIT_THRESHOLD;
    
    // Initial progress (0%)
    if (config.enable_progress) {
        progress_callback(0, total_size);
    }
    
    while (offset < total_size) {
        // Calculate chunk size
        size_t chunk = std::min(config.chunk_size, total_size - offset);
        
        // Perform transfer
        cudaError_t err = cudaMemcpy(
            static_cast<char*>(device_ptr) + offset,
            static_cast<const char*>(host_ptr) + offset,
            chunk,
            cudaMemcpyHostToDevice
        );
        
        if (err != cudaSuccess) {
            throw CudaError::kernel_launch_failed(
                "cudaMemcpy failed at offset " + std::to_string(offset) +
                " (chunk size: " + std::to_string(chunk) + "): " +
                cudaGetErrorString(err)
            );
        }
        
        offset += chunk;
        
        // Emit progress if threshold crossed
        if (config.enable_progress) {
            float progress_delta = static_cast<float>(offset - last_progress_offset);
            if (progress_delta >= progress_threshold || offset == total_size) {
                progress_callback(offset, total_size);
                last_progress_offset = offset;
            }
        }
    }
    
    // Synchronize if using default stream
    if (config.stream == nullptr) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw CudaError::kernel_launch_failed(
                "cudaDeviceSynchronize failed after transfer: " +
                std::string(cudaGetErrorString(err))
            );
        }
    }
    
    // Final progress (100%)
    if (config.enable_progress) {
        progress_callback(total_size, total_size);
    }
}

} // namespace io
} // namespace worker

// ---
// Implemented by Llama-Beta 🦙
