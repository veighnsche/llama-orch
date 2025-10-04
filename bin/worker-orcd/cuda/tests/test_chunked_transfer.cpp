/**
 * Chunked Transfer Tests
 * 
 * Unit tests for chunked host-to-device transfer.
 * 
 * Tests cover:
 * - Chunked transfer logic
 * - Progress tracking
 * - Boundary conditions
 * - Error handling
 * 
 * Note: These tests use GTEST_SKIP() when CUDA is not available.
 * 
 * Spec: M0-W-1222
 */

#include <gtest/gtest.h>
#include "io/chunked_transfer.h"
#include "cuda_error.h"
#include <vector>
#include <cuda_runtime.h>

using namespace worker::io;

class ChunkedTransferTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA not available, skipping test";
        }
    }
};

// Test: Validate transfer parameters
TEST_F(ChunkedTransferTest, ValidateTransferParams) {
    std::vector<uint8_t> host_data(1024, 0xAB);
    void* device_ptr = nullptr;
    
    // Allocate device memory
    cudaMalloc(&device_ptr, 1024);
    
    TransferConfig config;
    
    // Valid parameters (should not throw)
    EXPECT_NO_THROW({
        ChunkedTransfer::validate_transfer_params(
            device_ptr, host_data.data(), 1024, config
        );
    });
    
    // NULL device pointer
    EXPECT_THROW({
        ChunkedTransfer::validate_transfer_params(
            nullptr, host_data.data(), 1024, config
        );
    }, worker::CudaError);
    
    // NULL host pointer
    EXPECT_THROW({
        ChunkedTransfer::validate_transfer_params(
            device_ptr, nullptr, 1024, config
        );
    }, worker::CudaError);
    
    // Zero size
    EXPECT_THROW({
        ChunkedTransfer::validate_transfer_params(
            device_ptr, host_data.data(), 0, config
        );
    }, worker::CudaError);
    
    // Zero chunk size
    config.chunk_size = 0;
    EXPECT_THROW({
        ChunkedTransfer::validate_transfer_params(
            device_ptr, host_data.data(), 1024, config
        );
    }, worker::CudaError);
    
    cudaFree(device_ptr);
}

// Test: Calculate chunk size
TEST_F(ChunkedTransferTest, CalculateChunkSize) {
    TransferConfig config;
    config.chunk_size = 256 * 1024 * 1024;  // 256MB
    
    // Small transfer: use total size
    EXPECT_EQ(
        ChunkedTransfer::calculate_chunk_size(100 * 1024 * 1024, config),
        100 * 1024 * 1024
    );
    
    // Large transfer: use chunk size
    EXPECT_EQ(
        ChunkedTransfer::calculate_chunk_size(500 * 1024 * 1024, config),
        256 * 1024 * 1024
    );
}

// Test: Transfer small data (single chunk)
TEST_F(ChunkedTransferTest, TransferSmallDataSingleChunk) {
    std::vector<uint8_t> host_data(1024, 0xAB);
    void* device_ptr = nullptr;
    
    cudaMalloc(&device_ptr, 1024);
    
    TransferConfig config;
    config.chunk_size = 256 * 1024 * 1024;  // Much larger than data
    
    // Transfer
    EXPECT_NO_THROW({
        ChunkedTransfer::h2d_chunked(
            device_ptr, host_data.data(), 1024, config
        );
    });
    
    // Verify data
    std::vector<uint8_t> device_data(1024);
    cudaMemcpy(device_data.data(), device_ptr, 1024, cudaMemcpyDeviceToHost);
    
    EXPECT_EQ(device_data, host_data);
    
    cudaFree(device_ptr);
}

// Test: Transfer large data (multiple chunks)
TEST_F(ChunkedTransferTest, TransferLargeDataMultipleChunks) {
    size_t size = 10 * 1024 * 1024;  // 10MB
    std::vector<uint8_t> host_data(size, 0xCD);
    void* device_ptr = nullptr;
    
    cudaMalloc(&device_ptr, size);
    
    TransferConfig config;
    config.chunk_size = 1 * 1024 * 1024;  // 1MB chunks (10 chunks total)
    
    // Transfer
    EXPECT_NO_THROW({
        ChunkedTransfer::h2d_chunked(
            device_ptr, host_data.data(), size, config
        );
    });
    
    // Verify data
    std::vector<uint8_t> device_data(size);
    cudaMemcpy(device_data.data(), device_ptr, size, cudaMemcpyDeviceToHost);
    
    EXPECT_EQ(device_data, host_data);
    
    cudaFree(device_ptr);
}

// Test: Transfer with progress callback
TEST_F(ChunkedTransferTest, TransferWithProgressCallback) {
    size_t size = 5 * 1024 * 1024;  // 5MB
    std::vector<uint8_t> host_data(size, 0xEF);
    void* device_ptr = nullptr;
    
    cudaMalloc(&device_ptr, size);
    
    TransferConfig config;
    config.chunk_size = 1 * 1024 * 1024;  // 1MB chunks
    config.enable_progress = true;
    
    // Track progress callbacks
    std::vector<std::pair<size_t, size_t>> progress_events;
    auto callback = [&](size_t transferred, size_t total) {
        progress_events.push_back({transferred, total});
    };
    
    // Transfer
    ChunkedTransfer::h2d_with_progress(
        device_ptr, host_data.data(), size, config, callback
    );
    
    // Verify progress events
    EXPECT_GT(progress_events.size(), 0u);
    
    // First event should be 0%
    EXPECT_EQ(progress_events[0].first, 0u);
    EXPECT_EQ(progress_events[0].second, size);
    
    // Last event should be 100%
    EXPECT_EQ(progress_events.back().first, size);
    EXPECT_EQ(progress_events.back().second, size);
    
    cudaFree(device_ptr);
}

// Test: Transfer with exact chunk boundary
TEST_F(ChunkedTransferTest, TransferWithExactChunkBoundary) {
    size_t size = 4 * 1024 * 1024;  // 4MB
    std::vector<uint8_t> host_data(size, 0x12);
    void* device_ptr = nullptr;
    
    cudaMalloc(&device_ptr, size);
    
    TransferConfig config;
    config.chunk_size = 1 * 1024 * 1024;  // Exactly 4 chunks
    
    // Transfer
    EXPECT_NO_THROW({
        ChunkedTransfer::h2d_chunked(
            device_ptr, host_data.data(), size, config
        );
    });
    
    // Verify data
    std::vector<uint8_t> device_data(size);
    cudaMemcpy(device_data.data(), device_ptr, size, cudaMemcpyDeviceToHost);
    
    EXPECT_EQ(device_data, host_data);
    
    cudaFree(device_ptr);
}

// Test: Transfer with partial last chunk
TEST_F(ChunkedTransferTest, TransferWithPartialLastChunk) {
    size_t size = 4 * 1024 * 1024 + 512 * 1024;  // 4.5MB
    std::vector<uint8_t> host_data(size, 0x34);
    void* device_ptr = nullptr;
    
    cudaMalloc(&device_ptr, size);
    
    TransferConfig config;
    config.chunk_size = 1 * 1024 * 1024;  // 4 full chunks + 1 partial (512KB)
    
    // Transfer
    EXPECT_NO_THROW({
        ChunkedTransfer::h2d_chunked(
            device_ptr, host_data.data(), size, config
        );
    });
    
    // Verify data
    std::vector<uint8_t> device_data(size);
    cudaMemcpy(device_data.data(), device_ptr, size, cudaMemcpyDeviceToHost);
    
    EXPECT_EQ(device_data, host_data);
    
    cudaFree(device_ptr);
}

// Test: Transfer with very small chunk size
TEST_F(ChunkedTransferTest, TransferWithSmallChunkSize) {
    size_t size = 10 * 1024;  // 10KB
    std::vector<uint8_t> host_data(size, 0x56);
    void* device_ptr = nullptr;
    
    cudaMalloc(&device_ptr, size);
    
    TransferConfig config;
    config.chunk_size = 1024;  // 1KB chunks (10 chunks)
    
    // Transfer
    EXPECT_NO_THROW({
        ChunkedTransfer::h2d_chunked(
            device_ptr, host_data.data(), size, config
        );
    });
    
    // Verify data
    std::vector<uint8_t> device_data(size);
    cudaMemcpy(device_data.data(), device_ptr, size, cudaMemcpyDeviceToHost);
    
    EXPECT_EQ(device_data, host_data);
    
    cudaFree(device_ptr);
}

// Test: Progress callback invocation count
TEST_F(ChunkedTransferTest, ProgressCallbackInvocationCount) {
    size_t size = 10 * 1024 * 1024;  // 10MB
    std::vector<uint8_t> host_data(size, 0x78);
    void* device_ptr = nullptr;
    
    cudaMalloc(&device_ptr, size);
    
    TransferConfig config;
    config.chunk_size = 1 * 1024 * 1024;  // 1MB chunks
    config.enable_progress = true;
    
    int callback_count = 0;
    auto callback = [&](size_t transferred, size_t total) {
        callback_count++;
        EXPECT_LE(transferred, total);
    };
    
    // Transfer
    ChunkedTransfer::h2d_with_progress(
        device_ptr, host_data.data(), size, config, callback
    );
    
    // Should have multiple progress events
    EXPECT_GT(callback_count, 1);
    
    cudaFree(device_ptr);
}

// Test: Zero-size transfer validation
TEST_F(ChunkedTransferTest, ZeroSizeTransferValidation) {
    std::vector<uint8_t> host_data(1024, 0x9A);
    void* device_ptr = nullptr;
    
    cudaMalloc(&device_ptr, 1024);
    
    TransferConfig config;
    
    // Zero size should throw
    EXPECT_THROW({
        ChunkedTransfer::h2d_chunked(
            device_ptr, host_data.data(), 0, config
        );
    }, worker::CudaError);
    
    cudaFree(device_ptr);
}

// Test: Chunk size validation (too small)
TEST_F(ChunkedTransferTest, ChunkSizeTooSmall) {
    std::vector<uint8_t> host_data(1024, 0xBC);
    void* device_ptr = nullptr;
    
    cudaMalloc(&device_ptr, 1024);
    
    TransferConfig config;
    config.chunk_size = 512;  // < 1KB minimum
    
    EXPECT_THROW({
        ChunkedTransfer::h2d_chunked(
            device_ptr, host_data.data(), 1024, config
        );
    }, worker::CudaError);
    
    cudaFree(device_ptr);
}

// Test: Chunk size validation (too large)
TEST_F(ChunkedTransferTest, ChunkSizeTooLarge) {
    std::vector<uint8_t> host_data(1024, 0xDE);
    void* device_ptr = nullptr;
    
    cudaMalloc(&device_ptr, 1024);
    
    TransferConfig config;
    config.chunk_size = 2ULL * 1024 * 1024 * 1024;  // 2GB > 1GB maximum
    
    EXPECT_THROW({
        ChunkedTransfer::h2d_chunked(
            device_ptr, host_data.data(), 1024, config
        );
    }, worker::CudaError);
    
    cudaFree(device_ptr);
}

// Test: Transfer with pattern verification
TEST_F(ChunkedTransferTest, TransferWithPatternVerification) {
    size_t size = 1024 * 1024;  // 1MB
    std::vector<uint8_t> host_data(size);
    
    // Create pattern
    for (size_t i = 0; i < size; ++i) {
        host_data[i] = static_cast<uint8_t>(i % 256);
    }
    
    void* device_ptr = nullptr;
    cudaMalloc(&device_ptr, size);
    
    TransferConfig config;
    config.chunk_size = 256 * 1024;  // 256KB chunks
    
    // Transfer
    ChunkedTransfer::h2d_chunked(
        device_ptr, host_data.data(), size, config
    );
    
    // Verify pattern
    std::vector<uint8_t> device_data(size);
    cudaMemcpy(device_data.data(), device_ptr, size, cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(device_data[i], static_cast<uint8_t>(i % 256));
    }
    
    cudaFree(device_ptr);
}

// ---
// Implemented by Llama-Beta 🦙
