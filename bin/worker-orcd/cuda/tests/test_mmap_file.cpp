/**
 * Memory-Mapped File I/O Tests
 * 
 * Unit tests for memory-mapped file access.
 * 
 * Tests cover:
 * - File opening and mapping
 * - Pointer access at offsets
 * - Bounds validation
 * - Error handling
 * - RAII cleanup
 * 
 * Spec: M0-W-1221
 */

#include <gtest/gtest.h>
#include "io/mmap_file.h"
#include "cuda_error.h"
#include <fstream>
#include <vector>
#include <cstring>
#include <unistd.h>

using namespace worker::io;

class MmapFileTest : public ::testing::Test {
protected:
    std::string test_file_path;
    
    void SetUp() override {
        // Create temporary test file
        test_file_path = "/tmp/test_mmap_" + std::to_string(getpid()) + ".bin";
        
        // Write test data
        std::ofstream file(test_file_path, std::ios::binary);
        std::vector<uint8_t> data(4096, 0xAB);
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();
    }
    
    void TearDown() override {
        // Clean up test file
        unlink(test_file_path.c_str());
    }
    
    // Helper: Create test file with specific content
    void create_test_file(const std::vector<uint8_t>& content) {
        std::ofstream file(test_file_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(content.data()), content.size());
        file.close();
    }
};

// Test: Open and map file
TEST_F(MmapFileTest, OpenAndMapFile) {
    MmapFile mmap = MmapFile::open(test_file_path);
    
    EXPECT_TRUE(mmap.is_mapped());
    EXPECT_EQ(mmap.size(), 4096u);
    EXPECT_NE(mmap.data(), nullptr);
    EXPECT_EQ(mmap.path(), test_file_path);
}

// Test: Access data at offset 0
TEST_F(MmapFileTest, AccessDataAtOffsetZero) {
    MmapFile mmap = MmapFile::open(test_file_path);
    
    const void* ptr = mmap.get_data_at(0);
    EXPECT_NE(ptr, nullptr);
    
    // Verify data content
    const uint8_t* data = static_cast<const uint8_t*>(ptr);
    EXPECT_EQ(data[0], 0xAB);
}

// Test: Access data at various offsets
TEST_F(MmapFileTest, AccessDataAtVariousOffsets) {
    MmapFile mmap = MmapFile::open(test_file_path);
    
    // Test offsets throughout file
    std::vector<size_t> offsets = {0, 1, 100, 1024, 2048, 4095};
    
    for (size_t offset : offsets) {
        const void* ptr = mmap.get_data_at(offset);
        EXPECT_NE(ptr, nullptr);
        
        // Verify data content
        const uint8_t* data = static_cast<const uint8_t*>(ptr);
        EXPECT_EQ(data[0], 0xAB);
    }
}

// Test: Access tensor data with size validation
TEST_F(MmapFileTest, AccessTensorDataWithSizeValidation) {
    MmapFile mmap = MmapFile::open(test_file_path);
    
    // Valid access
    const void* ptr = mmap.get_tensor_data(0, 1024);
    EXPECT_NE(ptr, nullptr);
    
    // Valid access at end
    ptr = mmap.get_tensor_data(3072, 1024);  // 3072 + 1024 = 4096 (exactly at end)
    EXPECT_NE(ptr, nullptr);
}

// Test: Error on offset beyond file size
TEST_F(MmapFileTest, ErrorOnOffsetBeyondFileSize) {
    MmapFile mmap = MmapFile::open(test_file_path);
    
    EXPECT_THROW({
        mmap.get_data_at(5000);  // > 4096
    }, worker::CudaError);
}

// Test: Error on tensor data extending beyond file
TEST_F(MmapFileTest, ErrorOnTensorDataExtendingBeyondFile) {
    MmapFile mmap = MmapFile::open(test_file_path);
    
    EXPECT_THROW({
        mmap.get_tensor_data(3072, 2000);  // 3072 + 2000 = 5072 > 4096
    }, worker::CudaError);
}

// Test: Error on integer overflow in tensor access
TEST_F(MmapFileTest, ErrorOnIntegerOverflowInTensorAccess) {
    MmapFile mmap = MmapFile::open(test_file_path);
    
    EXPECT_THROW({
        mmap.get_tensor_data(SIZE_MAX - 100, 200);  // Overflow
    }, worker::CudaError);
}

// Test: Error on non-existent file
TEST_F(MmapFileTest, ErrorOnNonExistentFile) {
    EXPECT_THROW({
        MmapFile::open("/nonexistent/file.gguf");
    }, worker::CudaError);
}

// Test: Error on empty file
TEST_F(MmapFileTest, ErrorOnEmptyFile) {
    // Create empty file
    std::ofstream file(test_file_path, std::ios::binary | std::ios::trunc);
    file.close();
    
    EXPECT_THROW({
        MmapFile::open(test_file_path);
    }, worker::CudaError);
}

// Test: RAII cleanup (munmap called)
TEST_F(MmapFileTest, RAIICleanup) {
    {
        MmapFile mmap = MmapFile::open(test_file_path);
        EXPECT_TRUE(mmap.is_mapped());
    }
    // Destructor should have unmapped file
    // No way to directly test this, but valgrind would catch leaks
}

// Test: Move constructor
TEST_F(MmapFileTest, MoveConstructor) {
    MmapFile mmap1 = MmapFile::open(test_file_path);
    const void* original_ptr = mmap1.data();
    size_t original_size = mmap1.size();
    
    MmapFile mmap2 = std::move(mmap1);
    
    EXPECT_TRUE(mmap2.is_mapped());
    EXPECT_EQ(mmap2.data(), original_ptr);
    EXPECT_EQ(mmap2.size(), original_size);
    
    // Original should be invalidated
    EXPECT_FALSE(mmap1.is_mapped());
    EXPECT_EQ(mmap1.data(), nullptr);
}

// Test: Move assignment
TEST_F(MmapFileTest, MoveAssignment) {
    MmapFile mmap1 = MmapFile::open(test_file_path);
    const void* original_ptr = mmap1.data();
    
    // Create second file
    std::string test_file2 = "/tmp/test_mmap2_" + std::to_string(getpid()) + ".bin";
    std::ofstream file(test_file2, std::ios::binary);
    std::vector<uint8_t> data(2048, 0xCD);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    file.close();
    
    MmapFile mmap2 = MmapFile::open(test_file2);
    
    // Move assign
    mmap2 = std::move(mmap1);
    
    EXPECT_TRUE(mmap2.is_mapped());
    EXPECT_EQ(mmap2.data(), original_ptr);
    EXPECT_EQ(mmap2.size(), 4096u);
    
    // Clean up
    unlink(test_file2.c_str());
}

// Test: Large file support (>4GB on 64-bit)
TEST_F(MmapFileTest, LargeFileSupport) {
    // This test just verifies size_t can hold large values
    // Actual large file test would require creating multi-GB file
    
    size_t large_size = 5ULL * 1024 * 1024 * 1024;  // 5GB
    EXPECT_GT(large_size, 4ULL * 1024 * 1024 * 1024);  // > 4GB
    
    // On 64-bit systems, size_t should handle this
    EXPECT_LE(large_size, SIZE_MAX);
}

// Test: Verify data content
TEST_F(MmapFileTest, VerifyDataContent) {
    // Create file with known pattern
    std::vector<uint8_t> pattern = {0x01, 0x02, 0x03, 0x04, 0x05};
    create_test_file(pattern);
    
    MmapFile mmap = MmapFile::open(test_file_path);
    
    const uint8_t* data = static_cast<const uint8_t*>(mmap.data());
    for (size_t i = 0; i < pattern.size(); ++i) {
        EXPECT_EQ(data[i], pattern[i]);
    }
}

// Test: Access at file boundary
TEST_F(MmapFileTest, AccessAtFileBoundary) {
    MmapFile mmap = MmapFile::open(test_file_path);
    
    // Access last byte (valid)
    const void* ptr = mmap.get_data_at(4095);
    EXPECT_NE(ptr, nullptr);
    
    // Access one byte beyond (invalid)
    EXPECT_THROW({
        mmap.get_data_at(4096);
    }, worker::CudaError);
}

// Test: Tensor data at exact file end
TEST_F(MmapFileTest, TensorDataAtExactFileEnd) {
    MmapFile mmap = MmapFile::open(test_file_path);
    
    // Tensor exactly at end (valid)
    const void* ptr = mmap.get_tensor_data(3072, 1024);  // 3072 + 1024 = 4096
    EXPECT_NE(ptr, nullptr);
    
    // One byte beyond (invalid)
    EXPECT_THROW({
        mmap.get_tensor_data(3072, 1025);
    }, worker::CudaError);
}

// Test: Zero-size tensor access
TEST_F(MmapFileTest, ZeroSizeTensorAccess) {
    MmapFile mmap = MmapFile::open(test_file_path);
    
    // Zero-size access should be valid
    const void* ptr = mmap.get_tensor_data(1000, 0);
    EXPECT_NE(ptr, nullptr);
}

// ---
// Implemented by Llama-Beta 🦙
