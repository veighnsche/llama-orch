/**
 * GGUF Header Parser Tests
 * 
 * Comprehensive unit tests for GGUF header parsing with security validation.
 * 
 * Tests cover:
 * - Valid header parsing
 * - Magic bytes validation
 * - Version validation
 * - Tensor count validation
 * - Bounds validation (security-critical)
 * - Integer overflow detection
 * - Malformed file handling
 * 
 * Spec: M0-W-1211, M0-W-1211a (security)
 */

#include <gtest/gtest.h>
#include "gguf/header_parser.h"
#include "cuda_error.h"
#include <vector>
#include <cstring>

using namespace worker::gguf;

// Helper: Create minimal valid GGUF header
static std::vector<uint8_t> create_minimal_gguf() {
    std::vector<uint8_t> data;
    
    // Magic bytes: "GGUF" (0x47475546)
    uint32_t magic = 0x47475546;
    data.insert(data.end(),
                reinterpret_cast<uint8_t*>(&magic),
                reinterpret_cast<uint8_t*>(&magic) + 4);
    
    // Version: 3
    uint32_t version = 3;
    data.insert(data.end(),
                reinterpret_cast<uint8_t*>(&version),
                reinterpret_cast<uint8_t*>(&version) + 4);
    
    // Tensor count: 0
    uint64_t tensor_count = 0;
    data.insert(data.end(),
                reinterpret_cast<uint8_t*>(&tensor_count),
                reinterpret_cast<uint8_t*>(&tensor_count) + 8);
    
    // Metadata KV count: 0
    uint64_t metadata_count = 0;
    data.insert(data.end(),
                reinterpret_cast<uint8_t*>(&metadata_count),
                reinterpret_cast<uint8_t*>(&metadata_count) + 8);
    
    return data;
}

// Helper: Create GGUF with one tensor
static std::vector<uint8_t> create_gguf_with_tensor(
    const std::string& name,
    const std::vector<uint64_t>& dims,
    uint32_t type,
    uint64_t offset
) {
    std::vector<uint8_t> data;
    
    // Magic bytes: "GGUF" (0x47475546)
    uint32_t magic = 0x47475546;
    data.insert(data.end(),
                reinterpret_cast<uint8_t*>(&magic),
                reinterpret_cast<uint8_t*>(&magic) + 4);
    
    // Version: 3
    uint32_t version = 3;
    data.insert(data.end(),
                reinterpret_cast<uint8_t*>(&version),
                reinterpret_cast<uint8_t*>(&version) + 4);
    
    // Tensor count: 1
    uint64_t tensor_count = 1;
    data.insert(data.end(),
                reinterpret_cast<uint8_t*>(&tensor_count),
                reinterpret_cast<uint8_t*>(&tensor_count) + 8);
    
    // Metadata KV count: 0
    uint64_t metadata_count = 0;
    data.insert(data.end(),
                reinterpret_cast<uint8_t*>(&metadata_count),
                reinterpret_cast<uint8_t*>(&metadata_count) + 8);
    
    // Tensor name (length + string)
    uint64_t name_len = name.size();
    data.insert(data.end(),
                reinterpret_cast<uint8_t*>(&name_len),
                reinterpret_cast<uint8_t*>(&name_len) + 8);
    data.insert(data.end(), name.begin(), name.end());
    
    // Number of dimensions
    uint32_t n_dims = dims.size();
    data.insert(data.end(),
                reinterpret_cast<uint8_t*>(&n_dims),
                reinterpret_cast<uint8_t*>(&n_dims) + 4);
    
    // Dimensions
    for (uint64_t dim : dims) {
        data.insert(data.end(),
                    reinterpret_cast<const uint8_t*>(&dim),
                    reinterpret_cast<const uint8_t*>(&dim) + 8);
    }
    
    // Tensor type
    data.insert(data.end(),
                reinterpret_cast<uint8_t*>(&type),
                reinterpret_cast<uint8_t*>(&type) + 4);
    
    // Tensor offset
    data.insert(data.end(),
                reinterpret_cast<uint8_t*>(&offset),
                reinterpret_cast<uint8_t*>(&offset) + 8);
    
    return data;
}

class GGUFHeaderParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Tests run without GPU
    }
};

// Test: Parse minimal valid GGUF
TEST_F(GGUFHeaderParserTest, ParseMinimalValidGGUF) {
    auto data = create_minimal_gguf();
    
    GGUFHeader header = parse_gguf_header(data.data(), data.size());
    
    EXPECT_EQ(header.magic, 0x47475546u);
    EXPECT_EQ(header.version, 3u);
    EXPECT_EQ(header.tensor_count, 0u);
    EXPECT_EQ(header.metadata_kv_count, 0u);
    EXPECT_EQ(header.tensors.size(), 0u);
    EXPECT_EQ(header.metadata.size(), 0u);
}

// Test: Reject invalid magic bytes
TEST_F(GGUFHeaderParserTest, RejectInvalidMagicBytes) {
    auto data = create_minimal_gguf();
    
    // Corrupt magic bytes
    data[0] = 0xFF;
    
    EXPECT_THROW({
        parse_gguf_header(data.data(), data.size());
    }, worker::CudaError);
}

// Test: Reject unsupported version
TEST_F(GGUFHeaderParserTest, RejectUnsupportedVersion) {
    auto data = create_minimal_gguf();
    
    // Set version to 2 (unsupported)
    uint32_t version = 2;
    std::memcpy(&data[4], &version, 4);
    
    EXPECT_THROW({
        parse_gguf_header(data.data(), data.size());
    }, worker::CudaError);
}

// Test: Reject excessive tensor count
TEST_F(GGUFHeaderParserTest, RejectExcessiveTensorCount) {
    auto data = create_minimal_gguf();
    
    // Set tensor count to exceed maximum
    uint64_t tensor_count = 20000;  // > MAX_TENSOR_COUNT (10000)
    std::memcpy(&data[12], &tensor_count, 8);
    
    EXPECT_THROW({
        parse_gguf_header(data.data(), data.size());
    }, worker::CudaError);
}

// Test: Reject file too small
TEST_F(GGUFHeaderParserTest, RejectFileTooSmall) {
    std::vector<uint8_t> data = {0x47, 0x47, 0x55, 0x46};  // Just magic bytes
    
    EXPECT_THROW({
        parse_gguf_header(data.data(), data.size());
    }, worker::CudaError);
}

// Test: Reject NULL pointer
TEST_F(GGUFHeaderParserTest, RejectNullPointer) {
    EXPECT_THROW({
        parse_gguf_header(nullptr, 1024);
    }, worker::CudaError);
}

// Test: Parse tensor with valid bounds
TEST_F(GGUFHeaderParserTest, ParseTensorWithValidBounds) {
    // Create GGUF with one tensor: [128, 256] F32 at offset 0
    auto data = create_gguf_with_tensor("test.weight", {128, 256}, 0, 0);
    
    // Add padding to align data section
    while (data.size() % 32 != 0) {
        data.push_back(0);
    }
    
    // Add tensor data (128 * 256 * 4 bytes = 131072 bytes)
    size_t tensor_size = 128 * 256 * 4;
    data.resize(data.size() + tensor_size, 0);
    
    GGUFHeader header = parse_gguf_header(data.data(), data.size());
    
    EXPECT_EQ(header.tensor_count, 1u);
    EXPECT_EQ(header.tensors.size(), 1u);
    EXPECT_EQ(header.tensors[0].name, "test.weight");
    EXPECT_EQ(header.tensors[0].dimensions.size(), 2u);
    EXPECT_EQ(header.tensors[0].dimensions[0], 128u);
    EXPECT_EQ(header.tensors[0].dimensions[1], 256u);
    EXPECT_EQ(header.tensors[0].type, 0u);  // F32
    EXPECT_EQ(header.tensors[0].size, tensor_size);
}

// Security Test: Reject tensor offset beyond file
TEST_F(GGUFHeaderParserTest, RejectTensorOffsetBeyondFile) {
    // Create GGUF with tensor at invalid offset
    auto data = create_gguf_with_tensor("bad.weight", {10, 10}, 0, 999999);
    
    EXPECT_THROW({
        parse_gguf_header(data.data(), data.size());
    }, worker::CudaError);
}

// Security Test: Reject tensor extending beyond file
TEST_F(GGUFHeaderParserTest, RejectTensorExtendingBeyondFile) {
    // Create GGUF with large tensor that extends beyond file
    auto data = create_gguf_with_tensor("huge.weight", {1000, 1000}, 0, 0);
    
    // File is too small for this tensor (1000*1000*4 = 4MB)
    EXPECT_THROW({
        parse_gguf_header(data.data(), data.size());
    }, worker::CudaError);
}

// Security Test: Detect integer overflow in tensor size
TEST_F(GGUFHeaderParserTest, DetectIntegerOverflowInTensorSize) {
    // Create tensor with dimensions that overflow
    std::vector<uint64_t> huge_dims = {
        UINT64_MAX / 2,
        UINT64_MAX / 2
    };
    
    EXPECT_THROW({
        calculate_tensor_size(huge_dims, 0);  // F32
    }, worker::CudaError);
}

// Security Test: Validate tensor bounds function
TEST_F(GGUFHeaderParserTest, ValidateTensorBoundsFunction) {
    GGUFTensor tensor;
    tensor.name = "test";
    tensor.offset = 1000;
    tensor.size = 500;
    
    size_t file_size = 2000;
    size_t data_start = 100;
    
    // Valid tensor
    auto result = validate_tensor_bounds(tensor, file_size, data_start);
    EXPECT_TRUE(result.valid);
    
    // Offset before data start
    tensor.offset = 50;
    result = validate_tensor_bounds(tensor, file_size, data_start);
    EXPECT_FALSE(result.valid);
    
    // Offset beyond file
    tensor.offset = 3000;
    result = validate_tensor_bounds(tensor, file_size, data_start);
    EXPECT_FALSE(result.valid);
    
    // Tensor extends beyond file
    tensor.offset = 1800;
    tensor.size = 500;  // 1800 + 500 = 2300 > 2000
    result = validate_tensor_bounds(tensor, file_size, data_start);
    EXPECT_FALSE(result.valid);
}

// Security Test: Integer overflow in offset + size
TEST_F(GGUFHeaderParserTest, DetectOffsetPlusSizeOverflow) {
    GGUFTensor tensor;
    tensor.name = "overflow";
    tensor.offset = SIZE_MAX - 100;
    tensor.size = 200;  // offset + size wraps around
    
    size_t file_size = SIZE_MAX;
    size_t data_start = 0;
    
    auto result = validate_tensor_bounds(tensor, file_size, data_start);
    EXPECT_FALSE(result.valid);
    EXPECT_NE(result.error_message.find("overflow"), std::string::npos);
}

// Test: Calculate tensor size for various types
TEST_F(GGUFHeaderParserTest, CalculateTensorSizeForTypes) {
    std::vector<uint64_t> dims = {10, 20};  // 200 elements
    
    // F32 (type 0): 4 bytes per element
    EXPECT_EQ(calculate_tensor_size(dims, 0), 200 * 4);
    
    // F16 (type 1): 2 bytes per element
    EXPECT_EQ(calculate_tensor_size(dims, 1), 200 * 2);
    
    // I8 (type 16): 1 byte per element
    EXPECT_EQ(calculate_tensor_size(dims, 16), 200 * 1);
    
    // F64 (type 20): 8 bytes per element
    EXPECT_EQ(calculate_tensor_size(dims, 20), 200 * 8);
}

// Test: Get type size for GGML types
TEST_F(GGUFHeaderParserTest, GetTypeSizeForGGMLTypes) {
    EXPECT_EQ(get_type_size(0), 4u);   // F32
    EXPECT_EQ(get_type_size(1), 2u);   // F16
    EXPECT_EQ(get_type_size(16), 1u);  // I8
    EXPECT_EQ(get_type_size(20), 8u);  // F64
    EXPECT_EQ(get_type_size(30), 4u);  // MXFP4
}

// Test: Empty dimensions
TEST_F(GGUFHeaderParserTest, HandleEmptyDimensions) {
    std::vector<uint64_t> empty_dims;
    EXPECT_EQ(calculate_tensor_size(empty_dims, 0), 0u);
}

// Security Test: Fuzzing with random data
TEST_F(GGUFHeaderParserTest, FuzzingWithRandomData) {
    // Test with various sizes of random data
    std::vector<size_t> sizes = {0, 1, 10, 100, 1000};
    
    for (size_t size : sizes) {
        std::vector<uint8_t> random_data(size, 0xAB);
        
        // Should not crash, but will throw on invalid data
        try {
            parse_gguf_header(random_data.data(), random_data.size());
        } catch (const worker::CudaError&) {
            // Expected for random data
        }
    }
}

// Security Test: Boundary conditions
TEST_F(GGUFHeaderParserTest, BoundaryConditions) {
    GGUFTensor tensor;
    tensor.name = "boundary";
    
    size_t file_size = 1000;
    size_t data_start = 100;
    
    // Tensor exactly at end of file (valid)
    tensor.offset = 900;
    tensor.size = 100;
    auto result = validate_tensor_bounds(tensor, file_size, data_start);
    EXPECT_TRUE(result.valid);
    
    // Tensor one byte beyond (invalid)
    tensor.offset = 900;
    tensor.size = 101;
    result = validate_tensor_bounds(tensor, file_size, data_start);
    EXPECT_FALSE(result.valid);
    
    // Zero-size tensor (valid)
    tensor.offset = 500;
    tensor.size = 0;
    result = validate_tensor_bounds(tensor, file_size, data_start);
    EXPECT_TRUE(result.valid);
}

// ---
// Implemented by Llama-Beta 🦙
