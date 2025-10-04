/**
 * Pre-Load Validation Tests
 * 
 * Unit tests for GGUF pre-load validation.
 * 
 * Tests cover:
 * - File access validation
 * - Header validation
 * - Metadata validation
 * - VRAM requirement calculation
 * - Security validation
 * 
 * Spec: M0-W-1210
 */

#include <gtest/gtest.h>
#include "validation/pre_load.h"
#include "cuda_error.h"
#include <fstream>
#include <vector>
#include <unistd.h>

using namespace worker::validation;
using namespace worker::gguf;

class PreLoadValidationTest : public ::testing::Test {
protected:
    std::string test_file_path;
    
    void SetUp() override {
        test_file_path = "/tmp/test_validation_" + std::to_string(getpid()) + ".gguf";
    }
    
    void TearDown() override {
        unlink(test_file_path.c_str());
    }
    
    // Helper: Create minimal valid GGUF file
    void create_minimal_gguf() {
        std::vector<uint8_t> data;
        
        // Magic
        uint32_t magic = 0x47475546;
        data.insert(data.end(),
                    reinterpret_cast<uint8_t*>(&magic),
                    reinterpret_cast<uint8_t*>(&magic) + 4);
        
        // Version
        uint32_t version = 3;
        data.insert(data.end(),
                    reinterpret_cast<uint8_t*>(&version),
                    reinterpret_cast<uint8_t*>(&version) + 4);
        
        // Tensor count
        uint64_t tensor_count = 0;
        data.insert(data.end(),
                    reinterpret_cast<uint8_t*>(&tensor_count),
                    reinterpret_cast<uint8_t*>(&tensor_count) + 8);
        
        // Metadata count
        uint64_t metadata_count = 1;
        data.insert(data.end(),
                    reinterpret_cast<uint8_t*>(&metadata_count),
                    reinterpret_cast<uint8_t*>(&metadata_count) + 8);
        
        // Metadata: general.architecture = "llama"
        // Key: "general.architecture"
        uint64_t key_len = 21;
        data.insert(data.end(),
                    reinterpret_cast<uint8_t*>(&key_len),
                    reinterpret_cast<uint8_t*>(&key_len) + 8);
        std::string key = "general.architecture";
        data.insert(data.end(), key.begin(), key.end());
        
        // Value type: STRING (8)
        uint32_t value_type = 8;
        data.insert(data.end(),
                    reinterpret_cast<uint8_t*>(&value_type),
                    reinterpret_cast<uint8_t*>(&value_type) + 4);
        
        // Value: "llama"
        uint64_t value_len = 5;
        data.insert(data.end(),
                    reinterpret_cast<uint8_t*>(&value_len),
                    reinterpret_cast<uint8_t*>(&value_len) + 8);
        std::string value = "llama";
        data.insert(data.end(), value.begin(), value.end());
        
        // Write to file
        std::ofstream file(test_file_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();
    }
};

// Test: Validate file access - file exists
TEST_F(PreLoadValidationTest, ValidateFileAccessExists) {
    create_minimal_gguf();
    
    bool result = PreLoadValidator::validate_file_access(test_file_path);
    EXPECT_TRUE(result);
}

// Test: Validate file access - file not found
TEST_F(PreLoadValidationTest, ValidateFileAccessNotFound) {
    bool result = PreLoadValidator::validate_file_access("/nonexistent/file.gguf");
    EXPECT_FALSE(result);
}

// Test: Validate header - valid
TEST_F(PreLoadValidationTest, ValidateHeaderValid) {
    GGUFHeader header;
    header.magic = GGUF_MAGIC;
    header.version = GGUF_VERSION;
    header.tensor_count = 100;
    header.metadata_kv_count = 10;
    
    bool result = PreLoadValidator::validate_header(header);
    EXPECT_TRUE(result);
}

// Test: Validate header - invalid magic
TEST_F(PreLoadValidationTest, ValidateHeaderInvalidMagic) {
    GGUFHeader header;
    header.magic = 0xDEADBEEF;
    header.version = GGUF_VERSION;
    header.tensor_count = 100;
    
    bool result = PreLoadValidator::validate_header(header);
    EXPECT_FALSE(result);
}

// Test: Validate header - invalid version
TEST_F(PreLoadValidationTest, ValidateHeaderInvalidVersion) {
    GGUFHeader header;
    header.magic = GGUF_MAGIC;
    header.version = 2;  // Unsupported
    header.tensor_count = 100;
    
    bool result = PreLoadValidator::validate_header(header);
    EXPECT_FALSE(result);
}

// Test: Validate header - excessive tensor count
TEST_F(PreLoadValidationTest, ValidateHeaderExcessiveTensorCount) {
    GGUFHeader header;
    header.magic = GGUF_MAGIC;
    header.version = GGUF_VERSION;
    header.tensor_count = 20000;  // > MAX_TENSOR_COUNT
    
    bool result = PreLoadValidator::validate_header(header);
    EXPECT_FALSE(result);
}

// Test: Calculate VRAM requirement
TEST_F(PreLoadValidationTest, CalculateVRAMRequirement) {
    std::vector<GGUFTensor> tensors;
    
    // Add 3 tensors
    GGUFTensor t1;
    t1.size = 100 * 1024 * 1024;  // 100MB
    tensors.push_back(t1);
    
    GGUFTensor t2;
    t2.size = 200 * 1024 * 1024;  // 200MB
    tensors.push_back(t2);
    
    GGUFTensor t3;
    t3.size = 50 * 1024 * 1024;  // 50MB
    tensors.push_back(t3);
    
    // Total: 350MB + 10% overhead = 385MB
    size_t required = PreLoadValidator::calculate_vram_requirement(tensors);
    
    EXPECT_EQ(required, 385 * 1024 * 1024);
}

// Test: Validate VRAM availability - sufficient
TEST_F(PreLoadValidationTest, ValidateVRAMAvailabilitySufficient) {
    size_t required = 1 * 1024 * 1024 * 1024;  // 1GB
    size_t available = 2 * 1024 * 1024 * 1024;  // 2GB
    
    bool result = PreLoadValidator::validate_vram_availability(required, available);
    EXPECT_TRUE(result);
}

// Test: Validate VRAM availability - insufficient
TEST_F(PreLoadValidationTest, ValidateVRAMAvailabilityInsufficient) {
    size_t required = 2 * 1024 * 1024 * 1024;  // 2GB
    size_t available = 1 * 1024 * 1024 * 1024;  // 1GB
    
    bool result = PreLoadValidator::validate_vram_availability(required, available);
    EXPECT_FALSE(result);
}

// Test: Validate VRAM availability - exact match
TEST_F(PreLoadValidationTest, ValidateVRAMAvailabilityExactMatch) {
    size_t required = 1 * 1024 * 1024 * 1024;  // 1GB
    size_t available = 1 * 1024 * 1024 * 1024;  // 1GB
    
    bool result = PreLoadValidator::validate_vram_availability(required, available);
    EXPECT_TRUE(result);
}

// Test: Validate tensor bounds - all valid
TEST_F(PreLoadValidationTest, ValidateTensorBoundsAllValid) {
    std::vector<GGUFTensor> tensors;
    
    GGUFTensor t1;
    t1.name = "tensor1";
    t1.offset = 0;
    t1.size = 1000;
    tensors.push_back(t1);
    
    GGUFTensor t2;
    t2.name = "tensor2";
    t2.offset = 1000;
    t2.size = 500;
    tensors.push_back(t2);
    
    size_t file_size = 2000;
    size_t data_start = 0;
    
    bool result = PreLoadValidator::validate_tensor_bounds(
        tensors, file_size, data_start
    );
    EXPECT_TRUE(result);
}

// Test: Validate tensor bounds - one invalid
TEST_F(PreLoadValidationTest, ValidateTensorBoundsOneInvalid) {
    std::vector<GGUFTensor> tensors;
    
    GGUFTensor t1;
    t1.name = "tensor1";
    t1.offset = 0;
    t1.size = 1000;
    tensors.push_back(t1);
    
    GGUFTensor t2;
    t2.name = "bad_tensor";
    t2.offset = 1800;
    t2.size = 500;  // Extends beyond file (1800 + 500 = 2300 > 2000)
    tensors.push_back(t2);
    
    size_t file_size = 2000;
    size_t data_start = 0;
    
    bool result = PreLoadValidator::validate_tensor_bounds(
        tensors, file_size, data_start
    );
    EXPECT_FALSE(result);
}

// Test: Audit log rejection (smoke test)
TEST_F(PreLoadValidationTest, AuditLogRejection) {
    // Should not crash
    EXPECT_NO_THROW({
        PreLoadValidator::audit_log_rejection(
            "Test rejection",
            "/test/path.gguf"
        );
    });
}

// Test: VRAM calculation overflow detection
TEST_F(PreLoadValidationTest, VRAMCalculationOverflowDetection) {
    std::vector<GGUFTensor> tensors;
    
    GGUFTensor t1;
    t1.size = SIZE_MAX / 2;
    tensors.push_back(t1);
    
    GGUFTensor t2;
    t2.size = SIZE_MAX / 2;
    tensors.push_back(t2);
    
    // Should detect overflow
    EXPECT_THROW({
        PreLoadValidator::calculate_vram_requirement(tensors);
    }, worker::CudaError);
}

// ---
// Implemented by Llama-Beta 🦙
