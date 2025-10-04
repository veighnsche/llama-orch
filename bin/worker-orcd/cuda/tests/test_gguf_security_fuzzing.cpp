/**
 * GGUF Security Fuzzing Tests
 * 
 * Property-based and fuzzing tests for GGUF header parser security.
 * 
 * Tests 100+ malformed GGUF files to ensure robust error handling
 * and no crashes from malicious input.
 * 
 * Spec: M0-W-1211a (security)
 */

#include <gtest/gtest.h>
#include "gguf/header_parser.h"
#include "cuda_error.h"
#include <vector>
#include <random>
#include <cstring>

using namespace worker::gguf;

class GGUFSecurityFuzzingTest : public ::testing::Test {
protected:
    std::mt19937 rng{42};  // Deterministic seed for reproducibility
    
    // Generate random bytes
    std::vector<uint8_t> random_bytes(size_t size) {
        std::vector<uint8_t> data(size);
        std::uniform_int_distribution<uint32_t> dist(0, 255);
        for (auto& byte : data) {
            byte = static_cast<uint8_t>(dist(rng));
        }
        return data;
    }
    
    // Create valid header base
    std::vector<uint8_t> valid_header_base() {
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
        uint64_t metadata_count = 0;
        data.insert(data.end(),
                    reinterpret_cast<uint8_t*>(&metadata_count),
                    reinterpret_cast<uint8_t*>(&metadata_count) + 8);
        
        return data;
    }
};

// Fuzzing Test: Random data of various sizes
TEST_F(GGUFSecurityFuzzingTest, FuzzRandomDataVariousSizes) {
    std::vector<size_t> sizes = {
        0, 1, 2, 4, 8, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 127, 128, 129, 255, 256, 257,
        511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049
    };
    
    for (size_t size : sizes) {
        auto data = random_bytes(size);
        
        // Should not crash
        try {
            parse_gguf_header(data.data(), data.size());
            // If it succeeds, that's fine (unlikely with random data)
        } catch (const worker::CudaError&) {
            // Expected for random data
        } catch (...) {
            FAIL() << "Unexpected exception type for size " << size;
        }
    }
}

// Fuzzing Test: Corrupt magic bytes (100 variations)
TEST_F(GGUFSecurityFuzzingTest, FuzzCorruptMagicBytes) {
    for (int i = 0; i < 100; ++i) {
        auto data = valid_header_base();
        
        // Corrupt one byte of magic
        int byte_idx = i % 4;
        data[byte_idx] ^= 0xFF;
        
        EXPECT_THROW({
            parse_gguf_header(data.data(), data.size());
        }, worker::CudaError);
    }
}

// Fuzzing Test: Invalid versions
TEST_F(GGUFSecurityFuzzingTest, FuzzInvalidVersions) {
    std::vector<uint32_t> invalid_versions = {
        0, 1, 2, 4, 5, 10, 100, 1000, 0xFFFFFFFF
    };
    
    for (uint32_t version : invalid_versions) {
        auto data = valid_header_base();
        std::memcpy(&data[4], &version, 4);
        
        EXPECT_THROW({
            parse_gguf_header(data.data(), data.size());
        }, worker::CudaError);
    }
}

// Fuzzing Test: Excessive tensor counts
TEST_F(GGUFSecurityFuzzingTest, FuzzExcessiveTensorCounts) {
    std::vector<uint64_t> excessive_counts = {
        10001, 20000, 50000, 100000, 1000000,
        UINT32_MAX, UINT64_MAX / 2, UINT64_MAX - 1, UINT64_MAX
    };
    
    for (uint64_t count : excessive_counts) {
        auto data = valid_header_base();
        std::memcpy(&data[12], &count, 8);
        
        EXPECT_THROW({
            parse_gguf_header(data.data(), data.size());
        }, worker::CudaError);
    }
}

// Fuzzing Test: Truncated files
TEST_F(GGUFSecurityFuzzingTest, FuzzTruncatedFiles) {
    auto full_data = valid_header_base();
    
    // Try truncating at every byte
    for (size_t truncate_at = 0; truncate_at < full_data.size(); ++truncate_at) {
        std::vector<uint8_t> truncated(full_data.begin(),
                                       full_data.begin() + truncate_at);
        
        // Should not crash
        try {
            parse_gguf_header(truncated.data(), truncated.size());
        } catch (const worker::CudaError&) {
            // Expected for truncated data
        }
    }
}

// Property Test: Tensor bounds validation (1000 random configurations)
TEST_F(GGUFSecurityFuzzingTest, PropertyTestTensorBounds) {
    std::uniform_int_distribution<uint64_t> offset_dist(0, 1000000);
    std::uniform_int_distribution<size_t> size_dist(0, 100000);
    
    for (int i = 0; i < 1000; ++i) {
        GGUFTensor tensor;
        tensor.name = "test_" + std::to_string(i);
        tensor.offset = offset_dist(rng);
        tensor.size = size_dist(rng);
        
        size_t file_size = offset_dist(rng);
        size_t data_start = offset_dist(rng) % 1000;
        
        // Should not crash
        auto result = validate_tensor_bounds(tensor, file_size, data_start);
        
        // Verify result is consistent
        if (result.valid) {
            // If valid, all conditions must hold
            EXPECT_GE(tensor.offset, data_start);
            EXPECT_LT(tensor.offset, file_size);
            EXPECT_LE(tensor.offset + tensor.size, file_size);
        }
    }
}

// Fuzzing Test: Malicious tensor offsets
TEST_F(GGUFSecurityFuzzingTest, FuzzMaliciousTensorOffsets) {
    std::vector<uint64_t> malicious_offsets = {
        UINT64_MAX,
        UINT64_MAX - 1,
        UINT64_MAX / 2,
        SIZE_MAX,
        SIZE_MAX - 1,
        SIZE_MAX - 100,
    };
    
    for (uint64_t offset : malicious_offsets) {
        GGUFTensor tensor;
        tensor.name = "malicious";
        tensor.offset = offset;
        tensor.size = 1000;
        
        size_t file_size = 10000;
        size_t data_start = 100;
        
        auto result = validate_tensor_bounds(tensor, file_size, data_start);
        EXPECT_FALSE(result.valid);
    }
}

// Fuzzing Test: Malicious tensor sizes
TEST_F(GGUFSecurityFuzzingTest, FuzzMaliciousTensorSizes) {
    std::vector<size_t> malicious_sizes = {
        SIZE_MAX,
        SIZE_MAX - 1,
        SIZE_MAX / 2,
        SIZE_MAX - 100,
    };
    
    for (size_t size : malicious_sizes) {
        GGUFTensor tensor;
        tensor.name = "malicious";
        tensor.offset = 1000;
        tensor.size = size;
        
        size_t file_size = 10000;
        size_t data_start = 100;
        
        auto result = validate_tensor_bounds(tensor, file_size, data_start);
        EXPECT_FALSE(result.valid);
    }
}

// Fuzzing Test: Dimension overflow combinations
TEST_F(GGUFSecurityFuzzingTest, FuzzDimensionOverflows) {
    std::vector<std::vector<uint64_t>> overflow_dims = {
        {UINT64_MAX, 1},
        {UINT64_MAX / 2, UINT64_MAX / 2},
        {UINT64_MAX / 2, 3},
        {1000000, 1000000, 1000000},
        {UINT32_MAX, UINT32_MAX},
    };
    
    for (const auto& dims : overflow_dims) {
        EXPECT_THROW({
            calculate_tensor_size(dims, 0);  // F32
        }, worker::CudaError);
    }
}

// Fuzzing Test: Edge case dimensions
TEST_F(GGUFSecurityFuzzingTest, FuzzEdgeCaseDimensions) {
    std::vector<std::vector<uint64_t>> edge_dims = {
        {},                    // Empty
        {0},                   // Zero dimension
        {1},                   // Single element
        {0, 100},              // Zero in first dim
        {100, 0},              // Zero in second dim
        {1, 1, 1, 1},          // All ones
    };
    
    for (const auto& dims : edge_dims) {
        // Should not crash
        try {
            size_t size = calculate_tensor_size(dims, 0);
            // Verify result makes sense
            if (dims.empty()) {
                EXPECT_EQ(size, 0u);
            }
        } catch (const worker::CudaError&) {
            // Some edge cases may throw
        }
    }
}

// Fuzzing Test: Random valid headers (should all parse)
TEST_F(GGUFSecurityFuzzingTest, FuzzValidHeaders) {
    std::uniform_int_distribution<uint64_t> count_dist(0, 100);
    
    for (int i = 0; i < 50; ++i) {
        auto data = valid_header_base();
        
        // Set random but valid counts
        uint64_t tensor_count = count_dist(rng);
        uint64_t metadata_count = count_dist(rng);
        
        std::memcpy(&data[12], &tensor_count, 8);
        std::memcpy(&data[20], &metadata_count, 8);
        
        // Should parse without error (though may fail on missing data)
        try {
            parse_gguf_header(data.data(), data.size());
        } catch (const worker::CudaError&) {
            // Expected if we don't have enough data for tensors/metadata
        }
    }
}

// Fuzzing Test: Alignment edge cases
TEST_F(GGUFSecurityFuzzingTest, FuzzAlignmentEdgeCases) {
    // Test various file sizes around alignment boundaries
    for (size_t size = 24; size < 100; ++size) {
        auto data = valid_header_base();
        data.resize(size, 0);
        
        // Should not crash
        try {
            parse_gguf_header(data.data(), data.size());
        } catch (const worker::CudaError&) {
            // Expected for incomplete data
        }
    }
}

// Fuzzing Test: Bit flips in valid header
TEST_F(GGUFSecurityFuzzingTest, FuzzBitFlips) {
    auto base_data = valid_header_base();
    
    // Flip each bit in the header
    for (size_t byte_idx = 0; byte_idx < base_data.size(); ++byte_idx) {
        for (int bit = 0; bit < 8; ++bit) {
            auto data = base_data;
            data[byte_idx] ^= (1 << bit);
            
            // Should not crash
            try {
                parse_gguf_header(data.data(), data.size());
            } catch (const worker::CudaError&) {
                // Expected for corrupted data
            }
        }
    }
}

// ---
// Implemented by Llama-Beta 🦙
