/**
 * GGUF Header Parser
 * 
 * Parses GGUF file format headers with comprehensive bounds validation
 * to prevent heap overflow vulnerabilities (CWE-119/787).
 * 
 * Security: All tensor offsets and sizes are validated before memory access.
 * 
 * Spec: M0-W-1211, M0-W-1211a (security)
 */

#ifndef WORKER_GGUF_HEADER_PARSER_H
#define WORKER_GGUF_HEADER_PARSER_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace worker {
namespace gguf {

// GGUF magic bytes: "GGUF" in little-endian (0x46554747)
// In memory: 47 47 55 46 = 'G' 'G' 'U' 'F'
// As uint32_t (little-endian): 0x46554747
constexpr uint32_t GGUF_MAGIC = 0x46554747;

// GGUF version 3 (required for MXFP4 support)
constexpr uint32_t GGUF_VERSION = 3;

// Security limits
constexpr uint64_t MAX_TENSOR_COUNT = 10000;
constexpr size_t MAX_STRING_LENGTH = 1024 * 1024;  // 1MB
constexpr size_t MAX_ARRAY_LENGTH = 1000000;       // 1M elements
constexpr uint64_t MAX_TENSOR_ELEMENTS = 10000000000ULL;  // 10 billion elements (reasonable for largest models)

/**
 * GGUF metadata value types
 */
enum class GGUFValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

/**
 * GGUF tensor information
 */
struct GGUFTensor {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t type;          // GGML tensor type
    uint64_t offset;        // Offset from data_start
    size_t size;            // Tensor size in bytes
};

/**
 * GGUF metadata key-value pair
 */
struct GGUFMetadata {
    std::string key;
    GGUFValueType value_type;
    
    // Value storage (only one will be used based on value_type)
    uint64_t uint_value;
    int64_t int_value;
    double float_value;
    bool bool_value;
    std::string string_value;
    std::vector<uint8_t> array_value;
};

/**
 * GGUF file header
 */
struct GGUFHeader {
    uint32_t magic;              // 0x47475546 "GGUF"
    uint32_t version;            // 3
    uint64_t tensor_count;       // Number of tensors
    uint64_t metadata_kv_count;  // Number of metadata entries
    size_t header_size;          // Total header size in bytes
    size_t metadata_size;        // Total metadata size in bytes
    size_t data_start;           // Offset where tensor data begins
    
    std::vector<GGUFMetadata> metadata;
    std::vector<GGUFTensor> tensors;
};

/**
 * Security validation result
 */
struct ValidationResult {
    bool valid;
    std::string error_message;
    
    static ValidationResult ok() {
        return {true, ""};
    }
    
    static ValidationResult error(const std::string& msg) {
        return {false, msg};
    }
};

/**
 * Parse GGUF header from memory-mapped file
 * 
 * @param file_data Pointer to memory-mapped GGUF file
 * @param file_size Size of the file in bytes
 * @return Parsed header or throws on error
 * 
 * Security: Validates all offsets and sizes before access
 */
GGUFHeader parse_gguf_header(const void* file_data, size_t file_size);

/**
 * Validate tensor bounds (security-critical)
 * 
 * Checks:
 * - Offset >= data_start (tensor is after metadata)
 * - Offset < file_size (tensor starts within file)
 * - Offset + size <= file_size (tensor ends within file)
 * - No integer overflow (offset + size doesn't wrap)
 * 
 * @param tensor Tensor to validate
 * @param file_size Total file size
 * @param data_start Offset where tensor data begins
 * @return Validation result
 */
ValidationResult validate_tensor_bounds(
    const GGUFTensor& tensor,
    size_t file_size,
    size_t data_start
);

/**
 * Calculate tensor size in bytes from dimensions and type
 * 
 * @param dimensions Tensor dimensions
 * @param type GGML tensor type
 * @return Size in bytes
 */
size_t calculate_tensor_size(
    const std::vector<uint64_t>& dimensions,
    uint32_t type
);

/**
 * Get type size in bytes for GGML tensor types
 * 
 * @param type GGML tensor type
 * @return Bytes per element
 */
size_t get_type_size(uint32_t type);

} // namespace gguf
} // namespace worker

#endif // WORKER_GGUF_HEADER_PARSER_H

// ---
// Implemented by Llama-Beta 🦙
