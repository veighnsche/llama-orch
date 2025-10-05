/**
 * Pre-Load Validation
 * 
 * Comprehensive validation of GGUF files before loading to VRAM.
 * Prevents loading failures and security vulnerabilities.
 * 
 * Spec: M0-W-1210
 */

#ifndef WORKER_VALIDATION_PRE_LOAD_H
#define WORKER_VALIDATION_PRE_LOAD_H

#include "../gguf/header_parser.h"
#include "../gguf/llama_metadata.h"
#include <string>
#include <vector>

namespace worker {
namespace validation {

/**
 * Validation report
 */
struct ValidationReport {
    bool passed;
    std::string error_message;
    std::vector<std::string> warnings;
    
    // Validation details
    size_t total_vram_required;
    size_t available_vram;
    uint64_t tensor_count;
    std::string architecture;
    std::string file_path;
    
    // Helper: Create passing report
    static ValidationReport pass(
        size_t vram_required,
        size_t vram_available,
        uint64_t tensors,
        const std::string& arch,
        const std::string& path
    ) {
        return {
            true,
            "",
            {},
            vram_required,
            vram_available,
            tensors,
            arch,
            path
        };
    }
    
    // Helper: Create failing report
    static ValidationReport fail(
        const std::string& error,
        const std::string& path
    ) {
        return {
            false,
            error,
            {},
            0,
            0,
            0,
            "",
            path
        };
    }
};

/**
 * Pre-load validator
 */
class PreLoadValidator {
public:
    /**
     * Validate GGUF file before loading
     * 
     * Performs comprehensive validation:
     * 1. File access (exists, readable)
     * 2. Header validation (magic, version, counts)
     * 3. Metadata validation (required keys, architecture)
     * 4. Tensor bounds validation (security)
     * 5. VRAM requirement validation (fits in available VRAM)
     * 
     * @param gguf_path Path to GGUF file
     * @param available_vram Available VRAM in bytes
     * @return Validation report with pass/fail and details
     */
    static ValidationReport validate(
        const std::string& gguf_path,
        size_t available_vram
    );
    
    // Helper methods (public for testing)
    
    /**
     * Validate file access
     * 
     * @param path File path
     * @return true if file exists and is readable
     */
    static bool validate_file_access(const std::string& path);
    
    /**
     * Validate GGUF header
     * 
     * @param header Parsed header
     * @return true if header is valid
     */
    static bool validate_header(const gguf::GGUFHeader& header);
    
    /**
     * Validate Llama metadata
     * 
     * @param metadata Parsed metadata
     * @return true if metadata is valid
     */
    static bool validate_llama_metadata(const std::vector<gguf::GGUFMetadata>& metadata);
    
    /**
     * Validate tensor bounds (security-critical)
     * 
     * @param tensors Tensor list
     * @param file_size File size
     * @param data_start Data section offset
     * @return true if all tensors have valid bounds
     */
    static bool validate_tensor_bounds(
        const std::vector<gguf::GGUFTensor>& tensors,
        size_t file_size,
        size_t data_start
    );
    
    /**
     * Calculate total VRAM requirement
     * 
     * Sums all tensor sizes and adds 10% overhead for KV cache and buffers.
     * 
     * @param tensors Tensor list
     * @return Total VRAM required in bytes
     */
    static size_t calculate_vram_requirement(
        const std::vector<gguf::GGUFTensor>& tensors
    );
    
    /**
     * Validate VRAM availability
     * 
     * @param required Required VRAM
     * @param available Available VRAM
     * @return true if sufficient VRAM
     */
    static bool validate_vram_availability(
        size_t required,
        size_t available
    );
    
    /**
     * Write audit log entry for rejected file
     * 
     * @param reason Rejection reason
     * @param path File path
     */
    static void audit_log_rejection(
        const std::string& reason,
        const std::string& path
    );
};

} // namespace validation
} // namespace worker

#endif // WORKER_VALIDATION_PRE_LOAD_H

// ---
// Implemented by Llama-Beta 🦙
