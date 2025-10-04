/**
 * Architecture Detection
 * 
 * Detects specific Llama variants (Qwen, Phi-3, Llama 2/3) from model configuration.
 * Enables variant-specific optimizations and validation.
 * 
 * Spec: M0-W-1212
 */

#ifndef WORKER_MODEL_ARCH_DETECT_H
#define WORKER_MODEL_ARCH_DETECT_H

#include "../gguf/llama_metadata.h"
#include <string>

namespace worker {
namespace model {

/**
 * Llama model variants
 */
enum class LlamaVariant {
    Qwen,        // Qwen 2.5 series (32K context, GQA with 2 KV heads)
    Phi3,        // Microsoft Phi-3 (4K context, MHA)
    Llama2,      // Meta Llama 2 (4K context, GQA)
    Llama3,      // Meta Llama 3 (8K context, GQA)
    Unknown,     // Unknown Llama variant
};

/**
 * Architecture information
 */
struct ArchitectureInfo {
    std::string architecture;  // "llama"
    LlamaVariant variant;
    bool supports_gqa;         // Grouped Query Attention
    bool supports_mha;         // Multi-Head Attention
    uint32_t kv_heads;         // Number of KV heads
    std::string model_name;    // e.g., "Qwen2.5-0.5B", "Phi-3-mini"
};

/**
 * Architecture detector
 */
class ArchitectureDetector {
public:
    /**
     * Detect Llama variant and capabilities from config
     * 
     * Identifies specific Llama variants based on:
     * - Context length (32K = Qwen, 4K = Phi-3/Llama2, 8K = Llama3)
     * - KV head count (2 = Qwen GQA, 32 = Phi-3 MHA, etc.)
     * - Embedding dimensions (896 = Qwen-0.5B, 3072 = Phi-3, etc.)
     * 
     * @param config Parsed Llama configuration
     * @return Architecture information with variant and capabilities
     */
    static ArchitectureInfo detect(const gguf::LlamaConfig& config);

private:
    /**
     * Detect specific Llama variant
     * 
     * @param config Llama configuration
     * @return Detected variant
     */
    static LlamaVariant detect_variant(const gguf::LlamaConfig& config);
    
    /**
     * Infer model name from config and variant
     * 
     * @param config Llama configuration
     * @param variant Detected variant
     * @return Human-readable model name
     */
    static std::string infer_model_name(
        const gguf::LlamaConfig& config,
        LlamaVariant variant
    );
    
    /**
     * Get variant name as string
     * 
     * @param variant Llama variant
     * @return Variant name
     */
    static std::string variant_to_string(LlamaVariant variant);
};

} // namespace model
} // namespace worker

#endif // WORKER_MODEL_ARCH_DETECT_H

// ---
// Implemented by Llama-Beta 🦙
