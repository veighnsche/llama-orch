/**
 * Architecture Detection Implementation
 * 
 * Detects Llama variants from model configuration.
 * 
 * Spec: M0-W-1212
 */

#include "model/arch_detect.h"
#include <sstream>

namespace worker {
namespace model {

std::string ArchitectureDetector::variant_to_string(LlamaVariant variant) {
    switch (variant) {
        case LlamaVariant::Qwen:    return "Qwen";
        case LlamaVariant::Phi3:    return "Phi-3";
        case LlamaVariant::Llama2:  return "Llama-2";
        case LlamaVariant::Llama3:  return "Llama-3";
        case LlamaVariant::Unknown: return "Unknown";
        default:                    return "Unknown";
    }
}

LlamaVariant ArchitectureDetector::detect_variant(const gguf::LlamaConfig& config) {
    // Qwen: 32K context + GQA with 2 KV heads
    if (config.context_length == 32768 && config.attention_head_count_kv == 2) {
        return LlamaVariant::Qwen;
    }
    
    // Phi-3: 4K context + MHA (KV heads == attention heads)
    if (config.context_length == 4096 && 
        config.attention_head_count_kv == config.attention_head_count) {
        return LlamaVariant::Phi3;
    }
    
    // Llama 3: 8K context + GQA
    if (config.context_length == 8192 && 
        config.attention_head_count_kv < config.attention_head_count) {
        return LlamaVariant::Llama3;
    }
    
    // Llama 2: 4K context + GQA
    if (config.context_length == 4096 && 
        config.attention_head_count_kv < config.attention_head_count) {
        return LlamaVariant::Llama2;
    }
    
    return LlamaVariant::Unknown;
}

std::string ArchitectureDetector::infer_model_name(
    const gguf::LlamaConfig& config,
    LlamaVariant variant
) {
    std::ostringstream oss;
    
    switch (variant) {
        case LlamaVariant::Qwen:
            // Infer Qwen size from embedding dimensions
            if (config.embedding_length == 896) {
                oss << "Qwen2.5-0.5B";
            } else if (config.embedding_length == 1536) {
                oss << "Qwen2.5-1.5B";
            } else if (config.embedding_length == 2048) {
                oss << "Qwen2.5-3B";
            } else {
                oss << "Qwen2.5-" << config.embedding_length << "d";
            }
            break;
            
        case LlamaVariant::Phi3:
            // Infer Phi-3 size from embedding dimensions
            if (config.embedding_length == 3072) {
                oss << "Phi-3-mini";
            } else if (config.embedding_length == 4096) {
                oss << "Phi-3-small";
            } else if (config.embedding_length == 5120) {
                oss << "Phi-3-medium";
            } else {
                oss << "Phi-3-" << config.embedding_length << "d";
            }
            break;
            
        case LlamaVariant::Llama2:
            // Infer Llama 2 size from embedding dimensions
            if (config.embedding_length == 4096) {
                oss << "Llama-2-7B";
            } else if (config.embedding_length == 5120) {
                oss << "Llama-2-13B";
            } else if (config.embedding_length == 8192) {
                oss << "Llama-2-70B";
            } else {
                oss << "Llama-2-" << config.embedding_length << "d";
            }
            break;
            
        case LlamaVariant::Llama3:
            // Infer Llama 3 size from embedding dimensions
            if (config.embedding_length == 4096) {
                oss << "Llama-3-8B";
            } else if (config.embedding_length == 6144) {
                oss << "Llama-3-70B";
            } else {
                oss << "Llama-3-" << config.embedding_length << "d";
            }
            break;
            
        case LlamaVariant::Unknown:
            oss << "Llama-Unknown-" << config.embedding_length << "d";
            break;
    }
    
    return oss.str();
}

ArchitectureInfo ArchitectureDetector::detect(const gguf::LlamaConfig& config) {
    ArchitectureInfo info;
    
    info.architecture = config.architecture;
    info.variant = detect_variant(config);
    info.kv_heads = config.attention_head_count_kv;
    
    // Determine attention capabilities
    info.supports_gqa = (config.attention_head_count_kv < config.attention_head_count);
    info.supports_mha = (config.attention_head_count_kv == config.attention_head_count);
    
    // Infer model name
    info.model_name = infer_model_name(config, info.variant);
    
    return info;
}

} // namespace model
} // namespace worker

// ---
// Implemented by Llama-Beta 🦙
