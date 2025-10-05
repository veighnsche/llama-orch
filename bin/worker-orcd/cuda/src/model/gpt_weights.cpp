/**
 * GPT Weight Mapping and Loading Implementation
 * 
 * Implements weight loading for GPT models from GGUF files.
 * Supports Q4_K_M and MXFP4 quantization formats.
 * 
 * Spec: M0-W-1220, M0-W-1212, M0-W-1221, M0-W-1222
 * Story: GT-024, GT-025, GT-051
 */

#include "gpt_weights.h"
// GGUF parsing now done in Rust
// #include "../gguf/header_parser.h"
// #include "../gguf/llama_metadata.h"
// #include "../io/mmap_file.h"
// #include "device_memory.h"  // TODO: GT-052 - not needed for config parsing
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace worker {
namespace model {

// ============================================================================
// GPTConfig Implementation
// ============================================================================

bool GPTConfig::validate() const {
    if (vocab_size <= 0) {
        return false;
    }
    if (hidden_dim <= 0) {
        return false;
    }
    if (num_layers <= 0) {
        return false;
    }
    if (num_heads <= 0) {
        return false;
    }
    if (hidden_dim % num_heads != 0) {
        return false;  // Head dimension must be integer
    }
    if (head_dim != hidden_dim / num_heads) {
        return false;
    }
    if (ffn_dim <= 0) {
        return false;
    }
    if (max_seq_len <= 0) {
        return false;
    }
    return true;
}

// ============================================================================
// GPTLayerWeights Implementation
// ============================================================================

GPTLayerWeights::GPTLayerWeights()
    : attn_norm_weight(nullptr)
    , attn_norm_bias(nullptr)
    , attn_qkv_weight(nullptr)
    , attn_qkv_bias(nullptr)
    , attn_out_weight(nullptr)
    , attn_out_bias(nullptr)
    , ffn_norm_weight(nullptr)
    , ffn_norm_bias(nullptr)
    , ffn_up_weight(nullptr)
    , ffn_up_bias(nullptr)
    , ffn_down_weight(nullptr)
    , ffn_down_bias(nullptr)
    , total_vram_bytes(0)
{
}

GPTLayerWeights::~GPTLayerWeights() {
    // Free all VRAM allocations
    if (attn_norm_weight) cudaFree(attn_norm_weight);
    if (attn_norm_bias) cudaFree(attn_norm_bias);
    if (attn_qkv_weight) cudaFree(attn_qkv_weight);
    if (attn_qkv_bias) cudaFree(attn_qkv_bias);
    if (attn_out_weight) cudaFree(attn_out_weight);
    if (attn_out_bias) cudaFree(attn_out_bias);
    if (ffn_norm_weight) cudaFree(ffn_norm_weight);
    if (ffn_norm_bias) cudaFree(ffn_norm_bias);
    if (ffn_up_weight) cudaFree(ffn_up_weight);
    if (ffn_up_bias) cudaFree(ffn_up_bias);
    if (ffn_down_weight) cudaFree(ffn_down_weight);
    if (ffn_down_bias) cudaFree(ffn_down_bias);
}

// ============================================================================
// GPTModelWeights Implementation
// ============================================================================

GPTModelWeights::GPTModelWeights()
    : token_embeddings(nullptr)
    , position_embeddings(nullptr)
    , output_norm_weight(nullptr)
    , output_norm_bias(nullptr)
    , lm_head_weight(nullptr)
    , total_vram_bytes(0)
{
}

GPTModelWeights::~GPTModelWeights() {
    // Free embeddings
    if (token_embeddings) cudaFree(token_embeddings);
    if (position_embeddings) cudaFree(position_embeddings);
    
    // Layers freed by unique_ptr destructors
    
    // Free output head
    if (output_norm_weight) cudaFree(output_norm_weight);
    if (output_norm_bias) cudaFree(output_norm_bias);
    if (lm_head_weight) cudaFree(lm_head_weight);
}

// ============================================================================
// GPTWeightMapper Implementation
// ============================================================================

std::string GPTWeightMapper::get_tensor_name(
    const std::string& component,
    int layer_idx
) {
    if (layer_idx < 0) {
        // Global tensors
        return component;
    } else {
        // Layer-specific tensors
        return "blk." + std::to_string(layer_idx) + "." + component;
    }
}

std::vector<uint64_t> GPTWeightMapper::get_expected_shape(
    const std::string& tensor_name,
    const GPTConfig& config
) {
    // Token embeddings
    if (tensor_name == "token_embd.weight") {
        // GGUF stores embeddings as [n_in, n_out] => [hidden_dim, vocab_size]
        return {static_cast<uint64_t>(config.hidden_dim),
                static_cast<uint64_t>(config.vocab_size)};
    }
    
    // Position embeddings
    if (tensor_name == "position_embd.weight") {
        // GGUF stores embeddings as [n_in, n_out] => [hidden_dim, max_seq_len]
        return {static_cast<uint64_t>(config.hidden_dim),
                static_cast<uint64_t>(config.max_seq_len)};
    }
    
    // Output norm
    if (tensor_name == "output_norm.weight" || tensor_name == "output_norm.bias") {
        return {static_cast<uint64_t>(config.hidden_dim)};
    }
    
    // LM head
    if (tensor_name == "output.weight") {
        return {static_cast<uint64_t>(config.hidden_dim), 
                static_cast<uint64_t>(config.vocab_size)};
    }
    
    // Layer-specific tensors
    if (tensor_name.find("blk.") == 0) {
        if (tensor_name.find(".attn_norm.weight") != std::string::npos ||
            tensor_name.find(".attn_norm.bias") != std::string::npos ||
            tensor_name.find(".ffn_norm.weight") != std::string::npos ||
            tensor_name.find(".ffn_norm.bias") != std::string::npos) {
            return {static_cast<uint64_t>(config.hidden_dim)};
        }
        
        if (tensor_name.find(".attn_qkv.weight") != std::string::npos) {
            return {static_cast<uint64_t>(config.hidden_dim), 
                    static_cast<uint64_t>(3 * config.hidden_dim)};
        }
        
        if (tensor_name.find(".attn_qkv.bias") != std::string::npos) {
            return {static_cast<uint64_t>(3 * config.hidden_dim)};
        }
        
        if (tensor_name.find(".attn_output.weight") != std::string::npos) {
            return {static_cast<uint64_t>(config.hidden_dim), 
                    static_cast<uint64_t>(config.hidden_dim)};
        }
        
        if (tensor_name.find(".attn_output.bias") != std::string::npos) {
            return {static_cast<uint64_t>(config.hidden_dim)};
        }
        
        if (tensor_name.find(".ffn_up.weight") != std::string::npos) {
            return {static_cast<uint64_t>(config.hidden_dim), 
                    static_cast<uint64_t>(config.ffn_dim)};
        }
        
        if (tensor_name.find(".ffn_up.bias") != std::string::npos) {
            return {static_cast<uint64_t>(config.ffn_dim)};
        }
        
        if (tensor_name.find(".ffn_down.weight") != std::string::npos) {
            return {static_cast<uint64_t>(config.ffn_dim), 
                    static_cast<uint64_t>(config.hidden_dim)};
        }
        
        if (tensor_name.find(".ffn_down.bias") != std::string::npos) {
            return {static_cast<uint64_t>(config.hidden_dim)};
        }
    }
    
    // Unknown tensor
    return {};
}

bool GPTWeightMapper::is_valid_gpt_tensor(const std::string& name) {
    // Check for valid prefixes
    if (name.find("token_embd") == 0) return true;
    if (name.find("position_embd") == 0) return true;
    if (name.find("blk.") == 0) return true;
    if (name.find("output_norm") == 0) return true;
    if (name.find("output.") == 0) return true;
    
    return false;
}

int GPTWeightMapper::parse_layer_index(const std::string& name) {
    // Check if this is a layer tensor
    if (name.find("blk.") != 0) {
        return -1;
    }
    
    // Extract layer index
    size_t start = 4;  // After "blk."
    size_t end = name.find(".", start);
    if (end == std::string::npos) {
        return -1;
    }
    
    std::string idx_str = name.substr(start, end - start);
    try {
        return std::stoi(idx_str);
    } catch (...) {
        return -1;
    }
}

// ============================================================================
// GPTWeightLoader Implementation
// ============================================================================

std::unique_ptr<GPTModelWeights> GPTWeightLoader::load_from_gguf(
    const std::string& path
) {
    // Parse config from GGUF
    GPTConfig config = parse_config_from_gguf(path);
    
    // Validate config
    if (!config.validate()) {
        throw std::runtime_error("Invalid GPT configuration");
    }
    
    // Create model weights
    auto model = std::make_unique<GPTModelWeights>();
    model->config = config;
    
    // TODO: Parse GGUF tensors
    // This requires actual GGUF file to implement
    std::vector<GGUFTensorInfo> tensors;
    
    // TODO: Load embeddings
    // load_embeddings(model.get(), path, tensors);
    
    // TODO: Load transformer layers
    model->layers.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        auto layer = std::make_unique<GPTLayerWeights>();
        // load_layer(layer.get(), i, path, tensors, config);
        model->layers.push_back(std::move(layer));
    }
    
    // TODO: Load output head
    // load_output_head(model.get(), path, tensors);
    
    // Calculate total VRAM usage
    model->total_vram_bytes = calculate_vram_usage(config);
    
    return model;
}

GPTConfig GPTWeightLoader::validate_gguf(const std::string& path) {
    return parse_config_from_gguf(path);
}

size_t GPTWeightLoader::calculate_vram_usage(const GPTConfig& config) {
    size_t total = 0;
    
    // Token embeddings: vocab_size * hidden_dim * 2 bytes (FP16 or quantized)
    total += config.vocab_size * config.hidden_dim * 2;
    
    // Position embeddings: max_seq_len * hidden_dim * 2 bytes
    total += config.max_seq_len * config.hidden_dim * 2;
    
    // Per-layer weights
    for (int i = 0; i < config.num_layers; ++i) {
        // LayerNorm weights: 2 * hidden_dim * 2 (gamma + beta) * 2 LayerNorms
        total += 4 * config.hidden_dim * 2;
        
        // Attention: QKV + output projection
        total += config.hidden_dim * 3 * config.hidden_dim * 2;  // QKV
        total += 3 * config.hidden_dim * 2;                       // QKV bias
        total += config.hidden_dim * config.hidden_dim * 2;       // Output
        total += config.hidden_dim * 2;                           // Output bias
        
        // FFN: up + down projections
        total += config.hidden_dim * config.ffn_dim * 2;          // Up
        total += config.ffn_dim * 2;                              // Up bias
        total += config.ffn_dim * config.hidden_dim * 2;          // Down
        total += config.hidden_dim * 2;                           // Down bias
    }
    
    // Output head
    total += config.hidden_dim * 2;                               // Norm gamma
    total += config.hidden_dim * 2;                               // Norm beta
    total += config.hidden_dim * config.vocab_size * 2;           // LM head
    
    // KV cache: 2 (K+V) * num_layers * hidden_dim * max_seq_len * 2 bytes (FP16)
    total += 2 * config.num_layers * config.hidden_dim * config.max_seq_len * 2;
    
    // Activation buffers (conservative estimate)
    total += config.max_seq_len * config.hidden_dim * 4 * 10;  // ~10 intermediate buffers
    
    return total;
}

// DEPRECATED: Config parsing now done in Rust
GPTConfig GPTWeightLoader::parse_config_from_gguf(const std::string& path) {
    (void)path;  // Unused
    throw std::runtime_error("parse_config_from_gguf is deprecated - use Rust worker-gguf instead");
    
    /*
    // OLD CODE - DEPRECATED
    // Open and memory-map the GGUF file
    auto mmap = io::MmapFile::open(path);
    
    // Parse GGUF header
    auto header = gguf::parse_gguf_header(mmap.data(), mmap.size());
    
    // Extract architecture string
    std::string arch = gguf::get_required_string(header.metadata, "general.architecture");
    
    GPTConfig config;
    
    // Handle Qwen2 architecture (Qwen2.5-0.5B)
    if (arch == "qwen2") {
        // Extract config from qwen2.* metadata keys
        // Note: vocab_size comes from tokenizer, not qwen2.vocab_size
        config.vocab_size = gguf::get_array_length(header.metadata, "tokenizer.ggml.tokens");
        config.hidden_dim = gguf::get_required_uint32(header.metadata, "qwen2.embedding_length");
        config.num_layers = gguf::get_required_uint32(header.metadata, "qwen2.block_count");
        config.num_heads = gguf::get_required_uint32(header.metadata, "qwen2.attention.head_count");
        config.ffn_dim = gguf::get_required_uint32(header.metadata, "qwen2.feed_forward_length");
        config.context_length = gguf::get_required_uint32(header.metadata, "qwen2.context_length");
        config.max_seq_len = config.context_length;  // Use context_length as max_seq_len
        
        // Calculate head dimension
        config.head_dim = config.hidden_dim / config.num_heads;
    }
    // Handle Llama architecture (standard Llama, Phi-3, etc.)
    else if (arch == "llama") {
        // Use existing llama metadata parser
        auto llama_config = gguf::parse_llama_metadata(header.metadata);
        
        // Map to GPTConfig
        config.vocab_size = llama_config.vocab_size;
        config.hidden_dim = llama_config.embedding_length;
        config.num_layers = llama_config.block_count;
        config.num_heads = llama_config.attention_head_count;
        config.head_dim = llama_config.head_dim;
        config.ffn_dim = llama_config.ffn_length;
        config.max_seq_len = llama_config.context_length;
        config.context_length = llama_config.context_length;
    }
    // Handle GPT2/GPT architecture (if needed in future)
    else if (arch == "gpt2" || arch == "gpt") {
        // Extract config from gpt2.* metadata keys
        // Note: vocab_size comes from tokenizer, not gpt2.vocab_size
        config.vocab_size = gguf::get_array_length(header.metadata, "tokenizer.ggml.tokens");
        config.hidden_dim = gguf::get_required_uint32(header.metadata, "gpt2.embedding_length");
        config.num_layers = gguf::get_required_uint32(header.metadata, "gpt2.block_count");
        config.num_heads = gguf::get_required_uint32(header.metadata, "gpt2.attention.head_count");
        config.ffn_dim = gguf::get_required_uint32(header.metadata, "gpt2.feed_forward_length");
        config.context_length = gguf::get_required_uint32(header.metadata, "gpt2.context_length");
        config.max_seq_len = config.context_length;
        
        // Calculate head dimension
        config.head_dim = config.hidden_dim / config.num_heads;
    }
    else {
        throw std::runtime_error("Unsupported architecture: " + arch + 
                               " (supported: qwen2, llama, gpt2)");
    }
    
    // Extract quantization type from first tensor
    if (!header.tensors.empty()) {
        // Map GGML type to quantization string
        switch (header.tensors[0].type) {
            case 0:  config.quant_kind = "F32"; break;
            case 1:  config.quant_kind = "F16"; break;
            case 2:  config.quant_kind = "Q4_0"; break;
            case 3:  config.quant_kind = "Q4_1"; break;
            case 6:  config.quant_kind = "Q5_0"; break;
            case 7:  config.quant_kind = "Q5_1"; break;
            case 8:  config.quant_kind = "Q8_0"; break;
            case 9:  config.quant_kind = "Q8_1"; break;
            case 12: config.quant_kind = "Q4_K_M"; break;
            case 20: config.quant_kind = "MXFP4"; break;
            default: config.quant_kind = "UNKNOWN"; break;
        }
    } else {
        config.quant_kind = quant_kind;
    
    return config;
}

void GPTWeightLoader::load_embeddings(
    GPTModelWeights* model,
    const std::string& path,
) {
    // Find token embeddings
    const GGUFTensorInfo* token_emb = find_tensor(tensors, "token_embd.weight");
    if (!token_emb) {
        throw std::runtime_error("Missing token_embd.weight");
    }
    
    // Validate shape (GGUF stores as [hidden_dim, vocab_size])
    std::vector<uint64_t> expected_shape = {
        static_cast<uint64_t>(model->config.hidden_dim),
        static_cast<uint64_t>(model->config.vocab_size)
    };
    validate_tensor_shape(*token_emb, expected_shape, "token_embd.weight");
    
    // TODO: Load from file and copy to VRAM
    // model->token_embeddings = allocate_and_copy(...);
    
    // Find position embeddings
    const GGUFTensorInfo* pos_emb = find_tensor(tensors, "position_embd.weight");
    if (!pos_emb) {
        throw std::runtime_error("Missing position_embd.weight");
    }
    
    // Validate shape (GGUF stores as [hidden_dim, max_seq_len])
    expected_shape = {
        static_cast<uint64_t>(model->config.hidden_dim),
        static_cast<uint64_t>(model->config.max_seq_len)
    };
    validate_tensor_shape(*pos_emb, expected_shape, "position_embd.weight");
    
    // TODO: Load from file and copy to VRAM
    // model->position_embeddings = allocate_and_copy(...);
}

void GPTWeightLoader::load_layer(
    GPTLayerWeights* layer,
    int layer_idx,
    const std::string& path,
    const std::vector<GGUFTensorInfo>& tensors,
    const GPTConfig& config
) {
    // TODO: Load all layer weights
    // - attn_norm (weight + bias)
    // - attn_qkv (weight + bias)
    // - attn_output (weight + bias)
    // - ffn_norm (weight + bias)
    // - ffn_up (weight + bias)
    // - ffn_down (weight + bias)
}

void GPTWeightLoader::load_output_head(
    GPTModelWeights* model,
    const std::string& path,
    const std::vector<GGUFTensorInfo>& tensors
) {
    // TODO: Load output norm and LM head
}

void* GPTWeightLoader::allocate_and_copy(
    const void* host_data,
    size_t size_bytes,
    const std::string& tensor_name
) {
    void* device_ptr = nullptr;
    cudaError_t err = cudaMalloc(&device_ptr, size_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "Failed to allocate VRAM for " + tensor_name + 
            ": " + cudaGetErrorString(err)
        );
    }
    
    // Copy in chunks (1MB)
    const size_t CHUNK_SIZE = 1024 * 1024;
    for (size_t offset = 0; offset < size_bytes; offset += CHUNK_SIZE) {
        size_t chunk_size = std::min(CHUNK_SIZE, size_bytes - offset);
        err = cudaMemcpy(
            static_cast<char*>(device_ptr) + offset,
            static_cast<const char*>(host_data) + offset,
            chunk_size,
            cudaMemcpyHostToDevice
        );
        if (err != cudaSuccess) {
            cudaFree(device_ptr);
            throw std::runtime_error(
                "Failed to copy " + tensor_name + " to VRAM: " + 
                cudaGetErrorString(err)
            );
        }
    }
    
    return device_ptr;
}

void GPTWeightLoader::validate_tensor_shape(
    const GGUFTensorInfo& tensor,
    const std::vector<uint64_t>& expected_shape,
    const std::string& context
) {
    if (tensor.dimensions.size() != expected_shape.size()) {
        std::ostringstream oss;
        oss << "Shape mismatch for " << context << ": expected " 
            << expected_shape.size() << " dimensions, got " 
            << tensor.dimensions.size();
        throw std::runtime_error(oss.str());
    }
    
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (tensor.dimensions[i] != expected_shape[i]) {
            std::ostringstream oss;
            oss << "Shape mismatch for " << context << " at dimension " << i 
                << ": expected " << expected_shape[i] << ", got " 
                << tensor.dimensions[i];
            throw std::runtime_error(oss.str());
        }
    }
}

const GGUFTensorInfo* GPTWeightLoader::find_tensor(
    const std::vector<GGUFTensorInfo>& tensors,
    const std::string& name
) {
    auto it = std::find_if(tensors.begin(), tensors.end(),
        [&name](const GGUFTensorInfo& t) { return t.name == name; });
    
    if (it != tensors.end()) {
        return &(*it);
    }
    return nullptr;
}

    */
}

} // namespace model
} // namespace worker

// ---
// Implemented by Llama-Beta 
