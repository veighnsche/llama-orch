#include "qwen_weight_loader.h"
#include "vram_tracker.h"
#include "device_memory.h"
#include "../io/chunked_transfer.h"
#include <fstream>
#include <cstring>
#include <stdexcept>

namespace worker {
namespace model {

std::vector<std::string> QwenWeightLoader::get_tensor_names(int num_layers) {
    std::vector<std::string> names;
    
    // Token embeddings
    names.push_back("token_embd.weight");
    
    // Transformer layers (24 for Qwen2.5-0.5B)
    for (int i = 0; i < num_layers; i++) {
        std::string prefix = "blk." + std::to_string(i) + ".";
        
        // Attention
        names.push_back(prefix + "attn_norm.weight");
        names.push_back(prefix + "attn_q.weight");
        names.push_back(prefix + "attn_q.bias");
        names.push_back(prefix + "attn_k.weight");
        names.push_back(prefix + "attn_k.bias");
        names.push_back(prefix + "attn_v.weight");
        names.push_back(prefix + "attn_v.bias");
        names.push_back(prefix + "attn_output.weight");
        
        // FFN
        names.push_back(prefix + "ffn_norm.weight");
        names.push_back(prefix + "ffn_gate.weight");
        names.push_back(prefix + "ffn_up.weight");
        names.push_back(prefix + "ffn_down.weight");
    }
    
    // Output
    names.push_back("output_norm.weight");
    names.push_back("output.weight");
    
    return names;
}

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t type;
    uint64_t offset;
    size_t size_bytes;
};

// Read GGUF string
std::string read_gguf_string(std::ifstream& file) {
    uint64_t len;
    file.read(reinterpret_cast<char*>(&len), 8);
    std::string str(len, '\0');
    file.read(&str[0], len);
    return str;
}

// Skip GGUF metadata value
void skip_metadata_value(std::ifstream& file) {
    uint32_t value_type;
    file.read(reinterpret_cast<char*>(&value_type), 4);
    
    switch (value_type) {
        case 0: file.seekg(1, std::ios::cur); break;  // uint8
        case 1: file.seekg(1, std::ios::cur); break;  // int8
        case 2: file.seekg(2, std::ios::cur); break;  // uint16
        case 3: file.seekg(2, std::ios::cur); break;  // int16
        case 4: file.seekg(4, std::ios::cur); break;  // uint32
        case 5: file.seekg(4, std::ios::cur); break;  // int32
        case 6: file.seekg(4, std::ios::cur); break;  // float32
        case 7: file.seekg(1, std::ios::cur); break;  // bool
        case 8: read_gguf_string(file); break;        // string
        case 9: {  // array
            uint32_t elem_type;
            file.read(reinterpret_cast<char*>(&elem_type), 4);
            uint64_t count;
            file.read(reinterpret_cast<char*>(&count), 8);
            
            // Skip array elements
            for (uint64_t i = 0; i < count; i++) {
                if (elem_type == 8) {
                    read_gguf_string(file);
                } else if (elem_type <= 7) {
                    size_t elem_size = (elem_type == 0 || elem_type == 1) ? 1 :
                                      (elem_type == 2 || elem_type == 3) ? 2 : 4;
                    file.seekg(elem_size, std::ios::cur);
                } else {
                    file.seekg(8, std::ios::cur);
                }
            }
            break;
        }
        case 10: file.seekg(8, std::ios::cur); break;  // uint64
        case 11: file.seekg(8, std::ios::cur); break;  // int64
        case 12: file.seekg(8, std::ios::cur); break;  // float64
        default: throw std::runtime_error("Unknown value type");
    }
}

// Simple GGUF tensor finder
TensorInfo find_tensor(const char* path, const std::string& target_name) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open GGUF file");
    }
    
    // Read header
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x46554747) {  // "GGUF"
        throw std::runtime_error("Invalid GGUF magic");
    }
    
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    
    uint64_t tensor_count;
    file.read(reinterpret_cast<char*>(&tensor_count), 8);
    
    uint64_t metadata_count;
    file.read(reinterpret_cast<char*>(&metadata_count), 8);
    
    // Skip metadata section
    for (uint64_t i = 0; i < metadata_count; i++) {
        read_gguf_string(file);  // key
        skip_metadata_value(file);  // value
    }
    
    // Read tensor info
    for (uint64_t i = 0; i < tensor_count; i++) {
        std::string name = read_gguf_string(file);
        
        uint32_t n_dims;
        file.read(reinterpret_cast<char*>(&n_dims), 4);
        
        std::vector<uint64_t> dims(n_dims);
        file.read(reinterpret_cast<char*>(dims.data()), n_dims * 8);
        
        uint32_t type;
        file.read(reinterpret_cast<char*>(&type), 4);
        
        uint64_t offset;
        file.read(reinterpret_cast<char*>(&offset), 8);
        
        if (name == target_name) {
            // Calculate size
            size_t size = 1;
            for (auto d : dims) size *= d;
            
            // Estimate bytes per element based on type
            size_t bytes_per_elem = 2;  // Default to FP16
            if (type == 0) bytes_per_elem = 4;  // F32
            else if (type == 1) bytes_per_elem = 2;  // F16
            else if (type >= 2 && type <= 14) bytes_per_elem = 2;  // Quantized (approx)
            
            TensorInfo info;
            info.name = name;
            info.dimensions = dims;
            info.type = type;
            info.offset = offset;
            info.size_bytes = size * bytes_per_elem;
            
            return info;
        }
    }
    
    throw std::runtime_error("Tensor not found: " + target_name);
}

void* QwenWeightLoader::load_tensor_to_vram(
    const char* path,
    const std::string& tensor_name,
    VramTracker& tracker
) {
    // Find tensor in GGUF file
    TensorInfo info = find_tensor(path, tensor_name);
    
    // WARNING: This is loading quantized weights (Q4_K_M) directly without dequantization!
    // This will cause NaN and garbage values. Need to implement dequantization.
    fprintf(stderr, "⚠️  Loading %s: type=%u, size=%zu bytes (QUANTIZED - NOT DEQUANTIZED!)\n",
            tensor_name.c_str(), info.type, info.size_bytes);
    
    // Allocate GPU memory
    void* gpu_ptr = nullptr;
    cudaError_t err = cudaMalloc(&gpu_ptr, info.size_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed for " + tensor_name + ": " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    // Read tensor data from file
    std::ifstream file(path, std::ios::binary);
    file.seekg(info.offset);
    
    std::vector<char> host_data(info.size_bytes);
    file.read(host_data.data(), info.size_bytes);
    
    if (!file) {
        cudaFree(gpu_ptr);
        throw std::runtime_error("Failed to read tensor data for " + tensor_name);
    }
    
    // Copy to GPU
    err = cudaMemcpy(gpu_ptr, host_data.data(), info.size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(gpu_ptr);
        throw std::runtime_error("cudaMemcpy failed for " + tensor_name + ": " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    // Track allocation
    tracker.record_allocation(gpu_ptr, info.size_bytes, VramPurpose::ModelWeights, tensor_name);
    
    return gpu_ptr;
}

QwenModel* QwenWeightLoader::load(
    const char* path,
    const QwenConfig& config
) {
    auto model = new QwenModel();
    model->config = config;
    
    VramTracker tracker;
    
    // Get all tensor names
    auto tensor_names = get_tensor_names(config.num_layers);
    
    fprintf(stderr, "Loading %zu tensors for Qwen2.5-0.5B...\n", tensor_names.size());
    
    // Load embeddings
    model->weights.token_embd = load_tensor_to_vram(path, "token_embd.weight", tracker);
    
    // Load layers
    model->weights.layers.resize(config.num_layers);
    for (uint32_t i = 0; i < config.num_layers; i++) {
        std::string prefix = "blk." + std::to_string(i) + ".";
        
        auto& layer = model->weights.layers[i];
        layer.attn_norm = load_tensor_to_vram(path, prefix + "attn_norm.weight", tracker);
        layer.attn_q_weight = load_tensor_to_vram(path, prefix + "attn_q.weight", tracker);
        layer.attn_q_bias = load_tensor_to_vram(path, prefix + "attn_q.bias", tracker);
        layer.attn_k_weight = load_tensor_to_vram(path, prefix + "attn_k.weight", tracker);
        layer.attn_k_bias = load_tensor_to_vram(path, prefix + "attn_k.bias", tracker);
        layer.attn_v_weight = load_tensor_to_vram(path, prefix + "attn_v.weight", tracker);
        layer.attn_v_bias = load_tensor_to_vram(path, prefix + "attn_v.bias", tracker);
        layer.attn_output = load_tensor_to_vram(path, prefix + "attn_output.weight", tracker);
        
        layer.ffn_norm = load_tensor_to_vram(path, prefix + "ffn_norm.weight", tracker);
        layer.ffn_gate = load_tensor_to_vram(path, prefix + "ffn_gate.weight", tracker);
        layer.ffn_up = load_tensor_to_vram(path, prefix + "ffn_up.weight", tracker);
        layer.ffn_down = load_tensor_to_vram(path, prefix + "ffn_down.weight", tracker);
        
        if (i % 5 == 0) {
            fprintf(stderr, "  Loaded layer %u/%u\n", i + 1, config.num_layers);
        }
    }
    
    // Load output
    model->weights.output_norm = load_tensor_to_vram(path, "output_norm.weight", tracker);
    model->weights.lm_head = load_tensor_to_vram(path, "output.weight", tracker);

    
    model->vram_usage = tracker.total_usage();
    
    fprintf(stderr, "✅ Loaded %zu tensors, VRAM usage: %.2f MB\n",
            tensor_names.size(),
            model->vram_usage / 1024.0 / 1024.0);
    
    return model;
}

QwenModel* QwenWeightLoader::load_from_gpu_pointers(
    const std::map<std::string, void*>& gpu_pointers,
    const QwenConfig& config,
    uint64_t total_vram_bytes
) {
    auto model = new QwenModel();
    model->config = config;
    model->vram_usage = total_vram_bytes;
    
    fprintf(stderr, "🔗 [C++] Wiring %zu pre-loaded GPU pointers...\n", gpu_pointers.size());
    
    // Helper to get pointer with error checking
    auto get_ptr = [&](const std::string& name) -> void* {
        auto it = gpu_pointers.find(name);
        if (it == gpu_pointers.end()) {
            fprintf(stderr, "❌ Missing tensor: %s\n", name.c_str());
            throw std::runtime_error("Missing tensor: " + name);
        }
        return it->second;
    };
    
    // Wire embeddings
    model->weights.token_embd = get_ptr("token_embd.weight");
    fprintf(stderr, "🔍 [C++] Retrieved token_embd.weight pointer: %p\n", model->weights.token_embd);
    
    // Load layers
    model->weights.layers.resize(config.num_layers);
    for (uint32_t i = 0; i < config.num_layers; i++) {
        std::string prefix = "blk." + std::to_string(i) + ".";
        
        auto& layer = model->weights.layers[i];
        layer.attn_norm = get_ptr(prefix + "attn_norm.weight");
        layer.attn_q_weight = get_ptr(prefix + "attn_q.weight");
        layer.attn_q_bias = nullptr;  // Qwen2.5 doesn't use biases
        layer.attn_k_weight = get_ptr(prefix + "attn_k.weight");
        layer.attn_k_bias = nullptr;  // Qwen2.5 doesn't use biases
        layer.attn_v_weight = get_ptr(prefix + "attn_v.weight");
        layer.attn_v_bias = nullptr;  // Qwen2.5 doesn't use biases
        layer.attn_output = get_ptr(prefix + "attn_output.weight");
        
        layer.ffn_norm = get_ptr(prefix + "ffn_norm.weight");
        layer.ffn_gate = get_ptr(prefix + "ffn_gate.weight");
        layer.ffn_up = get_ptr(prefix + "ffn_up.weight");
    }
    
    // Wire output
    model->weights.output_norm = get_ptr("output_norm.weight");
    model->weights.lm_head = get_ptr("output.weight");
    
    fprintf(stderr, "✅ [C++] Wired all %u layers (VRAM: %.2f MB)\n",
            config.num_layers,
            total_vram_bytes / 1024.0 / 1024.0);
    
    return model;
}

} // namespace model
} // namespace worker
