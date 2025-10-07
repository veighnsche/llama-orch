#include "qwen_weight_loader.h"
#include "vram_tracker.h"
#include "device_memory.h"
#include "../io/chunked_transfer.h"
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <cuda_fp16.h>
#include <cmath>

// ============================================================================
// [TEAM_CHARLIE_BETA] ⚠️ MISSING WEIGHT LOADING - FIXED BUT NOT THE BUG (2025-10-06 17:07 UTC)
// ============================================================================
//
// ROOT CAUSE (HYPOTHESIS): Missing ffn_down weight loading in load_from_gpu_pointers()
//
// SYMPTOM: Model generates garbage tokens (mojibake, code tokens, foreign languages)
//
// THE BUG I FOUND:
// The load_from_gpu_pointers() function (line 280) loaded ffn_gate and ffn_up
// but FORGOT to load ffn_down! This would cause the FFN down projection to use
// uninitialized memory (garbage).
//
// THE FIX (line 389):
//   layer.ffn_down = get_ptr(prefix + "ffn_down.weight");
//
// WHY I THOUGHT THIS WAS THE BUG:
// 1. The load() function (line 224) correctly loads all 4 FFN weights
// 2. The struct definition includes ffn_down
// 3. The code compiles without errors
// 4. The program would run without crashing
// 5. But FFN output would be garbage due to uninitialized memory
//
// COMPARISON:
// load() function (line 256-259):          load_from_gpu_pointers() (line 320-389):
//   ✅ ffn_norm                               ✅ ffn_norm
//   ✅ ffn_gate                               ✅ ffn_gate
//   ✅ ffn_up                                 ✅ ffn_up
//   ✅ ffn_down                               ✅ ffn_down (ADDED - FIX APPLIED)
//
// TEST RESULT: ❌ Garbage tokens persist! This fixed a real issue but wasn't THE bug.
// The model still outputs mojibake and code tokens after this fix.
//
// CONCLUSION: This line was genuinely missing (good fix), but the garbage token bug
// is elsewhere. Bug is likely in attention mechanism or KV cache usage.
//
// See: investigation-teams/TEAM_CHARLIE_BETA_FALSE_ALARM.md
// ============================================================================

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
            // TEAM FREE [Review]
            // Category: Data parsing
            // Hypothesis: GGUF quantized tensor types (e.g., Q4_K_M) use blockwise layouts; treating them as 2 bytes/element miscomputes size_bytes.
            // Evidence: `type >= 2 && type <= 14` coerced to 2 bytes; QK* formats have headers/scales + packed nibbles → bytes per element varies by block.
            // Risk: Under/over-read when copying tensor data → corrupted weights or OOB file reads; downstream CUDA copies load garbage.
            // Confidence: High
            // Next step: Use GGUF type table to compute exact byte size per tensor (block size × block count) or read recorded byte size from header if available.
            
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
    // TEAM FREE [Review]
    // Category: Numerical correctness
    // Hypothesis: Copying quantized GGUF weights directly to GPU without dequantization yields invalid activations.
    // Evidence: Explicit warning above; Q4_K_M requires dequant into FP16/FP32 before GEMMs; downstream code assumes `half*` weights.
    // Risk: Systematic wrong logits, NaNs, or unstable attention; parity with llama.cpp impossible.
    // Confidence: High
    // Next step: Implement dequantization for supported GGUF types or ensure Rust pre-dequantizes before wiring pointers.
    
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
    // [TEAM VAN GOGH 2025-10-07] Read output_norm wiring here (evidence in VAN_GOGH report)
    // This loads output_norm.weight RAW from GGUF file without any normalization
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
        // [TEAM GREEN] 2025-10-06T20:43Z - BUG FOUND!
        // SUSPECT: We were setting biases to nullptr, but the model HAS biases!
        // OBSERVED: Test output shows "blk.0.attn_q.bias -> 0x7dfbaa5c1200"
        // FIXED: Load the biases from GPU pointers instead of nullptr
        layer.attn_q_bias = get_ptr(prefix + "attn_q.bias");
        layer.attn_k_weight = get_ptr(prefix + "attn_k.weight");
        layer.attn_k_bias = get_ptr(prefix + "attn_k.bias");
        layer.attn_v_weight = get_ptr(prefix + "attn_v.weight");
        layer.attn_v_bias = get_ptr(prefix + "attn_v.bias");
        layer.attn_output = get_ptr(prefix + "attn_output.weight");
        
        layer.ffn_norm = get_ptr(prefix + "ffn_norm.weight");
        layer.ffn_gate = get_ptr(prefix + "ffn_gate.weight");
        layer.ffn_up = get_ptr(prefix + "ffn_up.weight");
        // [TEAM_CHARLIE_BETA] MISSING WEIGHT LOADING - FIXED BUT NOT THE BUG (2025-10-06 17:07 UTC)
        // I found this line was MISSING and thought it was THE bug!
        // HYPOTHESIS: ffn_down not loaded → uninitialized memory → repetitive tokens
        // FIX APPLIED: Added the missing line below
        // TEST RESULT: ❌ Garbage tokens persist! This fixed a real issue but wasn't THE bug.
        // CONCLUSION: This line was genuinely missing (good fix), but the garbage token bug
        // is elsewhere. The model still outputs mojibake and code tokens.
        // NOTE: First 3 tokens DO work ("separately", "epoch", "aws"), then breaks!
        // → Bug is position-dependent, likely in attention mechanism or KV cache usage!
        layer.ffn_down = get_ptr(prefix + "ffn_down.weight");
    }
    
    // Wire output
    // [TEAM MONET 2025-10-07T14:22Z] Checked line 393: output_norm loaded raw (no normalization) ⚠️
    // [TEAM VAN GOGH 2025-10-07] Read output_norm wiring here (evidence in VAN_GOGH report)
    // This wires pre-loaded output_norm.weight pointer - weights come from Rust loader (RAW, no normalization)
    model->weights.output_norm = get_ptr("output_norm.weight");
    model->weights.lm_head = get_ptr("output.weight");
    
    // [TEAM VAN GOGH 2025-10-07T22:43Z] A/B Testing: RAW vs NORMALIZED
    // Check environment variable to enable weight normalization for testing
    const char* normalize_env = std::getenv("VAN_GOGH_NORMALIZE_OUTPUT_NORM");
    if (normalize_env != nullptr && std::string(normalize_env) == "1") {
        fprintf(stderr, "\n[TEAM VAN GOGH] ⚠️  NORMALIZING output_norm.weight for A/B testing\n");
        
        // Copy weights from GPU to host
        half h_weights[896];
        cudaError_t err = cudaMemcpy(h_weights, model->weights.output_norm, 
                                      config.hidden_dim * sizeof(half), 
                                      cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[TEAM VAN GOGH] ❌ Failed to copy output_norm for normalization: %s\n", 
                    cudaGetErrorString(err));
        } else {
            // Calculate mean
            float sum = 0.0f;
            for (uint32_t i = 0; i < config.hidden_dim; i++) {
                sum += __half2float(h_weights[i]);
            }
            float mean = sum / config.hidden_dim;
            
            fprintf(stderr, "[TEAM VAN GOGH] Original mean: %.6f\n", mean);
            
            // Normalize to mean=1.0
            float scale = 1.0f / mean;
            for (uint32_t i = 0; i < config.hidden_dim; i++) {
                float val = __half2float(h_weights[i]);
                h_weights[i] = __float2half(val * scale);
            }
            
            // Calculate new mean
            sum = 0.0f;
            for (uint32_t i = 0; i < config.hidden_dim; i++) {
                sum += __half2float(h_weights[i]);
            }
            float new_mean = sum / config.hidden_dim;
            
            fprintf(stderr, "[TEAM VAN GOGH] Normalized mean: %.6f (scale factor: %.6f)\n", 
                    new_mean, scale);
            fprintf(stderr, "[TEAM VAN GOGH] First 10 normalized values: ");
            for (int i = 0; i < 10; i++) {
                fprintf(stderr, "%.4f ", __half2float(h_weights[i]));
            }
            fprintf(stderr, "\n");
            
            // Copy back to GPU
            err = cudaMemcpy(model->weights.output_norm, h_weights, 
                           config.hidden_dim * sizeof(half), 
                           cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "[TEAM VAN GOGH] ❌ Failed to copy normalized weights back: %s\n", 
                        cudaGetErrorString(err));
            } else {
                fprintf(stderr, "[TEAM VAN GOGH] ✅ Normalized weights copied back to GPU\n\n");
            }
        }
    }
    
    // [TEAM VANGUARD] 2025-10-07T20:04Z
    // OBJECTIVE 1: Weight Integrity Verification
    // PLAN: Dump first 100 FP16 values from GPU memory for critical tensors
    // Compare byte-for-byte with llama.cpp to find dequantization bugs
    fprintf(stderr, "\n[TEAM VANGUARD] === WEIGHT INTEGRITY VERIFICATION ===\n");
    fprintf(stderr, "[TEAM VANGUARD] Dumping first 100 FP16 values from GPU memory for layer 0...\n\n");
    
    // Helper to dump first N values from GPU pointer
    auto dump_gpu_weights = [](const char* name, void* gpu_ptr, int count) {
        half h_weights[100];
        cudaError_t err = cudaMemcpy(h_weights, gpu_ptr, count * sizeof(half), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "❌ Failed to copy %s: %s\n", name, cudaGetErrorString(err));
            return;
        }
        
        fprintf(stderr, "[TEAM VANGUARD] %s (first %d FP16 values):\n", name, count);
        fprintf(stderr, "  Floats: ");
        for (int i = 0; i < count; i++) {
            fprintf(stderr, "%.6f ", __half2float(h_weights[i]));
            if ((i + 1) % 10 == 0 && i < count - 1) fprintf(stderr, "\n          ");
        }
        fprintf(stderr, "\n");
        
        // Also dump raw bytes for exact comparison
        fprintf(stderr, "  Bytes:  ");
        uint16_t* bytes = reinterpret_cast<uint16_t*>(h_weights);
        for (int i = 0; i < count; i++) {
            fprintf(stderr, "%04x ", bytes[i]);
            if ((i + 1) % 10 == 0 && i < count - 1) fprintf(stderr, "\n          ");
        }
        fprintf(stderr, "\n\n");
        
        // Calculate stats
        float sum = 0.0f, min_val = 1e9f, max_val = -1e9f;
        for (int i = 0; i < count; i++) {
            float val = __half2float(h_weights[i]);
            sum += val;
            min_val = fmin(min_val, val);
            max_val = fmax(max_val, val);
        }
        float mean = sum / count;
        fprintf(stderr, "  Stats: mean=%.6f, min=%.6f, max=%.6f\n\n", mean, min_val, max_val);
    };
    
    // Dump critical tensors from layer 0
    dump_gpu_weights("blk.0.attn_q.weight", model->weights.layers[0].attn_q_weight, 100);
    dump_gpu_weights("blk.0.attn_k.weight", model->weights.layers[0].attn_k_weight, 100);
    dump_gpu_weights("blk.0.attn_v.weight", model->weights.layers[0].attn_v_weight, 100);
    dump_gpu_weights("blk.0.attn_output.weight", model->weights.layers[0].attn_output, 100);
    dump_gpu_weights("blk.0.ffn_gate.weight", model->weights.layers[0].ffn_gate, 100);
    dump_gpu_weights("blk.0.ffn_up.weight", model->weights.layers[0].ffn_up, 100);
    dump_gpu_weights("blk.0.ffn_down.weight", model->weights.layers[0].ffn_down, 100);
    // [TEAM VAN GOGH 2025-10-07] Added output_norm.weight dump to investigate 16.75× amplification
    dump_gpu_weights("output_norm.weight", model->weights.output_norm, 100);
    dump_gpu_weights("output.weight", model->weights.lm_head, 100);
    
    fprintf(stderr, "[TEAM VANGUARD] Weight dump complete. Compare these with llama.cpp output.\n");
    fprintf(stderr, "[TEAM VANGUARD] Next: Run llama.cpp with same model and dump same tensors.\n\n");
    
    fprintf(stderr, "✅ [C++] Wired all %u layers (VRAM: %.2f MB)\n",
            config.num_layers,
            total_vram_bytes / 1024.0 / 1024.0);
    
    return model;
}

} // namespace model
} // namespace worker
