// ============================================================================
// [TEAM PRINTER] 2025-10-07T01:24Z - Checkpoint Logger for Parity Analysis
// ============================================================================
// MISSION: Collect clean, append-only checkpoint data for comparison with llama.cpp
// RULES: Non-invasive logging only, no math changes, FP32 output for precision
// ============================================================================

#ifndef CHECKPOINT_LOGGER_H
#define CHECKPOINT_LOGGER_H

#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

namespace team_printer {

// Global flag to enable checkpoint logging (set via environment variable)
extern bool g_checkpoint_logging_enabled;
extern int g_checkpoint_token_limit;  // Only log first N tokens
extern std::string g_checkpoint_output_path;

// Initialize checkpoint logging system
void init_checkpoint_logging();

// Checkpoint storage structure
struct Checkpoint {
    std::string name;
    int token_idx;
    std::vector<float> data;  // Always FP32 for precision
    
    Checkpoint(const std::string& n, int tok_idx) 
        : name(n), token_idx(tok_idx) {}
};

// Global checkpoint storage
extern std::vector<Checkpoint> g_checkpoints;

// Log a checkpoint from device memory (FP16 -> FP32 conversion)
inline void log_checkpoint_fp16(const char* name, int token_idx, const half* device_ptr, size_t count) {
    if (!g_checkpoint_logging_enabled) return;
    if (token_idx >= g_checkpoint_token_limit) return;
    
    // Copy from device to host
    std::vector<half> h_data(count);
    cudaMemcpy(h_data.data(), device_ptr, count * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Convert to FP32
    Checkpoint cp(name, token_idx);
    cp.data.reserve(count);
    for (size_t i = 0; i < count; i++) {
        cp.data.push_back(__half2float(h_data[i]));
    }
    
    g_checkpoints.push_back(std::move(cp));
    
    // Log summary to stderr
    float min_val = cp.data[0], max_val = cp.data[0], sum = 0.0f;
    for (float v : cp.data) {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
    }
    float mean = sum / count;
    
    fprintf(stderr, "[TEAM PRINTER] %s_tok%d: count=%zu, min=%.6f, max=%.6f, mean=%.6f\n",
            name, token_idx, count, min_val, max_val, mean);
}

// Log a checkpoint from device memory (FP32, no conversion needed)
inline void log_checkpoint_fp32(const char* name, int token_idx, const float* device_ptr, size_t count) {
    if (!g_checkpoint_logging_enabled) return;
    if (token_idx >= g_checkpoint_token_limit) return;
    
    Checkpoint cp(name, token_idx);
    cp.data.resize(count);
    cudaMemcpy(cp.data.data(), device_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
    
    g_checkpoints.push_back(std::move(cp));
    
    // Log summary to stderr
    float min_val = cp.data[0], max_val = cp.data[0], sum = 0.0f;
    for (float v : cp.data) {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
    }
    float mean = sum / count;
    
    fprintf(stderr, "[TEAM PRINTER] %s_tok%d: count=%zu, min=%.6f, max=%.6f, mean=%.6f\n",
            name, token_idx, count, min_val, max_val, mean);
}

// Log token IDs (special case for embedding input)
inline void log_token_ids(const char* name, int token_idx, const std::vector<uint32_t>& token_ids) {
    if (!g_checkpoint_logging_enabled) return;
    if (token_idx >= g_checkpoint_token_limit) return;
    
    Checkpoint cp(name, token_idx);
    for (uint32_t id : token_ids) {
        cp.data.push_back(static_cast<float>(id));
    }
    
    g_checkpoints.push_back(std::move(cp));
    
    fprintf(stderr, "[TEAM PRINTER] %s_tok%d: token_ids=[", name, token_idx);
    for (size_t i = 0; i < token_ids.size() && i < 10; i++) {
        fprintf(stderr, "%u%s", token_ids[i], (i < token_ids.size()-1) ? ", " : "");
    }
    fprintf(stderr, "]\n");
}

// Save all checkpoints to numpy .npz file
void save_checkpoints_to_npz(const std::string& output_path);

// Finalize and save checkpoints
void finalize_checkpoint_logging();

} // namespace team_printer

#endif // CHECKPOINT_LOGGER_H
