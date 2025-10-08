// TEAM-006: Implementation of checkpoint extraction callback
// Created by: TEAM-006
// Based on: TEAM-005 comprehensive analysis

#include "checkpoint_callback.h"
#include <cstdio>
#include <cstring>

namespace llorch {

// Checkpoint name mapping (from Phase 2)
static const char* CHECKPOINT_NAMES[] = {
    "attn_norm",      // Checkpoint 1: LayerNorm
    "Qcur",           // Checkpoint 2: Q
    "Kcur",           // Checkpoint 2: K
    "Vcur",           // Checkpoint 2: V
    "cache_k",        // Checkpoint 3: KV cache K
    "cache_v",        // Checkpoint 3: KV cache V
    "kq_soft_max",    // Checkpoint 4: Attention scores
    "attn_out_proj",  // Checkpoint 5: Attention output
    "ffn_out",        // Checkpoint 6: FFN output
};

bool is_checkpoint_tensor(const char * name) {
    if (!name) return false;
    
    for (const char* cp_name : CHECKPOINT_NAMES) {
        if (strcmp(name, cp_name) == 0) {
            return true;
        }
    }
    return false;
}

bool checkpoint_eval_callback(
    struct ggml_tensor * t,
    bool ask,
    void * user_data
) {
    if (ask) return true;  // Always allow execution
    
    auto * state = static_cast<CheckpointState*>(user_data);
    const char * name = ggml_get_name(t);
    
    if (!name || !is_checkpoint_tensor(name)) {
        return true;
    }
    
    // Check if already extracted (avoid duplicates)
    std::string key = std::string(name);
    if (state->extracted.count(key)) {
        return true;
    }
    
    // Extract checkpoint
    save_checkpoint(name, t, state->output_dir);
    state->extracted.insert(key);
    
    return true;
}

void save_checkpoint(
    const char * name,
    struct ggml_tensor * t,
    const std::string & output_dir
) {
    // Build filename
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/checkpoint_%s.bin", 
             output_dir.c_str(), name);
    
    FILE * f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "❌ TEAM-006: Failed to open %s\n", filename);
        return;
    }
    
    // Write shape metadata
    int32_t n_dims = ggml_n_dims(t);
    fwrite(&n_dims, sizeof(int32_t), 1, f);
    
    for (int i = 0; i < n_dims; i++) {
        int64_t dim = t->ne[i];
        fwrite(&dim, sizeof(int64_t), 1, f);
    }
    
    // Get tensor data (handles both CPU and GPU)
    size_t n_elements = ggml_nelements(t);
    float * data = new float[n_elements];
    ggml_backend_tensor_get(t, data, 0, n_elements * sizeof(float));
    
    // Write data
    fwrite(data, sizeof(float), n_elements, f);
    delete[] data;
    fclose(f);
    
    // Log success
    fprintf(stderr, "✅ TEAM-006: %s [", name);
    for (int i = 0; i < n_dims; i++) {
        fprintf(stderr, "%lld%s", (long long)t->ne[i], 
                i < n_dims-1 ? " × " : "");
    }
    fprintf(stderr, "] → %s\n", filename);
}

} // namespace llorch
