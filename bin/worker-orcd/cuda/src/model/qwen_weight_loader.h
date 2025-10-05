#ifndef WORKER_MODEL_QWEN_WEIGHT_LOADER_H
#define WORKER_MODEL_QWEN_WEIGHT_LOADER_H

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace worker {

// Forward declarations
class VramTracker;

namespace model {

struct QwenConfig {
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t context_length;
};

struct QwenWeights {
    // Embeddings
    void* token_embd;  // [vocab_size, hidden_dim]
    
    // Per-layer weights
    struct Layer {
        // Attention
        void* attn_norm;
        void* attn_q_weight;
        void* attn_q_bias;
        void* attn_k_weight;
        void* attn_k_bias;
        void* attn_v_weight;
        void* attn_v_bias;
        void* attn_output;
        
        // FFN
        void* ffn_norm;
        void* ffn_gate;
        void* ffn_up;
        void* ffn_down;
    };
    
    std::vector<Layer> layers;
    
    // Output
    void* output_norm;
    void* lm_head;
};

struct QwenModel {
    QwenConfig config;
    QwenWeights weights;
    uint64_t vram_usage;
};

class QwenWeightLoader {
public:
    static QwenModel* load(
        const char* path,
        const QwenConfig& config
    );
    
private:
    static std::vector<std::string> get_tensor_names(int num_layers);
    
    static void* load_tensor_to_vram(
        const char* path,
        const std::string& tensor_name,
        VramTracker& tracker
    );
};

} // namespace model
} // namespace worker

#endif
