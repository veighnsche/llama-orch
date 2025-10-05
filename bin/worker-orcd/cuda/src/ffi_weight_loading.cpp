#include "model/qwen_weight_loader.h"
#include <cstdio>

extern "C" {

struct CudaModel {
    worker::model::QwenModel* qwen_model;
};

CudaModel* cuda_load_model(
    void* ctx,
    const char* path,
    uint32_t vocab_size,
    uint32_t hidden_dim,
    uint32_t num_layers,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t context_length,
    int* error
) {
    try {
        fprintf(stderr, "Loading model from: %s\n", path);
        fprintf(stderr, "Config: vocab=%u, hidden=%u, layers=%u, heads=%u, kv_heads=%u\n",
                vocab_size, hidden_dim, num_layers, num_heads, num_kv_heads);
        
        worker::model::QwenConfig config;
        config.vocab_size = vocab_size;
        config.hidden_dim = hidden_dim;
        config.num_layers = num_layers;
        config.num_heads = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.context_length = context_length;
        
        auto qwen_model = worker::model::QwenWeightLoader::load(path, config);
        
        auto cuda_model = new CudaModel();
        cuda_model->qwen_model = qwen_model;
        
        *error = 0;
        return cuda_model;
    } catch (const std::exception& e) {
        fprintf(stderr, "❌ Model load failed: %s\n", e.what());
        *error = -1;
        return nullptr;
    }
}

uint64_t cuda_get_vram_usage(CudaModel* model) {
    if (!model || !model->qwen_model) {
        return 0;
    }
    return model->qwen_model->vram_usage;
}

void cuda_free_model(CudaModel* model) {
    if (model) {
        // TODO: Free all GPU allocations
        if (model->qwen_model) {
            delete model->qwen_model;
        }
        delete model;
    }
}

} // extern "C"
