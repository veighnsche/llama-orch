#pragma once

#include "model/qwen_weight_loader.h"

namespace worker {

// Minimal model wrapper - holds loaded QwenModel
class ModelImpl {
public:
    ModelImpl() : qwen_model_(nullptr), vram_bytes_(0) {}
    
    ~ModelImpl() {
        // QwenModel is owned and freed by weight loader
    }
    
    void set_qwen_model(model::QwenModel* model) {
        qwen_model_ = model;
    }
    
    model::QwenModel* get_qwen_model() const {
        return qwen_model_;
    }
    
    void set_vram_bytes(uint64_t bytes) {
        vram_bytes_ = bytes;
    }
    
    uint64_t vram_bytes() const {
        return vram_bytes_;
    }

private:
    model::QwenModel* qwen_model_;
    uint64_t vram_bytes_;
};

} // namespace worker
