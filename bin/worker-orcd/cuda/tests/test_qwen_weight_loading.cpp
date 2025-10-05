#include "../src/model/qwen_weight_loader.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda_runtime.h>

using namespace worker::model;

int main() {
    const char* model_path = "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    // Check if file exists
    std::ifstream test_file(model_path);
    if (!test_file) {
        std::cerr << "⚠️  Model file not found, skipping test: " << model_path << std::endl;
        return 0;
    }
    test_file.close();
    
    std::cout << "Testing Qwen2.5-0.5B weight loading..." << std::endl;
    
    // Initialize CUDA
    cudaSetDevice(0);
    
    // Create config
    QwenConfig config;
    config.vocab_size = 151936;
    config.hidden_dim = 896;
    config.num_layers = 24;
    config.num_heads = 14;
    config.num_kv_heads = 2;
    config.context_length = 32768;
    
    try {
        // Load model
        std::cout << "Loading weights from GGUF..." << std::endl;
        QwenModel* model = QwenWeightLoader::load(model_path, config);
        
        std::cout << "✅ Model loaded successfully!" << std::endl;
        std::cout << "   VRAM usage: " << (model->vram_usage / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "   Layers: " << model->weights.layers.size() << std::endl;
        
        // Verify pointers are not null
        assert(model->weights.token_embd != nullptr);
        assert(model->weights.output_norm != nullptr);
        assert(model->weights.lm_head != nullptr);
        assert(model->weights.layers.size() == 24);
        
        for (size_t i = 0; i < model->weights.layers.size(); i++) {
            assert(model->weights.layers[i].attn_q_weight != nullptr);
            assert(model->weights.layers[i].attn_k_weight != nullptr);
            assert(model->weights.layers[i].attn_v_weight != nullptr);
        }
        
        std::cout << "✅ All tensor pointers valid!" << std::endl;
        
        // Check VRAM usage is reasonable
        // Note: Currently estimating 2 bytes per element, actual Q4_K_M would be smaller
        assert(model->vram_usage > 400'000'000);  // At least 400 MB
        assert(model->vram_usage < 2'000'000'000);  // Less than 2 GB
        
        std::cout << "✅ VRAM usage in expected range!" << std::endl;
        std::cout << "✅ ALL TESTS PASSED!" << std::endl;
        
        // Cleanup
        delete model;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
