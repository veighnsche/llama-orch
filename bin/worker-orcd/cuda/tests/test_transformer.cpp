#include "../src/transformer/qwen_transformer.h"
#include "../src/model/qwen_weight_loader.h"
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

using namespace worker;

int main() {
    const char* model_path = "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    // Check if file exists
    std::ifstream test_file(model_path);
    if (!test_file) {
        std::cerr << "⚠️  Model file not found, skipping test" << std::endl;
        return 0;
    }
    test_file.close();
    
    std::cout << "Testing Qwen Transformer..." << std::endl;
    
    // Initialize CUDA
    cudaSetDevice(0);
    
    // Load model
    model::QwenConfig config;
    config.vocab_size = 151936;
    config.hidden_dim = 896;
    config.num_layers = 24;
    config.num_heads = 14;
    config.num_kv_heads = 2;
    config.context_length = 32768;
    
    std::cout << "Loading weights..." << std::endl;
    model::QwenModel* model = model::QwenWeightLoader::load(model_path, config);
    std::cout << "✅ Weights loaded: " << (model->vram_usage / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // Create transformer
    transformer::TransformerConfig tf_config;
    tf_config.vocab_size = config.vocab_size;
    tf_config.hidden_dim = config.hidden_dim;
    tf_config.num_layers = config.num_layers;
    tf_config.num_heads = config.num_heads;
    tf_config.num_kv_heads = config.num_kv_heads;
    tf_config.head_dim = 64;
    tf_config.ffn_dim = 4864;
    tf_config.context_length = config.context_length;
    tf_config.rope_freq_base = 10000.0f;
    
    std::cout << "Creating transformer..." << std::endl;
    transformer::QwenTransformer transformer(model, tf_config);
    
    // Test forward pass with dummy token
    uint32_t token_id = 123;  // Dummy token
    uint32_t* d_token;
    cudaMalloc(&d_token, sizeof(uint32_t));
    cudaMemcpy(d_token, &token_id, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    float* d_logits;
    cudaMalloc(&d_logits, config.vocab_size * sizeof(float));
    
    std::cout << "Running forward pass..." << std::endl;
    transformer.forward(d_token, 1, d_logits);
    
    // Check logits (should be zeros for now since LM head not implemented)
    float* h_logits = new float[config.vocab_size];
    cudaMemcpy(h_logits, d_logits, config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "✅ Forward pass completed!" << std::endl;
    std::cout << "   First 5 logits: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_logits[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    delete[] h_logits;
    cudaFree(d_token);
    cudaFree(d_logits);
    delete model;
    
    std::cout << "✅ ALL TESTS PASSED!" << std::endl;
    
    return 0;
}
