// llama.cpp Weight Dumper for Team VANGUARD
// Purpose: Dump first 100 FP16 values from llama.cpp loaded weights
// Compare with our dequantized weights to find bugs

#include "../../reference/llama.cpp/include/llama.h"
#include <stdio.h>
#include <vector>
#include <string>

// Compile: g++ -o dump_weights llama_cpp_weight_dumper.cpp -I../../reference/llama.cpp/include -L../../reference/llama.cpp/build/src -lllama -Wl,-rpath,../../reference/llama.cpp/build/src

void dump_tensor_values(struct llama_model * model, const char * tensor_name, int count) {
    // Get tensor from model
    struct ggml_tensor * tensor = llama_model_get_tensor(model, tensor_name);
    if (!tensor) {
        fprintf(stderr, "❌ Tensor not found: %s\n", tensor_name);
        return;
    }
    
    fprintf(stderr, "\n[LLAMA.CPP] %s (first %d FP16 values after dequant):\n", tensor_name, count);
    fprintf(stderr, "  Type: %d, Dims: [", tensor->type);
    for (int i = 0; i < tensor->n_dims; i++) {
        fprintf(stderr, "%lld", (long long)tensor->ne[i]);
        if (i < tensor->n_dims - 1) fprintf(stderr, ", ");
    }
    fprintf(stderr, "]\n");
    
    // llama.cpp stores weights as ggml_tensor, possibly quantized
    // We need to dequantize to FP32/FP16 for comparison
    // For now, just dump raw bytes
    
    size_t nelements = 1;
    for (int i = 0; i < tensor->n_dims; i++) {
        nelements *= tensor->ne[i];
    }
    
    // Get pointer to data
    void * data = tensor->data;
    if (!data) {
        fprintf(stderr, "❌ Tensor data is NULL\n");
        return;
    }
    
    // Dump first N values
    // Note: This assumes data is already in FP16 or we need to dequantize
    // For Q4_K, we need to dequantize first - this is complex
    
    fprintf(stderr, "  WARNING: This tool needs to be extended to handle Q4_K dequantization\n");
    fprintf(stderr, "  For now, showing raw bytes (first 100 bytes):\n  ");
    
    unsigned char * bytes = (unsigned char *)data;
    for (int i = 0; i < 100 && i < tensor->nbytes; i++) {
        fprintf(stderr, "%02x ", bytes[i]);
        if ((i + 1) % 16 == 0) fprintf(stderr, "\n  ");
    }
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    
    const char * model_path = argv[1];
    
    fprintf(stderr, "=== LLAMA.CPP WEIGHT DUMPER ===\n");
    fprintf(stderr, "Loading model: %s\n\n", model_path);
    
    // Initialize llama backend
    llama_backend_init();
    
    // Set model params
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU only for weight inspection
    
    // Load model
    struct llama_model * model = llama_load_model_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "❌ Failed to load model\n");
        llama_backend_free();
        return 1;
    }
    
    fprintf(stderr, "✅ Model loaded successfully\n\n");
    
    // Dump the same tensors that our code dumps
    dump_tensor_values(model, "blk.0.attn_q.weight", 100);
    dump_tensor_values(model, "blk.0.attn_k.weight", 100);
    dump_tensor_values(model, "blk.0.attn_v.weight", 100);
    dump_tensor_values(model, "blk.0.attn_output.weight", 100);
    dump_tensor_values(model, "blk.0.ffn_gate.weight", 100);
    dump_tensor_values(model, "blk.0.ffn_up.weight", 100);
    dump_tensor_values(model, "blk.0.ffn_down.weight", 100);
    dump_tensor_values(model, "output.weight", 100);
    
    // Cleanup
    llama_free_model(model);
    llama_backend_free();
    
    fprintf(stderr, "\n=== COMPARISON INSTRUCTIONS ===\n");
    fprintf(stderr, "1. Run this tool: ./dump_weights model.gguf > llama_weights.txt\n");
    fprintf(stderr, "2. Run our code: cargo test --release --features cuda 2> our_weights.txt\n");
    fprintf(stderr, "3. Compare: diff -u llama_weights.txt our_weights.txt\n");
    fprintf(stderr, "4. Look for mismatches in first 100 values of each tensor\n\n");
    
    return 0;
}
