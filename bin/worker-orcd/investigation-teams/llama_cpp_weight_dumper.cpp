//
// [APPEND-ONLY GUARD] Do not delete prior teams’ comments. Add new notes below existing blocks.
//
// [TEAM SENTINEL] 2025-10-07T22:56Z
// llama.cpp Weight Dumper for FP16 Weight Parity Verification
// Purpose: Dump first 100 FP16 values from llama.cpp loaded weights
// Compare with our FP16 weights byte-for-byte to find loading bugs

#include <llama.h>
#include <ggml.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <stdint.h>
#include <cmath>
#include <cstring>

// Compile: g++ -o dump_weights llama_cpp_weight_dumper.cpp -I../../../reference/llama.cpp/include -I../../../reference/llama.cpp/ggml/include -L../../../reference/llama.cpp/build/bin -lllama -Wl,-rpath,../../../reference/llama.cpp/build/bin -std=c++11

// [TEAM SENTINEL] 2025-10-07T22:56Z
// PLAN: Convert FP16 to float for comparison
// GGML uses uint16_t for FP16 storage, convert to float for display
static inline float fp16_to_float(uint16_t h) {
    // Extract sign, exponent, mantissa
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    // Handle special cases
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Subnormal
        float f = mant / 1024.0f;
        return sign ? -f * powf(2.0f, -14.0f) : f * powf(2.0f, -14.0f);
    }
    if (exp == 31) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    
    // Normalized
    uint32_t f32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

// [TEAM SENTINEL] 2025-10-07T22:56Z
// OBJECTIVE: Dump FP16 tensor values in exact same format as our code
// - Shows first 100 FP16 values as floats
// - Shows hex bytes for byte-by-byte comparison
// - Shows min/max/mean statistics
void dump_tensor_values(struct llama_model * model, const char * tensor_name, int count) {
    // Get tensor from model
    struct ggml_tensor * tensor = llama_model_get_tensor(model, tensor_name);
    if (!tensor) {
        fprintf(stderr, "❌ Tensor not found: %s\n", tensor_name);
        return;
    }
    
    fprintf(stderr, "\n[LLAMA.CPP] %s (first %d values):\n", tensor_name, count);
    fprintf(stderr, "  Type: %d (", tensor->type);
    if (tensor->type == GGML_TYPE_F16) fprintf(stderr, "FP16");
    else if (tensor->type == GGML_TYPE_F32) fprintf(stderr, "FP32");
    else fprintf(stderr, "QUANTIZED");
    fprintf(stderr, "), Dims: [");
    for (int i = 0; i < tensor->n_dims; i++) {
        fprintf(stderr, "%lld", (long long)tensor->ne[i]);
        if (i < tensor->n_dims - 1) fprintf(stderr, ", ");
    }
    fprintf(stderr, "]\n");
    
    // Get pointer to data
    void * data = tensor->data;
    if (!data) {
        fprintf(stderr, "❌ Tensor data is NULL\n");
        return;
    }
    
    // [TEAM SENTINEL] 2025-10-07T22:56Z
    // SUSPECT: FP16 model should have GGML_TYPE_F16 (type==1)
    // If type is quantized, we need different approach
    if (tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_F32) {
        fprintf(stderr, "  ⚠️  WARNING: Tensor is quantized (type=%d), need dequantization\n", tensor->type);
        fprintf(stderr, "  PLAN: This tool currently only handles FP16/FP32 tensors\n");
        return;
    }
    
    // Dump FP16 values
    uint16_t * fp16_data = (uint16_t *)data;
    
    fprintf(stderr, "  Floats: ");
    float sum = 0.0f, min_val = INFINITY, max_val = -INFINITY;
    for (int i = 0; i < count && i < (tensor->nbytes / 2); i++) {
        float val = fp16_to_float(fp16_data[i]);
        fprintf(stderr, "%.6f ", val);
        if ((i + 1) % 10 == 0 && i < count - 1) fprintf(stderr, "\n          ");
        
        sum += val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    fprintf(stderr, "\n");
    
    // Dump raw hex bytes
    fprintf(stderr, "  Bytes:  ");
    for (int i = 0; i < count && i < (tensor->nbytes / 2); i++) {
        fprintf(stderr, "%04x ", fp16_data[i]);
        if ((i + 1) % 10 == 0 && i < count - 1) fprintf(stderr, "\n          ");
    }
    fprintf(stderr, "\n");
    
    // Statistics
    float mean = sum / count;
    fprintf(stderr, "  Stats: mean=%.6f, min=%.6f, max=%.6f\n\n", mean, min_val, max_val);
}

// [TEAM SENTINEL] 2025-10-07T22:56Z
// MAIN: Load FP16 model and dump same tensors as our code
int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        fprintf(stderr, "Example: %s qwen2.5-0.5b-instruct-fp16.gguf\n", argv[0]);
        return 1;
    }
    
    const char * model_path = argv[1];
    
    fprintf(stderr, "=== [TEAM SENTINEL] LLAMA.CPP FP16 WEIGHT DUMPER ===\n");
    fprintf(stderr, "Loading model: %s\n", model_path);
    fprintf(stderr, "PLAN: Dump first 100 FP16 values from 8 critical tensors\n");
    fprintf(stderr, "OBJECTIVE: Compare byte-for-byte with our weight loading\n\n");
    
    // Initialize llama backend
    llama_backend_init();
    
    // [TEAM SENTINEL] 2025-10-07T22:56Z
    // PLAN: Load model on CPU to inspect weight values after llama.cpp's loading
    // llama.cpp may apply transformations during load - we need to match those
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
    
    // [TEAM SENTINEL] 2025-10-07T22:56Z
    // OBJECTIVE: Dump exact same tensors that our code dumps (qwen_weight_loader.cpp lines 428-435)
    // These are the 8 critical tensors for layer 0 forward pass
    fprintf(stderr, "=== DUMPING LAYER 0 TENSORS (FP16) ===\n");
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
    
    fprintf(stderr, "\n=== [TEAM SENTINEL] COMPARISON INSTRUCTIONS ===\n");
    fprintf(stderr, "1. Build this tool:\n");
    fprintf(stderr, "   cd investigation-teams\n");
    fprintf(stderr, "   g++ -o dump_weights llama_cpp_weight_dumper.cpp \\\n");
    fprintf(stderr, "       -I../../reference/llama.cpp/include \\\n");
    fprintf(stderr, "       -I../../reference/llama.cpp/ggml/include \\\n");
    fprintf(stderr, "       -L../../reference/llama.cpp/build/src \\\n");
    fprintf(stderr, "       -lllama -Wl,-rpath,../../reference/llama.cpp/build/src\n\n");
    fprintf(stderr, "2. Run llama.cpp dumper:\n");
    fprintf(stderr, "   ./dump_weights /path/to/qwen2.5-0.5b-instruct-fp16.gguf 2>&1 | tee llama_weights.txt\n\n");
    fprintf(stderr, "3. Run our test (captures our dumps to stderr):\n");
    fprintf(stderr, "   REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \\\n");
    fprintf(stderr, "       --test haiku_generation_anti_cheat \\\n");
    fprintf(stderr, "       test_haiku_generation_stub_pipeline_only \\\n");
    fprintf(stderr, "       -- --ignored --nocapture --test-threads=1 2>&1 | tee our_weights.txt\n\n");
    fprintf(stderr, "4. Compare hex bytes for each tensor:\n");
    fprintf(stderr, "   grep -A 15 'blk.0.attn_q.weight' llama_weights.txt > llama_q.txt\n");
    fprintf(stderr, "   grep -A 15 'blk.0.attn_q.weight' our_weights.txt > our_q.txt\n");
    fprintf(stderr, "   diff -u llama_q.txt our_q.txt\n\n");
    fprintf(stderr, "5. If hex bytes match → weight loading is correct\n");
    fprintf(stderr, "   If hex bytes differ → found the FP16 loading bug!\n\n");
    
    return 0;
}
