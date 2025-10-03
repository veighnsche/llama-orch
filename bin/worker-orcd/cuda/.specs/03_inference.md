# CUDA Inference Execution (CUDA-5300)

**Status**: Draft  
**Module**: `inference`  
**Files**: `src/inference.cu`, `include/inference.hpp`, `kernels/*.cu`  
**Conformance**: RFC-2119

---

## 0. Scope

The inference module executes LLM inference on GPU, generating tokens one-by-one via CUDA kernels. It manages KV cache, runs forward passes, and samples tokens.

**Parent**: `00_cuda_overview.md`

---

## 1. Responsibilities

### [CUDA-5301] Token Generation
The module MUST generate tokens one-by-one via autoregressive inference.

### [CUDA-5302] KV Cache Management
The module MUST allocate and manage KV cache in VRAM for each inference session.

### [CUDA-5303] Reproducible Sampling for Testing
The module MUST produce reproducible results for testing when given same seed, prompt, and temperature=0.0.

**Clarification**: This is for TESTING validation only. Temperature-based sampling (0.0-2.0) is the product feature. Determinism is not guaranteed in production due to model and hardware limitations.

### [CUDA-5304] Streaming
The module MUST support streaming token generation (not batch generation).

---

## 2. C API

### [CUDA-5310] Start Inference
```c
// Start inference session
// Returns: Opaque inference handle, or NULL on error
InferenceResult* cuda_inference_start(
    CudaModel* model,
    const char* prompt,
    int max_tokens,
    float temperature,
    uint64_t seed,
    int* error_code
);
```

### [CUDA-5311] Generate Next Token
```c
// Generate next token (blocking)
// Returns: true if token generated, false if done
// token_out: Buffer for UTF-8 token string
// token_index: Set to token position (0, 1, 2, ...)
bool cuda_inference_next_token(
    InferenceResult* result,
    char* token_out,
    int token_buffer_size,
    int* token_index,
    int* error_code
);
```

### [CUDA-5312] Cleanup
```c
// Free inference resources (KV cache, etc.)
void cuda_inference_free(InferenceResult* result);
```

---

## 3. C++ Implementation

### [CUDA-5320] Inference Class
```cpp
// include/inference.hpp
namespace worker_cuda {

struct InferenceConfig {
    int max_tokens;
    float temperature;
    uint64_t seed;
    int top_k = 50;
    float top_p = 0.95f;
};

class InferenceResult {
public:
    InferenceResult(
        const Model& model,
        const std::string& prompt,
        const InferenceConfig& config
    );
    ~InferenceResult();
    
    // Non-copyable, non-movable (holds CUDA resources)
    InferenceResult(const InferenceResult&) = delete;
    InferenceResult& operator=(const InferenceResult&) = delete;
    InferenceResult(InferenceResult&&) = delete;
    InferenceResult& operator=(InferenceResult&&) = delete;
    
    // Generate next token
    bool next_token(std::string& token_out, int& token_index);
    
    // Check if inference is complete
    bool is_done() const { return current_token_ >= config_.max_tokens; }
    
private:
    void tokenize_prompt(const std::string& prompt);
    void allocate_kv_cache();
    void run_forward_pass();
    int sample_token();
    std::string detokenize(int token_id);
    
    const Model& model_;
    InferenceConfig config_;
    
    // State
    std::vector<int> prompt_tokens_;
    int current_token_ = 0;
    
    // CUDA resources
    std::unique_ptr<DeviceMemory> kv_cache_;
    std::unique_ptr<DeviceMemory> logits_;
    cudaStream_t stream_;
    
    // RNG state
    std::mt19937_64 rng_;
};

} // namespace worker_cuda
```

### [CUDA-5321] Forward Pass
```cpp
// src/inference.cu
namespace worker_cuda {

void InferenceResult::run_forward_pass() {
    // 1. Embedding lookup
    embedding_kernel<<<grid, block, 0, stream_>>>(
        model_.weights(),
        prompt_tokens_.data(),
        embeddings_
    );
    
    // 2. Transformer layers
    for (int layer = 0; layer < model_.metadata().num_layers; ++layer) {
        // Self-attention
        attention_kernel<<<grid, block, 0, stream_>>>(
            model_.layer_weights(layer),
            embeddings_,
            kv_cache_->get(),
            current_token_,
            attention_output_
        );
        
        // Feed-forward
        ffn_kernel<<<grid, block, 0, stream_>>>(
            model_.layer_weights(layer),
            attention_output_,
            embeddings_
        );
    }
    
    // 3. Output projection
    output_kernel<<<grid, block, 0, stream_>>>(
        model_.output_weights(),
        embeddings_,
        logits_->get()
    );
    
    // 4. Synchronize
    cudaStreamSynchronize(stream_);
}

int InferenceResult::sample_token() {
    // Copy logits to host
    std::vector<float> host_logits(model_.metadata().vocab_size);
    cudaMemcpy(
        host_logits.data(),
        logits_->get(),
        host_logits.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    
    // Apply temperature
    for (float& logit : host_logits) {
        logit /= config_.temperature;
    }
    
    // Softmax
    softmax(host_logits);
    
    // Sample with top-k/top-p
    return sample_top_k_top_p(host_logits, config_.top_k, config_.top_p, rng_);
}

} // namespace worker_cuda
```

---

## 4. CUDA Kernels

### [CUDA-5330] Attention Kernel
```cuda
// kernels/attention.cu
__global__ void attention_kernel(
    const float* weights,
    const float* input,
    float* kv_cache,
    int position,
    float* output
) {
    // Multi-head self-attention
    // - Query, Key, Value projections
    // - Scaled dot-product attention
    // - Output projection
    
    // Implementation details...
}
```

### [CUDA-5331] Matrix Multiplication
```cuda
// kernels/matmul.cu
__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    // Optimized matrix multiplication
    // - Shared memory tiling
    // - Coalesced memory access
    
    // Implementation details...
}
```

### [CUDA-5332] Sampling Kernel
```cuda
// kernels/sampling.cu
__global__ void softmax_kernel(
    float* logits,
    int vocab_size
) {
    // Numerically stable softmax
    // - Find max
    // - Exp and sum
    // - Normalize
    
    // Implementation details...
}
```

---

## 5. KV Cache Management

### [CUDA-5340] Cache Allocation
```cpp
void InferenceResult::allocate_kv_cache() {
    // Calculate KV cache size
    // For each layer: 2 (K and V) * context_length * hidden_dim
    size_t cache_size = 2 * 
                       model_.metadata().num_layers *
                       model_.metadata().context_length *
                       model_.metadata().embedding_length *
                       sizeof(float);
    
    kv_cache_ = std::make_unique<DeviceMemory>(cache_size);
    
    // Initialize to zero
    cudaMemset(kv_cache_->get(), 0, cache_size);
}
```

### [CUDA-5341] Cache Layout
```
KV Cache Layout (per layer):
┌─────────────────────────────────────┐
│ Keys   [context_length × hidden_dim] │
├─────────────────────────────────────┤
│ Values [context_length × hidden_dim] │
└─────────────────────────────────────┘
```

---

## 6. Test Reproducibility (NOT a Product Guarantee)

### [CUDA-5350] Seeded RNG
```cpp
InferenceResult::InferenceResult(..., uint64_t seed) 
    : rng_(seed) {
    // Initialize RNG with seed
    // Sampling will be reproducible for testing
}
```

### [CUDA-5351] Reproducible Kernels for Testing
All CUDA kernels MUST be reproducible for testing:
- No atomics with race conditions
- No non-deterministic reductions
- Fixed execution order

**Clarification**: This is for TESTING validation only. Temperature-based sampling (0.0-2.0) is the product feature. Reproducibility is not guaranteed in production due to model and hardware limitations.

### [CUDA-5352] Verification
The module MUST produce identical output for testing when:
- Same model
- Same prompt
- Same seed
- temperature=0.0

**Note**: This is a testing requirement, NOT a product guarantee.

---

## 7. FFI Implementation

### [CUDA-5360] C API Wrapper
```cpp
// src/ffi.cpp
extern "C" {

InferenceResult* cuda_inference_start(
    CudaModel* model,
    const char* prompt,
    int max_tokens,
    float temperature,
    uint64_t seed,
    int* error_code
) {
    try {
        auto model_ptr = reinterpret_cast<worker_cuda::Model*>(model);
        
        worker_cuda::InferenceConfig config;
        config.max_tokens = max_tokens;
        config.temperature = temperature;
        config.seed = seed;
        
        auto result = std::make_unique<worker_cuda::InferenceResult>(
            *model_ptr,
            std::string(prompt),
            config
        );
        
        *error_code = CUDA_SUCCESS;
        return reinterpret_cast<InferenceResult*>(result.release());
    } catch (const worker_cuda::CudaError& e) {
        *error_code = e.code();
        return nullptr;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    }
}

bool cuda_inference_next_token(
    InferenceResult* result,
    char* token_out,
    int token_buffer_size,
    int* token_index,
    int* error_code
) {
    try {
        auto ptr = reinterpret_cast<worker_cuda::InferenceResult*>(result);
        
        std::string token;
        bool has_token = ptr->next_token(token, *token_index);
        
        if (has_token) {
            strncpy(token_out, token.c_str(), token_buffer_size - 1);
            token_out[token_buffer_size - 1] = '\0';
        }
        
        *error_code = CUDA_SUCCESS;
        return has_token;
    } catch (const worker_cuda::CudaError& e) {
        *error_code = e.code();
        return false;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return false;
    }
}

void cuda_inference_free(InferenceResult* result) {
    if (result) {
        auto ptr = reinterpret_cast<worker_cuda::InferenceResult*>(result);
        delete ptr;
    }
}

} // extern "C"
```

---

## 8. Error Handling

### [CUDA-5370] Error Codes
- `CUDA_SUCCESS` (0) — Token generated successfully
- `CUDA_ERROR_INFERENCE_FAILED` (5) — Kernel launch failed
- `CUDA_ERROR_OUT_OF_MEMORY` (2) — Insufficient VRAM for KV cache

### [CUDA-5371] Kernel Error Checking
```cpp
void InferenceResult::run_forward_pass() {
    // Launch kernel
    attention_kernel<<<grid, block, 0, stream_>>>(...);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_KERNEL_LAUNCH_FAILED,
                       cudaGetErrorString(err));
    }
    
    // Synchronize and check for execution errors
    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        throw CudaError(CUDA_ERROR_INFERENCE_FAILED,
                       cudaGetErrorString(err));
    }
}
```

---

## 9. Performance Optimization

### [CUDA-5380] Kernel Fusion
Fuse small kernels to reduce launch overhead:
- Combine RoPE + attention
- Combine LayerNorm + FFN

### [CUDA-5381] Memory Optimization
- Reuse intermediate buffers
- Use shared memory for small tensors
- Coalesce memory accesses

### [CUDA-5382] Stream Parallelism
Use CUDA streams to overlap:
- Kernel execution
- Memory transfers (if needed)

---

## 10. Testing

### [CUDA-5390] Unit Tests
```cpp
// tests/test_inference.cpp
TEST(InferenceTest, GenerateTokens) {
    auto ctx = create_test_context();
    auto model = load_test_model(ctx);
    
    int error_code;
    auto inference = cuda_inference_start(
        model,
        "Hello",
        10,      // max_tokens
        0.7f,    // temperature
        42,      // seed
        &error_code
    );
    
    ASSERT_NE(inference, nullptr);
    ASSERT_EQ(error_code, CUDA_SUCCESS);
    
    // Generate tokens
    char token[256];
    int token_index;
    int count = 0;
    
    while (cuda_inference_next_token(
        inference, token, sizeof(token), &token_index, &error_code
    )) {
        ASSERT_EQ(error_code, CUDA_SUCCESS);
        ASSERT_STREQ(token, expected_tokens[count]);
        count++;
    }
    
    ASSERT_EQ(count, 10);
    
    cuda_inference_free(inference);
}

TEST(InferenceTest, ReproducibleOutputForTesting) {
    // Run inference twice with same seed and temp=0
    auto output1 = run_inference("Hello", /*seed=*/42, /*temp=*/0.0f);
    auto output2 = run_inference("Hello", /*seed=*/42, /*temp=*/0.0f);
    
    // Outputs must be identical for testing validation
    ASSERT_EQ(output1, output2);
}
```

---

## 11. Traceability

**Code**: `cuda/src/inference.cu`, `cuda/include/inference.hpp`, `cuda/kernels/*.cu`  
**Tests**: `cuda/tests/test_inference.cpp`  
**Parent**: `00_cuda_overview.md`  
**Spec IDs**: CUDA-5301 to CUDA-5390

---

**End of Specification**
