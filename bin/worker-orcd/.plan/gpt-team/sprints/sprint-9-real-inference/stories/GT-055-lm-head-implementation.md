# GT-055: LM Head Implementation

**Team**: GPT-Gamma 🤖  
**Sprint**: Sprint 9 - Real Inference  
**Size**: S (2-3 hours)  
**Priority**: P0 (M0 blocker)  
**Spec Ref**: M0-W-1213 (implied)

---

## Story Description

Implement LM head projection (hidden_dim -> vocab_size) using cuBLAS GEMM. Replace stub that just copies data.

---

## Current State (STUB)

**File**: `cuda/src/model/gpt_model.cpp` line 307

```cpp
void GPTModel::apply_output_head(
    const half* input,
    half* logits
) {
    const GPTConfig& cfg = weights_->config;
    
    // Allocate temp buffer for normalized output
    half* normalized = workspace_ + (cfg.max_seq_len * cfg.hidden_dim * 2);
    
    // Apply final LayerNorm
    cuda_layernorm(
        normalized,
        input,
        weights_->output_norm_weight,
        weights_->output_norm_bias,
        1,  // batch_size
        1,  // seq_len (single token)
        cfg.hidden_dim,
        1e-5f,
        stream_
    );
    
    // TODO: Apply LM head (GEMM: normalized @ lm_head_weight)
    // For now, just copy (stub)
    cudaMemcpy(logits, normalized,
               cfg.hidden_dim * sizeof(half),
               cudaMemcpyDeviceToDevice);
    
    cudaStreamSynchronize(stream_);
}
```

---

## Acceptance Criteria

- [ ] Implement LM head GEMM using cuBLAS
- [ ] Project from hidden_dim to vocab_size
- [ ] Handle both prefill and decode modes
- [ ] Remove stub code
- [ ] Unit test verifies logits shape
- [ ] Integration test verifies sampling works

---

## Technical Details

### Implementation

```cpp
void GPTModel::apply_output_head(
    const half* input,
    half* logits
) {
    const GPTConfig& cfg = weights_->config;
    
    // Allocate temp buffer for normalized output
    half* normalized = workspace_ + (cfg.max_seq_len * cfg.hidden_dim * 2);
    
    // Apply final LayerNorm
    cuda_layernorm(
        normalized,
        input,
        weights_->output_norm_weight,
        weights_->output_norm_bias,
        1,  // batch_size
        1,  // seq_len (single token)
        cfg.hidden_dim,
        1e-5f,
        stream_
    );
    
    // Apply LM head: logits = normalized @ lm_head_weight^T
    // Matrix dimensions:
    //   normalized: [1, hidden_dim]
    //   lm_head_weight: [vocab_size, hidden_dim] (stored transposed)
    //   logits: [1, vocab_size]
    
    const half alpha = 1.0f;
    const half beta = 0.0f;
    
    cublasStatus_t status = cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_T,  // Transpose lm_head_weight
        CUBLAS_OP_N,  // Don't transpose normalized
        cfg.vocab_size,  // M: rows of output
        1,               // N: cols of output (single token)
        cfg.hidden_dim,  // K: shared dimension
        &alpha,
        weights_->lm_head_weight,  // A: [hidden_dim, vocab_size]
        CUDA_R_16F,
        cfg.hidden_dim,  // lda
        normalized,      // B: [1, hidden_dim]
        CUDA_R_16F,
        cfg.hidden_dim,  // ldb
        &beta,
        logits,          // C: [1, vocab_size]
        CUDA_R_16F,
        cfg.vocab_size,  // ldc
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CudaError::inference_failed(
            "LM head GEMM failed: " + std::to_string(status)
        );
    }
    
    cudaStreamSynchronize(stream_);
}
```

### Existing Code to Use

**Already implemented**:
- ✅ `cublas_handle_` - cuBLAS handle in GPTModel
- ✅ `cuda_layernorm()` - Final norm (already working)
- ✅ `cublasGemmEx()` - cuBLAS GEMM function

**Just need to call cuBLAS!**

---

## Testing

### Unit Test

```cpp
TEST(GPTModel, ApplyOutputHead) {
    GPTConfig config;
    config.hidden_dim = 128;
    config.vocab_size = 1000;
    
    auto weights = create_test_weights(config);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    GPTModel model(std::move(weights), cublas);
    
    // Allocate test input (hidden state)
    half* input;
    cudaMalloc(&input, config.hidden_dim * sizeof(half));
    
    // Allocate output (logits)
    half* logits;
    cudaMalloc(&logits, config.vocab_size * sizeof(half));
    
    // Apply output head
    model.apply_output_head(input, logits);
    
    // Verify logits shape
    std::vector<half> host_logits(config.vocab_size);
    cudaMemcpy(host_logits.data(), logits, 
               config.vocab_size * sizeof(half), D2H);
    
    // Logits should have vocab_size elements
    EXPECT_EQ(host_logits.size(), config.vocab_size);
    
    // At least some logits should be non-zero
    int non_zero = 0;
    for (const auto& logit : host_logits) {
        if (logit != 0.0f) non_zero++;
    }
    EXPECT_GT(non_zero, 0) << "Some logits should be non-zero";
    
    cudaFree(input);
    cudaFree(logits);
    cublasDestroy(cublas);
}
```

---

## Dependencies

**Upstream**: GT-052 (needs lm_head_weight loaded)  
**Downstream**: GT-056 (needs logits for sampling)

---

## Definition of Done

- [ ] Stub code removed
- [ ] cuBLAS GEMM implemented
- [ ] Logits shape correct (vocab_size)
- [ ] Unit test passes
- [ ] No TODOs remain

---

## Estimated Time

**Realistic**: 2-3 hours

---

**Created by**: Project Management Team 📋  
**Assigned to**: GPT-Gamma 🤖  
**Status**: TODO
