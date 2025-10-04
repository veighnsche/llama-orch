# GT-010: LayerNorm Variance + Normalize

**Team**: GPT-Gamma  
**Sprint**: Sprint 2 (GPT Kernels)  
**Size**: M (1.5 days)  
**Days**: 30-31 (partial)  
**Spec Ref**: M0-W-1432

---

## Story Description

Implement the variance computation and normalization components of LayerNorm for GPT architecture. This completes LayerNorm by computing variance from the mean, then normalizing and scaling the activations.

---

## Acceptance Criteria

- [ ] CUDA kernel computes variance across embedding dimension
- [ ] CUDA kernel normalizes activations using mean and variance
- [ ] Kernel applies learned scale (gamma) and bias (beta) parameters
- [ ] Kernel uses epsilon for numerical stability (1e-5)
- [ ] Kernel supports FP16 input/output with FP32 intermediate
- [ ] Unit test validates variance calculation correctness
- [ ] Unit test validates normalization correctness
- [ ] Unit test validates scale/bias application
- [ ] Performance: <0.1ms per layer for d_model=2048
- [ ] Error handling for zero variance

---

## Dependencies

### Upstream (Blocks This Story)
- GT-009: LayerNorm Mean Reduction (needs mean computation)

### Downstream (This Story Blocks)
- GT-011: LayerNorm Unit Tests (needs complete LayerNorm)
- GT-014: GPT FFN Kernel (needs LayerNorm)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/layernorm.cu` - Add variance and normalize kernels
- `bin/worker-orcd/cuda/kernels/layernorm.h` - Update interface
- `bin/worker-orcd/cuda/src/kernels/layernorm_test.cu` - Add tests

### Key Interfaces
```cpp
// Complete LayerNorm: mean, variance, normalize, scale, bias
void layernorm_forward(
    const half* input,      // [batch, seq_len, d_model]
    const half* gamma,      // [d_model] - learned scale
    const half* beta,       // [d_model] - learned bias
    half* output,           // [batch, seq_len, d_model]
    int batch_size,
    int seq_len,
    int d_model,
    float epsilon,          // 1e-5 for numerical stability
    cudaStream_t stream
);
```

### CUDA Kernel Implementation
```cuda
__global__ void layernorm_kernel(
    const half* input,      // [batch, seq_len, d_model]
    const half* gamma,      // [d_model]
    const half* beta,       // [d_model]
    half* output,           // [batch, seq_len, d_model]
    int d_model,
    float epsilon
) {
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float shared_mean;
    __shared__ float shared_var;
    
    // Step 1: Compute mean (from GT-009)
    float sum = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        int idx = (batch_idx * gridDim.x + seq_idx) * d_model + i;
        sum += __half2float(input[idx]);
    }
    // ... reduction code ...
    if (tid == 0) shared_mean = sum / d_model;
    __syncthreads();
    
    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        int idx = (batch_idx * gridDim.x + seq_idx) * d_model + i;
        float val = __half2float(input[idx]);
        float diff = val - shared_mean;
        var_sum += diff * diff;
    }
    // ... reduction code ...
    if (tid == 0) shared_var = var_sum / d_model;
    __syncthreads();
    
    // Step 3: Normalize and scale
    float inv_std = rsqrtf(shared_var + epsilon);
    for (int i = tid; i < d_model; i += blockDim.x) {
        int idx = (batch_idx * gridDim.x + seq_idx) * d_model + i;
        float val = __half2float(input[idx]);
        float normalized = (val - shared_mean) * inv_std;
        float scaled = normalized * __half2float(gamma[i]) + __half2float(beta[i]);
        output[idx] = __float2half(scaled);
    }
}
```

### Implementation Notes
- LayerNorm formula: `y = gamma * (x - mean) / sqrt(var + eps) + beta`
- Use FP32 for mean and variance computation
- Use `rsqrtf()` for fast reciprocal square root
- Epsilon prevents division by zero (1e-5 standard)
- Gamma and beta are learned parameters from GGUF
- Fuse mean, variance, and normalize in single kernel for efficiency

---

## Testing Strategy

### Unit Tests
- Test variance calculation correctness
- Test normalization with known mean/variance
- Test scale (gamma) and bias (beta) application
- Test epsilon handling (prevent div by zero)
- Test numerical stability with extreme values

### Integration Tests
- Test full LayerNorm with GPT-OSS-20B parameters
- Test against reference implementation
- Test with real model weights

### Manual Verification
1. Run LayerNorm with known inputs
2. Verify output matches reference
3. Check numerical stability
4. Profile kernel performance

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7.3 (GPT Kernels)
- LayerNorm Paper: https://arxiv.org/abs/1607.06450
- Related Stories: GT-009 (mean), GT-011 (tests), GT-014 (FFN)

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
