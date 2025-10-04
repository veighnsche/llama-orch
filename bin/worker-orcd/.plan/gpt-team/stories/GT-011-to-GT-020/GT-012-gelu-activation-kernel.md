# GT-012: GELU Activation Kernel

**Team**: GPT-Gamma  
**Sprint**: Sprint 2 (GPT Kernels)  
**Size**: M (2 days)  
**Days**: 33-34  
**Spec Ref**: M0-W-1433

---

## Story Description

Implement GELU (Gaussian Error Linear Unit) activation function kernel for GPT architecture. GPT uses GELU activation in FFN layers, unlike Llama which uses SwiGLU. Implement the exact GELU formula (not the tanh approximation).

---

## Acceptance Criteria

- [ ] CUDA kernel implements exact GELU formula
- [ ] Kernel supports FP16 input/output
- [ ] Kernel handles tensors up to [batch, seq_len, ffn_dim]
- [ ] Unit test validates GELU output matches reference
- [ ] Unit test validates numerical accuracy (error <0.1%)
- [ ] Performance: <0.05ms for 2048 x 8192 tensor
- [ ] Error handling for invalid dimensions
- [ ] Documentation explains GELU vs other activations

---

## Dependencies

### Upstream (Blocks This Story)
- GT-011: LayerNorm Unit Tests (needs validated LayerNorm)

### Downstream (This Story Blocks)
- GT-013: GELU Unit Tests (needs GELU kernel)
- GT-014: GPT FFN Kernel (needs GELU activation)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/gelu.cu` - GELU kernel
- `bin/worker-orcd/cuda/kernels/gelu.h` - GELU interface

### Key Interfaces
```cpp
// Apply GELU activation element-wise
void gelu_forward(
    const half* input,    // [batch, seq_len, dim]
    half* output,         // [batch, seq_len, dim]
    int total_elements,   // batch * seq_len * dim
    cudaStream_t stream
);
```

### CUDA Kernel Implementation
```cuda
__device__ float gelu_exact(float x) {
    // GELU(x) = x * Phi(x)
    // where Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    const float sqrt_2_inv = 0.70710678118f;  // 1 / sqrt(2)
    return 0.5f * x * (1.0f + erff(x * sqrt_2_inv));
}

__global__ void gelu_kernel(
    const half* input,
    half* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = __half2float(input[idx]);
        float y = gelu_exact(x);
        output[idx] = __float2half(y);
    }
}
```

### Implementation Notes
- Use exact GELU formula (not tanh approximation)
- GELU formula: `GELU(x) = x * Phi(x)` where `Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))`
- Use CUDA `erff()` for error function
- Convert FP16 to FP32 for computation, back to FP16 for output
- Launch with 1D grid covering all elements
- Block size: 256 threads

---

## Testing Strategy

### Unit Tests
- Test GELU output for known inputs
- Test numerical accuracy vs reference
- Test edge cases (zero, negative, large values)
- Test FP16 precision handling

### Integration Tests
- Test with GPT-OSS-20B FFN dimensions
- Test full tensor activation
- Compare with reference implementation

### Manual Verification
1. Run GELU with known inputs
2. Verify output matches reference
3. Check numerical accuracy
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
- GELU Paper: https://arxiv.org/abs/1606.08415
- Related Stories: GT-013 (tests), GT-014 (FFN)

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
