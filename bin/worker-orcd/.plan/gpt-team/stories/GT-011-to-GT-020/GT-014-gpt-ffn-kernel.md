# GT-014: GPT FFN Kernel

**Team**: GPT-Gamma  
**Sprint**: Sprint 2 (GPT Kernels)  
**Size**: L (3 days)  
**Days**: 36-38  
**Spec Ref**: M0-W-1433

---

## Story Description

Implement GPT feed-forward network (FFN) kernel. GPT FFN uses two linear projections with GELU activation: up projection (d_model → ffn_dim), GELU, down projection (ffn_dim → d_model). This differs from Llama's SwiGLU FFN.

---

## Acceptance Criteria

- [ ] CUDA kernel implements up projection (d_model → ffn_dim)
- [ ] CUDA kernel applies GELU activation
- [ ] CUDA kernel implements down projection (ffn_dim → d_model)
- [ ] Kernel integrates with cuBLAS GEMM for matrix multiplications
- [ ] Kernel supports FP16 weights and activations
- [ ] Unit test validates FFN output correctness
- [ ] Integration test validates full FFN layer
- [ ] Performance: <2ms per layer for GPT-OSS-20B dimensions
- [ ] Error handling for dimension mismatches

---

## Dependencies

### Upstream (Blocks This Story)
- GT-013: GELU Unit Tests (needs validated GELU)
- FT-016: cuBLAS GEMM Wrapper (needs GEMM operations)

### Downstream (This Story Blocks)
- GT-021: GPT Kernel Suite Integration (needs all GPT kernels)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/gpt_ffn.cu` - GPT FFN kernel
- `bin/worker-orcd/cuda/kernels/gpt_ffn.h` - FFN interface

### Key Interfaces
```cpp
void gpt_ffn_forward(
    const half* input,        // [batch, seq_len, d_model]
    const half* w_up,         // [d_model, ffn_dim]
    const half* b_up,         // [ffn_dim]
    const half* w_down,       // [ffn_dim, d_model]
    const half* b_down,       // [d_model]
    half* output,             // [batch, seq_len, d_model]
    half* workspace,          // Intermediate buffer
    int batch_size,
    int seq_len,
    int d_model,
    int ffn_dim,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
);
```

---

## Testing Strategy

### Unit Tests
- Test up projection correctness
- Test GELU activation
- Test down projection correctness
- Test full FFN pipeline

### Integration Tests
- Test with GPT-OSS-20B dimensions
- Compare with reference implementation

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Story marked complete

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7.3
- Related Stories: GT-012, GT-013, GT-015

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
