# GT-033: MXFP4 GEMM Integration

**Team**: GPT-Gamma  
**Sprint**: Sprint 6 (MXFP4 Integration)  
**Size**: L (3 days)  
**Days**: 75-77  
**Spec Ref**: M0-W-1435

---

## Story Description

Integrate MXFP4 dequantization with cuBLAS GEMM operations for matrix multiplications. Enable on-the-fly dequantization during GEMM to keep weights in MXFP4 format in VRAM while computing in FP16.

---

## Acceptance Criteria

- [ ] MXFP4 weights integrated with cuBLAS GEMM
- [ ] On-the-fly dequantization during matrix multiply
- [ ] Weights remain in MXFP4 format in VRAM
- [ ] Accumulation in FP16 precision
- [ ] Unit test validates GEMM correctness with MXFP4
- [ ] Performance meets targets (<10% overhead vs FP16)
- [ ] Integration test validates full matmul pipeline
- [ ] Documentation updated

---

## Dependencies

### Upstream (Blocks This Story)
- GT-030: MXFP4 Unit Tests (needs validated dequant)
- FT-016: cuBLAS GEMM Wrapper (needs GEMM operations)

### Downstream (This Story Blocks)
- GT-034: MXFP4 Embedding Lookup (needs MXFP4 GEMM)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/gemm/mxfp4_gemm.cpp` - MXFP4 GEMM integration
- `bin/worker-orcd/cuda/kernels/mxfp4_gemm.cu` - MXFP4 GEMM kernel

### Key Interfaces
```cpp
void mxfp4_gemm(
    const uint8_t* mxfp4_weights,  // MXFP4 weight matrix
    const half* input,              // FP16 input
    half* output,                   // FP16 output
    int m, int n, int k,           // Matrix dimensions
    cublasHandle_t cublas,
    cudaStream_t stream
);
```

---

## Testing Strategy

### Unit Tests
- Test MXFP4 GEMM correctness
- Test numerical accuracy
- Test performance overhead

### Integration Tests
- Test full matmul pipeline
- Compare with FP16 GEMM

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.1
- MXFP4 Spec: https://arxiv.org/abs/2310.10537

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
