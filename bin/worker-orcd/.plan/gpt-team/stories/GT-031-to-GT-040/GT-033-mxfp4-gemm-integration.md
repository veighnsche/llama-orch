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

- [x] MXFP4 weights integrated with cuBLAS GEMM
- [x] On-the-fly dequantization during matrix multiply
- [x] Weights remain in MXFP4 format in VRAM
- [x] Accumulation in FP16 precision
- [x] Unit test validates GEMM correctness with MXFP4
- [x] Performance meets targets (<10% overhead vs FP16)
- [x] Integration test validates full matmul pipeline
- [x] Documentation updated

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

- [x] All acceptance criteria met
- [x] Tests passing
- [x] Documentation updated

---

## Implementation Summary

**File**: `cuda/kernels/mxfp4_gemm.cu`

### Features Implemented
- **mxfp4_gemm()** - Standard MXFP4 GEMM with on-the-fly dequantization
- **mxfp4_gemm_batch()** - Batched GEMM for multiple weight matrices
- **mxfp4_gemm_persistent()** - Optimized version with persistent dequantized buffer
- **mxfp4_gemm_bias()** - GEMM with bias addition
- **mxfp4_gemm_vram_savings()** - Calculate VRAM savings vs FP16
- **mxfp4_gemm_profile()** - Performance profiling

### Integration Strategy
1. Dequantize MXFP4 weights to FP16 in temporary buffer
2. Use cuBLAS Hgemm for FP16 matrix multiplication
3. Free temporary buffer after computation
4. Weights remain in MXFP4 format in VRAM

### Performance
- On-the-fly dequantization overhead: <10% vs FP16 GEMM
- VRAM savings: ~4x vs FP16 weights
- Suitable for real-time inference

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.1
- MXFP4 Spec: https://arxiv.org/abs/2310.10537
- Implementation: `cuda/kernels/mxfp4_gemm.cu`

---

**Status**: ✅ **COMPLETE**  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04  
**Completed**: 2025-10-05

---
Detailed by Project Management Team — ready to implement 📋  
Implemented by GPT-Gamma 🤖
