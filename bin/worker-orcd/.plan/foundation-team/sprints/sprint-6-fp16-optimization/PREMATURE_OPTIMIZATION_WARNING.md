# ⚠️ PREMATURE OPTIMIZATION WARNING

**Sprint**: Sprint 6 - FP16 Optimization  
**Status**: BLOCKED until M0 complete  
**Created**: 2025-10-05

---

## Problem Statement

This sprint optimizes code that **does not exist yet**.

### Current State (2025-10-05)

**CUDA Kernels Status:**
- ❌ `cuda/kernels/gemm.cu` - **STUB** (TODO comments only)
- ❌ `cuda/kernels/attention.cu` - **STUB** (TODO comments only)
- ❌ `cuda/kernels/gqa_attention.cu` - **Simplified stub** (no real computation)
- ⚠️ `cuda/kernels/rmsnorm.cu` - Has FP16, but not integrated
- ⚠️ `cuda/kernels/sampling.cu` - Has FP16, but no inference pipeline

**Inference Pipeline:**
- ❌ No working FP32 baseline
- ❌ No functional GEMM operations
- ❌ No complete attention implementation
- ❌ M0 haiku test (FT-050) not passing

---

## Why This Is Premature

### Classic Premature Optimization

> "Premature optimization is the root of all evil" - Donald Knuth

**We are optimizing:**
- Non-existent GEMM operations
- Stub attention kernels
- Incomplete inference pipeline

**We should be:**
1. Implementing FP32 baseline first
2. Getting M0 working (haiku test passing)
3. Establishing performance baseline
4. **Then** optimizing with FP16

---

## Correct Sequence

### Phase 1: Implement FP32 Baseline (Sprint 5 or earlier)
```
1. Implement cuda/kernels/gemm.cu (FP32 cuBLAS)
2. Implement cuda/kernels/gqa_attention.cu (FP32 attention)
3. Complete inference pipeline
4. Wire everything together
5. Get haiku test passing (FT-050)
```

### Phase 2: Establish Baseline (Sprint 5)
```
1. Measure FP32 performance
2. Profile bottlenecks
3. Identify optimization targets
4. Document baseline metrics
```

### Phase 3: Optimize with FP16 (Sprint 6 - THIS SPRINT)
```
1. Implement FP16 GEMM (FT-051)
2. Implement FP16 attention (FT-052)
3. Implement FP16 KV cache (FT-053)
4. Profile improvements (FT-054)
5. Validate performance (FT-056)
```

---

## What Needs to Happen First

### Blockers for Sprint 6

1. **FP32 GEMM Implementation**
   - Replace stub in `cuda/kernels/gemm.cu`
   - Implement `cublasSgemm` wrapper
   - Validate correctness
   - Integrate with attention/FFN

2. **FP32 Attention Implementation**
   - Replace stubs in `cuda/kernels/gqa_attention.cu`
   - Implement prefill (Q @ K^T, softmax, @ V)
   - Implement decode (single token + cache)
   - Validate correctness

3. **Complete Inference Pipeline**
   - Wire GEMM → Attention → FFN
   - Implement token generation loop
   - Add KV cache management
   - Get end-to-end inference working

4. **M0 Validation**
   - Pass haiku test (FT-050)
   - Validate real GPU inference
   - Establish performance baseline
   - Document FP32 metrics

---

## Decision: DEFER TO POST-M0

**Sprint 6 is BLOCKED until:**
- ✅ M0 complete (haiku test passing)
- ✅ FP32 baseline working
- ✅ Performance baseline established
- ✅ All CUDA kernels implemented

**Estimated Unblock Date:** After Sprint 5 (Final Integration) completes

---

## Lessons Learned

### Red Flags We Ignored

1. **Stub Code**: Optimizing TODOs is pointless
2. **No Baseline**: Can't measure improvement without baseline
3. **Missing Dependencies**: FP16 requires working FP32
4. **Premature Planning**: Detailed optimization plans before basic functionality

### What We Should Have Done

1. Check if FP32 baseline exists ❌
2. Verify M0 is complete ❌
3. Confirm performance baseline established ❌
4. **Then** plan FP16 optimization ✅

---

## Action Items

- [x] Rename sprint folder: `sprint-5-fp16-optimization` → `sprint-6-fp16-optimization`
- [x] Add prerequisite warnings to all FT-051 through FT-056 stories
- [x] Update sprint README with M0 dependency
- [x] Create this warning document
- [ ] Focus on implementing FP32 baseline first
- [ ] Return to Sprint 6 after M0 complete

---

## References

- **M0 Success Criteria**: FT-050 (Haiku generation test)
- **Sprint 5**: Final Integration (M0 completion)
- **Current Sprint Status**: `../sprint-7-final-integration/README.md`

---

**Bottom Line**: Come back to this sprint after M0 is working. First make it work, then make it fast.

---
Built by Foundation-Alpha 🏗️ (with humility)
