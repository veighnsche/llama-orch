# Sprint 6: FP16 Optimization & Polish

**Team**: Foundation-Alpha  
**Duration**: 6-9 days  
**Start**: Post-M0 (After Sprint 5 - Final Integration)  
**Status**: 📋 Planned (Post-M0)  
**Goal**: Performance optimization through FP16 acceleration and memory bandwidth reduction

---

## ⚠️ IMPORTANT: Post-M0 Sprint

**This sprint requires:**
- ✅ Working FP32 baseline implementation
- ✅ M0 haiku test passing (FT-050)
- ✅ Complete CUDA kernel implementations (GEMM, attention, FFN)
- ✅ Functional inference pipeline

**Do NOT start this sprint until M0 is complete.** FP16 optimization assumes we have working FP32 code to optimize.

---

## Sprint Overview

Implement comprehensive FP16 optimizations across the inference pipeline: GEMM operations, attention kernels, KV cache storage, and optional kernel fusion. Target 1.4-1.8x speedup and 40-50% VRAM reduction for long-context generation.

---

## Sprint Goals

### Primary Goals
1. ✅ FP16 GEMM operations (cuBLAS)
2. ✅ FP16 attention kernels (prefill/decode)
3. ✅ FP16 KV cache storage
4. ✅ Memory bandwidth profiling
5. ✅ End-to-end performance validation

### Secondary Goals
6. ✅ Numerical accuracy validation
7. ✅ Performance report generation

### Stretch Goals
8. ⏸️ Fused kernel optimization (RMSNorm+GEMM, GEMM+SwiGLU)

---

## Stories

### FT-051: FP16 GEMM Optimization
**Size**: M (2 days)  
**Priority**: High  
**Owner**: Foundation-Alpha

**Scope**:
- cuBLAS FP16 GEMM wrapper (`cublasGemmEx`)
- Mixed precision support (FP16 compute, FP32 accumulate)
- Integration with attention Q·K^T, attention·V, FFN layers
- Numerical accuracy validation (tolerance: 1e-3)
- Performance benchmarks (expect 1.5-2x speedup)
- Unit tests for all GEMM variants

**Target**: 1.5-2x speedup, 50% memory bandwidth reduction

---

### FT-052: FP16 Attention Kernels
**Size**: M (2 days)  
**Priority**: High  
**Owner**: Foundation-Alpha

**Scope**:
- FP16 attention prefill kernel (full sequence)
- FP16 attention decode kernel (single token with cache)
- FP16 softmax with numerical stability
- Causal masking support
- Integration with FP16 GEMM (FT-051)
- Numerical accuracy validation (tolerance: 1e-2)
- Performance benchmarks (expect 1.4-1.8x speedup)

**Target**: 1.4-1.8x speedup, 50% memory bandwidth reduction

---

### FT-053: KV Cache FP16 Optimization
**Size**: S (1 day)  
**Priority**: Medium  
**Owner**: Foundation-Alpha

**Scope**:
- FP16 KV cache storage (keys and values)
- Automatic FP32↔FP16 conversion on read/write
- VRAM usage calculation update
- Integration with FP16 attention kernels
- Unit tests for cache operations
- VRAM savings validation

**Target**: 50% VRAM reduction for KV cache (critical for long-context)

**Impact**:
- Qwen2.5-0.5B @ 8K tokens: 8 MB → 4 MB (4 MB saved)
- Phi-3 Mini @ 8K tokens: 192 MB → 96 MB (96 MB saved)

---

### FT-054: Memory Bandwidth Profiling
**Size**: S (1 day)  
**Priority**: Medium  
**Owner**: Foundation-Alpha

**Scope**:
- CUDA event-based timing infrastructure
- Memory bandwidth calculation (bytes/time)
- Profiling for GEMM, attention, KV cache
- CSV/JSON output for benchmark results
- Visualization script (Python/matplotlib)
- Integration with proof-bundle system

**Deliverables**:
- Bandwidth comparison report (FP32 vs FP16)
- Performance visualization graphs
- Benchmark data in proof-bundle

---

### FT-055: Fused Kernel Optimization (Stretch)
**Size**: L (3 days)  
**Priority**: Medium (Stretch Goal)  
**Owner**: Foundation-Alpha

**Scope**:
- Fused RMSNorm + GEMM kernel (attention projections)
- Fused GEMM + SwiGLU kernel (FFN activation)
- Fused attention score + softmax kernel
- Numerical accuracy validation
- Performance benchmarks (expect 1.2-1.5x additional speedup)
- Memory bandwidth reduction measurement

**Target**: 1.2-1.5x additional speedup, 25-30% bandwidth reduction

**Note**: Defer to Sprint 6 if time-constrained. FT-051 through FT-054 are sufficient for Sprint 5 goals.

---

### FT-056: FP16 Performance Validation
**Size**: S (1 day)  
**Priority**: High  
**Owner**: Foundation-Alpha

**Scope**:
- End-to-end inference benchmarks (FP32 vs FP16)
- First-token latency comparison
- Tokens/sec throughput comparison
- VRAM usage comparison
- Haiku test with FP16 pipeline
- Performance report generation (Markdown + CSV)
- Numerical accuracy validation on real prompts

**Deliverables**:
- Performance report (Markdown)
- Benchmark data (CSV)
- Proof-bundle artifacts

---

## Sprint Timeline

### Days 1-2: FP16 GEMM (FT-051)
- Day 1: cuBLAS wrapper, Tensor Core integration
- Day 2: Numerical validation, performance benchmarks

### Days 3-4: FP16 Attention (FT-052)
- Day 3: Prefill kernel, softmax
- Day 4: Decode kernel, causal masking, integration

### Day 5: KV Cache FP16 (FT-053)
- FP16 storage, VRAM calculation, integration

### Day 6: Bandwidth Profiling (FT-054)
- Profiling infrastructure, benchmarks, visualization

### Days 7-9: Fused Kernels (FT-055) - Optional
- Day 7: RMSNorm+GEMM fusion
- Day 8: GEMM+SwiGLU fusion
- Day 9: Attention score+softmax fusion

### Day 6 or 10: Performance Validation (FT-056)
- End-to-end benchmarks, haiku test, report generation

---

## Deliverables

### Core Deliverables (Required)
- [x] FP16 GEMM operations (FT-051)
- [x] FP16 attention kernels (FT-052)
- [x] FP16 KV cache storage (FT-053)
- [x] Memory bandwidth profiling (FT-054)
- [x] Performance validation report (FT-056)

### Stretch Deliverables (Optional)
- [ ] Fused RMSNorm+GEMM kernel (FT-055)
- [ ] Fused GEMM+SwiGLU kernel (FT-055)
- [ ] Fused attention kernel (FT-055)

---

## Performance Targets

### Overall Targets
- **Speedup**: 1.4-1.8x for full inference pipeline
- **VRAM Reduction**: 40-50% for long-context generation (8K+ tokens)
- **Memory Bandwidth**: 50% reduction across all operations
- **Numerical Accuracy**: < 1e-2 difference in final output

### Per-Operation Targets

| Operation | FP32 (ms) | FP16 (ms) | Target Speedup |
|-----------|-----------|-----------|----------------|
| Decode GEMM | 0.05 | 0.03 | 1.67x |
| Prefill GEMM | 0.8 | 0.5 | 1.6x |
| Attention Prefill | 1.5 | 1.0 | 1.5x |
| Attention Decode | 0.3 | 0.2 | 1.5x |
| KV Cache Write | 0.1 | 0.1 | 1.0x (VRAM savings) |

### End-to-End Targets

**Qwen2.5-0.5B**:
- First token: 50 ms → 35 ms (1.4x)
- Throughput: 30 tok/s → 45 tok/s (1.5x)
- VRAM @ 512 tok: 1.2 GB → 0.8 GB (-33%)

**Phi-3 Mini**:
- First token: 120 ms → 80 ms (1.5x)
- Throughput: 20 tok/s → 30 tok/s (1.5x)
- VRAM @ 512 tok: 3.5 GB → 2.3 GB (-34%)

---

## Testing Strategy

### Unit Tests (25+ tests)
- FP16 GEMM: 8 tests (accuracy, edge cases, tensor cores)
- FP16 Attention: 6 tests (prefill, decode, softmax, masking)
- KV Cache: 5 tests (write, read, append, overflow)
- Profiling: 3 tests (timing, bandwidth, CSV)
- Fused kernels: 6 tests (if implemented)

### Integration Tests (7+ tests)
- FP16 transformer layer: 2 tests
- FP16 inference pipeline: 2 tests
- Haiku test with FP16: 1 test
- Performance validation: 2 tests

### Performance Tests (6+ benchmarks)
- GEMM benchmarks: 3 (decode, prefill, FFN)
- Attention benchmarks: 2 (prefill, decode)
- End-to-end benchmarks: 3 (Qwen, Phi-3, long-context)

---

## References

- **Backlog**: `../DEFERRED_WORK_BACKLOG.md` (FT-021 FP16 Sampling)
- **Sprint 5**: `../sprint-7-final-integration/README.md` (M0 completion)
- **Stories**: `../../stories/FT-051-to-FT-060/`
- **Prerequisites**: M0 complete, FP32 baseline working
- **cuBLAS**: https://docs.nvidia.com/cuda/cublas/
- **Tensor Cores**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-cores

---

## Success Criteria

Sprint 5 is complete when:
1. ✅ All core stories (FT-051 through FT-054, FT-056) are complete
2. ✅ Performance targets met (1.4-1.8x speedup)
3. ✅ VRAM reduction validated (40-50%)
4. ✅ Numerical accuracy validated (< 1e-2 error)
5. ✅ Haiku test passes with FP16
6. ✅ Performance report generated
7. ✅ All tests passing (25+ unit, 7+ integration)

---
Built by Foundation-Alpha 🏗️
