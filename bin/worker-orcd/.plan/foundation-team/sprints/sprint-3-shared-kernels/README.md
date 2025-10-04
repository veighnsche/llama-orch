# Sprint 3: Shared Kernels

**Team**: Foundation-Alpha  
**Days**: 23-38 (16 agent-days)  
**Goal**: Implement VRAM enforcement, memory management, and shared CUDA kernels

---

## Sprint Overview

Sprint 3 implements the core CUDA infrastructure that both Llama and GPT architectures will use. This includes VRAM-only enforcement (no RAM fallback), device memory RAII wrappers, and shared kernels for embedding lookup, matrix multiplication, sampling, and RNG.

This sprint establishes the deterministic, VRAM-resident compute foundation that M0 requires.

---

## Stories in This Sprint

| ID | Title | Size | Days | Day Range |
|----|-------|------|------|-----------|
| FT-011 | VRAM-Only Enforcement | M | 2 | 23-24 |
| FT-012 | FFI Integration Tests | S | 1 | 25 |
| FT-013 | Device Memory RAII Wrapper | S | 1 | 26 |
| FT-014 | VRAM Residency Verification | S | 1 | 27 |
| FT-015 | Embedding Lookup Kernel | M | 2 | 28-29 |
| FT-016 | cuBLAS GEMM Wrapper | M | 2 | 30-31 |
| FT-017 | Temperature Scaling Kernel | S | 1 | 32 |
| FT-018 | Greedy Sampling | S | 1 | 33 |
| FT-019 | Stochastic Sampling | M | 2 | 34-35 |
| FT-020 | Seeded RNG | S | 1 | 36 |

**Total**: 10 stories, 16 agent-days (Days 23-38)

---

## Story Execution Order

### Days 23-24: FT-011 - VRAM-Only Enforcement
**Goal**: Enforce VRAM-only allocation, no RAM fallback  
**Key Deliverable**: VramTracker that fails on RAM allocation attempts  
**Blocks**: FT-012 (FFI integration tests)

### Day 25: FT-012 - FFI Integration Tests
**Goal**: End-to-end FFI integration tests  
**Key Deliverable**: Test suite validating FFI boundary  
**Blocks**: FT-013 (device memory RAII)

### Day 26: FT-013 - Device Memory RAII Wrapper
**Goal**: RAII wrapper for device memory with automatic cleanup  
**Key Deliverable**: DeviceMemory<T> with Drop implementation  
**Blocks**: FT-014 (VRAM residency verification)

### Day 27: FT-014 - VRAM Residency Verification
**Goal**: Runtime verification that all memory is VRAM-resident  
**Key Deliverable**: Verification function checking memory location  
**Blocks**: FT-015 (embedding lookup kernel)

### Days 28-29: FT-015 - Embedding Lookup Kernel
**Goal**: CUDA kernel for embedding lookup (FP16)  
**Key Deliverable**: Optimized embedding lookup kernel  
**Blocks**: FT-016 (cuBLAS GEMM wrapper)

### Days 30-31: FT-016 - cuBLAS GEMM Wrapper
**Goal**: Wrapper for cuBLAS GEMM with deterministic mode  
**Key Deliverable**: Deterministic matrix multiplication  
**Blocks**: FT-017 (temperature scaling)

### Day 32: FT-017 - Temperature Scaling Kernel
**Goal**: CUDA kernel for temperature scaling of logits  
**Key Deliverable**: Temperature scaling kernel  
**Blocks**: FT-018 (greedy sampling)

### Day 33: FT-018 - Greedy Sampling
**Goal**: Greedy sampling (argmax) implementation  
**Key Deliverable**: Greedy sampling kernel  
**Blocks**: FT-019 (stochastic sampling)

### Days 34-35: FT-019 - Stochastic Sampling
**Goal**: Stochastic sampling with top-k/top-p  
**Key Deliverable**: Stochastic sampling kernel  
**Blocks**: FT-020 (seeded RNG)

### Day 36: FT-020 - Seeded RNG
**Goal**: Seeded RNG for reproducible sampling  
**Key Deliverable**: Deterministic RNG with seed control  
**Blocks**: Sprint 4 (integration tests)

---

## Dependencies

### Upstream (Blocks This Sprint)
- FT-010: CUDA Context Initialization (provides CUDA context)

### Downstream (This Sprint Blocks)
- Sprint 4: Integration + Gate 1 (needs all shared kernels)
- LT-015: GQA Attention (needs embedding lookup)
- GT-017: MHA Attention (needs embedding lookup)

---

## Success Criteria

Sprint is complete when:
- [ ] All 10 stories marked complete
- [ ] VRAM-only enforcement operational
- [ ] FFI integration tests passing
- [ ] Device memory RAII wrapper working
- [ ] VRAM residency verification working
- [ ] Embedding lookup kernel implemented and tested
- [ ] cuBLAS GEMM wrapper working in deterministic mode
- [ ] Temperature scaling kernel implemented
- [ ] Greedy sampling working
- [ ] Stochastic sampling working with top-k/top-p
- [ ] Seeded RNG providing reproducible results
- [ ] All unit tests passing
- [ ] Ready for Sprint 4 (integration)

---

## Next Sprint

**Sprint 4**: Integration + Gate 1  
**Starts**: Day 39  
**Focus**: KV cache, integration tests, Gate 1 checkpoint

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Coordinated by Project Management Team 📋
