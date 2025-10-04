# Sprint 2: GPT Kernels

**Team**: GPT-Gamma  
**Days**: 27-41 (15 agent-days)  
**Goal**: Implement all GPT-specific CUDA kernels (LayerNorm, GELU, FFN, residual)

---

## Sprint Overview

Sprint 2 implements the core GPT-specific CUDA kernels that differentiate GPT architecture from Llama. Key differences:
- **LayerNorm** (not RMSNorm)
- **GELU activation** (not SwiGLU)
- **Absolute positional embeddings** (not RoPE)
- **Standard FFN** (not gated FFN)

These kernels are foundational for all subsequent GPT inference work.

---

## Stories in This Sprint

| ID | Title | Size | Days | Day Range |
|----|-------|------|------|-----------|
| GT-008 | Absolute Positional Embedding | M | 2 | 27-28 |
| GT-009 | LayerNorm Mean Reduction | M | 1.5 | 29-30 |
| GT-010 | LayerNorm Variance + Normalize | M | 1.5 | 30-31 |
| GT-011 | LayerNorm Unit Tests | S | 1 | 32 |
| GT-012 | GELU Activation Kernel | M | 2 | 33-34 |
| GT-013 | GELU Unit Tests | S | 1 | 35 |
| GT-014 | GPT FFN Kernel | L | 3 | 36-38 |
| GT-015 | Residual Connection Kernel | S | 1 | 39 |
| GT-016 | Kernel Integration Tests | M | 2 | 40-41 |

**Total**: 9 stories, 15 agent-days (Days 27-41)

---

## Story Execution Order

### Days 27-28: GT-008 - Absolute Positional Embedding
**Goal**: Implement learned position embeddings  
**Key Deliverable**: Position embedding addition kernel  
**Blocks**: GT-021 (kernel suite integration)

### Days 29-31: GT-009 + GT-010 - LayerNorm (Mean + Variance)
**Goal**: Implement full LayerNorm (mean, variance, normalize, scale, bias)  
**Key Deliverable**: Complete LayerNorm kernel  
**Blocks**: GT-011 (LayerNorm tests)

### Day 32: GT-011 - LayerNorm Unit Tests
**Goal**: Validate LayerNorm correctness  
**Key Deliverable**: Comprehensive test suite  
**Blocks**: GT-012 (GELU kernel)

### Days 33-34: GT-012 - GELU Activation Kernel
**Goal**: Implement exact GELU formula  
**Key Deliverable**: GELU kernel using `erff()`  
**Blocks**: GT-013 (GELU tests)

### Day 35: GT-013 - GELU Unit Tests
**Goal**: Validate GELU correctness  
**Key Deliverable**: GELU test suite  
**Blocks**: GT-014 (FFN kernel)

### Days 36-38: GT-014 - GPT FFN Kernel
**Goal**: Implement GPT feed-forward network  
**Key Deliverable**: Up projection + GELU + down projection  
**Blocks**: GT-015 (residual)

### Day 39: GT-015 - Residual Connection Kernel
**Goal**: Implement element-wise addition for residuals  
**Key Deliverable**: Residual connection kernel  
**Blocks**: GT-016 (integration tests)

### Days 40-41: GT-016 - Kernel Integration Tests
**Goal**: Validate all kernels work together  
**Key Deliverable**: Full transformer layer integration test  
**Blocks**: Sprint 3 (MHA attention)

---

## Dependencies

### Upstream (Blocks This Sprint)
- Sprint 1: HF Tokenizer (needs architecture detection)
- FT-015: Embedding Lookup Kernel (needs token embeddings)

### Downstream (This Sprint Blocks)
- Sprint 3: MHA + Gate 1 (needs validated kernel suite)
- GT-017: MHA Attention Prefill (needs LayerNorm, residual)

---

## Success Criteria

Sprint is complete when:
- [ ] All 9 stories marked complete
- [ ] Absolute positional embedding working
- [ ] LayerNorm implemented and tested
- [ ] GELU activation implemented and tested
- [ ] GPT FFN implemented
- [ ] Residual connections working
- [ ] Integration tests passing
- [ ] Ready for Sprint 3 (MHA attention)

---

## Next Sprint

**Sprint 3**: MHA + Gate 1  
**Starts**: Day 42  
**Focus**: Implement Multi-Head Attention and validate Gate 1

---

**Status**: 📋 Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Coordinated by Project Management Team 📋
