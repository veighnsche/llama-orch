# Sprint 3: UTF-8 Safety + Llama Kernels

**Team**: Llama-Beta  
**Days**: 36-41 (6 agent-days)  
**Goal**: Complete tokenizer and implement Llama-specific CUDA kernels

---

## Sprint Overview

Sprint 3 completes the tokenizer with UTF-8 safe streaming and implements core Llama-specific CUDA kernels. These kernels (RoPE, RMSNorm, residual connections) are fundamental building blocks for Llama architecture inference.

This sprint establishes the kernel foundation required for GQA attention implementation.

---

## Stories in This Sprint

| ID | Title | Size | Days | Day Range |
|----|-------|------|------|-----------|
| LT-011 | UTF-8 Safe Streaming Decode | M | 2 | 36-37 |
| LT-012 | RoPE Kernel | M | 2 | 38-39 |
| LT-013 | RMSNorm Kernel | S | 1 | 40 |
| LT-014 | Residual Connection Kernel | S | 1 | 41 |

**Total**: 4 stories, 6 agent-days (Days 36-41)

---

## Story Execution Order

### Days 36-37: LT-011 - UTF-8 Safe Streaming Decode
**Goal**: Handle UTF-8 partial sequences in streaming  
**Key Deliverable**: UTF-8 safe streaming decoder  
**Blocks**: Sprint 5 (Qwen integration)

### Days 38-39: LT-012 - RoPE Kernel
**Goal**: Implement Rotary Position Embedding kernel  
**Key Deliverable**: RoPE CUDA kernel  
**Blocks**: LT-015 (GQA attention)

### Day 40: LT-013 - RMSNorm Kernel
**Goal**: Implement RMSNorm kernel  
**Key Deliverable**: RMSNorm CUDA kernel  
**Blocks**: LT-015 (GQA attention)

### Day 41: LT-014 - Residual Connection Kernel
**Goal**: Implement residual connection kernel  
**Key Deliverable**: Residual connection CUDA kernel  
**Blocks**: LT-015 (GQA attention)

---

## Dependencies

### Upstream (Blocks This Sprint)
- LT-010: Byte-Level BPE Decoder (provides tokenizer)
- FT-010: CUDA Context Initialization (provides CUDA context)

### Downstream (This Sprint Blocks)
- Sprint 4: GQA Attention + Gate 1 (needs these kernels)
- LT-015: GQA Attention Kernel (needs RoPE, RMSNorm, residual)

---

## Success Criteria

Sprint is complete when:
- [ ] All 4 stories marked complete
- [ ] UTF-8 streaming handles partial sequences correctly
- [ ] RoPE kernel working
- [ ] RMSNorm kernel working
- [ ] Residual connection kernel working
- [ ] All unit tests passing
- [ ] Ready for Sprint 4 (GQA attention)

---

## Next Sprint

**Sprint 4**: GQA Attention + Gate 1  
**Starts**: Day 42  
**Focus**: GQA attention, SwiGLU, conformance tests, Gate 1

---

**Status**: 📋 Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-05

---
Coordinated by Project Management Team 📋
