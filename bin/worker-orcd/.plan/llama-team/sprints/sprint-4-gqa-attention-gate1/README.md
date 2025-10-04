# Sprint 4: GQA Attention + Gate 1

**Team**: Llama-Beta  
**Days**: 42-54 (13 agent-days)  
**Goal**: Complete Llama kernels and reach Gate 1

---

## Sprint Overview

Sprint 4 implements the most complex Llama kernels (GQA attention, SwiGLU FFN) and establishes comprehensive testing with tokenizer conformance tests and kernel unit tests. This sprint culminates in Gate 1 validation, proving all Llama kernels are complete and working.

This is a critical milestone that enables Qwen integration in Sprint 5.

---

## Stories in This Sprint

| ID | Title | Size | Days | Day Range |
|----|-------|------|------|-----------|
| LT-015 | GQA Attention Kernel (Prefill) | L | 4 | 42-45 |
| LT-016 | GQA Attention Kernel (Decode) | M | 2 | 46-47 |
| LT-017 | SwiGLU FFN Kernel | M | 2 | 48-49 |
| LT-018 | Tokenizer Conformance Tests (Qwen) | M | 2 | 50-51 |
| LT-019 | Kernel Unit Tests | M | 2 | 52-53 |
| LT-020 | Gate 1 Participation | S | 1 | 54 |

**Total**: 6 stories, 13 agent-days (Days 42-54)

---

## Story Execution Order

### Days 42-45: LT-015 - GQA Attention Kernel (Prefill)
**Goal**: Implement GQA attention for prefill phase  
**Key Deliverable**: GQA prefill kernel  
**Blocks**: LT-016 (GQA decode)

### Days 46-47: LT-016 - GQA Attention Kernel (Decode)
**Goal**: Implement GQA attention for decode phase  
**Key Deliverable**: GQA decode kernel  
**Blocks**: LT-017 (SwiGLU)

### Days 48-49: LT-017 - SwiGLU FFN Kernel
**Goal**: Implement SwiGLU feed-forward network kernel  
**Key Deliverable**: SwiGLU FFN kernel  
**Blocks**: LT-018 (conformance tests)

### Days 50-51: LT-018 - Tokenizer Conformance Tests (Qwen)
**Goal**: Create conformance tests for Qwen tokenizer  
**Key Deliverable**: 20-30 test pairs validating tokenizer  
**Blocks**: LT-019 (kernel tests)

### Days 52-53: LT-019 - Kernel Unit Tests
**Goal**: Comprehensive unit tests for all kernels  
**Key Deliverable**: Unit tests for all Llama kernels  
**Blocks**: LT-020 (Gate 1)

### Day 54: LT-020 - Gate 1 Participation
**Goal**: Participate in Gate 1 validation  
**Key Deliverable**: Gate 1 checkpoint passed  
**Blocks**: Sprint 5 (Qwen integration)

---

## Dependencies

### Upstream (Blocks This Sprint)
- LT-012: RoPE Kernel (provides RoPE)
- LT-013: RMSNorm Kernel (provides RMSNorm)
- LT-014: Residual Connection Kernel (provides residual)
- FT-052: Integration Framework (Day 52, provides Gate 1 framework)

### Downstream (This Sprint Blocks)
- Sprint 5: Qwen Integration (needs all kernels)
- LT-022: Qwen Weight Mapping (needs complete kernel set)

---

## Critical Milestone: Gate 1 (Day 54)

**What**: Llama kernels complete and validated  
**Why Critical**: Proves Llama kernel foundation is solid  
**Deliverable**: Gate 1 validation report

**Checklist**:
- [ ] All Llama kernels implemented
- [ ] All kernel unit tests passing
- [ ] Tokenizer conformance tests passing
- [ ] Performance within budget
- [ ] Documentation complete
- [ ] Gate 1 report published

---

## Success Criteria

Sprint is complete when:
- [ ] All 6 stories marked complete
- [ ] GQA attention working (prefill + decode)
- [ ] SwiGLU FFN working
- [ ] Tokenizer conformance tests passing (20-30 pairs)
- [ ] All kernel unit tests passing
- [ ] Gate 1 checkpoint passed
- [ ] All unit tests passing
- [ ] Ready for Sprint 5 (Qwen integration)

---

## Next Sprint

**Sprint 5**: Qwen Integration  
**Starts**: Day 55  
**Focus**: First complete model pipeline - Qwen2.5-0.5B

---

**Status**: 📋 Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-05

---
Coordinated by Project Management Team 📋
