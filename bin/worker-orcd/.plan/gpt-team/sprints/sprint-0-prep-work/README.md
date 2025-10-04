# Sprint 0: Prep Work

**Team**: GPT-Gamma  
**Days**: 1-3 (3 agent-days)  
**Goal**: Study MXFP4 specification and prepare for HF tokenizer integration

---

## Sprint Overview

Sprint 0 is preparatory work conducted while waiting for the FFI lock (Day 15). The GPT team studies the MXFP4 quantization specification to understand the novel 4-bit format that will be critical for fitting GPT-OSS-20B in 24GB VRAM.

This sprint has no dependencies on other teams and can be completed independently during the Foundation team's FFI layer development.

---

## Stories in This Sprint

| ID | Title | Size | Days | Day Range |
|----|-------|------|------|-----------|
| GT-000 | MXFP4 Spec Study | S | 1 | 1-3 |

**Total**: 1 story, 3 agent-days

---

## Story Execution Order

### Days 1-3: GT-000 - MXFP4 Spec Study
**Goal**: Understand MXFP4 format specification  
**Key Deliverable**: Technical notes on MXFP4 block structure, dequantization algorithm, numerical accuracy expectations  
**Blocks**: GT-029 (MXFP4 Dequantization Kernel)

**What to Study**:
- MXFP4 block structure: 32 FP4 values + 1 FP8 scale = 17 bytes
- Dequantization formula: `fp16 = fp4_mantissa * fp8_scale`
- Numerical accuracy: ±1% tolerance vs FP16
- GGUF v3 tensor format for MXFP4
- Integration points with cuBLAS GEMM

---

## Dependencies

### Upstream (Blocks This Sprint)
- None (independent prep work)

### Downstream (This Sprint Blocks)
- Sprint 1: HF Tokenizer (starts Day 15 after FFI lock)
- GT-029: MXFP4 Dequantization Kernel (needs spec knowledge)

---

## Success Criteria

Sprint is complete when:
- [ ] MXFP4 specification studied and documented
- [ ] Technical notes created with block structure details
- [ ] Dequantization algorithm understood
- [ ] Numerical accuracy expectations documented
- [ ] Ready to implement MXFP4 kernels in Sprint 5

---

## Next Sprint

**Sprint 1**: HF Tokenizer  
**Starts**: Day 15 (after FFI lock)  
**Focus**: Integrate HuggingFace tokenizers crate for GPT-OSS-20B

---

**Status**: 📋 Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Coordinated by Project Management Team 📋
