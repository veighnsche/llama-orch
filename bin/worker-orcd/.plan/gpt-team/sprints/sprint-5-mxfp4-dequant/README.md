# Sprint 5: MXFP4 Dequant

**Team**: GPT-Gamma  
**Days**: 67-74 (8 agent-days)  
**Goal**: Implement MXFP4 dequantization kernel (novel 4-bit format)

---

## Sprint Overview

Sprint 5 implements the critical MXFP4 dequantization kernel. MXFP4 is a novel quantization format (4-bit mantissa + shared 8-bit exponent per 32-element block) that enables fitting GPT-OSS-20B in 24GB VRAM.

This is the foundation for Sprint 6's full MXFP4 integration.

---

## Stories in This Sprint

| ID | Title | Size | Days | Day Range |
|----|-------|------|------|-----------|
| GT-029 | MXFP4 Dequantization Kernel | L | 3 | 67-69 |
| GT-030 | MXFP4 Unit Tests | M | 2 | 70-71 |
| GT-031 | UTF-8 Streaming Safety Tests | S | 1 | 72 |

**Total**: 3 stories, 8 agent-days (Days 67-74)

---

## Technical Highlights

### MXFP4 Format
- **Block size**: 32 FP4 values + 1 FP8 scale = 17 bytes
- **Dequantization**: `fp16 = fp4_mantissa * fp8_scale`
- **Accuracy target**: ±1% vs FP16
- **Memory savings**: ~4x vs FP16

---

## Success Criteria

Sprint is complete when:
- [ ] MXFP4 dequantization kernel implemented
- [ ] Unit tests validate correctness (±1%)
- [ ] UTF-8 streaming safety validated
- [ ] Ready for Sprint 6 (MXFP4 integration)

---

## Next Sprint

**Sprint 6**: MXFP4 Integration  
**Starts**: Day 75  
**Focus**: Integrate MXFP4 with all weight consumers (GEMM, embeddings, attention, FFN)

---

**Status**: 📋 Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Coordinated by Project Management Team 📋
