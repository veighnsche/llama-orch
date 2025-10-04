# Sprint 4: GPT Basic

**Team**: GPT-Gamma  
**Days**: 56-66 (11 agent-days)  
**Goal**: Implement basic GPT-OSS-20B inference pipeline with Q4_K_M quantization

---

## Sprint Overview

Sprint 4 brings together all GPT kernels into a working inference pipeline. Load GPT-OSS-20B with Q4_K_M quantization and generate text. This validates the complete GPT architecture implementation before adding MXFP4 support.

This sprint culminates in Gate 2: GPT Basic Working.

---

## Stories in This Sprint

| ID | Title | Size | Days | Day Range |
|----|-------|------|------|-----------|
| GT-024 | GPT Weight Mapping (Q4_K_M) | L | 3 | 58-60 |
| GT-025 | GPT Weight Loading | M | 2 | 61-62 |
| GT-026 | GPT Forward Pass (Q4_K_M) | L | 3 | 63-65 |
| GT-027 | GPT Basic Generation Test | S | 1 | 66 |
| GT-028 | Gate 2 Checkpoint | M | 1 | 66 |

**Total**: 5 stories, 11 agent-days (Days 56-66, Gate 2 on Day 66)

---

## Critical Milestones

### Gate 2: GPT Basic Working (Day 66)
**What**: Validate GPT-OSS-20B loads and generates with Q4_K_M  
**Why Critical**: Proves GPT architecture works before MXFP4  
**Deliverable**: Gate 2 validation report, working text generation  
**Checklist**: See `integration-gates/gate-2-gpt-basic.md`

---

## Success Criteria

Sprint is complete when:
- [ ] GPT-OSS-20B weight mapping complete
- [ ] Model loads successfully
- [ ] Forward pass executes correctly
- [ ] Text generation works
- [ ] **Gate 2 passed**
- [ ] Ready for Sprint 5 (MXFP4 dequantization)

---

## Next Sprint

**Sprint 5**: MXFP4 Dequant  
**Starts**: Day 67  
**Focus**: Implement MXFP4 dequantization kernel

---

**Status**: 📋 Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Coordinated by Project Management Team 📋
