# GPT Team (GPT-Gamma) - Planning Artifacts

**Team**: GPT-Gamma  
**Mission**: Implement GPT architecture support for worker-orcd M0  
**Timeline**: 110 agent-days (Days 1-110)  
**Status**: 📋 **PLANNING COMPLETE - READY FOR EXECUTION**

---

## Overview

The GPT team is responsible for implementing GPT architecture support in worker-orcd, including:
- HuggingFace tokenizer integration
- GPT-specific CUDA kernels (LayerNorm, GELU, MHA)
- MXFP4 quantization (novel 4-bit format)
- GPTInferenceAdapter (architecture adapter pattern)
- GPT-OSS-20B model support (20B parameters in 24GB VRAM)

---

## Planning Artifacts

### ✅ Story Cards (48 total)
Detailed story cards with acceptance criteria, technical specifications, and testing strategies.

**Location**: `stories/`

**Breakdown**:
- `GT-000-prep/` - 1 story (MXFP4 spec study)
- `GT-001-to-GT-010/` - 10 stories (HF tokenizer + GPT metadata)
- `GT-011-to-GT-020/` - 10 stories (GPT kernels: LayerNorm, GELU, MHA)
- `GT-021-to-GT-030/` - 10 stories (Kernel integration + Gate 1 & 2)
- `GT-031-to-GT-040/` - 9 stories (MXFP4 integration)
- `GT-041-to-GT-048/` - 8 stories (Adapter + final integration)

**Note**: GT-032 skipped per PM plan

---

### ✅ Sprint READMEs (9 total)
Sprint goals, story sequences, and coordination points.

**Location**: `sprints/`

**Sprints**:
1. **Sprint 0**: Prep Work (Days 1-3)
2. **Sprint 1**: HF Tokenizer (Days 15-26)
3. **Sprint 2**: GPT Kernels (Days 27-41)
4. **Sprint 3**: MHA + Gate 1 (Days 42-57)
5. **Sprint 4**: GPT Basic (Days 56-66)
6. **Sprint 5**: MXFP4 Dequant (Days 67-74)
7. **Sprint 6**: MXFP4 Integration (Days 75-89)
8. **Sprint 7**: Adapter + E2E (Days 90-96)
9. **Sprint 8**: Final Integration (Days 97-110)

---

### ✅ Gate Checklists (3 total)
Validation checklists for major gates.

**Location**: `integration-gates/`

**Gates**:
1. **Gate 1**: GPT Kernels Complete (Day 53)
2. **Gate 2**: GPT Basic Working (Day 66)
3. **Gate 3**: MXFP4 + Adapter Complete (Day 96)

---

### ✅ Execution Templates (4 total)
Day-to-day tracking and coordination documents.

**Location**: `execution/`

**Templates**:
1. **day-tracker.md** - Daily progress tracking
2. **dependencies.md** - Upstream/downstream dependency tracking
3. **milestones.md** - Sprint and gate milestone tracking
4. **mxfp4-validation-framework.md** - MXFP4 validation strategy

---

## Quick Start for Engineers

### 1. Understand the Mission
Read: `STORY_GENERATION_SUMMARY.md`

### 2. Review Your Sprint
Navigate to: `sprints/sprint-N-name/README.md`

### 3. Pick Your Story
Navigate to: `stories/GT-XXX-to-GT-YYY/GT-XXX-story-name.md`

### 4. Track Your Progress
Update: `execution/day-tracker.md` daily

### 5. Check Dependencies
Review: `execution/dependencies.md` for blockers

### 6. Validate Gates
When ready: `integration-gates/gate-N-name.md`

---

## Key Milestones

| Milestone | Day | Description |
|-----------|-----|-------------|
| FFI Lock | 15 | FFI interface frozen, Sprint 1 starts |
| Gate 1 | 53 | GPT kernels complete and validated |
| Gate 2 | 66 | GPT-OSS-20B basic inference working |
| Adapter Lock | 71 | InferenceAdapter interface frozen |
| Gate 3 | 96 | MXFP4 + adapter complete |
| M0 Delivery | 110 | GPT-OSS-20B production-ready |

---

## Critical Path

```
Sprint 0 (Prep) → Sprint 1 (HF Tokenizer) → Sprint 2 (GPT Kernels) 
  → Sprint 3 (MHA + Gate 1) → Sprint 4 (GPT Basic + Gate 2)
  → Sprint 5 (MXFP4 Dequant) → Sprint 6 (MXFP4 Integration)
  → Sprint 7 (Adapter + Gate 3) → Sprint 8 (Final Integration)
  → M0 Delivery
```

---

## Technical Highlights

### HuggingFace Tokenizer
- Pure Rust implementation (no Python)
- tokenizer.json format support
- Conformance tests against reference

### GPT-Specific Kernels
- **LayerNorm** (not RMSNorm like Llama)
- **GELU** (exact formula, not tanh approximation)
- **MHA** (Multi-Head Attention, not GQA)
- **Absolute positional embeddings** (not RoPE)

### MXFP4 Quantization
- Novel 4-bit format (4-bit mantissa + 8-bit shared exponent)
- ~4x memory savings vs FP16
- Target: ±1% accuracy vs FP16
- Critical for fitting 20B model in 24GB VRAM

### Architecture Adapter
- GPTInferenceAdapter implements InferenceAdapter interface
- Architecture detection from GGUF metadata
- Automatic routing to GPT-specific kernels

---

## Team Coordination

### Dependencies on Foundation Team
- FFI Interface Lock (Day 15)
- cuBLAS GEMM Wrapper (Day 30)
- InferenceAdapter Interface (Day 61)

### Dependencies on Llama Team
- Memory-Mapped I/O (Day 20)
- GQA reference for comparison (Day 40)

### What GPT Team Provides
- Architecture Detection (GT-007)
- GPTInferenceAdapter (GT-039)
- MHA vs GQA documentation (GT-019)

---

## Success Metrics

- **48 stories** created with detailed acceptance criteria
- **9 sprints** planned with clear goals
- **3 gates** defined with validation procedures
- **110 agent-days** estimated timeline
- **100% planning coverage** - no ambiguity

---

## Document Index

### Planning Documents
- `README.md` - This file (team overview)
- `STORY_GENERATION_SUMMARY.md` - Complete story inventory
- `PM_RESPONSIBILITIES.md` - PM role and standards
- `PM_ARTIFACT_GENERATION_PLAN.md` - Artifact generation plan
- `PM_WORK_BREAKDOWN.md` - Work breakdown structure

### Story Cards
- `stories/GT-000-prep/` - Prep work
- `stories/GT-001-to-GT-010/` - HF tokenizer + metadata
- `stories/GT-011-to-GT-020/` - GPT kernels
- `stories/GT-021-to-GT-030/` - Integration + gates
- `stories/GT-031-to-GT-040/` - MXFP4 integration
- `stories/GT-041-to-GT-048/` - Final integration

### Sprint Plans
- `sprints/sprint-0-prep-work/`
- `sprints/sprint-1-hf-tokenizer/`
- `sprints/sprint-2-gpt-kernels/`
- `sprints/sprint-3-mha-gate1/`
- `sprints/sprint-4-gpt-basic/`
- `sprints/sprint-5-mxfp4-dequant/`
- `sprints/sprint-6-mxfp4-integration/`
- `sprints/sprint-7-adapter-e2e/`
- `sprints/sprint-8-final-integration/`

### Gate Checklists
- `integration-gates/gate-1-gpt-kernels.md`
- `integration-gates/gate-2-gpt-basic.md`
- `integration-gates/gate-3-mxfp4-adapter.md`

### Execution Templates
- `execution/day-tracker.md`
- `execution/dependencies.md`
- `execution/milestones.md`
- `execution/mxfp4-validation-framework.md`

---

## Contact

**Team**: GPT-Gamma  
**PM**: Project Management Team 📋  
**Status**: Ready for execution

---

**Version**: 1.0.0  
**Last Updated**: 2025-10-04  
**Planning Complete**: ✅

---
Planned by Project Management Team 📋
