# GPT Team Dependencies

**Team**: GPT-Gamma  
**Purpose**: Track upstream and downstream dependencies

---

## Instructions

This document tracks:
1. **Upstream dependencies**: What blocks GPT team work
2. **Downstream dependencies**: What GPT team blocks
3. **Dependency status**: Resolved, waiting, blocked

---

## Upstream Dependencies (Blocks GPT Team)

### From Foundation Team (Foundation-Alpha)

| Dependency | Story | Status | Expected | Actual | Impact |
|------------|-------|--------|----------|--------|--------|
| FFI Interface Lock | FT-006, FT-007 | ⬜ Waiting | Day 15 | - | Blocks Sprint 1 |
| cuBLAS GEMM Wrapper | FT-016 | ⬜ Waiting | Day 30 | - | Blocks GT-014, GT-033 |
| Embedding Lookup Kernel | FT-015 | ⬜ Waiting | Day 28 | - | Blocks GT-008 |
| InferenceAdapter Interface | FT-033 | ⬜ Waiting | Day 61 | - | Blocks GT-039 |
| Device Memory RAII | FT-013 | ⬜ Waiting | Day 26 | - | Blocks GT-006 |

### From Llama Team (Llama-Beta)

| Dependency | Story | Status | Expected | Actual | Impact |
|------------|-------|--------|----------|--------|--------|
| Memory-Mapped I/O | LT-003 | ⬜ Waiting | Day 20 | - | Blocks GT-025 |
| GQA Attention (for comparison) | LT-016 | ⬜ Waiting | Day 40 | - | Blocks GT-019 |

---

## Downstream Dependencies (GPT Team Blocks)

### Blocks Foundation Team

| What GPT Provides | Story | Status | Needed By | Impact |
|-------------------|-------|--------|-----------|--------|
| Architecture Detection | GT-007 | ⬜ Pending | FT-035 | Blocks adapter integration |
| GPTInferenceAdapter | GT-039 | ⬜ Pending | FT-035 | Blocks architecture routing |

### Blocks Llama Team

| What GPT Provides | Story | Status | Needed By | Impact |
|-------------------|-------|--------|-----------|--------|
| MHA vs GQA Documentation | GT-019 | ⬜ Pending | LT-019 | Reference for GQA validation |

---

## Internal Dependencies (Within GPT Team)

### Critical Path

```
GT-000 (Prep)
  ↓
GT-001 → GT-002 → GT-003 → GT-004 (HF Tokenizer)
  ↓
GT-005 → GT-006 → GT-007 (GPT Metadata + Architecture)
  ↓
GT-008 → GT-009 → GT-010 → GT-011 (LayerNorm)
  ↓
GT-012 → GT-013 → GT-014 (GELU + FFN)
  ↓
GT-015 → GT-016 (Residual + Integration)
  ↓
GT-017 → GT-018 → GT-019 → GT-020 (MHA)
  ↓
GT-021 → GT-022 (Kernel Suite + Gate 1)
  ↓
GT-024 → GT-025 → GT-026 → GT-027 → GT-028 (GPT Basic + Gate 2)
  ↓
GT-029 → GT-030 (MXFP4 Dequant)
  ↓
GT-033 → GT-034 → GT-035 → GT-036 → GT-037 → GT-038 (MXFP4 Integration)
  ↓
GT-039 → GT-040 → GT-041 (Adapter + Gate 3)
  ↓
GT-042 → GT-043 → GT-044 → GT-045 → GT-046 → GT-047 → GT-048 (Final Integration)
```

---

## Dependency Status Summary

### Resolved
- [None yet]

### Waiting
- FFI Interface Lock (Day 15)
- cuBLAS GEMM Wrapper (Day 30)
- InferenceAdapter Interface (Day 61)

### Blocked
- [None currently]

---

## Coordination Points

### FFI Lock (Day 15)
**Participants**: Foundation-Alpha, Llama-Beta, GPT-Gamma  
**What**: FFI interface frozen, no more changes  
**Impact**: Unblocks all team Sprint 1 work

### Adapter Pattern Lock (Day 71)
**Participants**: Foundation-Alpha, Llama-Beta, GPT-Gamma  
**What**: InferenceAdapter interface frozen  
**Impact**: Enables architecture-specific adapters

---

## Escalation

If a dependency is blocked or delayed:
1. Document the blocker in this file
2. Update day-tracker.md with blocker
3. Notify PM (update milestones.md)
4. Coordinate with blocking team

---

**Last Updated**: [Date]  
**Updated By**: GPT-Gamma

---
Tracked by Project Management Team 📋
