# Foundation Team Dependencies

**Team**: Foundation-Alpha  
**Purpose**: Track all upstream and downstream dependencies

---

## Dependency Overview

Foundation team is the **first team to start** and **blocks all other teams**. This document tracks what blocks Foundation work and what Foundation blocks.

---

## Upstream Dependencies (What Blocks Foundation)

### None - Foundation Starts First

Foundation-Alpha has **no upstream dependencies**. The team starts on Day 1 and establishes the infrastructure that all other teams depend on.

---

## Downstream Dependencies (What Foundation Blocks)

### Critical Milestone: FFI Lock (Day 11)

**Story**: FT-006 - FFI Interface Definition  
**Deliverable**: `coordination/FFI_INTERFACE_LOCKED.md`

**Blocks**:
- **Llama-Beta prep work**: Cannot start until FFI is stable
- **GPT-Gamma prep work**: Cannot start until FFI is stable

**Why Critical**: FFI interface must be frozen before model teams can build against it.

---

### Critical Milestone: Gate 1 (Day 52)

**Story**: FT-027 - Gate 1 Checkpoint  
**Validates**: HTTP + FFI + CUDA foundation working end-to-end

**Blocks**:
- **Llama-Beta Gate 1** (LT-020): Cannot participate until Foundation Gate 1 passes
- **GPT-Gamma Gate 1** (GT-022): Cannot participate until Foundation Gate 1 passes

**Why Critical**: Model teams need stable foundation before they can validate their implementations.

---

### Critical Milestone: Gate 3 (Day 71)

**Story**: FT-038 - Gate 3 Checkpoint  
**Validates**: InferenceAdapter pattern operational

**Blocks**:
- **Llama-Beta Gate 3** (LT-034): Cannot participate until Foundation Gate 3 passes
- **GPT-Gamma Gate 3** (GT-041): Cannot participate until Foundation Gate 3 passes

**Why Critical**: Adapter pattern must work before model teams can validate their adapters.

---

## Story-Level Dependencies

### Sprint 1: HTTP Foundation (Days 1-9)
**No blockers** - Foundation starts independently

**Blocks**:
- Integration test framework (needs HTTP server)
- All HTTP-based testing

---

### Sprint 2: FFI Layer (Days 10-22)

**Internal Dependencies**:
- FT-007 (Rust FFI) depends on FT-006 (FFI definition)
- FT-008 (Error codes C++) depends on FT-006 (FFI definition)
- FT-009 (Error conversion Rust) depends on FT-008 (Error codes)
- FT-010 (CUDA context) depends on FT-007 (Rust FFI)

**Blocks**:
- **All Llama-Beta work**: Waits for FFI lock
- **All GPT-Gamma work**: Waits for FFI lock

---

### Sprint 3: Shared Kernels (Days 23-38)

**Internal Dependencies**:
- FT-012 (FFI integration tests) depends on FT-010 (CUDA context)
- FT-013 (Device memory) depends on FT-010 (CUDA context)
- FT-014 (VRAM residency) depends on FT-011 (VRAM enforcement)
- FT-015-020 (Kernels) depend on FT-013 (Device memory)

**Blocks**:
- Llama kernel development (needs shared kernels)
- GPT kernel development (needs shared kernels)

---

### Sprint 4: Integration + Gate 1 (Days 39-52)

**Internal Dependencies**:
- FT-021 (KV cache alloc) depends on FT-013 (Device memory)
- FT-022 (KV cache mgmt) depends on FT-021 (KV cache alloc)
- FT-023 (Test framework) depends on FT-001-005 (HTTP layer)
- FT-024 (Integration test) depends on FT-023 (Test framework)
- FT-025 (Gate 1 validation) depends on FT-024 (Integration test)
- FT-027 (Gate 1) depends on FT-025 (Gate 1 validation)

**Blocks**:
- **Llama Gate 1 participation**: Waits for FT-027
- **GPT Gate 1 participation**: Waits for FT-027

---

### Sprint 5: Support + Prep (Days 53-60)

**Internal Dependencies**:
- FT-028 (Llama support) depends on FT-027 (Gate 1)
- FT-029 (GPT support) depends on FT-027 (Gate 1)

**Blocks**:
- Llama integration issues
- GPT integration issues

---

### Sprint 6: Adapter + Gate 3 (Days 61-71)

**Internal Dependencies**:
- FT-032 (Gate 2) depends on Llama Gate 2 + GPT Gate 2
- FT-033 (Adapter interface) depends on FT-032 (Gate 2)
- FT-034 (Factory pattern) depends on FT-033 (Adapter interface)
- FT-035 (Architecture detection) depends on FT-034 (Factory)
- FT-036 (Integration tests) depends on FT-035 (Architecture detection)
- FT-038 (Gate 3) depends on FT-036 (Integration tests)

**Blocks**:
- **Llama Gate 3 participation**: Waits for FT-038
- **GPT Gate 3 participation**: Waits for FT-038

---

### Sprint 7: Final Integration (Days 72-89)

**Internal Dependencies**:
- FT-040 (Performance baseline) depends on FT-031 (Baseline prep)
- FT-041 (All models test) depends on FT-040 (Baseline)
- FT-046 (Final validation) depends on FT-041-045 (All tests)
- FT-047 (Gate 4) depends on FT-046 (Final validation)

**Blocks**:
- **M0 release**: Waits for FT-047 (Gate 4)

---

## Critical Path

The critical path through Foundation team:

```
Day 1: Start
  ↓
Day 11: FFI Lock (FT-006) 🔒
  ↓ [Unblocks Llama + GPT prep]
Day 52: Gate 1 (FT-027) 🎯
  ↓ [Unblocks Llama + GPT Gate 1]
Day 62: Gate 2 (FT-032) 🎯
  ↓ [Validates both architectures]
Day 71: Gate 3 (FT-038) 🎯
  ↓ [Unblocks Llama + GPT Gate 3]
Day 89: Gate 4 / M0 (FT-047) 🎯
  ↓ [M0 Complete]
```

**Total Duration**: 89 days  
**Critical Milestones**: 5 (FFI Lock + 4 Gates)

---

## Cross-Team Coordination Points

### With Llama-Beta

**Foundation Provides**:
- FFI interface (Day 11)
- Shared kernels (Days 23-38)
- Integration test framework (Days 43-44)
- Adapter interface (Days 63-64)

**Foundation Needs**:
- Llama Gate 1 completion (for FT-032 Gate 2)
- Llama Gate 3 completion (for FT-047 Gate 4)

---

### With GPT-Gamma

**Foundation Provides**:
- FFI interface (Day 11)
- Shared kernels (Days 23-38)
- Integration test framework (Days 43-44)
- Adapter interface (Days 63-64)

**Foundation Needs**:
- GPT Gate 1 completion (for FT-032 Gate 2)
- GPT Gate 3 completion (for FT-047 Gate 4)

---

## Dependency Management

### How to Handle Blockers

1. **Identify blocker**: Document what's blocking progress
2. **Escalate**: Notify blocking team via coordination channel
3. **Workaround**: If possible, work on non-blocked stories
4. **Track**: Update this document with blocker status

### How to Avoid Blocking Others

1. **Publish early**: FFI lock on Day 11 (not later)
2. **Communicate**: Notify downstream teams of milestone completion
3. **Validate**: Ensure gates pass before marking complete
4. **Document**: Keep coordination docs up to date

---

**Last Updated**: 2025-10-04  
**Updated By**: Foundation-Alpha
