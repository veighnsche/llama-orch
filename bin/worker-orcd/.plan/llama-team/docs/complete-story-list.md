# Llama Team - Complete Story List

**Agent**: Llama-Beta (Autonomous Development Agent)  
**Total Stories**: 38 stories  
**Total Estimated Effort**: ~72 agent-days (sequential execution)  
**Timeline**: ~72 calendar days starting after FFI lock (day 15)  
**Completion**: Day 87 (15 + 72)

---

## Sprint 1: GGUF Foundation (6 stories, 11 agent-days)

**Starts**: Day 15 (after Foundation-Alpha locks FFI)  
**Goal**: Parse GGUF format and establish loader infrastructure

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| LT-001 | GGUF Header Parser | M | 2 | M0-W-1211 |
| LT-002 | GGUF Metadata Extraction (Llama) | M | 2 | M0-W-1211, M0-W-1212 |
| LT-003 | Memory-Mapped I/O Implementation | M | 2 | M0-W-1221 |
| LT-004 | Chunked H2D Transfer | M | 2 | M0-W-1222 |
| LT-005 | Pre-Load Validation | M | 2 | M0-W-1210 |
| LT-006 | Architecture Detection (Llama) | S | 1 | M0-W-1212 |

**Sequential Execution**: Agent completes LT-001 fully before starting LT-002, etc.  
**Timeline**: Days 15-25  
**Blocks**: None (first sprint after FFI lock)  
**Dependency**: Requires Foundation-Alpha's FFI interface (locked day 15)

---

## Sprint 2: GGUF-BPE Tokenizer (5 stories, 9 agent-days)

**Goal**: Implement pure Rust byte-level BPE tokenizer

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| LT-007 | GGUF Vocab Parsing | M | 2 | M0-W-1362 |
| LT-008 | GGUF Merges Parsing | M | 2 | M0-W-1362 |
| LT-009 | Byte-Level BPE Encoder | M | 3 | M0-W-1362 |
| LT-010 | Byte-Level BPE Decoder | M | 2 | M0-W-1362 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 26-34  
**Dependency**: Requires LT-001-006 (GGUF loader) to extract vocab/merges

---

## Sprint 3: UTF-8 Safety + Llama Kernels (4 stories, 6 agent-days)

**Goal**: Complete tokenizer and implement Llama-specific CUDA kernels

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| LT-011 | UTF-8 Safe Streaming Decode | M | 2 | M0-W-1362 |
| LT-012 | RoPE Kernel | M | 2 | M0-W-1214, M0-W-1430 |
| LT-013 | RMSNorm Kernel | S | 1 | M0-W-1214, M0-W-1430 |
| LT-014 | Residual Connection Kernel | S | 1 | M0-W-1214 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 35-40  
**Note**: Tokenizer must be complete before starting Qwen integration

---

## Sprint 4: GQA Attention + Integration (7 stories, 13 agent-days)

**Goal**: Complete Llama kernels and reach Gate 1

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| LT-015 | GQA Attention Kernel (Prefill) | L | 4 | M0-W-1214, M0-W-1430 |
| LT-016 | GQA Attention Kernel (Decode) | M | 2 | M0-W-1214 |
| LT-017 | SwiGLU FFN Kernel | M | 2 | M0-W-1214 |
| LT-018 | Tokenizer Conformance Tests (Qwen) | M | 2 | M0-W-1363 |
| LT-019 | Kernel Unit Tests | M | 2 | M0-W-1430 |
| LT-020 | **Gate 1 Participation** | S | 1 | Gate 1 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 41-53  
**Critical Milestone**: Gate 1 validation - Llama kernels complete  
**Dependency**: Requires Foundation-Alpha's integration framework (day 52)

---

## Sprint 5: Qwen Integration (6 stories, 13 agent-days) 🔴 CRITICAL

**Goal**: First complete model pipeline - Qwen2.5-0.5B working end-to-end

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| LT-022 | Qwen Weight Mapping | M | 3 | M0-W-1230 |
| LT-023 | Qwen Weight Loading to VRAM | M | 2 | M0-W-1220, M0-W-1221 |
| LT-024 | Qwen Forward Pass Implementation | L | 4 | M0-W-1214, M0-W-1420 |
| LT-025 | Qwen Haiku Generation Test | M | 2 | M0-W-1800 |
| LT-026 | Qwen Reproducibility Validation | M | 2 | M0-W-1826 |
| LT-027 | **Gate 2 Checkpoint** | - | - | Gate 2 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 54-66  
**Critical Milestone**: Gate 2 - First Llama model working (Qwen)  
**Note**: This validates entire Llama pipeline

---

## Sprint 6: Phi-3 + Adapter (6 stories, 11 agent-days)

**Goal**: Second Llama model + LlamaInferenceAdapter implementation

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| LT-029 | Phi-3 Metadata Analysis | S | 1 | M0-W-1230 |
| LT-030 | Phi-3 Weight Loading | M | 2 | M0-W-1230 |
| LT-031 | Phi-3 Forward Pass (Adapt from Qwen) | M | 2 | M0-W-1214 |
| LT-032 | Tokenizer Conformance Tests (Phi-3) | M | 2 | M0-W-1363 |
| LT-033 | LlamaInferenceAdapter Implementation | M | 3 | M0-W-1213, M0-W-1214 |
| LT-034 | **Gate 3 Participation** | S | 1 | Gate 3 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 67-77  
**Critical Milestone**: Gate 3 - Adapter pattern complete  
**Dependency**: Requires Foundation-Alpha's adapter pattern (day 71)

---

## Sprint 7: Final Integration (4 stories, 9 agent-days)

**Goal**: Complete Llama testing and documentation

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------| 
| LT-035 | Llama Integration Test Suite | M | 3 | M0-W-1818 |
| LT-036 | Reproducibility Tests (10 runs × 2 models) | M | 2 | M0-W-1826 |
| LT-037 | VRAM Pressure Tests (Phi-3) | M | 2 | M0-W-1230 |
| LT-038 | Documentation (GGUF, BPE, Llama) | M | 2 | (Docs) |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 78-86  
**Note**: Llama-Beta finishes day 86, before GPT-Gamma (day 102)

---

## Agent Execution Reality

### ✅ **NO OVERCOMMITMENT ISSUE**

**Reality**: Llama-Beta is a single agent working sequentially, not a team with parallel capacity.

**Agent Characteristics**:
- Works sequentially through all 38 stories
- Completes each story fully before moving to next
- Can work on multiple files simultaneously within a story
- No parallel work across stories
- No "overcommitment" - agent works until done

**Timeline**: 72 agent-days starting day 15 = completes day 87 (15 + 72)

**Critical Path Position**: Llama-Beta (87 days total) finishes before GPT-Gamma (102 days)

### Key Milestones

**Day 15 (Start)**: FFI Interface Locked by Foundation-Alpha
- **Unblocks**: Llama-Beta can begin GGUF loader work
- **Action**: Begin LT-001

**Day 53 (After LT-020)**: Gate 1 Complete
- **Validation**: Llama kernels complete and tested
- **Enables**: Qwen integration can proceed

**Day 66 (After LT-027)**: Gate 2 Complete  
- **Validation**: First Llama model (Qwen) working end-to-end
- **Critical**: Proves Llama pipeline works

**Day 77 (After LT-034)**: Gate 3 Complete
- **Validation**: LlamaInferenceAdapter implemented
- **Enables**: Refactoring to use adapter pattern

**Day 87 (After LT-038)**: Llama-Beta Complete
- **Note**: GPT-Gamma still working (finishes day 102)

---

## Dependency Analysis

### Sequential Execution Chain

```
Day 15: FFI Lock (Foundation-Alpha) ← START POINT
  ↓
Days 15-25: GGUF Loader (LT-001 → LT-006)
  ↓
Days 26-34: Tokenizer (LT-007 → LT-010)
  ↓
Days 35-40: UTF-8 + Kernels (LT-011 → LT-014)
  ↓
Days 41-53: GQA + Gate 1 (LT-015 → LT-020)
  ↓
Days 54-66: Qwen + Gate 2 (LT-022 → LT-027) ← FIRST MODEL
  ↓
Days 67-77: Phi-3 + Adapter + Gate 3 (LT-029 → LT-034)
  ↓
Days 78-87: Final Testing (LT-035 → LT-038)
```

**Total Duration**: 72 agent-days (starting day 15)

**Critical Dependencies**:
- **Day 15**: FFI lock blocks start
- **Day 25**: GGUF loader blocks tokenizer
- **Day 34**: Tokenizer blocks Qwen integration
- **Day 52**: Foundation's integration framework blocks Gate 1
- **Day 71**: Foundation's adapter pattern blocks Gate 3

**No "Slack Time"**: Agent works sequentially until done

---

## Risk Register (Revised for AI Agent Reality)

### Actual Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| FFI lock delayed beyond day 15 | Llama-Beta blocked | Coordinate with Foundation-Alpha |
| GGUF format edge cases | Parsing failures | Comprehensive tests, reference llama.cpp |
| BPE algorithm bugs | Tokenization broken | Conformance test vectors, study reference |
| GQA attention complexity | Integration delays | Reference llama.cpp, unit tests |
| Qwen forward pass bugs | Gate 2 failure | Incremental integration, unit tests |

### ❌ Not Risks (Human Team Assumptions)

- ~~"Week 3 overcommitment"~~ - Agent works sequentially
- ~~"Need 3rd person"~~ - Cannot scale agent count
- ~~"Utilization percentage"~~ - Meaningless for sequential execution
- ~~"Parallel work streams"~~ - Agent works one story at a time

---

## Key Actions for Llama-Beta

### 1. **Wait for FFI Lock (Day 15)** 🔴 CRITICAL

**Action**: Monitor Foundation-Alpha's progress on FT-006 and FT-007
- Begin work immediately when `FFI_INTERFACE_LOCKED.md` published
- Study FFI interface documentation during wait time

### 2. **Study Reference Implementations**

**Action**: Research llama.cpp during prep time (days 1-14)
- GGUF format parsing patterns
- BPE tokenization algorithm
- GQA attention implementation
- RoPE, RMSNorm, SwiGLU kernels

### 3. **Build Conformance Tests First**

**Action**: Create test vectors before implementation
- Tokenizer conformance tests (20-30 pairs)
- Kernel validation tests
- Reproducibility test framework

---

## Timeline Summary

**Llama-Beta**: 72 agent-days starting day 15 = completes day 87  
**Critical Dependency**: Day 15 FFI lock  
**Gate 2**: Day 66 (First Llama model working)  
**Completion**: Day 87 (before GPT-Gamma's day 102)

---

**Status**: ✅ **REVISED FOR AI AGENT REALITY**  
**Next Action**: Wait for FFI lock, then begin LT-001  
**Owner**: Llama-Beta

---

*Implemented by Llama-Beta 🦙*

