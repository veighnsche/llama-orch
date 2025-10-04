# GPT Team - Complete Story List

**Agent**: GPT-Gamma (Autonomous Development Agent)  
**Total Stories**: 51 stories (48 original + 3 security)  
**Total Estimated Effort**: ~95 agent-days (sequential execution)  
**Timeline**: ~95 calendar days starting after FFI lock (day 15)  
**Completion**: Day 110 (15 + 95) ← **CRITICAL PATH FOR M0**

**Security Updates**: Added 3 security stories (+3 days) for GGUF parsing bounds validation and MXFP4 quantization attack mitigation.

---

## Sprint 1: HF Tokenizer + GPT Metadata (8 stories, 13 agent-days)

**Starts**: Day 15 (after Foundation-Alpha locks FFI)  
**Goal**: Integrate HuggingFace tokenizers crate and parse GPT GGUF metadata

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| GT-001 | HF Tokenizers Crate Integration | S | 1 | M0-W-1361 |
| GT-002 | tokenizer.json Loading | M | 2 | M0-W-1361 |
| GT-003 | Tokenizer Metadata Exposure | M | 2 | M0-W-1361 |
| GT-004 | HF Tokenizer Conformance Tests | M | 2 | M0-W-1363 |
| GT-005 | GPT GGUF Metadata Parsing | M | 2 | M0-W-1211, M0-W-1212 |
| GT-005a | GGUF Bounds Validation (Security) | M | 1 | M0-W-1211a (heap overflow fix) |
| GT-006 | GGUF v3 Tensor Support (MXFP4 Parsing) | M | 2 | M0-W-1201, M0-W-1211 |
| GT-007 | Architecture Detection (GPT) | S | 1 | M0-W-1212 |

**Sequential Execution**: Agent completes GT-001 fully before starting GT-002, etc.  
**Timeline**: Days 15-27  
**Blocks**: None (first sprint after FFI lock)  
**Dependency**: Requires Foundation-Alpha's FFI interface (locked day 15)

**Security Note**: GT-005a adds GGUF bounds validation to prevent heap overflow vulnerabilities (CWE-119/787). Includes fuzzing tests, property tests, and edge case validation.

---

## Sprint 2: GPT Kernels Foundation (8 stories, 15 agent-days)

**Goal**: Implement GPT-specific CUDA kernels (LayerNorm, GELU, etc.)

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| GT-008 | Absolute Positional Embedding Kernel | S | 1 | M0-W-1434, M0-W-1215 |
| GT-009 | LayerNorm Kernel (Mean Reduction) | M | 2 | M0-W-1432, M0-W-1215 |
| GT-010 | LayerNorm Kernel (Variance + Normalize) | M | 2 | M0-W-1432 |
| GT-011 | LayerNorm Unit Tests | M | 2 | M0-W-1432 |
| GT-012 | GELU Activation Kernel | S | 1 | M0-W-1433, M0-W-1215 |
| GT-013 | GELU Unit Tests | S | 1 | M0-W-1433 |
| GT-014 | GPT FFN Kernel (fc1 + GELU + fc2) | M | 3 | M0-W-1215 |
| GT-015 | Residual Connection Kernel (if not shared) | S | 1 | M0-W-1215 |
| GT-016 | Kernel Integration Tests | M | 2 | (Testing) |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 28-42  
**Note**: LayerNorm is more complex than Llama's RMSNorm (two reduction passes)

---

## Sprint 3: MHA Attention + Integration (7 stories, 14 agent-days)

**Goal**: Complete MHA attention and reach Gate 1

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| GT-017 | MHA Attention Kernel (Prefill Phase) | L | 4 | M0-W-1215 |
| GT-018 | MHA Attention Kernel (Decode Phase) | M | 2 | M0-W-1215 |
| GT-019 | MHA vs GQA Differences Validation | M | 2 | M0-W-1215 |
| GT-020 | MHA Unit Tests | M | 2 | (Testing) |
| GT-021 | GPT Kernel Suite Integration | M | 2 | (Integration) |
| GT-022 | **Gate 1 Participation** | S | 1 | Gate 1 |
| GT-023 | FFI Integration Tests (GPT Kernels) | S | 1 | (Testing) |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 43-56  
**Critical Milestone**: Gate 1 validation - GPT kernels complete  
**Dependency**: Requires Foundation-Alpha's integration framework (day 52)

---

## Sprint 4: GPT Basic Pipeline (5 stories, 11 agent-days)

**Goal**: Get GPT-OSS-20B working with Q4_K_M fallback (before MXFP4)

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| GT-024 | GPT Weight Mapping (Q4_K_M Fallback) | M | 3 | M0-W-1230, M0-W-1215 |
| GT-025 | GPT Weight Loading to VRAM | M | 2 | M0-W-1220, M0-W-1221 |
| GT-026 | GPT Forward Pass (Q4_K_M) | L | 4 | M0-W-1215 |
| GT-027 | GPT Basic Generation Test | M | 2 | (Testing) |
| GT-028 | **Gate 2 Checkpoint** | - | - | Gate 2 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 57-67  
**Critical Milestone**: Gate 2 - GPT basic working (Q4_K_M fallback)  
**Note**: Validates GPT pipeline before tackling MXFP4

---

## Sprint 5: MXFP4 Dequantization (4 stories, 9 agent-days) 🔴 CRITICAL

**Goal**: Implement novel MXFP4 dequantization kernel (no reference implementation)

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| GT-029 | MXFP4 Dequantization Kernel | L | 4 | M0-W-1201, M0-W-1820 |
| GT-030 | MXFP4 Unit Tests (Dequant Correctness) | M | 2 | M0-W-1820 |
| GT-030a | MXFP4 Behavioral Security Tests | M | 1 | MXFP4_QUANT_ATTACK.md |
| GT-031 | UTF-8 Streaming Safety Tests | M | 2 | M0-W-1312, M0-W-1361 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 68-76  
**Note**: MXFP4 is novel format - no reference implementation, build validation framework first

**Security Note**: GT-030a adds behavioral testing for quantization attack mitigation. Compares FP32 vs MXFP4 outputs for code generation safety and content integrity. See `.security/MXFP4_QUANT_ATTACK.md`.

---

## Sprint 6: MXFP4 Integration (6 stories, 15 agent-days) 🔴 CRITICAL

**Goal**: Wire MXFP4 into all weight consumers and validate numerically

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------|
| GT-033 | MXFP4 GEMM Integration | M | 3 | M0-W-1201, M0-W-1008 |
| GT-034 | MXFP4 Embedding Lookup | M | 2 | M0-W-1201 |
| GT-035 | MXFP4 Attention (Q/K/V Projections) | M | 3 | M0-W-1201 |
| GT-036 | MXFP4 FFN (Up/Down Projections) | M | 2 | M0-W-1201 |
| GT-037 | MXFP4 LM Head | M | 2 | M0-W-1201 |
| GT-038 | MXFP4 Numerical Validation (±1%) | M | 3 | M0-W-1820 |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 77-91  
**Note**: Must wire MXFP4 into ALL weight consumers sequentially

---

## Sprint 7: Adapter + End-to-End (3 stories, 8 agent-days)

**Goal**: Implement GPTInferenceAdapter and validate GPT-OSS-20B with MXFP4

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------| 
| GT-039 | GPTInferenceAdapter Implementation | M | 3 | M0-W-1213, M0-W-1215 |
| GT-040 | GPT-OSS-20B MXFP4 End-to-End | L | 4 | M0-W-1230, M0-W-1201 |
| GT-040a | Model Provenance Verification | S | 1 | MXFP4_QUANT_ATTACK.md (supply chain) |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 92-99  
**Critical Milestone**: Gate 3 - MXFP4 + Adapter working  
**Dependency**: Requires Foundation-Alpha's adapter pattern (day 71)

**Security Note**: GT-040a adds model hash verification and provenance logging for supply chain security. Verifies GPT-OSS-20B from official OpenAI source.

---

## Sprint 8: Final Integration (7 stories, 16 agent-days)

**Goal**: Complete GPT testing, large model validation, and documentation

| ID | Title | Size | Days | Spec Ref |
|----|-------|------|------|---------| 
| GT-041 | **Gate 3 Participation** | S | 1 | Gate 3 |
| GT-042 | GPT Integration Test Suite | M | 3 | M0-W-1818 |
| GT-043 | MXFP4 Regression Tests | M | 2 | M0-W-1820 |
| GT-044 | 24 GB VRAM Boundary Tests | M | 2 | M0-W-1230 |
| GT-045 | OOM Recovery Tests (GPT-OSS-20B) | M | 2 | M0-W-1021 |
| GT-046 | UTF-8 Multibyte Edge Cases | M | 2 | M0-W-1312 |
| GT-047 | Documentation (GPT, MXFP4, HF) | M | 3 | (Docs) |
| GT-048 | Performance Baseline (GPT-OSS-20B) | M | 2 | M0-W-1012 (FT-012) |

**Sequential Execution**: Agent completes each story fully before next  
**Timeline**: Days 100-115  
**Note**: GPT-Gamma finishes day 110 (last agent) - **M0 CRITICAL PATH**

---

## Agent Execution Reality

### ✅ **NO OVERCOMMITMENT ISSUE**

**Reality**: GPT-Gamma is a single agent working sequentially, not a team with parallel capacity.

**Agent Characteristics**:
- Works sequentially through all 48 stories
- Completes each story fully before moving to next
- Can work on multiple files simultaneously within a story
- No parallel work across stories
- No "overcommitment" - agent works until done

**Timeline**: 95 agent-days starting day 15 = completes day 110 (15 + 95)

**Critical Path Position**: GPT-Gamma (110 days total) is **THE CRITICAL PATH** for M0

**Security Impact**: +3 days for GGUF bounds validation (GT-005a), MXFP4 behavioral tests (GT-030a), and provenance verification (GT-040a)

### Key Milestones

**Day 15 (Start)**: FFI Interface Locked by Foundation-Alpha
- **Unblocks**: GPT-Gamma can begin HF tokenizer work
- **Action**: Begin GT-001

**Day 56 (After GT-022)**: Gate 1 Complete
- **Validation**: GPT kernels complete and tested
- **Enables**: GPT basic pipeline can proceed

**Day 67 (After GT-028)**: Gate 2 Complete  
- **Validation**: GPT basic working with Q4_K_M fallback
- **Critical**: Validates GPT pipeline before MXFP4

**Day 76 (After GT-031)**: MXFP4 Dequant Complete
- **Validation**: Novel MXFP4 kernel working
- **Enables**: Integration into weight consumers

**Day 91 (After GT-038)**: MXFP4 Integration Complete
- **Validation**: MXFP4 wired into all weight consumers
- **Critical**: Numerical correctness validated (±1%)

**Day 99 (After GT-040a)**: Gate 3 Complete
- **Validation**: GPT-OSS-20B with MXFP4 working end-to-end
- **Critical**: Proves MXFP4 path works

**Day 110 (After GT-048)**: GPT-Gamma Complete ← **M0 DELIVERY**
- **Note**: Last agent to finish - determines M0 timeline
- **Security**: All 3 security stories completed (bounds validation, behavioral tests, provenance)

---

## Dependency Analysis

### Sequential Execution Chain

```
Day 15: FFI Lock (Foundation-Alpha) ← START POINT
  ↓
Days 15-27: HF Tokenizer + GPT Metadata (GT-001 → GT-007) [+1 day security]
  ↓
Days 28-42: GPT Kernels (GT-008 → GT-016)
  ↓
Days 43-56: MHA + Gate 1 (GT-017 → GT-023)
  ↓
Days 57-67: GPT Basic + Gate 2 (GT-024 → GT-028)
  ↓
Days 68-76: MXFP4 Dequant (GT-029 → GT-031) [+1 day security] ← NOVEL FORMAT
  ↓
Days 77-91: MXFP4 Integration (GT-033 → GT-038) ← CRITICAL
  ↓
Days 92-99: Adapter + E2E + Gate 3 (GT-039 → GT-040a) [+1 day security]
  ↓
Days 100-110: Final Testing (GT-041 → GT-048) ← M0 DELIVERY
```

**Total Duration**: 95 agent-days (starting day 15, includes +3 days for security)

**Critical Dependencies**:
- **Day 15**: FFI lock blocks start
- **Day 52**: Foundation's integration framework blocks Gate 1
- **Day 67**: GPT basic validates pipeline before MXFP4
- **Day 71**: Foundation's adapter pattern blocks Gate 3
- **Day 91**: MXFP4 integration is sequential (cannot parallelize)

**No "Slack Time"**: Agent works sequentially until done

---

## Why GPT-Gamma Has Most Work (26 Days More Than Llama)

### 1. MXFP4 Complexity (20 days) - UNIQUE TO GPT

**No other agent has this**:
- Novel quantization format (no reference implementation)
- Dequantization kernel (4 days)
- Unit tests (2 days)
- Wire into 5 weight consumers sequentially (12 days)
- Numerical validation ±1% (3 days)

### 2. Large Model Complexity (6 days) - UNIQUE TO GPT

**GPT-OSS-20B is 34x larger than Qwen**:
- 12 GB model (vs 352 MB)
- ~16 GB VRAM total (close to 24 GB limit)
- OOM recovery tests (2 days)
- 24 GB boundary tests (2 days)
- UTF-8 multibyte edge cases (2 days)

### 3. GPT Kernels More Complex (4 days more than Llama)

- LayerNorm (6 days) vs RMSNorm (1 day)
- MHA (8 days) vs GQA (6 days)

**Total Unique Work**: 20 + 6 = 26 days more than Llama-Beta

---

## Risk Register (Revised for AI Agent Reality)

### Actual Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| FFI lock delayed beyond day 15 | GPT-Gamma blocked | Coordinate with Foundation-Alpha |
| MXFP4 numerical bugs | GPT-OSS-20B broken | Extensive unit tests, ±1% tolerance, Q4_K_M baseline |
| GPT-OSS-20B OOM | Model won't load | Memory profiling, chunked loading, OOM tests |
| MHA complexity | Integration delays | Reference llama.cpp, unit tests |
| LayerNorm numerical stability | Inference errors | Epsilon tuning, reference implementations |
| MXFP4 no reference implementation | Unknown correctness | Build validation framework first, compare with Q4_K_M |
| **GGUF heap overflow** | **Worker crash/RCE** | **GT-005a: Bounds validation, fuzzing tests** |
| **MXFP4 quantization attack** | **Malicious model behavior** | **GT-030a: Behavioral tests, GT-040a: Provenance verification** |
| **Supply chain compromise** | **Poisoned models** | **Trusted sources only (OpenAI, Qwen, Microsoft)** |

### ❌ Not Risks (Human Team Assumptions)

- ~~"Weeks 5-6 overcommitment"~~ - Agent works sequentially
- ~~"Need 4th person"~~ - Cannot scale agent count
- ~~"Utilization percentage"~~ - Meaningless for sequential execution
- ~~"Parallel work streams"~~ - Agent works one story at a time

---

## Key Actions for GPT-Gamma

### 1. **Wait for FFI Lock (Day 15)** 🔴 CRITICAL

**Action**: Monitor Foundation-Alpha's progress on FT-006 and FT-007
- Begin work immediately when `FFI_INTERFACE_LOCKED.md` published
- Study FFI interface documentation during wait time

### 2. **Research MXFP4 Format (Days 1-14)**

**Action**: Study MXFP4 spec during prep time
- Microscaling FP4 format details
- Block-based quantization patterns
- Scale factor handling
- FP16 accumulation requirements
- Build validation framework design

### 3. **Establish Q4_K_M Baseline First**

**Action**: Get GPT working with Q4_K_M before MXFP4 (Sprint 4)
- Validates GPT pipeline independently
- Provides numerical baseline for MXFP4 comparison
- Risk mitigation if MXFP4 takes longer

---

## Timeline Summary

**GPT-Gamma**: 95 agent-days starting day 15 = completes day 110  
**Critical Dependency**: Day 15 FFI lock  
**Gate 2**: Day 67 (GPT basic with Q4_K_M)  
**Gate 3**: Day 99 (MXFP4 + Adapter working)  
**Completion**: Day 110 ← **M0 CRITICAL PATH**

**M0 Delivery Date**: Day 110 (GPT-Gamma determines M0 timeline)

**Security Timeline**:
- Day 27: GGUF bounds validation complete (GT-005a)
- Day 76: MXFP4 behavioral tests complete (GT-030a)
- Day 99: Model provenance verification complete (GT-040a)

---

**Status**: ✅ **REVISED FOR AI AGENT REALITY**  
**Next Action**: Wait for FFI lock, then begin GT-001  
**Owner**: GPT-Gamma

---

*Crafted by GPT-Gamma 🤖*
