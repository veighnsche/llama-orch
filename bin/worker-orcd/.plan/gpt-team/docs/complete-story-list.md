# GPT Team - Complete Story List

**Total Stories**: 48 stories across 6 weeks (Weeks 2-7)  
**Total Estimated Effort**: ~92 days (with 2-3 people = **TIGHT** for 6 weeks)

---

## Week 2: HF Tokenizer + GPT Metadata (7 stories, 12 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| GT-001 | HF Tokenizers Crate Integration | S | 1 | Rust/C++ Dev | M0-W-1361 |
| GT-002 | tokenizer.json Loading | M | 2 | Rust/C++ Dev | M0-W-1361 |
| GT-003 | Tokenizer Metadata Exposure | M | 2 | Rust/C++ Dev | M0-W-1361 |
| GT-004 | HF Tokenizer Conformance Tests | M | 2 | Rust/C++ Dev | M0-W-1363 |
| GT-005 | GPT GGUF Metadata Parsing | M | 2 | Rust/C++ Dev | M0-W-1211, M0-W-1212 |
| GT-006 | GGUF v3 Tensor Support (MXFP4 Parsing) | M | 2 | Quantization Spec | M0-W-1201, M0-W-1211 |
| GT-007 | Architecture Detection (GPT) | S | 1 | Rust/C++ Dev | M0-W-1212 |

**Week 2 Capacity**: 2-3 people × 5 days = 10-15 days available, 12 days committed (80-120% utilization)

---

## Week 3: GPT Kernels Foundation (8 stories, 15 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| GT-008 | Absolute Positional Embedding Kernel | S | 1 | C++ Lead | M0-W-1434, M0-W-1215 |
| GT-009 | LayerNorm Kernel (Mean Reduction) | M | 2 | C++ Lead | M0-W-1432, M0-W-1215 |
| GT-010 | LayerNorm Kernel (Variance + Normalize) | M | 2 | C++ Lead | M0-W-1432 |
| GT-011 | LayerNorm Unit Tests | M | 2 | C++ Lead | M0-W-1432 |
| GT-012 | GELU Activation Kernel | S | 1 | C++ Lead | M0-W-1433, M0-W-1215 |
| GT-013 | GELU Unit Tests | S | 1 | C++ Lead | M0-W-1433 |
| GT-014 | GPT FFN Kernel (fc1 + GELU + fc2) | M | 3 | C++ Lead | M0-W-1215 |
| GT-015 | Residual Connection Kernel (if not shared) | S | 1 | C++ Lead | M0-W-1215 |
| GT-016 | Kernel Integration Tests | M | 2 | C++ Lead | (Testing) |

**Week 3 Capacity**: 10-15 days, 15 days committed (100-150% utilization) ⚠️

---

## Week 4: MHA Attention + Integration (7 stories, 14 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| GT-017 | MHA Attention Kernel (Prefill Phase) | L | 4 | C++ Lead | M0-W-1215 |
| GT-018 | MHA Attention Kernel (Decode Phase) | M | 2 | C++ Lead | M0-W-1215 |
| GT-019 | MHA vs GQA Differences Validation | M | 2 | C++ Lead | M0-W-1215 |
| GT-020 | MHA Unit Tests | M | 2 | C++ Lead | (Testing) |
| GT-021 | GPT Kernel Suite Integration | M | 2 | C++ Lead | (Integration) |
| GT-022 | **Gate 1 Participation** | S | 1 | Team Lead | Gate 1 |
| GT-023 | FFI Integration Tests (GPT Kernels) | S | 1 | Rust/C++ Dev | (Testing) |

**Week 4 Capacity**: 10-15 days, 14 days committed (93-140% utilization)

---

## Week 5: GPT Basic + MXFP4 Start (10 stories, 19 days) ⚠️ **OVERCOMMITTED**

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| GT-024 | GPT Weight Mapping (Q4_K_M Fallback) | M | 3 | C++ Lead | M0-W-1230, M0-W-1215 |
| GT-025 | GPT Weight Loading to VRAM | M | 2 | C++ Lead | M0-W-1220, M0-W-1221 |
| GT-026 | GPT Forward Pass (Q4_K_M) | L | 4 | C++ Lead | M0-W-1215 |
| GT-027 | GPT Basic Generation Test | M | 2 | Rust/C++ Dev | (Testing) |
| GT-028 | UTF-8 Streaming Safety Tests | M | 2 | Rust/C++ Dev | M0-W-1312, M0-W-1361 |
| GT-029 | MXFP4 Dequantization Kernel | L | 4 | Quantization Spec | M0-W-1201, M0-W-1820 |
| GT-030 | MXFP4 Unit Tests (Dequant Correctness) | M | 2 | Quantization Spec | M0-W-1820 |
| GT-031 | **Gate 2 Checkpoint** | - | - | Team Lead | Gate 2 |
| GT-032 | Bug Fixes from GPT Basic Integration | - | - | All | (Reactive, buffer) |

**Week 5 Capacity**: 10-15 days, 19 days committed (127-190% utilization) 🔴 **SEVERELY OVERCOMMITTED**

---

## Week 6: MXFP4 Integration + Adapter (9 stories, 17 days) ⚠️ **OVERCOMMITTED**

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| GT-033 | MXFP4 GEMM Integration | M | 3 | Quantization Spec | M0-W-1201, M0-W-1008 |
| GT-034 | MXFP4 Embedding Lookup | M | 2 | Quantization Spec | M0-W-1201 |
| GT-035 | MXFP4 Attention (Q/K/V Projections) | M | 3 | Quantization Spec | M0-W-1201 |
| GT-036 | MXFP4 FFN (Up/Down Projections) | M | 2 | Quantization Spec | M0-W-1201 |
| GT-037 | MXFP4 LM Head | M | 2 | Quantization Spec | M0-W-1201 |
| GT-038 | MXFP4 Numerical Validation (±1%) | M | 3 | Quantization Spec | M0-W-1820 |
| GT-039 | GPTInferenceAdapter Implementation | M | 3 | C++ Lead | M0-W-1213, M0-W-1215 |
| GT-040 | GPT-OSS-20B MXFP4 End-to-End | L | 4 | All | M0-W-1230, M0-W-1201 |
| GT-041 | **Gate 3 Participation** | S | 1 | Team Lead | Gate 3 |

**Week 6 Capacity**: 10-15 days, 17 days committed (113-170% utilization) 🔴 **OVERCOMMITTED**

---

## Week 7: Final Integration (7 stories, 15 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| GT-042 | GPT Integration Test Suite | M | 3 | Rust/C++ Dev | M0-W-1818 |
| GT-043 | MXFP4 Regression Tests | M | 2 | Quantization Spec | M0-W-1820 |
| GT-044 | 24 GB VRAM Boundary Tests | M | 2 | C++ Lead | M0-W-1230 |
| GT-045 | OOM Recovery Tests (GPT-OSS-20B) | M | 2 | C++ Lead | M0-W-1021 |
| GT-046 | UTF-8 Multibyte Edge Cases | M | 2 | Rust/C++ Dev | M0-W-1312 |
| GT-047 | Documentation (GPT, MXFP4, HF) | M | 3 | All | (Docs) |
| GT-048 | Performance Baseline (GPT-OSS-20B) | M | 2 | C++ Lead | M0-W-1012 (FT-012) |

**Week 7 Capacity**: 10-15 days, 15 days committed (100-150% utilization) 🟡

---

## Planning Gap Analysis

### 🔴 **CRITICAL FINDING: Weeks 5 & 6 SEVERELY Overcommitted**

**Problem**: GPT Team has MORE work than Foundation or Llama, but same timeline.

#### Week 5: 19 Days Committed

**With 2 people**: 10 days available, 19 days committed = **190% utilization** 🔴  
**With 3 people**: 15 days available, 19 days committed = **127% utilization** 🔴  
**With 4 people**: 20 days available, 19 days committed = **95% utilization** ✅

**Why So Much**:
- GPT basic pipeline (9 days): Weight mapping + loading + forward pass + testing
- MXFP4 dequant kernel (6 days): Complex new format + unit tests
- **Both are critical path** - can't defer either

#### Week 6: 17 Days Committed

**With 2 people**: 10 days available, 17 days committed = **170% utilization** 🔴  
**With 3 people**: 15 days available, 17 days committed = **113% utilization** 🔴  
**With 4 people**: 20 days available, 17 days committed = **85% utilization** ✅

**Why So Much**:
- MXFP4 integration (15 days): Wire into ALL weight consumers (5 stories)
- Numerical validation (3 days): Critical for correctness
- GPTAdapter (3 days): Required by spec
- End-to-end MXFP4 (4 days): Integration complexity

#### Week 7: 15 Days Committed

**With 2 people**: 10 days available, 15 days committed = **150% utilization** 🔴  
**With 3 people**: 15 days available, 15 days committed = **100% utilization** 🟡  
**With 4 people**: 20 days available, 15 days committed = **75% utilization** ✅

---

### 🟡 **MEDIUM FINDING: Week 3 Tight**

**With 2 people**: 10 days available, 15 days committed = **150% utilization** 🔴  
**With 3 people**: 15 days available, 15 days committed = **100% utilization** 🟡  
**With 4 people**: 20 days available, 15 days committed = **75% utilization** ✅

---

## Overall Utilization Analysis

### Scenario A: 2 People (NOT FEASIBLE)

| Week | Stories | Days | Capacity | Util % | Status |
|------|---------|------|----------|--------|--------|
| 2 | 7 | 12 | 10 | 120% | 🔴 Overcommitted |
| 3 | 8 | 15 | 10 | 150% | 🔴 Overcommitted |
| 4 | 7 | 14 | 10 | 140% | 🔴 Overcommitted |
| 5 | 10 | 19 | 10 | **190%** | 🔴 **SEVERELY Overcommitted** |
| 6 | 9 | 17 | 10 | 170% | 🔴 Overcommitted |
| 7 | 7 | 15 | 10 | 150% | 🔴 Overcommitted |

**Total**: 48 stories, 92 days of work, 60 days available (2 people × 6 weeks × 5 days)

**Overall Utilization**: 92 / 60 = **153%** 🔴 **COMPLETELY INFEASIBLE**

---

### Scenario B: 3 People (STILL TIGHT)

| Week | Stories | Days | Capacity | Util % | Status |
|------|---------|------|----------|--------|--------|
| 2 | 7 | 12 | 15 | 80% | ✅ Good |
| 3 | 8 | 15 | 15 | 100% | 🟡 Tight |
| 4 | 7 | 14 | 15 | 93% | ✅ Good |
| 5 | 10 | 19 | 15 | **127%** | 🔴 **Overcommitted** |
| 6 | 9 | 17 | 15 | **113%** | 🔴 **Overcommitted** |
| 7 | 7 | 15 | 15 | 100% | 🟡 Tight |

**Total**: 48 stories, 92 days of work, 90 days available (3 people × 6 weeks × 5 days)

**Overall Utilization**: 92 / 90 = **102%** 🔴 **OVERCOMMITTED**

---

### Scenario C: 4 People (FEASIBLE)

| Week | Stories | Days | Capacity | Util % | Status |
|------|---------|------|----------|--------|--------|
| 2 | 7 | 12 | 20 | 60% | ✅ Good |
| 3 | 8 | 15 | 20 | 75% | ✅ Good |
| 4 | 7 | 14 | 20 | 70% | ✅ Good |
| 5 | 10 | 19 | 20 | 95% | ✅ Good |
| 6 | 9 | 17 | 20 | 85% | ✅ Good |
| 7 | 7 | 15 | 20 | 75% | ✅ Good |

**Total**: 48 stories, 92 days of work, 120 days available (4 people × 6 weeks × 5 days)

**Overall Utilization**: 92 / 120 = **77%** ✅ **HEALTHY**

---

## Root Cause Analysis

### Why GPT Team Has More Work

**1. MXFP4 Complexity** (20 days total):
- Dequantization kernel (4 days)
- Unit tests (2 days)
- GEMM integration (3 days)
- Wire into 5 weight consumers (9 days: embedding, Q/K/V, FFN, LM head)
- Numerical validation (3 days)
- **No other team has this**

**2. GPT-Specific Kernels** (15 days):
- LayerNorm (6 days: implementation + tests, more complex than RMSNorm)
- GELU (2 days)
- MHA (8 days: more complex than GQA, all heads unique K/V)
- Absolute pos emb (1 day)
- **Llama team has similar, but GPT kernels are more complex**

**3. HF Tokenizer** (7 days):
- New crate integration (1 day)
- tokenizer.json loading (2 days)
- Metadata exposure (2 days)
- Conformance tests (2 days)
- **Llama team has GGUF-BPE (9 days), but HF has its own quirks**

**4. Large Model Complexity** (6 days):
- GPT-OSS-20B is 12 GB (vs Qwen 352 MB, Phi-3 2.3 GB)
- Memory profiling (implicit in stories)
- OOM tests (2 days)
- 24 GB boundary tests (2 days)
- UTF-8 streaming edge cases (2 days)
- **Llama team doesn't have this scale**

**Total Unique Work**: 20 (MXFP4) + 6 (large model) = **26 days more than Llama**

---

## Dependency Analysis

### Critical Path

```
Week 2: HF Tokenizer + GPT Metadata
  ↓
Week 3: GPT Kernels (LayerNorm, GELU, MHA) ← BOTTLENECK
  ↓
Week 4: MHA Complete + Gate 1
  ↓
Week 5: GPT Basic (9 days) + MXFP4 Dequant (6 days) ← BOTTLENECK
  ↓
Week 5: Gate 2 (GPT Basic working)
  ↓
Week 6: MXFP4 Integration (15 days) ← BOTTLENECK
  ↓
Week 6: Gate 3 (MXFP4 + Adapter)
  ↓
Week 7: Final Testing
  ↓
Week 7: Gate 4 (M0 Complete)
```

**Bottlenecks**:
1. **Week 5**: GPT basic + MXFP4 dequant (19 days) - Can't parallelize much
2. **Week 6**: MXFP4 integration (15 days) - Sequential wiring into weight consumers
3. **Week 3**: GPT kernels (15 days) - LayerNorm complexity

**With 3 people**: Bottlenecks still sequential, limited parallelization  
**With 4 people**: Can parallelize GPT basic + MXFP4, better throughput

---

## Risk Register

### High Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Weeks 5-6 overcommitment (3 people) | Gate 3 failure, M0 delay | **HIGH** | **Add 4th person OR extend to 8 weeks** |
| MXFP4 numerical bugs | GPT-OSS-20B broken | Medium | Extensive unit tests, ±1% tolerance |
| GPT-OSS-20B OOM | Model won't load | Medium | Memory profiling, OOM tests, chunked loading |
| MHA complexity | Gate 1 delay | Medium | Reference llama.cpp, unit tests |

### Medium Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| LayerNorm numerical stability | Inference errors | Medium | Epsilon tuning, reference implementations |
| HF tokenizer quirks | Tokenization bugs | Low-Medium | Conformance tests, compare with upstream |
| GGUF v3 parsing | MXFP4 load failures | Low-Medium | Comprehensive tests, reference llama.cpp |

---

## Recommendations

### 1. **Add 4th Team Member** (Critical) ⭐ **RECOMMENDED**

**Rationale**: Weeks 5-6 are severely overcommitted even with 3 people (127%, 113%).

**Team Composition** (4 people):
- **C++/CUDA Lead**: GPT kernels, forward pass
- **Quantization Specialist**: MXFP4 (full-time focus)
- **Rust/C++ Developer**: HF tokenizer, integration
- **QA/Integration**: Tests, validation, documentation

**Timeline**: 6 weeks (Weeks 2-7)

**Utilization**: 77% (healthy)

**Confidence**: 85%

---

### 2. **Extend to 8 Weeks** (If 4th Person Not Available)

**Changes**:
- Split Week 5 into 2 weeks:
  - Week 5a: GPT basic only (9 days)
  - Week 5b: MXFP4 dequant only (6 days)
- Split Week 6 into 2 weeks:
  - Week 6a: MXFP4 integration (10 days)
  - Week 6b: Numerical validation + Adapter (7 days)

**New Timeline** (8 weeks, 3 people):
| Week | Days | Capacity | Util % |
|------|------|----------|--------|
| 2 | 12 | 15 | 80% |
| 3 | 15 | 15 | 100% |
| 4 | 14 | 15 | 93% |
| 5a | 9 | 15 | 60% |
| 5b | 6 | 15 | 40% |
| 6a | 10 | 15 | 67% |
| 6b | 7 | 15 | 47% |
| 7 | 15 | 15 | 100% |

**Total**: 8 weeks, 92 days of work, 120 days available

**Utilization**: 92 / 120 = 77% ✅

**Confidence**: 75%

---

### 3. **Reduce Scope** (NOT RECOMMENDED)

**Options**:
- Defer MXFP4 to M1 (use Q4_K_M only)
  - **Problem**: GPT-OSS-20B validation is M0 requirement
  - **Savings**: 20 days (MXFP4 work)
  
- Defer GPT-OSS-20B entirely (only Qwen + Phi-3 in M0)
  - **Problem**: Violates M0 spec (3 models required)
  - **Savings**: 40 days (all GPT work)

**NOT RECOMMENDED**: Both options violate M0 requirements

---

## Revised Timeline (Option A: 4 People, 6 Weeks)

| Week | Stories | Days | Capacity | Util % | Gate |
|------|---------|------|----------|--------|------|
| 2 | 7 | 12 | 20 | 60% | - |
| 3 | 8 | 15 | 20 | 75% | - |
| 4 | 7 | 14 | 20 | 70% | Gate 1 ✓ |
| 5 | 10 | 19 | 20 | 95% | Gate 2 ✓ |
| 6 | 9 | 17 | 20 | 85% | Gate 3 ✓ |
| 7 | 7 | 15 | 20 | 75% | Gate 4 ✓ |

**Total**: 6 weeks, 92 days of work, 120 days available (4 people × 6 weeks × 5 days)

**Utilization**: 77% (healthy)

---

## Revised Timeline (Option B: 3 People, 8 Weeks)

| Week | Stories | Days | Capacity | Util % | Gate |
|------|---------|------|----------|--------|------|
| 2 | 7 | 12 | 15 | 80% | - |
| 3 | 8 | 15 | 15 | 100% | - |
| 4 | 7 | 14 | 15 | 93% | Gate 1 ✓ |
| 5a | 5 (GPT basic) | 9 | 15 | 60% | - |
| 5b | 2 (MXFP4 dequant) | 6 | 15 | 40% | Gate 2 ✓ |
| 6a | 5 (MXFP4 integration) | 10 | 15 | 67% | - |
| 6b | 4 (Validation + Adapter) | 7 | 15 | 47% | Gate 3 ✓ |
| 7 | 7 | 15 | 15 | 100% | Gate 4 ✓ |

**Total**: 8 weeks, 92 days of work, 120 days available (3 people × 8 weeks × 5 days)

**Utilization**: 77% (healthy)

---

## Final Recommendation

### **Option A: 4 People, 6 Weeks** ⭐ RECOMMENDED

**Why**:
1. **Stays within 6 weeks** (no M0 delay)
2. **Healthy utilization** (77%)
3. **Clear work streams**:
   - Person 1: GPT kernels
   - Person 2: MXFP4 (full-time)
   - Person 3: HF tokenizer + integration
   - Person 4: QA + tests
4. **Highest confidence** (85%)

**Cost**: 4 people × 6 weeks = 24 person-weeks

---

### Alternative: 3 People, 8 Weeks

**If 4th person not available**:
- Extend to 8 weeks
- Split Weeks 5-6 into 4 weeks
- Accept 2-week delay to M0

**Cost**: 3 people × 8 weeks = 24 person-weeks (same total)

**Confidence**: 75%

---

## Next Actions

1. **Immediate**: Decide on team size (3 or 4 people)
2. **Immediate**: If 3 people, decide on 6 weeks (not feasible) or 8 weeks (feasible)
3. **Week 0**: Create all 48 story cards
4. **Week 0**: Story sizing workshop
5. **Week 2**: Sprint planning, start HF tokenizer

---

**Status**: ✅ Analysis Complete  
**Recommendation**: **4 people, 6 weeks**  
**Alternative**: **3 people, 8 weeks**  
**Next Action**: Team size decision
