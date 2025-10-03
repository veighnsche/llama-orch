# Llama Team - Complete Story List

**Total Stories**: 38 stories across 6 weeks (Weeks 2-7)  
**Total Estimated Effort**: ~72 days (with 2-3 people = realistic for 6 weeks)

---

## Week 2: GGUF Foundation (6 stories, 11 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| LT-001 | GGUF Header Parser | M | 2 | Rust/C++ Dev | M0-W-1211 |
| LT-002 | GGUF Metadata Extraction (Llama) | M | 2 | Rust/C++ Dev | M0-W-1211, M0-W-1212 |
| LT-003 | Memory-Mapped I/O Implementation | M | 2 | C++ Lead | M0-W-1221 |
| LT-004 | Chunked H2D Transfer | M | 2 | C++ Lead | M0-W-1222 |
| LT-005 | Pre-Load Validation | M | 2 | Rust/C++ Dev | M0-W-1210 |
| LT-006 | Architecture Detection (Llama) | S | 1 | Rust/C++ Dev | M0-W-1212 |

**Week 2 Capacity**: 2-3 people × 5 days = 10-15 days available, 11 days committed (73-110% utilization)

---

## Week 3: Tokenization + Kernels Start (8 stories, 15 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| LT-007 | GGUF Vocab Parsing | M | 2 | Rust/C++ Dev | M0-W-1362 |
| LT-008 | GGUF Merges Parsing | M | 2 | Rust/C++ Dev | M0-W-1362 |
| LT-009 | Byte-Level BPE Encoder | M | 3 | Rust/C++ Dev | M0-W-1362 |
| LT-010 | Byte-Level BPE Decoder | M | 2 | Rust/C++ Dev | M0-W-1362 |
| LT-011 | UTF-8 Safe Streaming Decode | M | 2 | Rust/C++ Dev | M0-W-1362 |
| LT-012 | RoPE Kernel | M | 2 | C++ Lead | M0-W-1214, M0-W-1430 |
| LT-013 | RMSNorm Kernel | S | 1 | C++ Lead | M0-W-1214, M0-W-1430 |
| LT-014 | Residual Connection Kernel | S | 1 | C++ Lead | M0-W-1214 |

**Week 3 Capacity**: 10-15 days, 15 days committed (100-150% utilization) ⚠️

---

## Week 4: Kernels Complete + Integration (7 stories, 13 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| LT-015 | GQA Attention Kernel (Prefill) | L | 4 | C++ Lead | M0-W-1214, M0-W-1430 |
| LT-016 | GQA Attention Kernel (Decode) | M | 2 | C++ Lead | M0-W-1214 |
| LT-017 | SwiGLU FFN Kernel | M | 2 | C++ Lead | M0-W-1214 |
| LT-018 | Tokenizer Conformance Tests (Qwen) | M | 2 | Rust/C++ Dev | M0-W-1363 |
| LT-019 | Kernel Unit Tests | M | 2 | C++ Lead | M0-W-1430 |
| LT-020 | **Gate 1 Participation** | S | 1 | Team Lead | Gate 1 |
| LT-021 | FFI Integration Tests (Llama Kernels) | - | - | QA | (Included in LT-019) |

**Week 4 Capacity**: 10-15 days, 13 days committed (87-130% utilization)

---

## Week 5: Qwen Integration (7 stories, 13 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| LT-022 | Qwen Weight Mapping | M | 3 | C++ Lead | M0-W-1230 |
| LT-023 | Qwen Weight Loading to VRAM | M | 2 | C++ Lead | M0-W-1220, M0-W-1221 |
| LT-024 | Qwen Forward Pass Implementation | L | 4 | C++ Lead | M0-W-1214, M0-W-1420 |
| LT-025 | Qwen Haiku Generation Test | M | 2 | QA | M0-W-1800 |
| LT-026 | Qwen Reproducibility Validation | M | 2 | QA | M0-W-1826 |
| LT-027 | **Gate 2 Checkpoint** | - | - | Team Lead | Gate 2 |
| LT-028 | Bug Fixes from Qwen Integration | - | - | All | (Reactive, buffer) |

**Week 5 Capacity**: 10-15 days, 13 days committed (87-130% utilization)

---

## Week 6: Phi-3 + Adapter (6 stories, 11 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| LT-029 | Phi-3 Metadata Analysis | S | 1 | Rust/C++ Dev | M0-W-1230 |
| LT-030 | Phi-3 Weight Loading | M | 2 | C++ Lead | M0-W-1230 |
| LT-031 | Phi-3 Forward Pass (Adapt from Qwen) | M | 2 | C++ Lead | M0-W-1214 |
| LT-032 | Tokenizer Conformance Tests (Phi-3) | M | 2 | Rust/C++ Dev | M0-W-1363 |
| LT-033 | LlamaInferenceAdapter Implementation | M | 3 | C++ Lead | M0-W-1213, M0-W-1214 |
| LT-034 | **Gate 3 Participation** | S | 1 | Team Lead | Gate 3 |

**Week 6 Capacity**: 10-15 days, 11 days committed (73-110% utilization)

---

## Week 7: Final Integration (4 stories, 9 days)

| ID | Title | Size | Days | Owner | Spec Ref |
|----|-------|------|------|-------|----------|
| LT-035 | Llama Integration Test Suite | M | 3 | QA | M0-W-1818 |
| LT-036 | Reproducibility Tests (10 runs × 2 models) | M | 2 | QA | M0-W-1826 |
| LT-037 | VRAM Pressure Tests (Phi-3) | M | 2 | QA | M0-W-1230 |
| LT-038 | Documentation (GGUF, BPE, Llama) | M | 2 | All | (Docs) |

**Week 7 Capacity**: 10-15 days, 9 days committed (60-90% utilization)

---

## Planning Gap Analysis

### ⚠️ **CRITICAL FINDING: Week 3 Overcommitted**

**Problem**: Week 3 has 15 days of work committed but only 10-15 days available depending on team size.

**With 2 people**: 10 days available, 15 days committed = **150% utilization** 🔴  
**With 3 people**: 15 days available, 15 days committed = **100% utilization** 🟡

**Impact**:
- High risk of Week 4 (Gate 1) delay
- Tokenizer + kernels both critical path
- No buffer for issues

**Root Cause**:
- Tokenization is 5 stories (9 days) - complex pure Rust BPE
- Kernels starting in parallel (3 stories, 4 days)
- Both are new, unfamiliar territory

---

### 🟡 **MEDIUM FINDING: Week 4 Tight**

**Problem**: Week 4 has 13 days committed, 10-15 days available.

**With 2 people**: 10 days available, 13 days committed = **130% utilization** 🔴  
**With 3 people**: 15 days available, 13 days committed = **87% utilization** ✅

**Impact**:
- Gate 1 at risk if only 2 people
- GQA attention is complex (4 days for prefill alone)

---

### 🟡 **MEDIUM FINDING: Week 5 Tight**

**Problem**: Week 5 has 13 days committed, 10-15 days available.

**With 2 people**: 10 days available, 13 days committed = **130% utilization** 🔴  
**With 3 people**: 15 days available, 13 days committed = **87% utilization** ✅

**Impact**:
- Gate 2 (Qwen working) at risk if only 2 people
- Qwen forward pass is complex (4 days)

---

## Dependency Analysis

### Critical Path

```
Week 2: GGUF Loader (LT-001, LT-002, LT-003, LT-004)
  ↓
Week 3: Tokenizer (LT-007 → LT-011) + Kernels Start (LT-012, LT-013)
  ↓
Week 4: GQA Attention (LT-015, LT-016) + SwiGLU (LT-017)
  ↓
Week 4: **GATE 1** (LT-020) ← CRITICAL MILESTONE
  ↓
Week 5: Qwen Integration (LT-022 → LT-026)
  ↓
Week 5: **GATE 2** (LT-027) ← CRITICAL MILESTONE (First Model Working)
  ↓
Week 6: Phi-3 + Adapter (LT-029 → LT-033)
  ↓
Week 6: **GATE 3** (LT-034)
  ↓
Week 7: Final Testing (LT-035 → LT-038)
  ↓
Week 7: **GATE 4** (M0 Complete)
```

**Critical Path Duration**: 6 weeks (minimum)

**Slack Time**:
- Week 2: 0-4 days slack (depends on team size)
- Week 7: 1-6 days slack
- **Total slack**: 1-10 days (highly dependent on team size)

---

## Risk Register

### High Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Week 3 overcommitment (2 people) | Gate 1 delay | **HIGH** | **Add 3rd person OR split Week 3 work** |
| BPE algorithm bugs | Tokenization broken | Medium | Conformance test vectors, reference impl |
| GQA attention complexity | Gate 1 delay | Medium | Reference llama.cpp, start early |
| Qwen forward pass bugs | Gate 2 failure | Medium | Unit tests, incremental integration |

### Medium Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| GGUF format edge cases | Parsing failures | Medium | Comprehensive tests, multiple models |
| Phi-3 architecture differences | Week 6 delays | Low-Medium | Research early (Week 3-4) |
| FFI interface changes | Blocked | Low | Participate in Week 2 FFI lock |

---

## Recommendations

### 1. **Add 3rd Team Member** (Critical)

**Rationale**: Weeks 3, 4, 5 are all tight with 2 people (130-150% utilization).

**Options**:
- **Option A**: 3 people full-time (Weeks 2-7)
  - Week 3: 15 days / 15 days = 100% util ✅
  - Week 4: 13 days / 15 days = 87% util ✅
  - Week 5: 13 days / 15 days = 87% util ✅

- **Option B**: 2 people + 1 part-time (50%) for Weeks 3-5 only
  - Week 3: 15 days / 12.5 days = 120% util 🟡 (still tight)
  - Week 4: 13 days / 12.5 days = 104% util 🟡
  - Week 5: 13 days / 12.5 days = 104% util 🟡

- **Option C**: 2 people + extend timeline
  - Split Week 3 into 2 weeks (Week 3a: Tokenizer, Week 3b: Kernels)
  - Total: 7 weeks instead of 6

**Recommendation**: **Option A (3 people)** or **Option C (extend to 7 weeks)**

---

### 2. **Split Week 3 Work** (If Staying with 2 People)

**Current Week 3**: Tokenizer (9 days) + Kernels (4 days) + Misc (2 days) = 15 days

**Proposed Split**:
- **Week 3a**: Tokenizer focus (LT-007 through LT-011) = 9 days
- **Week 3b**: Kernels focus (LT-012, LT-013, LT-014) = 4 days
- **Total**: 7 weeks instead of 6

**Impact**: Gate 1 moves from Week 4 to Week 5

---

### 3. **Start Tokenizer in Week 2** (Recommended)

**Rationale**: Tokenizer is 9 days of work, critical path, unfamiliar territory.

**Action**:
- Move LT-007 (Vocab parsing) to Week 2
- Move LT-008 (Merges parsing) to Week 2
- Week 2 becomes 15 days (100% util with 3 people)
- Week 3 becomes 11 days (73% util with 3 people) ✅

**Benefit**: Reduces Week 3 pressure, gives more time for BPE algorithm

---

### 4. **Parallelize Tokenizer and Kernels** (If 3 People)

**Rationale**: With 3 people, can split work streams.

**Work Streams**:
- **Person 1**: Tokenizer (Weeks 2-3)
- **Person 2**: Kernels (Weeks 3-4)
- **Person 3**: GGUF loader + Integration (Weeks 2-4)

**Benefit**: Better utilization, clear ownership

---

## Revised Timeline (Option A: 3 People)

| Week | Stories | Days | Capacity | Util % | Gate |
|------|---------|------|----------|--------|------|
| 2 | 6 + 2 (tokenizer start) | 15 | 15 | 100% | - |
| 3 | 6 (tokenizer finish + kernels) | 11 | 15 | 73% | - |
| 4 | 7 | 13 | 15 | 87% | Gate 1 ✓ |
| 5 | 7 | 13 | 15 | 87% | Gate 2 ✓ |
| 6 | 6 | 11 | 15 | 73% | Gate 3 ✓ |
| 7 | 4 | 9 | 15 | 60% | Gate 4 ✓ |

**Total**: 6 weeks, 72 days of work, 90 days available (3 people × 6 weeks × 5 days)

**Utilization**: 72 / 90 = 80% (healthy)

---

## Revised Timeline (Option C: 2 People, 7 Weeks)

| Week | Stories | Days | Capacity | Util % | Gate |
|------|---------|------|----------|--------|------|
| 2 | 6 | 11 | 10 | 110% | - |
| 3 | 5 (tokenizer only) | 9 | 10 | 90% | - |
| 4 | 3 (kernels only) | 6 | 10 | 60% | - |
| 5 | 7 (GQA + SwiGLU + Gate 1) | 13 | 10 | 130% | Gate 1 ✓ |
| 6 | 7 (Qwen) | 13 | 10 | 130% | Gate 2 ✓ |
| 7 | 6 (Phi-3 + adapter) | 11 | 10 | 110% | Gate 3 ✓ |
| 8 | 4 (final testing) | 9 | 10 | 90% | Gate 4 ✓ |

**Total**: 7 weeks, 72 days of work, 70 days available (2 people × 7 weeks × 5 days)

**Utilization**: 72 / 70 = 103% (overcommitted) 🔴

**Conclusion**: 2 people + 7 weeks is STILL tight. Need 3 people OR 8 weeks.

---

## Final Recommendation

### **Option A: 3 People, 6 Weeks** ⭐ RECOMMENDED

**Team Composition**:
- C++/CUDA Lead (full-time)
- Rust/C++ Developer (full-time)
- QA/Integration (full-time, or shared with Foundation Week 5+)

**Timeline**: Weeks 2-7 (6 weeks)

**Utilization**: 80% average (healthy)

**Confidence**: High (90%)

---

### Alternative: 2 People, 8 Weeks

**If 3rd person not available**:
- Extend to 8 weeks
- Split Week 3 into 2 weeks
- Accept 103% utilization (still risky)

**Confidence**: Medium (60%)

---

## Next Actions

1. **Immediate**: Decide on team size (2 or 3 people)
2. **Immediate**: If 2 people, decide on 6 weeks (risky) or 7-8 weeks (safer)
3. **Week 0**: Create all 38 story cards
4. **Week 0**: Story sizing workshop
5. **Week 2**: Sprint planning, start GGUF loader

---

**Status**: ✅ Analysis Complete  
**Recommendation**: **3 people, 6 weeks**  
**Alternative**: **2 people, 8 weeks**  
**Next Action**: Team size decision
