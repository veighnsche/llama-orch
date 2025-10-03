# GPT Team - Planning Gap Analysis

**Analysis Date**: 2025-10-03  
**Analyst**: AI Project Manager  
**Status**: 🔴 **CRITICAL FINDINGS - GPT TEAM HAS MOST WORK**

---

## Executive Summary

**Bottom Line**: GPT Team has **20% MORE work** than Llama Team, **8% MORE** than Foundation Team.

**Key Finding**: With 3 people, Weeks 5-6 are severely overcommitted (127%, 113%). **Need 4 people** for 6-week timeline.

**Recommendation**: **Add 4th team member** OR **extend to 8 weeks**.

---

## Work Comparison Across Teams

| Team | Stories | Days | Team Size | Weeks | Util % | Status |
|------|---------|------|-----------|-------|--------|--------|
| **Foundation** | 47 | 85 | 3 (fixed) | 7-8 | 81-101% | 🟡 Week 7 tight |
| **Llama** | 38 | 72 | 2-3 | 6-9 | 80-120% | 🔴 Need 3 people |
| **GPT** | **48** | **92** | 3-4 | 6-8 | **77-153%** | 🔴 **Need 4 people** |

**GPT Team has**:
- **+20 days** more than Llama (28% more)
- **+7 days** more than Foundation (8% more)
- **Most stories**: 48 vs 47 (Foundation) vs 38 (Llama)

---

## Detailed Analysis

### Work Breakdown by Team Size

#### Scenario A: 2 People (COMPLETELY INFEASIBLE)

| Week | Stories | Estimated Days | Available Days | Utilization | Status |
|------|---------|----------------|----------------|-------------|--------|
| 2 | 7 | 12 | 10 | 120% | 🔴 Overcommitted |
| 3 | 8 | 15 | 10 | 150% | 🔴 Overcommitted |
| 4 | 7 | 14 | 10 | 140% | 🔴 Overcommitted |
| 5 | 10 | 19 | 10 | **190%** | 🔴 **SEVERELY Overcommitted** |
| 6 | 9 | 17 | 10 | 170% | 🔴 Overcommitted |
| 7 | 7 | 15 | 10 | 150% | 🔴 Overcommitted |

**Total**: 48 stories, 92 days of work, 60 days available (2 people × 6 weeks × 5 days)

**Overall Utilization**: 92 / 60 = **153%** 🔴 **COMPLETELY INFEASIBLE**

**Every single week is overcommitted.** This is not a viable option.

---

#### Scenario B: 3 People (OVERCOMMITTED)

| Week | Stories | Estimated Days | Available Days | Utilization | Status |
|------|---------|----------------|----------------|-------------|--------|
| 2 | 7 | 12 | 15 | 80% | ✅ Good |
| 3 | 8 | 15 | 15 | 100% | 🟡 Tight |
| 4 | 7 | 14 | 15 | 93% | ✅ Good |
| 5 | 10 | 19 | 15 | **127%** | 🔴 **Overcommitted** |
| 6 | 9 | 17 | 15 | **113%** | 🔴 **Overcommitted** |
| 7 | 7 | 15 | 15 | 100% | 🟡 Tight |

**Total**: 48 stories, 92 days of work, 90 days available (3 people × 6 weeks × 5 days)

**Overall Utilization**: 92 / 90 = **102%** 🔴 **OVERCOMMITTED**

**Weeks 5-6 are critical bottlenecks.** High risk of gate failures.

---

#### Scenario C: 4 People (FEASIBLE) ⭐

| Week | Stories | Estimated Days | Available Days | Utilization | Status |
|------|---------|----------------|----------------|-------------|--------|
| 2 | 7 | 12 | 20 | 60% | ✅ Good |
| 3 | 8 | 15 | 20 | 75% | ✅ Good |
| 4 | 7 | 14 | 20 | 70% | ✅ Good |
| 5 | 10 | 19 | 20 | 95% | ✅ Good |
| 6 | 9 | 17 | 20 | 85% | ✅ Good |
| 7 | 7 | 15 | 20 | 75% | ✅ Good |

**Total**: 48 stories, 92 days of work, 120 days available (4 people × 6 weeks × 5 days)

**Overall Utilization**: 92 / 120 = **77%** ✅ **HEALTHY**

**All weeks are sustainable.** This is the recommended option.

---

## Critical Finding: Week 5 Bottleneck

### The Problem

**Week 5 Committed Work** (19 days):

**GPT Basic Pipeline** (9 days):
- GT-024: GPT Weight Mapping (3 days)
- GT-025: GPT Weight Loading (2 days)
- GT-026: GPT Forward Pass (4 days)

**MXFP4 Dequantization** (6 days):
- GT-029: MXFP4 Dequant Kernel (4 days)
- GT-030: MXFP4 Unit Tests (2 days)

**Testing** (4 days):
- GT-027: GPT Basic Generation Test (2 days)
- GT-028: UTF-8 Streaming Tests (2 days)

**Total**: 19 days

**Week 5 Available Capacity**:
- 2 people: 10 days → **190% utilization** 🔴
- 3 people: 15 days → **127% utilization** 🔴
- 4 people: 20 days → **95% utilization** ✅

### Why This Is Critical

1. **Both Work Streams Are Critical Path**:
   - GPT basic needed for Gate 2 (Week 5)
   - MXFP4 dequant needed for Week 6 integration
   - Can't defer either

2. **Limited Parallelization**:
   - GPT forward pass (4 days) is single-threaded work
   - MXFP4 dequant (4 days) is single-threaded work
   - With 3 people, one person idle while others blocked

3. **Gate 2 Risk**:
   - Week 5 slip → Gate 2 failure → M0 delayed
   - MXFP4 slip → Week 6 (Gate 3) at risk

4. **No Buffer**:
   - Zero slack time for bugs, rework, or learning curve
   - Any issue cascades to Week 6

---

## Critical Finding: Week 6 Bottleneck

### The Problem

**Week 6 Committed Work** (17 days):

**MXFP4 Integration** (15 days):
- GT-033: MXFP4 GEMM Integration (3 days)
- GT-034: MXFP4 Embedding Lookup (2 days)
- GT-035: MXFP4 Attention (Q/K/V) (3 days)
- GT-036: MXFP4 FFN (Up/Down) (2 days)
- GT-037: MXFP4 LM Head (2 days)
- GT-038: MXFP4 Numerical Validation (3 days)

**Adapter + Integration** (7 days):
- GT-039: GPTInferenceAdapter (3 days)
- GT-040: GPT-OSS-20B MXFP4 End-to-End (4 days)

**Total**: 17 days (some overlap in GT-040)

**Week 6 Available Capacity**:
- 2 people: 10 days → **170% utilization** 🔴
- 3 people: 15 days → **113% utilization** 🔴
- 4 people: 20 days → **85% utilization** ✅

### Why This Is Critical

1. **MXFP4 Wiring Is Sequential**:
   - Must wire into embedding, then attention, then FFN, then LM head
   - Each depends on previous (numerical validation needs all)
   - Limited parallelization

2. **Numerical Validation Is Complex**:
   - Compare MXFP4 vs Q4_K_M baseline
   - ±1% tolerance across all layers
   - Debugging numerical issues time-consuming

3. **Gate 3 Risk**:
   - Week 6 is Gate 3 week (MXFP4 + Adapter)
   - Slip → M0 delivery at risk

---

## Root Cause Analysis

### Why GPT Team Has Most Work

#### 1. MXFP4 Complexity (20 days total)

**No other team has this**:
- Dequantization kernel (4 days)
- Unit tests (2 days)
- GEMM integration (3 days)
- Wire into 5 weight consumers (9 days):
  - Embedding lookup (2 days)
  - Attention Q/K/V (3 days)
  - FFN up/down (2 days)
  - LM head (2 days)
- Numerical validation (3 days)

**Why it's hard**:
- Novel quantization format (no reference implementation)
- Must wire into ALL weight consumers
- FP16 accumulation paths required
- Numerical correctness critical (±1% tolerance)

**Comparison**:
- Foundation Team: No quantization work
- Llama Team: No quantization work (Q4_K_M is standard)
- GPT Team: **20 days of MXFP4 work**

---

#### 2. GPT-Specific Kernels (15 days)

**More complex than Llama kernels**:
- **LayerNorm** (6 days): Two reduction passes (mean + variance), learnable scale/bias
  - vs Llama's RMSNorm (1 day): Single reduction pass
- **MHA** (8 days): All heads have unique K/V, more memory
  - vs Llama's GQA (6 days): Grouped K/V heads, less memory
- **GELU** (2 days): Approximate formula with tanh
  - vs Llama's SwiGLU (2 days): Similar complexity
- **Absolute Pos Emb** (1 day): Learned embeddings
  - vs Llama's RoPE (2 days): Rotary embeddings

**Total**: 15 days (GPT) vs 11 days (Llama kernels)

---

#### 3. Large Model Complexity (6 days)

**GPT-OSS-20B is 34x larger than Qwen**:
- Qwen: 352 MB → ~400 MB VRAM
- Phi-3: 2.3 GB → ~3.5 GB VRAM
- GPT-OSS-20B: 12 GB → ~16 GB VRAM

**Additional work**:
- 24 GB VRAM boundary tests (2 days)
- OOM recovery tests (2 days)
- UTF-8 multibyte edge cases (2 days)
- Memory profiling (implicit in stories)

**Llama team doesn't have this scale**.

---

#### 4. HF Tokenizer (7 days)

**Different from GGUF-BPE**:
- HF tokenizers crate integration (1 day)
- tokenizer.json loading (2 days)
- Metadata exposure (2 days)
- Conformance tests (2 days)

**Comparison**:
- Llama Team: GGUF-BPE (9 days) - pure Rust implementation
- GPT Team: HF tokenizer (7 days) - crate integration

**Similar complexity, different approach**.

---

### Total Unique Work

| Category | Days | Unique to GPT? |
|----------|------|----------------|
| MXFP4 | 20 | ✅ YES |
| GPT Kernels | 15 | Partially (4 days more than Llama) |
| Large Model | 6 | ✅ YES |
| HF Tokenizer | 7 | Different approach (vs GGUF-BPE) |
| **Total Unique** | **26 days** | **More than Llama** |

**This is why GPT Team needs 4 people (vs Llama's 3).**

---

## Options Analysis

### Option A: 4 People, 6 Weeks ⭐ **RECOMMENDED**

**Changes**:
- Add 4th person (Quantization Specialist full-time)
- Week 5: 19 days / 20 days = 95% ✅
- Week 6: 17 days / 20 days = 85% ✅
- Week 7: 15 days / 20 days = 75% ✅

**Work Distribution** (Week 5):
- **Person 1 (C++)**: GPT forward pass (4 days)
- **Person 2 (Quant)**: MXFP4 dequant kernel (4 days)
- **Person 3 (Rust)**: Weight mapping + loading (5 days)
- **Person 4 (QA)**: Testing + validation (4 days)

**Pros**:
- ✅ Stays within 6 weeks
- ✅ Clear work streams (GPT vs MXFP4 vs Testing)
- ✅ Buffer for unknowns
- ✅ 85% confidence

**Cons**:
- ❌ Additional cost: 24 person-weeks (vs 18 for Llama)

**Cost**: 4 people × 6 weeks = 24 person-weeks

**Confidence**: 85%

---

### Option B: 3 People, 8 Weeks

**Changes**:
- Split Week 5 into 2 weeks:
  - Week 5a: GPT basic only (9 days)
  - Week 5b: MXFP4 dequant only (6 days)
- Split Week 6 into 2 weeks:
  - Week 6a: MXFP4 integration (10 days)
  - Week 6b: Validation + Adapter (7 days)
- Total: 8 weeks instead of 6

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

**Pros**:
- ✅ No additional headcount
- ✅ Sustainable utilization (77%)

**Cons**:
- ❌ 2 extra weeks (8 weeks vs 6 weeks)
- ❌ M0 delivery delayed
- ❌ Gates move significantly

**Cost**: 3 people × 8 weeks = 24 person-weeks (same as Option A)

**Confidence**: 75%

---

### Option C: 3 People, 6 Weeks (NOT FEASIBLE)

**Problem**: Weeks 5-6 overcommitted (127%, 113%)

**Expected Outcome**:
- Week 5 slip → Gate 2 delayed
- Week 6 slip → Gate 3 delayed
- Likely 7-8 weeks actual delivery

**Confidence**: 40%

**NOT RECOMMENDED**

---

## Comparison Matrix

| Factor | Option A (4p, 6w) | Option B (3p, 8w) | Option C (3p, 6w) |
|--------|-------------------|-------------------|-------------------|
| **Timeline** | 6 weeks ✅ | 8 weeks ❌ | 6 weeks (planned) |
| **Cost** | 24 pw | 24 pw | 18 pw (planned) |
| **Risk** | 🟢 Low | 🟡 Medium | 🔴 High |
| **Quality** | 🟢 High | 🟢 High | 🔴 Low |
| **Team Health** | 🟢 Good | 🟢 Good | 🔴 Burnout |
| **Confidence** | 🟢 85% | 🟡 75% | 🔴 40% |
| **Expected Delivery** | Week 6 | Week 8 | Week 7-8 (slip) |

**Winner**: **Option A (4 people, 6 weeks)**

---

## Financial Impact

### Option A: 4 People, 6 Weeks

**Direct Cost**: 4 × 6 = 24 person-weeks

**Risk Cost**: 15% probability of 1-week delay
- Expected delay: 0.15 weeks
- Risk cost: 4 × 0.15 = 0.6 person-weeks
- **Total Expected Cost**: 24 + 0.6 = **24.6 person-weeks**

**Timeline**: 6 weeks

---

### Option B: 3 People, 8 Weeks

**Direct Cost**: 3 × 8 = 24 person-weeks

**Risk Cost**: 25% probability of 1-week delay
- Expected delay: 0.25 weeks
- Risk cost: 3 × 0.25 = 0.75 person-weeks
- **Total Expected Cost**: 24 + 0.75 = **24.75 person-weeks**

**Timeline**: 8 weeks

**Opportunity Cost**: 2 extra weeks (M0 delayed)

---

### Option C: 3 People, 6 Weeks (Risky)

**Direct Cost**: 3 × 6 = 18 person-weeks

**Risk Cost**: 60% probability of 1-2 week delay
- Expected delay: 0.9 weeks
- Risk cost: 3 × 0.9 = 2.7 person-weeks
- **Total Expected Cost**: 18 + 2.7 = **20.7 person-weeks**

**Timeline**: 6 weeks (planned), 6.9 weeks (expected)

**Problem**: Still cheaper than Options A/B, but **high risk of failure**

---

### Conclusion

**Option A is fastest and most reliable, with similar cost to Option B.**

---

## Recommendation

### **Choose Option A: 4 People, 6 Weeks** ⭐

**Rationale**:
1. **Lowest Risk**: No overcommitment, buffer for unknowns
2. **Fastest**: 6 weeks vs 8 weeks (Option B)
3. **Clear Work Streams**: GPT vs MXFP4 vs Testing vs Integration
4. **Sustainable**: 77% utilization (healthy)
5. **Highest Confidence**: 85% vs 75% (B) vs 40% (C)

**Team Composition**:
- **C++/CUDA Lead**: GPT kernels, forward pass, integration
- **Quantization Specialist**: MXFP4 (full-time focus)
- **Rust/C++ Developer**: HF tokenizer, GGUF v3, integration
- **QA/Integration**: Tests, validation, documentation

**Timeline**: Weeks 2-7 (6 weeks)

**Confidence**: 85%

---

### Alternative: 3 People, 8 Weeks

**If 4th person absolutely not available**:
- Extend to 8 weeks (split Weeks 5-6)
- Accept 2-week delay to M0
- Still risky (75% confidence)

**Confidence**: 75%

---

## Action Items

### Immediate (Week 0)

- [ ] **DECISION REQUIRED**: 4 people or 3 people?
- [ ] If 4 people: Allocate Quantization Specialist (full-time MXFP4 focus)
- [ ] If 3 people: Extend timeline to 8 weeks, update M0 date
- [ ] Communicate decision to Foundation & Llama teams
- [ ] Update stakeholder expectations

### Week 1 (Foundation Team Only)

- [ ] GPT Team prepares for Week 2 start
- [ ] Review Foundation Team's FFI interface
- [ ] Research MXFP4 format (OpenAI docs, GGUF v3 spec)
- [ ] Research HF tokenizers crate

### Week 2 (GPT Team Starts)

- [ ] Sprint planning Monday
- [ ] Begin HF tokenizer integration
- [ ] Begin GPT GGUF metadata parsing
- [ ] Participate in FFI interface lock

---

## Conclusion

**GPT Team has the most work of all 3 teams** (92 days vs 85 Foundation, 72 Llama).

**Root cause**: MXFP4 complexity (20 days unique work) + large model (6 days) = 26 days more than Llama.

**With 3 people, Weeks 5-6 are severely overcommitted** (127%, 113%).

**Recommendation**: **Add 4th team member (Quantization Specialist)**

**Rationale**:
- Eliminates overcommitment
- Stays within 6-week timeline
- Clear work streams (GPT vs MXFP4)
- Highest confidence (85%)
- Same cost as 3 people × 8 weeks

**Alternative**: If 4th person not available, extend to 8 weeks (Option B), but accept:
- 2-week delay to M0
- Lower confidence (75%)

**Next Action**: **Team size decision required before Week 2**

---

**Status**: 🔴 **CRITICAL - DECISION REQUIRED**  
**Priority**: **P0 - BLOCKS GPT TEAM START**  
**Owner**: [Project Manager + Tech Lead]  
**Deadline**: Before Week 2 (GPT Team start)
