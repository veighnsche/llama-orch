# Llama Team - Planning Gap Analysis

**Analysis Date**: 2025-10-03  
**Analyst**: AI Project Manager  
**Status**: 🔴 **CRITICAL FINDINGS - TEAM SIZE DECISION REQUIRED**

---

## Executive Summary

**Bottom Line**: The 6-week timeline is **ONLY FEASIBLE with 3 people**.

**Key Finding**: With 2 people, Weeks 3, 4, and 5 are overcommitted (130-150% utilization).

**Recommendation**: **Add 3rd team member** OR **extend to 8 weeks**.

---

## Detailed Analysis

### Work Breakdown by Team Size

#### Scenario A: 2 People

| Week | Stories | Estimated Days | Available Days | Utilization | Status |
|------|---------|----------------|----------------|-------------|--------|
| 2 | 6 | 11 | 10 | 110% | 🟡 Tight |
| 3 | 8 | 15 | 10 | **150%** | 🔴 **OVERCOMMITTED** |
| 4 | 7 | 13 | 10 | **130%** | 🔴 **OVERCOMMITTED** |
| 5 | 7 | 13 | 10 | **130%** | 🔴 **OVERCOMMITTED** |
| 6 | 6 | 11 | 10 | 110% | 🟡 Tight |
| 7 | 4 | 9 | 10 | 90% | ✅ Good |

**Total**: 38 stories, 72 days of work, 60 days available (2 people × 6 weeks × 5 days)

**Overall Utilization**: 72 / 60 = **120%** 🔴 **NOT FEASIBLE**

---

#### Scenario B: 3 People (Recommended)

| Week | Stories | Estimated Days | Available Days | Utilization | Status |
|------|---------|----------------|----------------|-------------|--------|
| 2 | 6 + 2 | 15 | 15 | 100% | ✅ Good |
| 3 | 6 | 11 | 15 | 73% | ✅ Good |
| 4 | 7 | 13 | 15 | 87% | ✅ Good |
| 5 | 7 | 13 | 15 | 87% | ✅ Good |
| 6 | 6 | 11 | 15 | 73% | ✅ Good |
| 7 | 4 | 9 | 15 | 60% | ✅ Good |

**Total**: 38 stories, 72 days of work, 90 days available (3 people × 6 weeks × 5 days)

**Overall Utilization**: 72 / 90 = **80%** ✅ **FEASIBLE**

---

## Critical Finding: Week 3 Overcommitment

### The Problem

**Week 3 Committed Work** (with 2 people):
- LT-007: GGUF Vocab Parsing (2 days)
- LT-008: GGUF Merges Parsing (2 days)
- LT-009: Byte-Level BPE Encoder (3 days)
- LT-010: Byte-Level BPE Decoder (2 days)
- LT-011: UTF-8 Safe Streaming Decode (2 days)
- LT-012: RoPE Kernel (2 days)
- LT-013: RMSNorm Kernel (1 day)
- LT-014: Residual Connection Kernel (1 day)
- **Total**: 15 days

**Week 3 Available Capacity** (2 people): 10 days

**Overcommitment**: 15 - 10 = **5 days (50% over)** 🔴

### Why This Is Critical

1. **Tokenizer is Complex**: Pure Rust BPE implementation (9 days total)
   - Unfamiliar territory (no reference implementation in Rust)
   - UTF-8 boundary handling tricky
   - Conformance tests required

2. **Kernels are Critical Path**: RoPE, RMSNorm needed for Week 4 GQA
   - Can't start GQA without RoPE
   - Can't test forward pass without RMSNorm

3. **No Buffer**: Zero slack time for:
   - BPE algorithm bugs
   - Tokenization edge cases
   - Kernel debugging
   - Learning curve

4. **Gate 1 Risk**: Week 3 slip → Week 4 (Gate 1) failure → Entire project delayed

---

## Root Cause Analysis

### Why Week 3 Is Overloaded

**Reason 1: Tokenizer Complexity Underestimated**
- BPE algorithm is 9 days of work
- Pure Rust implementation (no existing crate)
- UTF-8 safety adds complexity
- Conformance tests required

**Reason 2: Kernels Can't Wait**
- RoPE needed for Week 4 GQA attention
- RMSNorm needed for forward pass
- Can't defer to Week 4 (already full)

**Reason 3: GGUF Loader Dependency**
- Tokenizer needs vocab/merges from GGUF (Week 2)
- Can't start tokenizer until GGUF loader done
- Forces tokenizer into Week 3

---

## Options Analysis

### Option A: Add 3rd Team Member ⭐ **RECOMMENDED**

**Changes**:
- Add 3rd person (C++/CUDA or Rust specialist)
- Week 3 capacity becomes 15 days
- 15 days / 15 days = 100% utilization ✅

**Work Distribution** (Week 3 with 3 people):
- **Person 1 (Rust)**: Tokenizer (LT-007 through LT-011) = 9 days
- **Person 2 (C++)**: Kernels (LT-012, LT-013, LT-014) = 4 days
- **Person 3 (C++/Rust)**: Integration tests, support = 2 days

**Pros**:
- ✅ Eliminates overcommitment
- ✅ Clear work streams (tokenizer vs kernels)
- ✅ Buffer for unknowns
- ✅ Stays within 6 weeks

**Cons**:
- ❌ Additional cost (1 person × 6 weeks = 6 person-weeks)
- ❌ May not be available (hiring/allocation)

**Risk**: LOW

**Cost**: 6 person-weeks (Weeks 2-7)

---

### Option B: Extend to 7-8 Weeks (2 People)

**Changes**:
- Split Week 3 into 2 weeks:
  - Week 3a: Tokenizer (LT-007 through LT-011) = 9 days
  - Week 3b: Kernels (LT-012, LT-013, LT-014) = 4 days
- Total: 7 weeks instead of 6

**New Timeline** (7 weeks, 2 people):
| Week | Days | Capacity | Util % |
|------|------|----------|--------|
| 2 | 11 | 10 | 110% |
| 3a | 9 | 10 | 90% |
| 3b | 4 | 10 | 40% |
| 4 | 13 | 10 | 130% | ← Still overcommitted! |
| 5 | 13 | 10 | 130% | ← Still overcommitted! |
| 6 | 11 | 10 | 110% |
| 7 | 9 | 10 | 90% |

**Problem**: Weeks 4 and 5 still overcommitted (130%)

**Need to extend to 8 weeks**:
- Week 4a: GQA Prefill + Unit Tests (6 days)
- Week 4b: GQA Decode + SwiGLU + Gate 1 (7 days)

**New Timeline** (8 weeks, 2 people):
| Week | Days | Capacity | Util % |
|------|------|----------|--------|
| 2 | 11 | 10 | 110% |
| 3a | 9 | 10 | 90% |
| 3b | 4 | 10 | 40% |
| 4a | 6 | 10 | 60% |
| 4b | 7 | 10 | 70% |
| 5 | 13 | 10 | 130% | ← Qwen still tight! |
| 6 | 11 | 10 | 110% |
| 7 | 9 | 10 | 90% |

**Still need to split Week 5**:
- Week 5a: Qwen Weight Loading + Forward Pass (6 days)
- Week 5b: Qwen Testing + Gate 2 (7 days)

**Final Timeline** (9 weeks, 2 people):
| Week | Days | Capacity | Util % |
|------|------|----------|--------|
| 2 | 11 | 10 | 110% |
| 3a | 9 | 10 | 90% |
| 3b | 4 | 10 | 40% |
| 4a | 6 | 10 | 60% |
| 4b | 7 | 10 | 70% |
| 5a | 6 | 10 | 60% |
| 5b | 7 | 10 | 70% |
| 6 | 11 | 10 | 110% |
| 7 | 9 | 10 | 90% |

**Total**: 9 weeks, 72 days of work, 90 days available

**Utilization**: 72 / 90 = 80% ✅

**Pros**:
- ✅ No additional headcount
- ✅ Sustainable utilization

**Cons**:
- ❌ 3 extra weeks (9 weeks vs 6 weeks)
- ❌ M0 delivery delayed significantly
- ❌ Gates move: Gate 1 Week 4b → Week 5b, Gate 2 Week 5b → Week 6b

**Risk**: MEDIUM

**Cost**: 3 extra weeks (but same total person-weeks: 2 × 9 = 18 vs 3 × 6 = 18)

---

### Option C: Reduce Scope

**Changes**:
- Defer Phi-3 to M1 (only Qwen in M0)
- Simplify tokenizer (use existing crate, not pure Rust)
- Reduce test coverage

**Scope Cuts**:
- Remove LT-029 through LT-032 (Phi-3, 7 days)
- Remove LT-033 (LlamaAdapter, 3 days)
- Simplify tokenizer (save 3 days)
- **Total savings**: 13 days

**New Timeline** (2 people, 6 weeks):
| Week | Days | Capacity | Util % |
|------|------|----------|--------|
| 2 | 11 | 10 | 110% |
| 3 | 12 (15 - 3 tokenizer savings) | 10 | 120% | ← Still over! |
| 4 | 13 | 10 | 130% | ← Still over! |
| 5 | 13 | 10 | 130% | ← Still over! |
| 6 | 4 (11 - 7 Phi-3) | 10 | 40% |
| 7 | 9 | 10 | 90% |

**Problem**: Still overcommitted in Weeks 3-5

**Pros**:
- ✅ Reduces total work

**Cons**:
- ❌ Loses Phi-3 validation (stretch goal)
- ❌ Loses LlamaAdapter (needed for M0 spec)
- ❌ Still overcommitted (doesn't solve problem)
- ❌ Technical debt

**Risk**: HIGH (doesn't solve core issue)

**NOT RECOMMENDED**

---

### Option D: Work Overtime

**Changes**:
- Team works extra hours in Weeks 3-5
- 5 days overtime in Week 3 = 150% → 100%
- 3 days overtime in Week 4 = 130% → 100%
- 3 days overtime in Week 5 = 130% → 100%

**Pros**:
- ✅ Stays within 6 weeks
- ✅ No additional cost

**Cons**:
- ❌ **NOT RECOMMENDED** - Burnout risk
- ❌ Quality suffers under pressure
- ❌ Unsustainable (3 weeks of overtime)
- ❌ Team morale impact

**Risk**: HIGH

**NOT RECOMMENDED**

---

## Comparison Matrix

| Factor | Option A (3 people) | Option B (9 weeks) | Option C (Reduce scope) | Option D (Overtime) |
|--------|---------------------|--------------------|-----------------------|---------------------|
| **Timeline** | 6 weeks ✅ | 9 weeks ❌ | 6 weeks ✅ | 6 weeks ✅ |
| **Cost** | +6 pw | Same | Same | Same |
| **Risk** | 🟢 Low | 🟡 Medium | 🔴 High | 🔴 High |
| **Quality** | 🟢 High | 🟢 High | 🔴 Low | 🔴 Low |
| **Team Health** | 🟢 Good | 🟢 Good | 🟡 OK | 🔴 Burnout |
| **Scope** | 🟢 Full | 🟢 Full | 🔴 Reduced | 🟢 Full |
| **Confidence** | 🟢 90% | 🟡 70% | 🔴 40% | 🔴 30% |

**Winner**: **Option A (3 people, 6 weeks)**

---

## Dependency Analysis

### Critical Path

```
Week 2: GGUF Loader (Foundation for everything)
  ↓
Week 3: Tokenizer (9 days) ← BOTTLENECK
  ↓
Week 3: Kernels (RoPE, RMSNorm) ← BOTTLENECK
  ↓
Week 4: GQA Attention (4 days prefill) ← COMPLEX
  ↓
Week 4: Gate 1 ← CRITICAL MILESTONE
  ↓
Week 5: Qwen Forward Pass (4 days) ← COMPLEX
  ↓
Week 5: Gate 2 (Qwen working) ← CRITICAL MILESTONE
  ↓
Week 6: Phi-3 + Adapter
  ↓
Week 7: Final Testing
```

**Bottlenecks**:
1. **Week 3 Tokenizer** (9 days) - Can't parallelize, single-threaded work
2. **Week 4 GQA Attention** (4 days) - Complex, unfamiliar
3. **Week 5 Qwen Forward Pass** (4 days) - Integration complexity

**With 2 people**: All bottlenecks sequential, no parallelization  
**With 3 people**: Tokenizer + Kernels parallel, better throughput

---

## Financial Impact

### Option A: 3 People, 6 Weeks

**Cost**: 3 people × 6 weeks = 18 person-weeks

**Risk Cost**: Low (10% probability of 1-week delay)
- Expected delay: 0.1 weeks
- **Total Expected Cost**: 18 + 0.3 = **18.3 person-weeks**

---

### Option B: 2 People, 9 Weeks

**Cost**: 2 people × 9 weeks = 18 person-weeks

**Risk Cost**: Medium (30% probability of 1-2 week delay)
- Expected delay: 0.45 weeks
- **Total Expected Cost**: 18 + 0.9 = **18.9 person-weeks**

**Also**: 3 weeks longer timeline (opportunity cost)

---

### Conclusion

**Option A is cheaper AND faster when accounting for risk.**

---

## Recommendation

### **Choose Option A: 3 People, 6 Weeks** ⭐

**Rationale**:
1. **Lowest Risk**: Eliminates overcommitment, buffer for unknowns
2. **Fastest**: 6 weeks vs 9 weeks
3. **Sustainable**: 80% utilization (healthy)
4. **Clear Work Streams**: Tokenizer vs Kernels vs Integration
5. **Cost-Effective**: Actually cheaper when accounting for risk

**Team Composition**:
- **C++/CUDA Lead**: Kernels (RoPE, GQA, RMSNorm, SwiGLU), Qwen/Phi-3 integration
- **Rust/C++ Developer**: GGUF loader, Tokenizer (pure Rust BPE)
- **QA/Integration**: Tests, conformance vectors, documentation (can be shared with Foundation Team Week 5+)

**Timeline**: Weeks 2-7 (6 weeks)

**Confidence**: 90%

---

### Alternative: 2 People, 9 Weeks

**If 3rd person absolutely not available**:
- Extend to 9 weeks (split Weeks 3, 4, 5)
- Accept 3-week delay to M0
- Still risky (80% utilization but long timeline)

**Confidence**: 70%

---

## Action Items

### Immediate (Week 0)

- [ ] **DECISION REQUIRED**: 3 people or 2 people?
- [ ] If 3 people: Allocate 3rd person (C++/CUDA or Rust specialist)
- [ ] If 2 people: Decide on 6 weeks (not feasible) or 9 weeks (feasible but slow)
- [ ] Communicate decision to Foundation Team (affects overall M0 timeline)
- [ ] Update stakeholder expectations

### Week 1 (Foundation Team Only)

- [ ] Llama Team prepares for Week 2 start
- [ ] Review Foundation Team's FFI interface design
- [ ] Research GGUF format (llama.cpp reference)
- [ ] Research BPE algorithm (prepare for Week 3)

### Week 2 (Llama Team Starts)

- [ ] Sprint planning Monday
- [ ] Begin GGUF loader development
- [ ] Participate in FFI interface lock (critical)

---

## Conclusion

**The 6-week timeline for Llama Team is ONLY FEASIBLE with 3 people.**

**With 2 people, need 9 weeks to be realistic.**

**Recommendation**: **Add 3rd team member (Option A)**

**Rationale**:
- Eliminates overcommitment (Weeks 3-5)
- Stays within 6-week timeline
- Sustainable pace for team
- Clear work streams (tokenizer vs kernels)
- Actually cheaper when accounting for risk
- Higher confidence in delivery (90%)

**Alternative**: If 3rd person not available, extend to 9 weeks (Option B), but accept:
- 3-week delay to M0
- Longer timeline (opportunity cost)
- Still some risk (70% confidence)

**Next Action**: **Team size decision required before Week 2**

---

**Status**: 🔴 **CRITICAL - DECISION REQUIRED**  
**Priority**: **P0 - BLOCKS LLAMA TEAM START**  
**Owner**: [Project Manager + Tech Lead]  
**Deadline**: Before Week 2 (Llama Team start)
