# Performance Team Proposal - PM Analysis

**Date**: 2025-10-04  
**From**: Performance Team (Worker-orcd Lead)  
**To**: Project Manager (M0 Worker-orcd)  
**Subject**: M0 Performance Enhancement Proposal (+4 weeks)

---

## Executive Summary

**PM Decision**: ❌ **REJECT** - Proposal conflicts with locked M0 scope and timeline

**Rationale**:
1. M0 scope was **explicitly locked** on 2025-10-03 with Performance Bundle deferred to M1+
2. Current M0 timeline is **6-7 weeks** (89-110 days across 3 teams)
3. Proposal adds **+4 weeks** (28 days) = **10-11 weeks total** (+57% timeline increase)
4. All proposed features are **already documented as M1+ scope** in spec
5. Teams have **139 story cards ready** for current M0 scope
6. Adding scope now would require **complete re-planning** (189 artifacts to revise)

---

## Proposal Summary

### Requested M0 Additions (+4 weeks)

Performance team proposes adding 5 foundational performance features to M0:

1. **Paged KV Cache Block Allocator**
   - Fixed-page VRAM allocator for KV cache
   - Prepares for continuous batching in M1

2. **Per-Token Step Function (Decode Loop Refactor)**
   - Step-based inference loop (start → next_token → free)
   - Enables future continuous batching without API changes

3. **FlashAttention / CUDA Graph Path**
   - Fast attention kernel path
   - CUDA Graph wrapping for decode loop
   - Immediate per-token latency boost

4. **Prefix Cache (System Prompt KV Reuse)**
   - KV reuse for static prefixes
   - Improves first-token latency

5. **Metrics Hooks & Performance Tests**
   - No-op performance hooks
   - Minimal Criterion benchmark suite
   - Baseline data collection

**Claimed Benefits**:
- Competitive parity with llama.cpp/vLLM from day 1
- Clean seams for M1 advanced features
- Avoids "giant rework" later
- Localized to worker (no orchestrator impact)

---

## Current M0 Scope (Locked 2025-10-03)

### What's IN M0 (18 items - 6-7 weeks)

**Core Functionality**:
- ✅ 3 models: Qwen2.5-0.5B, Phi-3-Mini, GPT-OSS-20B
- ✅ 3 quantization formats: Q4_K_M, MXFP4, Q4_0
- ✅ 2 tokenizer backends: GGUF byte-BPE, tokenizer.json
- ✅ Architecture adapters: LlamaInferenceAdapter + GPTInferenceAdapter
- ✅ Architecture detection from GGUF metadata

**Critical Safety**:
- ✅ VRAM OOM handling (M0-W-1021)
- ✅ VRAM residency verification (M0-W-1012)
- ✅ Kernel safety validation (M0-W-1431)

**User Experience**:
- ✅ Model load progress events (M0-W-1621)
- ✅ Narration-core logging (basic events)

**Correctness**:
- ✅ MXFP4 numerical correctness validation (M0-W-1822)
- ✅ Minimal same-device reproducibility check (seeded RNG, temp=0)

### What's DEFERRED to M1+ (15 items)

**Performance Bundle** (explicitly deferred):
1. ❌ Prometheus metrics endpoint (M0-W-1350)
2. ❌ Performance metrics in logs (M0-W-1901)
3. ❌ Graceful shutdown endpoint (M0-W-1340)
4. ❌ First token latency target (M0-W-1600)
5. ❌ Token generation rate target (M0-W-1601)
6. ❌ Per-token latency target (M0-W-1602)
7. ❌ Execute endpoint performance (M0-W-1603)
8. ❌ Health endpoint performance (M0-W-1604)
9. ❌ Cancellation latency target (M0-W-1610)
10. ❌ Client disconnect detection (M0-W-1611)
11. ❌ Model loading time target (M0-W-1620)
12. ❌ Performance test suite (M0-W-1830)
13. ❌ Deep CUDA determinism audit (M0-W-1031)
14. ❌ Sensitive data handling in logs (M0-W-1902)
15. ❌ Graceful shutdown performance target (M0-W-1630)

**Reference**: `bin/.specs/01_M0_worker_orcd.md` §0.0 (lines 21-36)

---

## Conflict Analysis

### 1. Scope Lock Violation

**Current State**:
- M0 scope locked on **2025-10-03**
- Documented in spec: "Performance Bundle Deferral (Hybrid)"
- Rationale: "Balance faster delivery (4-5 weeks) with critical safety features"

**Proposal Impact**:
- Reopens scope discussion
- Adds 5 new features (all performance-related)
- Contradicts explicit deferral decision

**PM Assessment**: ❌ **Violates locked scope**

---

### 2. Timeline Impact

**Current M0 Timeline** (from execution docs):
- Foundation-Alpha: Days 1-89 (89 days / 12.7 weeks)
- Llama-Beta: Days 1-90 (90 days / 12.9 weeks)
- GPT-Gamma: Days 1-110 (110 days / 15.7 weeks) ← **Critical path**

**Note**: Teams work in **parallel**, so total calendar time is **110 days / 15.7 weeks**

**Proposal Timeline**:
- Current: 110 days
- Proposed addition: +28 days (+4 weeks)
- New total: **138 days / 19.7 weeks** (+25% increase)

**PM Assessment**: ❌ **Unacceptable timeline extension**

---

### 3. Planning Impact

**Current Planning Status** (from PM docs):
- ✅ 139 story cards created (49 Foundation, 39 Llama, 49 GPT, 2 prep)
- ✅ 24 sprint READMEs created
- ✅ 12 gate checklists created
- ✅ 16 execution templates created
- ✅ **189 total artifacts** ready for execution

**Proposal Impact**:
- Requires **new story cards** for 5 features
- Requires **sprint re-planning** (story sequencing changes)
- Requires **gate criteria updates** (new validation items)
- Requires **dependency re-mapping** (new blockers/blocked items)
- Estimated re-planning effort: **3-4 days** (PM team)

**PM Assessment**: ❌ **Massive re-planning overhead**

---

### 4. Feature-by-Feature Analysis

#### Proposed Feature 1: Paged KV Cache Block Allocator

**Current M0 Scope**:
- ✅ FT-021: KV Cache Allocation (simple contiguous VRAM)
- ✅ FT-022: KV Cache Management (basic lifecycle)

**Proposal**:
- Paged allocator with fixed-page blocks
- Prepares for continuous batching

**PM Assessment**:
- ❌ **Out of scope** - M0 is single-request only
- ❌ Continuous batching is M1+ feature
- ❌ Current simple allocator sufficient for M0
- ✅ Valid M1 feature (defer)

---

#### Proposed Feature 2: Per-Token Step Function

**Current M0 Scope**:
- ✅ Basic inference loop (prompt → tokens → done)
- ✅ Temperature scaling (M0-W-1032)
- ✅ Sampling kernel (FT-018)

**Proposal**:
- Refactor to step-based loop (start → next_token → free)
- Enables future continuous batching

**PM Assessment**:
- ❌ **Architectural change** - affects all inference code
- ❌ No M0 benefit (single-request only)
- ❌ Adds complexity without value
- ✅ Valid M1 refactor (defer)

---

#### Proposed Feature 3: FlashAttention / CUDA Graph Path

**Current M0 Scope**:
- ✅ Basic attention kernels (Llama: GQA, GPT: MHA)
- ✅ cuBLAS GEMM wrapper (FT-016)

**Proposal**:
- FlashAttention kernel path
- CUDA Graph wrapping

**PM Assessment**:
- ❌ **Performance optimization** - explicitly deferred
- ❌ M0 has no latency targets (deferred to M1)
- ❌ FlashAttention adds significant complexity
- ❌ CUDA Graphs require extensive testing
- ✅ Valid M1 optimization (defer)

---

#### Proposed Feature 4: Prefix Cache (System Prompt KV Reuse)

**Current M0 Scope**:
- ✅ Basic KV cache (no reuse)
- ✅ Single-request inference

**Proposal**:
- KV reuse for static prefixes
- Improves first-token latency

**PM Assessment**:
- ❌ **Performance optimization** - explicitly deferred
- ❌ M0 has no first-token latency target (M0-W-1600 deferred)
- ❌ Prefix caching requires cache key management
- ❌ Single-request M0 doesn't benefit
- ✅ Valid M1 feature (defer)

---

#### Proposed Feature 5: Metrics Hooks & Performance Tests

**Current M0 Scope**:
- ❌ Prometheus metrics (M0-W-1350 deferred)
- ❌ Performance test suite (M0-W-1830 deferred)
- ❌ Performance metrics in logs (M0-W-1901 deferred)

**Proposal**:
- No-op performance hooks
- Minimal Criterion benchmark suite

**PM Assessment**:
- ❌ **Explicitly deferred** - Performance Bundle
- ❌ Even "no-op hooks" add API surface
- ❌ Criterion benchmarks require maintenance
- ❌ M0 has no performance targets to validate
- ✅ Valid M1 feature (defer)

---

## Alternative: M0.5 Intermediate Milestone

**If performance features are critical**, consider:

### M0.5 Scope (Post-M0, Pre-M1)
- Duration: +2 weeks (not +4 weeks)
- Focus: Performance foundation only
- Features:
  1. ✅ Metrics hooks (no-op in M0.5, wired in M1)
  2. ✅ Basic Criterion benchmarks (baseline only)
  3. ✅ Per-token step function refactor (no batching yet)

**Benefits**:
- M0 delivers on time (110 days)
- Performance foundation added incrementally
- No re-planning of M0 (139 stories unchanged)
- M1 continuous batching builds on M0.5

**Timeline**:
- M0: Days 1-110 (current plan)
- M0.5: Days 111-124 (+14 days)
- M1: Days 125+ (continuous batching, etc.)

**PM Assessment**: ✅ **Acceptable compromise** (if stakeholders approve)

---

## Recommendation

### Primary Recommendation: ❌ REJECT Proposal

**Reasons**:
1. **Scope lock**: M0 scope was explicitly locked on 2025-10-03
2. **Timeline**: +4 weeks is unacceptable (+25% increase)
3. **Planning overhead**: 189 artifacts would need revision
4. **No M0 value**: All features benefit M1+, not M0
5. **Spec alignment**: All proposed features already documented as M1+ scope

**Action Items**:
1. ✅ Acknowledge Performance team's concerns
2. ✅ Confirm M0 scope remains locked
3. ✅ Document proposed features as M1 priorities
4. ✅ Proceed with current M0 execution (Day 1 launch)

---

### Alternative Recommendation: M0.5 Intermediate Milestone

**If stakeholders insist on performance foundation**:

**Proposal**:
- M0: Days 1-110 (current plan, unchanged)
- M0.5: Days 111-124 (+14 days, performance foundation)
- M1: Days 125+ (continuous batching, advanced features)

**M0.5 Scope** (reduced from +4 weeks to +2 weeks):
1. ✅ Per-token step function refactor (no batching)
2. ✅ Metrics hooks (no-op, wired in M1)
3. ✅ Basic Criterion benchmarks (baseline only)

**Deferred to M1**:
- ❌ Paged KV cache (requires continuous batching)
- ❌ FlashAttention (requires extensive testing)
- ❌ Prefix cache (requires cache management)

**Action Items**:
1. ⬜ Stakeholder approval required
2. ⬜ Create M0.5 planning documents (3 story cards, 1 sprint, 1 gate)
3. ⬜ Update timeline: M0 → M0.5 → M1
4. ⬜ Proceed with M0 execution (unchanged)

---

## Response to Performance Team

### Draft Response

```
Hi Performance Team,

Thank you for the detailed performance proposal. I've reviewed the 5 proposed M0 additions and their rationale.

**Decision**: I cannot approve adding these features to M0 at this time.

**Rationale**:

1. **Scope Lock**: M0 scope was explicitly locked on 2025-10-03 with "Performance Bundle Deferral" documented in the spec (§0.0). All 5 proposed features fall under the deferred Performance Bundle.

2. **Timeline Impact**: Adding +4 weeks extends M0 from 110 days to 138 days (+25% increase). This is unacceptable given our commitment to faster delivery.

3. **Planning Overhead**: We have 139 story cards, 24 sprint plans, and 12 gate checklists ready for execution. Adding scope now requires complete re-planning (189 artifacts to revise, 3-4 days PM effort).

4. **M0 Value**: All proposed features benefit M1+ (continuous batching, advanced scheduling), not M0 (single-request, correctness-focused). M0 has no performance targets to validate against.

**What's Already in M0**:
- ✅ 3 models, 3 quantization formats, 2 tokenizer backends
- ✅ Architecture adapters (clean design from day 1)
- ✅ Critical safety features (VRAM monitoring, OOM handling)
- ✅ Correctness validation (MXFP4 numerical tests, reproducibility)

**Alternative: M0.5 Intermediate Milestone**

If performance foundation is critical, I propose:
- **M0**: Days 1-110 (current plan, unchanged)
- **M0.5**: Days 111-124 (+2 weeks, not +4)
  - Per-token step function refactor
  - Metrics hooks (no-op, wired in M1)
  - Basic Criterion benchmarks (baseline only)
- **M1**: Days 125+ (continuous batching, FlashAttention, prefix cache)

This approach:
- ✅ Delivers M0 on time (110 days)
- ✅ Adds performance foundation incrementally
- ✅ Avoids M0 re-planning
- ✅ Prepares clean seams for M1

**Next Steps**:

1. **Immediate**: Proceed with M0 execution (Day 1 launch with current scope)
2. **M0.5 Discussion**: If stakeholders approve M0.5, we can plan it post-M0
3. **M1 Planning**: Document all 5 proposed features as M1 priorities

I appreciate your focus on competitive parity and clean architecture. Let's deliver a correct, safe M0 first, then build performance on that foundation.

Best,
Project Manager (M0 Worker-orcd)
```

---

## Supporting Evidence

### From Spec (01_M0_worker_orcd.md)

**§0.0 Scope Decision Summary**:
> **DEFERRED to M1+ (14 items - Performance Bundle)**:
> 1. ✅ Prometheus metrics endpoint (M0-W-1350)
> 5. ✅ First token latency target (M0-W-1600)
> 6. ✅ Token generation rate target (M0-W-1601)
> 7. ✅ Per-token latency target (M0-W-1602)
> 13. ✅ Performance test suite (M0-W-1830)

**§0.0 Key Trade-offs**:
> **Deferred to M1**:
> - ❌ Performance validation and benchmarking
> - ❌ Performance metrics collection

### From PM Responsibilities

**PM Mandate** (PM_RESPONSIBILITIES.md):
> We exist to ensure that **engineers never have to think about what to build next**. Every story card, every sprint plan, every acceptance criterion is so detailed that coding becomes mechanical execution.

**Current Status**:
- ✅ 139 story cards created (100% of M0 scope)
- ✅ 24 sprint READMEs created (100% of sprints)
- ✅ 12 gate checklists created (100% of gates)
- ✅ Planning complete, ready for execution

**Adding scope now violates our mandate**: Engineers would be blocked while we re-plan.

### From Execution Guide

**Day 1 Launch** (EXECUTION_GUIDE.md):
> **Morning (9:00 AM)**
> 
> **Step 1: Foundation-Alpha (Monitor 1)**
> - Current Story: FT-001 (HTTP Server Setup)
> - Sprint: Sprint 1 - HTTP Foundation
> - Launch agent prompt: "Begin FT-001: HTTP Server Setup"

**We are ready to execute**. Adding scope now delays Day 1 launch.

---

## Decision Matrix

| Criterion | Current M0 | Proposal (+4 weeks) | M0.5 Alternative (+2 weeks) |
|-----------|------------|---------------------|------------------------------|
| **Timeline** | 110 days | 138 days (+25%) ❌ | 124 days (+13%) ⚠️ |
| **Scope Lock** | Respected ✅ | Violated ❌ | Respected ✅ |
| **Planning Ready** | Yes ✅ | No (re-plan) ❌ | Yes (M0 unchanged) ✅ |
| **M0 Value** | High ✅ | Low (perf deferred) ❌ | Medium (foundation) ⚠️ |
| **M1 Prep** | Adequate ✅ | Better ✅ | Better ✅ |
| **Risk** | Low ✅ | High (scope creep) ❌ | Medium (timeline) ⚠️ |

**PM Recommendation**: **Current M0** (reject proposal) or **M0.5 Alternative** (if stakeholders approve)

---

## Conclusion

**Primary Decision**: ❌ **REJECT** Performance Team proposal

**Rationale**:
1. Violates locked M0 scope (Performance Bundle explicitly deferred)
2. Unacceptable timeline extension (+25%)
3. Massive re-planning overhead (189 artifacts)
4. No M0 value (all features benefit M1+)
5. Teams ready to execute current plan (Day 1 launch)

**Alternative**: M0.5 intermediate milestone (+2 weeks, not +4)
- Requires stakeholder approval
- Delivers M0 on time, adds performance foundation incrementally

**Next Action**: 
- ✅ Respond to Performance team (draft above)
- ✅ Proceed with M0 Day 1 launch (current scope)
- ⬜ Escalate M0.5 discussion to stakeholders (if needed)

---

**Status**: Analysis Complete  
**Decision**: Reject proposal, proceed with current M0 scope  
**Prepared By**: Project Manager (M0 Worker-orcd) 📋

---

Planned by Project Management Team 📋
