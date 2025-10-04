# Management Summary: Performance Team Proposal

**Date**: 2025-10-04  
**To**: Management  
**From**: Project Manager (M0 Worker-orcd)  
**Subject**: Performance Team M0 Scope Addition Request - Decision Required

---

## TL;DR

**Performance team requests adding 5 features to M0, extending timeline by +4 weeks (+25%)**

**PM Recommendation**: ❌ **REJECT** - Violates locked scope, unacceptable timeline impact

**Alternative**: M0.5 intermediate milestone (+2 weeks) if performance foundation is critical

---

## The Request

Performance team proposes adding to M0:

1. **Paged KV Cache Block Allocator** - Prepares for continuous batching
2. **Per-Token Step Function** - Refactor inference loop for future batching
3. **FlashAttention / CUDA Graphs** - Fast attention kernels, latency optimization
4. **Prefix Cache** - System prompt KV reuse
5. **Metrics Hooks & Performance Tests** - Baseline benchmarking

**Claimed Benefits**:
- Competitive parity with llama.cpp/vLLM from day 1
- Clean seams for M1 advanced features
- Avoids "giant rework" later

**Timeline Impact**: +4 weeks (110 days → 138 days, +25% increase)

---

## Current M0 Status

### Scope (Locked 2025-10-03)

**What's IN M0** (6-7 weeks):
- ✅ 3 models: Qwen2.5-0.5B, Phi-3-Mini, GPT-OSS-20B
- ✅ 3 quantization formats: Q4_K_M, MXFP4, Q4_0
- ✅ Architecture adapters (clean design)
- ✅ Critical safety (VRAM monitoring, OOM handling)
- ✅ Correctness validation (MXFP4 numerical tests)

**What's DEFERRED to M1+** (Performance Bundle):
- ❌ Performance metrics & monitoring
- ❌ Performance test suite
- ❌ Latency targets (first-token, per-token, etc.)
- ❌ Graceful shutdown
- ❌ Advanced optimizations

### Planning Status

- ✅ **139 story cards** created (49 Foundation, 39 Llama, 49 GPT, 2 prep)
- ✅ **24 sprint READMEs** created
- ✅ **12 gate checklists** created
- ✅ **189 total artifacts** ready for Day 1 execution

### Timeline

- **Foundation-Alpha**: Days 1-89 (12.7 weeks)
- **Llama-Beta**: Days 1-90 (12.9 weeks)
- **GPT-Gamma**: Days 1-110 (15.7 weeks) ← Critical path
- **Total**: 110 days (teams work in parallel)

---

## PM Analysis

### Why REJECT?

#### 1. Scope Lock Violation
- M0 scope **explicitly locked** on 2025-10-03
- Spec documents "Performance Bundle Deferral (Hybrid)"
- All 5 proposed features are **already documented as M1+ scope**

#### 2. Timeline Impact
- Current: 110 days
- Proposed: 138 days (+28 days, +25%)
- **Unacceptable** for "faster delivery" goal

#### 3. Planning Overhead
- **189 artifacts** would need revision
- **3-4 days** PM re-planning effort
- Delays Day 1 execution launch

#### 4. No M0 Value
- All features benefit **M1+ only** (continuous batching, advanced scheduling)
- M0 is **single-request, correctness-focused**
- M0 has **no performance targets** to validate against

#### 5. Execution Ready
- Teams ready for **Day 1 launch** with current scope
- Adding scope now **blocks engineers** during re-planning

---

## Alternative: M0.5 Intermediate Milestone

**If performance foundation is deemed critical**:

### M0.5 Scope (+2 weeks, not +4)

**Timeline**:
- M0: Days 1-110 (current plan, **unchanged**)
- M0.5: Days 111-124 (+14 days)
- M1: Days 125+ (continuous batching, advanced features)

**M0.5 Features** (reduced from 5 to 3):
1. ✅ Per-token step function refactor (no batching yet)
2. ✅ Metrics hooks (no-op, wired in M1)
3. ✅ Basic Criterion benchmarks (baseline only)

**Deferred to M1**:
- ❌ Paged KV cache (requires continuous batching)
- ❌ FlashAttention (requires extensive testing)
- ❌ Prefix cache (requires cache management)

**Benefits**:
- ✅ M0 delivers on time (110 days)
- ✅ Performance foundation added incrementally
- ✅ No M0 re-planning (139 stories unchanged)
- ✅ Clean seams for M1

**Drawbacks**:
- ⚠️ Still extends timeline (+13% vs +25%)
- ⚠️ Requires stakeholder approval
- ⚠️ Adds planning overhead (3 story cards, 1 sprint, 1 gate)

---

## Decision Matrix

| Option | Timeline | Scope Lock | Planning | M0 Value | M1 Prep | Risk |
|--------|----------|------------|----------|----------|---------|------|
| **Current M0** | 110 days ✅ | Respected ✅ | Ready ✅ | High ✅ | Adequate ✅ | Low ✅ |
| **+4 weeks proposal** | 138 days ❌ | Violated ❌ | Re-plan ❌ | Low ❌ | Better ✅ | High ❌ |
| **M0.5 (+2 weeks)** | 124 days ⚠️ | Respected ✅ | Ready ✅ | Medium ⚠️ | Better ✅ | Medium ⚠️ |

---

## Recommendations

### Primary Recommendation: ❌ REJECT Proposal

**Action Items**:
1. ✅ Acknowledge Performance team's concerns
2. ✅ Confirm M0 scope remains locked
3. ✅ Document proposed features as M1 priorities
4. ✅ Proceed with M0 Day 1 launch (current scope)

**Rationale**: Violates locked scope, unacceptable timeline impact, no M0 value

---

### Alternative Recommendation: M0.5 Intermediate Milestone

**If management approves**:

**Action Items**:
1. ⬜ Management approval required
2. ⬜ Create M0.5 planning documents (3 story cards, 1 sprint, 1 gate)
3. ⬜ Update timeline: M0 (110d) → M0.5 (+14d) → M1
4. ⬜ Proceed with M0 execution (unchanged)

**Rationale**: Balances performance foundation with timeline constraints

---

## Questions for Management

1. **Scope Lock**: Do we respect the 2025-10-03 scope lock, or reopen M0 scope?

2. **Timeline**: Is +25% timeline extension acceptable for performance features?

3. **M0.5 Option**: Should we consider M0.5 intermediate milestone (+13% timeline)?

4. **Competitive Parity**: Is day-1 parity with llama.cpp/vLLM a hard requirement?

5. **M1 Planning**: Can performance features wait until M1 (post-M0)?

---

## PM Position

**I recommend REJECTING the proposal** for the following reasons:

1. **Scope discipline**: We locked M0 scope on 2025-10-03 after careful analysis. Reopening scope now sets a bad precedent.

2. **Timeline commitment**: We committed to "faster delivery" (4-5 weeks foundation). +25% extension violates that commitment.

3. **Planning investment**: We invested significant effort (189 artifacts) in current M0 plan. Re-planning wastes that investment.

4. **M0 mission**: M0 is about **correctness and safety**, not performance. Performance is M1+ scope.

5. **Execution readiness**: Teams are ready for Day 1 launch. Adding scope now blocks execution.

**If management insists on performance foundation**, M0.5 intermediate milestone is the **least disruptive** option:
- M0 delivers on time (110 days)
- Performance foundation added incrementally (+14 days)
- No M0 re-planning required

---

## Next Steps

### If REJECT (Recommended):
1. ✅ Respond to Performance team (see detailed analysis doc)
2. ✅ Proceed with M0 Day 1 launch (current scope)
3. ✅ Document proposed features as M1 priorities

### If M0.5 APPROVED:
1. ⬜ Notify Performance team of M0.5 approval
2. ⬜ Create M0.5 planning documents (1 day PM effort)
3. ⬜ Proceed with M0 Day 1 launch (unchanged)
4. ⬜ Plan M0.5 execution for Days 111-124

### If +4 weeks APPROVED (Not Recommended):
1. ⬜ Notify all teams of scope change
2. ⬜ Re-plan M0 (3-4 days PM effort)
3. ⬜ Delay Day 1 launch until re-planning complete
4. ⬜ Update timeline: 110 days → 138 days

---

## Supporting Documents

- **Detailed Analysis**: `.docs/PERFORMANCE_TEAM_PROPOSAL_ANALYSIS.md`
- **M0 Spec**: `bin/.specs/01_M0_worker_orcd.md` (§0.0 Scope Decision)
- **PM Responsibilities**: `.docs/PM_RESPONSIBILITIES.md`
- **Execution Guide**: `.docs/EXECUTION_GUIDE.md`
- **Current Planning**: 139 story cards, 24 sprints, 12 gates (ready)

---

## Conclusion

**Performance team's concerns are valid** - competitive parity and clean architecture matter.

**However, the proposal conflicts with locked M0 scope and timeline commitments.**

**PM Recommendation**: 
1. **REJECT** +4 weeks proposal
2. **PROCEED** with current M0 (110 days)
3. **CONSIDER** M0.5 intermediate milestone (+2 weeks) if stakeholders approve
4. **DOCUMENT** all 5 features as M1 priorities

**Decision Required**: Management approval to proceed with current M0 scope, or approval for M0.5 alternative.

---

**Prepared By**: Project Manager (M0 Worker-orcd) 📋  
**Date**: 2025-10-04  
**Status**: Awaiting Management Decision

---

Planned by Project Management Team 📋
