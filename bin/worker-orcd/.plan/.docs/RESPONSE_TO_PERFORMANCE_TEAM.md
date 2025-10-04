# Response to Performance Team - M0 Scope Addition Request

**Date**: 2025-10-04  
**From**: Project Manager (M0 Worker-orcd)  
**To**: Performance Team (Worker-orcd Lead)  
**Re**: M0 Performance Enhancement Proposal (+4 weeks)

---

## Email Draft

```
Hi Performance Team,

Thank you for the detailed performance proposal. I've reviewed the 5 proposed M0 additions (paged KV cache, per-token step function, FlashAttention/CUDA Graphs, prefix cache, metrics hooks) and their rationale for competitive parity.

**Decision**: I cannot approve adding these features to M0 at this time.

**Rationale**:

1. **Scope Lock**: M0 scope was explicitly locked on 2025-10-03 with "Performance Bundle Deferral" documented in the spec (§0.0, lines 21-36). All 5 proposed features fall under the deferred Performance Bundle.

2. **Timeline Impact**: Adding +4 weeks extends M0 from 110 days to 138 days (+25% increase). This contradicts our commitment to faster delivery.

3. **Planning Overhead**: We have 139 story cards, 24 sprint plans, and 12 gate checklists ready for Day 1 execution. Adding scope now requires complete re-planning (189 artifacts to revise, 3-4 days PM effort).

4. **M0 Value**: All proposed features benefit M1+ (continuous batching, advanced scheduling), not M0 (single-request, correctness-focused). M0 has no performance targets to validate against.

**What's Already in M0**:
- ✅ 3 models (Qwen, Phi-3, GPT-OSS-20B)
- ✅ 3 quantization formats (Q4_K_M, MXFP4, Q4_0)
- ✅ Architecture adapters (clean design from day 1)
- ✅ Critical safety features (VRAM monitoring, OOM handling)
- ✅ Correctness validation (MXFP4 numerical tests, reproducibility)

**Alternative: M0.5 Intermediate Milestone**

If performance foundation is critical, I propose:

**Timeline**:
- **M0**: Days 1-110 (current plan, unchanged)
- **M0.5**: Days 111-124 (+2 weeks, not +4)
- **M1**: Days 125+ (continuous batching, FlashAttention, prefix cache)

**M0.5 Scope** (reduced from 5 to 3 features):
1. ✅ Per-token step function refactor (no batching yet)
2. ✅ Metrics hooks (no-op, wired in M1)
3. ✅ Basic Criterion benchmarks (baseline only)

**Deferred to M1**:
- ❌ Paged KV cache (requires continuous batching)
- ❌ FlashAttention (requires extensive testing)
- ❌ Prefix cache (requires cache management)

**M0.5 Benefits**:
- ✅ M0 delivers on time (110 days)
- ✅ Performance foundation added incrementally
- ✅ No M0 re-planning (139 stories unchanged)
- ✅ Clean seams for M1

**Next Steps**:

1. **Immediate**: Proceed with M0 Day 1 launch (current scope)
2. **M0.5 Discussion**: If management approves M0.5, we can plan it post-M0
3. **M1 Planning**: Document all 5 proposed features as M1 priorities

I appreciate your focus on competitive parity and clean architecture. Let's deliver a correct, safe M0 first, then build performance on that foundation.

**Supporting Documents**:
- Detailed analysis: `.docs/PERFORMANCE_TEAM_PROPOSAL_ANALYSIS.md`
- Management summary: `.docs/MANAGEMENT_SUMMARY_PERFORMANCE_PROPOSAL.md`
- M0 spec: `bin/.specs/01_M0_worker_orcd.md` (§0.0)

Best,
Project Manager (M0 Worker-orcd) 📋
```

---

## Key Points to Emphasize

### 1. Acknowledge Validity
- ✅ Competitive parity matters
- ✅ Clean architecture matters
- ✅ Avoiding rework matters

### 2. Explain Constraints
- ❌ Scope locked on 2025-10-03
- ❌ Timeline extension unacceptable (+25%)
- ❌ Planning overhead significant (189 artifacts)

### 3. Offer Alternative
- ✅ M0.5 intermediate milestone (+2 weeks, not +4)
- ✅ Reduced scope (3 features, not 5)
- ✅ M0 unchanged (110 days)

### 4. Maintain Relationship
- ✅ Appreciate their input
- ✅ Document features as M1 priorities
- ✅ Collaborate on M0.5 if approved

---

## Follow-Up Actions

### If Performance Team Accepts Rejection:
1. ✅ Document 5 features as M1 priorities
2. ✅ Proceed with M0 Day 1 launch
3. ✅ Collaborate on M1 planning post-M0

### If Performance Team Pushes Back:
1. ⬜ Escalate to management (use MANAGEMENT_SUMMARY doc)
2. ⬜ Request management decision on scope lock
3. ⬜ Hold M0 Day 1 launch until decision made

### If M0.5 Approved by Management:
1. ⬜ Notify Performance team of approval
2. ⬜ Create M0.5 planning documents (3 story cards, 1 sprint, 1 gate)
3. ⬜ Proceed with M0 Day 1 launch (unchanged)
4. ⬜ Plan M0.5 execution for Days 111-124

---

## Talking Points (If Discussion Needed)

### On Competitive Parity

**Performance Team**: "We need parity with llama.cpp/vLLM from day 1"

**PM Response**: 
> "M0 is about correctness and safety, not performance. We're building a foundation that works correctly first. M1 adds performance optimizations on that foundation. This is a deliberate sequencing choice, not a gap."

### On "Giant Rework" Concern

**Performance Team**: "Adding these later requires giant rework"

**PM Response**:
> "The proposed features (paged KV cache, step function, etc.) are localized to the worker inference loop. They don't affect HTTP layer, FFI boundary, or model loading. M0.5 or M1 can add them incrementally without 'giant rework'. The architecture adapters in M0 already provide clean seams."

### On Timeline

**Performance Team**: "Only +4 weeks for competitive parity"

**PM Response**:
> "+4 weeks is +25% timeline extension. We committed to 'faster delivery' (4-5 weeks foundation). If performance foundation is critical, M0.5 (+2 weeks) is a better compromise. It delivers M0 on time, then adds performance incrementally."

### On Scope Lock

**Performance Team**: "Scope lock was premature"

**PM Response**:
> "Scope lock on 2025-10-03 was deliberate after analyzing contradictions. We chose 'Performance Bundle Deferral (Hybrid)' to balance speed with critical safety. Reopening scope now sets a bad precedent and wastes planning investment (189 artifacts)."

---

## Escalation Path

### Level 1: PM → Performance Team (This Email)
- ✅ Explain decision
- ✅ Offer M0.5 alternative
- ✅ Request acceptance or escalation

### Level 2: Management Decision
- ⬜ Use MANAGEMENT_SUMMARY doc
- ⬜ Request decision on scope lock
- ⬜ Options: Current M0 / M0.5 / +4 weeks

### Level 3: Stakeholder Alignment
- ⬜ Present trade-offs (timeline vs features)
- ⬜ Align on M0 mission (correctness vs performance)
- ⬜ Finalize scope and timeline

---

## Success Criteria

**Email is successful if**:
1. ✅ Performance team understands decision rationale
2. ✅ Performance team accepts M0 scope lock
3. ✅ Performance team collaborates on M1 planning
4. ✅ M0 Day 1 launch proceeds on schedule

**Email requires escalation if**:
1. ❌ Performance team rejects decision
2. ❌ Performance team insists on +4 weeks
3. ❌ Performance team blocks M0 execution

---

**Status**: Draft Ready  
**Next Action**: Send email to Performance team  
**Prepared By**: Project Manager (M0 Worker-orcd) 📋

---

Planned by Project Management Team 📋
