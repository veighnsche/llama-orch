# M0 Planning Gaps and Issues
**Date**: 2025-10-04  
**Context**: Three autonomous AI agents (Foundation-Alpha, Llama-Beta, GPT-Gamma)  
**Status**: 🔴 **3 CRITICAL GAPS IDENTIFIED**
---
## Executive Summary
**Total Gaps**: 3 critical issues that impact M0 delivery
**Impact**: 
- M0 timeline is **107 days** (not 8 weeks as originally planned)
- 15 days of idle time for Llama-Beta and GPT-Gamma waiting for FFI lock
- 2 spec requirements missing explicit stories
**Action Required**: Address gaps before agents begin work
---
## Gap 1: FFI Lock Timing (15 Days Idle) 🔴 CRITICAL
### The Problem
**What**: Llama-Beta and GPT-Gamma cannot start work until Foundation-Alpha locks FFI interface
**When**: Day 15 (after Foundation-Alpha completes FT-006 and FT-007)
**Impact**:
- **15 days of idle time** for two agents (days 1-14)
- Llama-Beta blocked from starting GGUF loader
- GPT-Gamma blocked from starting HF tokenizer
- No workaround - FFI interface is fundamental dependency
### Root Cause
Foundation-Alpha must complete 6 stories sequentially before FFI can be locked:
1. FT-001: HTTP Server Setup (2 days)
2. FT-002: POST /execute Endpoint (2 days)
3. FT-003: SSE Streaming (3 days)
4. FT-004: Correlation ID Middleware (1 day)
5. FT-005: Request Validation (1 day)
6. FT-006: FFI Interface Definition (2 days)
7. FT-007: Rust FFI Bindings (2 days)
**Total**: 13 days to reach FT-007 completion
### Mitigation Options
**Option A: Prep Work During Idle Time** ⭐ RECOMMENDED
- Llama-Beta: Study llama.cpp, GGUF format, design test framework
- GPT-Gamma: Study MXFP4 spec, HF tokenizers crate, design validation framework
- **Benefit**: Agents are productive during wait
- **Cost**: None
- **Action**: Add "prep work" stories LT-000 and GT-000
**Option B: Reorder Foundation Stories**
- Move FFI stories (FT-006, FT-007) earlier in sequence
- **Problem**: FFI depends on HTTP server context (FT-001-005)
- **Not feasible**: Dependencies prevent reordering
**Option C: Accept Idle Time**
- Do nothing, agents wait
- **Cost**: 15 days × 2 agents = 30 agent-days idle
- **Not recommended**: Wasteful
### Recommended Action
✅ **Add prep work stories**:
- **LT-000**: GGUF Format Research & Test Framework Design (3 days, days 1-3)
- **GT-000**: MXFP4 Spec Study & Validation Framework Design (3 days, days 1-3)
**Reduces idle time from 15 days to 12 days** (still some idle, but agents are productive)
---
## Gap 2: Missing Spec Coverage Stories 🟡 MEDIUM
### The Problem
**What**: 2 spec requirements have no explicit implementation stories
**Impact**: 
- M0 acceptance criteria incomplete (16/17 covered)
- Observability gaps
- User feedback missing during model loading
### Missing Story 1: Model Load Progress Events (M0-W-1621) 🔴 CRITICAL
**Spec Requirement**:
> Worker SHOULD emit progress events during model loading.
> Progress points: 0%, 25%, 50%, 75%, 100%
**Why Critical**: 
- GPT-OSS-20B takes significant time to load (12 GB model)
- User needs feedback during long-running operation
- Explicitly called out in M0 acceptance criteria (§15.1 item 17)
**Current Coverage**: ❌ Not covered by any story
**Recommended Action**:
- **Story ID**: FT-049
- **Title**: Model Load Progress Events
- **Owner**: Foundation-Alpha
- **Sprint**: 3 (days 23-38, alongside shared kernels)
- **Size**: S (1 day)
- **Spec Ref**: M0-W-1621
- **Acceptance Criteria**:
  - [ ] Progress callback in CUDA model loader (C++)
  - [ ] Rust FFI wrapper surfaces progress events
  - [ ] Log events with `event="model_load_progress"` and `percent` field
  - [ ] Progress emitted at 0%, 25%, 50%, 75%, 100%
  - [ ] Integration test validates all 5 progress points
**Impact on Timeline**: +1 day to Foundation-Alpha (87 days → 88 days)
---
### Missing Story 2: Narration-Core Logging (M0-W-1900) 🟡 MEDIUM
**Spec Requirement**:
> Worker-orcd MUST emit narration-core logs with basic event tracking.
> Event types: startup, model_load_start, model_load_progress, model_load_complete, ready, execute_start, execute_end, error, shutdown
**Why Important**: 
- Basic observability and debugging capability
- Required for production operations
- Enables correlation across system
**Current Coverage**: ⚠️ Implicit in FT-037 (API Documentation) but no explicit implementation
**Recommended Action**:
- **Story ID**: FT-050
- **Title**: Narration-Core Logging Implementation
- **Owner**: Foundation-Alpha
- **Sprint**: 2-3 (days 10-22, early in HTTP layer)
- **Size**: S (1 day)
- **Spec Ref**: M0-W-1900
- **Acceptance Criteria**:
  - [ ] Structured logging framework set up (e.g., `tracing` crate)
  - [ ] Context fields: `worker_id`, `job_id`, `model_ref`, `gpu_device`, `event`
  - [ ] Event types implemented: startup, model_load_start, model_load_progress, model_load_complete, ready, execute_start, execute_end, error, shutdown
  - [ ] All events emit at appropriate lifecycle points
  - [ ] Integration test validates event sequence
**Impact on Timeline**: +1 day to Foundation-Alpha (87 days → 88 days)
---
### Combined Impact
**Total Additional Stories**: 2 (FT-049, FT-050)  
**Total Additional Effort**: 2 days  
**New Foundation-Alpha Timeline**: 87 days → 89 days  
**New M0 Timeline**: 107 days → 109 days (if Foundation becomes critical path - it doesn't, GPT-Gamma still longer)
**Actual M0 Impact**: None (Foundation-Alpha still finishes before GPT-Gamma)
---
## Gap 3: Unclear Validation Criteria 🟡 MEDIUM
### The Problem
**What**: Some stories lack clear, testable success criteria
**Impact**:
- Agents may implement incorrectly
- Rework required
- Timeline extensions
### Examples of Unclear Criteria
**1. MXFP4 Numerical Correctness (GT-038)**
**Current**: "MXFP4 Numerical Validation (±1%)"
**Problem**: 
- ±1% of what? (per-token logits? final output? layer outputs?)
- How to measure? (MSE? max absolute error? percentile?)
- What's the baseline? (Q4_K_M? FP16?)
**Recommended Clarification**:
```
Acceptance Criteria:
- [ ] Compare MXFP4 vs Q4_K_M baseline on 10 test prompts
- [ ] Per-token logit difference < 1% (mean absolute error)
- [ ] Final output token matches Q4_K_M in 95% of cases
- [ ] Layer-wise activation differences documented
- [ ] Numerical validation report generated
```
**2. Tokenizer Conformance Tests (LT-018, GT-004)**
**Current**: "Tokenizer Conformance Tests"
**Problem**:
- How many test cases? (spec says 20-30 pairs)
- What coverage? (ASCII, UTF-8, multibyte, edge cases?)
- What's the reference? (upstream tokenizer? manual vectors?)
**Recommended Clarification**:
```
Acceptance Criteria:
- [ ] 25 test pairs covering: ASCII (10), UTF-8 (10), edge cases (5)
- [ ] Test vectors from upstream reference implementation
- [ ] Round-trip validation (encode → decode → original)
- [ ] Conformance test suite runs in CI
- [ ] All tests pass with 100% match
```
**3. Reproducibility Tests (LT-026, LT-036)**
**Current**: "Reproducibility Validation"
**Problem**:
- How many runs? (spec says 10)
- What's the tolerance? (exact match? statistical?)
- What seeds? (fixed? random?)
**Recommended Clarification**:
```
Acceptance Criteria:
- [ ] 10 runs with same seed (seed=42)
- [ ] All runs produce identical output (byte-for-byte)
- [ ] Test on both Qwen and Phi-3
- [ ] Reproducibility report generated
- [ ] Seed recorded in 
```
### Recommended Action
**Before agents start work**:
1. Review all story cards for unclear acceptance criteria
2. Add explicit, measurable success criteria
3. Define test vectors and baselines
4. Document validation methods
**Estimated Effort**: 2-3 days of planning work (before agent execution)
**Impact on Timeline**: None (done before agents start)
---
## Summary of All Gaps
| Gap | Severity | Impact | Mitigation | Timeline Impact |
|-----|----------|--------|------------|-----------------|
| **Gap 1**: FFI Lock Timing | 🔴 CRITICAL | 15 days idle | Add prep work stories | None (agents productive) |
| **Gap 2**: Missing Stories | 🟡 MEDIUM | Incomplete M0 | Add FT-049, FT-050 | +2 days (absorbed by buffer) |
| **Gap 3**: Unclear Criteria | 🟡 MEDIUM | Rework risk | Clarify before start | None (planning work) |
---
## Recommended Actions
### Immediate (Before Agent Start)
1. ✅ **Add prep work stories** (LT-000, GT-000)
   - Reduces idle time from 15 days to 12 days
   - Agents productive during FFI lock wait
2. ✅ **Add missing spec stories** (FT-049, FT-050)
   - Achieves 100% spec coverage
   - Adds 2 days to Foundation-Alpha
3. ✅ **Clarify validation criteria**
   - Review all 135 story cards
   - Add explicit acceptance criteria
   - Define test vectors and baselines
### During Execution
4. ✅ **Publish FFI lock document**
   - Foundation-Alpha publishes `FFI_INTERFACE_LOCKED.md` after FT-007
   - Includes C header, Rust bindings, usage examples
   - Notifies Llama-Beta and GPT-Gamma to begin
5. ✅ **Monitor dependencies**
   - Track Foundation-Alpha progress on critical milestones
   - Alert if FFI lock delayed beyond day 15
   - Coordinate integration framework (day 52) and adapter pattern (day 71)
---
## Revised M0 Timeline
**With all gaps addressed**:
| Agent | Stories | Days | Start | Finish | Notes |
|-------|---------|------|-------|--------|-------|
| Foundation-Alpha | 49 | 89 | Day 1 | Day 89 | +2 days for FT-049, FT-050 |
| Llama-Beta | 39 | 75 | Day 1 | Day 90 | +3 days for LT-000 prep work |
| GPT-Gamma | 49 | 95 | Day 1 | Day 110 | +3 days for GT-000 prep work |
**M0 Delivery**: **Day 110** (GPT-Gamma critical path)
**Note**: Prep work (LT-000, GT-000) happens days 1-3, then agents wait until day 15 for FFI lock. Total timeline increases by 3 days due to prep work being added to the front.
---
## What's NOT a Gap
### ❌ These are NOT planning gaps (human team assumptions):
1. ~~"Week 7 overcommitted"~~ - Agent works sequentially, no overcommitment
2. ~~"Need 4th person for GPT team"~~ - Cannot scale agent count
3. ~~"Team burnout"~~ - Not applicable to AI agents
4. ~~"Utilization percentage"~~ - Meaningless for sequential execution
5. ~~"Parallel work streams"~~ - Agents work one story at a time
6. ~~"Sprint velocity"~~ - Agents work at consistent pace
7. ~~"Resource allocation"~~ - Fixed: 3 agents, cannot change
---
## Conclusion
**3 actual gaps identified**:
1. 🔴 FFI lock timing (15 days idle) - **Mitigated with prep work**
2. 🟡 Missing spec stories (2 stories) - **Add FT-049, FT-050**
3. 🟡 Unclear validation criteria - **Clarify before start**
**Revised M0 timeline**: **110 days** (with all gaps addressed)
**Next action**: Address gaps before agents begin work
---
**Status**: ✅ **GAPS IDENTIFIED AND MITIGATED**  
**M0 Timeline**: Day 110 (revised from day 107)  
**Ready to Execute**: After gap mitigation complete
---
*Analysis by Project Manager, reviewed by Narration Core Team 🎀*
