# What the PM Missed: FT-049 Analysis

**Date**: 2025-10-04  
**Story**: FT-049: Narration-Core Logging Integration  
**Reviewer**: Narration-Core Team  
**Status**: ⚠️ Incomplete Planning

---

## The Problem

The PM wrote **FT-049** as if Foundation-Alpha (worker-orcd team) needs to "integrate narration-core logging patterns" internally. But this misses a critical insight:

> **We ARE narration-core. We don't integrate with ourselves. Other teams integrate with US.**

The story should have been split into **two perspectives**:

1. **Narration-Core's responsibilities** (provide taxonomy, guides, BDD tests)
2. **Worker-Orcd's responsibilities** (add dependency, emit narration, propagate correlation IDs)

---

## What the PM Missed

### 1. No Dependency Specification

**Missing**: How does worker-orcd add narration-core as a dependency?

**Fixed**: Created integration guide with explicit `Cargo.toml` changes:

```toml
[dependencies]
observability-narration-core = { path = "../shared-crates/narration-core" }
```

### 2. No Taxonomy Extensions

**Missing**: Does narration-core provide worker-specific actors/actions?

**Fixed**: Added to `src/lib.rs`:

```rust
pub const ACTOR_WORKER_ORCD: &str = "worker-orcd";
pub const ACTION_INFERENCE_START: &str = "inference_start";
pub const ACTION_INFERENCE_COMPLETE: &str = "inference_complete";
pub const ACTION_HEARTBEAT_SEND: &str = "heartbeat_send";
pub const ACTION_READY_CALLBACK: &str = "ready_callback";
pub const ACTION_CANCEL: &str = "cancel";
```

### 3. No Integration Guide

**Missing**: How does worker-orcd actually use narration-core?

**Fixed**: Created `docs/WORKER_ORCD_INTEGRATION.md` with:
- Step-by-step dependency setup
- Correlation ID extraction from HTTP headers
- Critical path narration examples
- Editorial guidelines
- Testing examples
- Verification commands

### 4. No BDD Scenarios

**Missing**: How do we verify worker-orcd is using narration correctly?

**Fixed**: Created `bdd/features/worker_orcd_integration.feature` with 25 scenarios covering:
- Correlation ID propagation
- Performance metrics emission
- Editorial standards compliance
- Error context quality
- Secret redaction
- Distributed tracing support

### 5. No Editorial Review Process

**Missing**: Who verifies narration quality?

**Fixed**: Documented in integration guide:
- Narration-core has **ultimate editorial authority**
- Editorial checklist (clarity, specificity, brevity, tense, voice, context, secrets, correlation)
- Review examples (good vs. bad narration)
- Submission process for editorial review

### 6. No Correlation ID Propagation Spec

**Missing**: How do correlation IDs flow rbees-orcd → worker-orcd → pool-managerd?

**Fixed**: Integration guide includes:
- HTTP header extraction: `X-Correlation-Id`
- Outgoing request propagation
- Testing correlation ID presence
- Verification commands

### 7. No Acceptance Criteria Split

**Missing**: What does narration-core deliver vs. what does worker-orcd deliver?

**Fixed**: Split into two checklists:

**Narration-Core delivers**:
- ✅ Worker-specific taxonomy
- ✅ Integration guide
- ✅ BDD scenarios
- ✅ Editorial review process
- ✅ Example narrations

**Worker-Orcd delivers**:
- ⏳ Dependency added
- ⏳ Critical path narrations
- ⏳ Correlation ID extraction
- ⏳ Performance metrics
- ⏳ Error context

### 8. No Example Narrations

**Missing**: What should worker-orcd's narration actually look like?

**Fixed**: Integration guide includes complete examples:
- Inference start
- Inference complete (with metrics)
- Heartbeat
- Ready callback
- Error handling (with context)

---

## Root Cause Analysis

### Why the PM Missed This

1. **Perspective confusion**: PM wrote from consumer perspective, not provider perspective
2. **No team boundary clarity**: Didn't distinguish narration-core's work from worker-orcd's work
3. **Missing integration pattern**: Didn't recognize this as a "library provides, consumer integrates" story
4. **No editorial authority recognition**: Didn't account for narration-core's review responsibilities

### How to Prevent This

1. **Team-specific stories**: When a foundational library is involved, write TWO stories:
   - One for the library team (provide support)
   - One for the consumer team (integrate)

2. **Clear acceptance criteria split**: Always distinguish "what we provide" from "what they do"

3. **Integration guides first**: For any library integration story, write the integration guide BEFORE the story

4. **Editorial review process**: For narration-core specifically, always include editorial review as an acceptance criterion

---

## Corrective Actions Taken

### Narration-Core Deliverables (Completed)

1. ✅ **Extended taxonomy** (`src/lib.rs`)
   - Added `ACTOR_WORKER_ORCD`, `ACTOR_INFERENCE_ENGINE`
   - Added worker-specific actions (inference, heartbeat, ready, cancel)
   - Added VRAM and pool management actions

2. ✅ **Created integration guide** (`docs/WORKER_ORCD_INTEGRATION.md`)
   - 9 steps from dependency to verification
   - Complete code examples
   - Editorial guidelines
   - Testing patterns
   - Verification commands

3. ✅ **Created BDD scenarios** (`bdd/features/worker_orcd_integration.feature`)
   - 25 scenarios covering all integration aspects
   - Correlation ID propagation tests
   - Editorial standards verification
   - Performance metrics validation
   - Error context quality checks

4. ✅ **Updated README** (`README.md`)
   - Added "Integration Guides" section
   - Linked to worker-orcd guide
   - Placeholder for future guides (rbees-orcd, pool-managerd)

5. ✅ **Created planning doc** (`.plan/FT-049-worker-orcd-integration.md`)
   - Narration-core perspective on the story
   - Clear responsibilities
   - Acceptance criteria split
   - Example narrations

6. ✅ **Created this analysis** (`.plan/PM_MISSED_ANALYSIS.md`)
   - Root cause analysis
   - Corrective actions
   - Prevention guidelines

### Worker-Orcd Deliverables (Pending)

These are **worker-orcd Foundation-Alpha's** responsibilities:

1. ⏳ Add `observability-narration-core` dependency to `Cargo.toml`
2. ⏳ Import taxonomy constants
3. ⏳ Extract correlation IDs from HTTP headers
4. ⏳ Emit narration at critical paths (start, complete, heartbeat, ready, error)
5. ⏳ Include performance metrics (duration_ms, tokens_out)
6. ⏳ Propagate correlation IDs in outgoing HTTP requests
7. ⏳ Write unit tests using `CaptureAdapter`
8. ⏳ Submit for editorial review

---

## Lessons Learned

### For PMs

1. **Identify the provider team**: When a foundational library is involved, identify who PROVIDES vs. who CONSUMES
2. **Write two stories**: One for provider, one for consumer
3. **Integration guides are deliverables**: Not just code, but documentation
4. **Editorial review is work**: For narration-core, editorial review is a deliverable

### For Narration-Core

1. **We're a service provider**: Our job is to make integration easy
2. **Guides are as important as code**: Integration guides prevent misuse
3. **BDD tests verify integration**: Not just our code, but consumer usage patterns
4. **Editorial authority is real**: We need to review and approve narration quality

### For Worker-Orcd

1. **Follow the integration guide**: We wrote it for a reason
2. **Correlation IDs are mandatory**: Extract from headers, include in all narration
3. **Performance metrics matter**: Always include duration_ms, tokens_out
4. **Submit for editorial review**: We'll help make your narration excellent

---

## Next Steps

### Immediate (Day 73)

1. **Narration-Core**: Notify worker-orcd team that integration guide is ready
2. **Worker-Orcd**: Review integration guide and begin implementation
3. **PM**: Update FT-049 to reference narration-core's deliverables

### Short-Term (Day 74)

1. **Worker-Orcd**: Complete integration (add dependency, emit narration)
2. **Narration-Core**: Conduct editorial review of worker-orcd's narration
3. **Both teams**: Verify BDD scenarios pass

### Medium-Term (Sprint 8)

1. **Narration-Core**: Create integration guides for rbees-orcd and pool-managerd
2. **All teams**: Adopt narration-core consistently
3. **PM**: Learn from this experience for future library integration stories

---

## Success Metrics

### Narration-Core

- ✅ Integration guide published
- ✅ BDD scenarios written
- ✅ Taxonomy extended
- ⏳ Worker-orcd successfully integrated
- ⏳ Editorial review completed

### Worker-Orcd

- ⏳ Dependency added
- ⏳ All critical paths emit narration
- ⏳ Correlation IDs propagated
- ⏳ Performance metrics included
- ⏳ Editorial review passed

### Project

- ⏳ Distributed tracing works end-to-end (rbees-orcd → worker-orcd → pool-managerd)
- ⏳ Correlation IDs traceable across services
- ⏳ Debugging is delightful (human-readable stories)

---

## Conclusion

The PM's story was **well-intentioned but incomplete**. It correctly identified that worker-orcd needs narration integration, but it missed the **provider/consumer split**.

As narration-core, we've now:
1. ✅ Provided the taxonomy
2. ✅ Written the integration guide
3. ✅ Created BDD scenarios
4. ✅ Documented the editorial process

The ball is now in **worker-orcd's court** to integrate. We're here to help! 🎀

---

**Status**: Analysis complete, corrective actions taken  
**Owner**: Narration-Core Team  
**Next Action**: Notify worker-orcd Foundation-Alpha team

---

*Analysis prepared by the Narration Core Team — may your planning be thorough and your stories be well-scoped! 🎀*
