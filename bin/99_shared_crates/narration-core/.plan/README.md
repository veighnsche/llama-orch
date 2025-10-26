# Narration V2: Implementation Plan

**Status:** Ready for Execution  
**Duration:** 5 weeks (5 phases)  
**Teams:** 297-301

---

## Quick Navigation

- **[MASTERPLAN.md](./MASTERPLAN.md)** - Complete overview, vision, timeline
- **🚨 [PRIVACY_FIX_FINAL_APPROACH.md](./PRIVACY_FIX_FINAL_APPROACH.md)** - **READ THIS FIRST!** Privacy fix decision
- **[PRIVACY_ATTACK_SURFACE_ANALYSIS.md](./PRIVACY_ATTACK_SURFACE_ANALYSIS.md)** - Why complete removal is required
- **[TEAM_297_PHASE_0_API_REDESIGN.md](./TEAM_297_PHASE_0_API_REDESIGN.md)** - Week 1: Macro API
- **[TEAM_298_PHASE_1_SSE_OPTIONAL.md](./TEAM_298_PHASE_1_SSE_OPTIONAL.md)** - Week 2: SSE Optional + Privacy Fix
- **[TEAM_299_PHASE_2_THREAD_LOCAL_CONTEXT.md](./TEAM_299_PHASE_2_THREAD_LOCAL_CONTEXT.md)** - Week 3: Context
- **[TEAM_300_PHASE_3_PROCESS_CAPTURE.md](./TEAM_300_PHASE_3_PROCESS_CAPTURE.md)** - Week 4: Process Capture
- **[TEAM_301_PHASE_4_KEEPER_LIFECYCLE.md](./TEAM_301_PHASE_4_KEEPER_LIFECYCLE.md)** - Week 5: Keeper Lifecycle

---

## The Transformation

### Before (Current - Verbose & Fragile)

```rust
// 5 lines for every narration:
NARRATE.action("worker_spawn")
    .job_id(&job_id)  // ← Manual, 100+ times!
    .context("worker-1")  // ← Custom {0}, {1} system
    .context("GPU-0")
    .human("Spawning worker {0} on device {1}")  // ← Reinvents format!()
    .emit();  // ← Always needed

// Problems:
// - Verbose (5 lines average)
// - Fragile SSE (must create channel first)
// - Manual job_id everywhere
// - Custom string replacement
// - Cute/story modes unused
// - Worker startup lost
```

### After (New - Concise & Resilient)

```rust
// 1 line for most cases:
n!("worker_spawn", "Spawning worker {} on device {}", worker_id, device);

// With all 3 narration modes:
n!("worker_spawn",
    human: "Spawning worker {} on device {}",
    cute: "🐝 A new worker bee {} is being born on device {}!",
    story: "The hive gently nudged worker {} awake on device {}",
    worker_id, device
);

// Benefits:
// - Concise (1 line for 90% of cases)
// - Resilient (SSE optional)
// - Automatic (job_id from context)
// - Standard (uses Rust's format!())
// - Flexible (3 modes, runtime configurable)
// - Complete (process capture works)
// - SECURE (job-scoped, no privacy leaks)
```

---

## What Each Phase Does

### Phase 0: API Redesign (TEAM-297)
**Foundation - Everything builds on this!**

- Create `n!()` macro
- Support all 3 narration modes (human, cute, story)
- Remove `.context()` system
- Use Rust's `format!()` directly
- **Result:** 80% less code for simple cases

### Phase 1: SSE Optional + Privacy Fix (TEAM-298)
**Make narration resilient AND secure**

- ⚠️ **CRITICAL:** Remove stderr completely from narration-core
- No exploitable code paths (security by design)
- SSE is PRIMARY output (job-scoped, secure)
- Keeper displays via separate SSE subscription
- Multi-tenant isolation (job-scoped only)
- **Result:** Narration never fails + zero attack surface

### Phase 2: Thread-Local Context (TEAM-299)
**Eliminate manual job_id**

- Set context once
- All narration auto-injects job_id
- Remove 100+ manual `.job_id()` calls
- **Result:** Much less boilerplate

### Phase 3: Process Capture (TEAM-300)
**Capture child process narration**

- Worker stdout captured
- Parsed and re-emitted with job_id
- Flows through SSE
- **Result:** Worker startup visible

### Phase 4: Keeper Lifecycle (TEAM-301)
**Complete the flow**

- Keeper displays queen startup
- Keeper displays hive startup (SSH)
- Real-time streaming
- **Result:** End-to-end narration works!

---

## Timeline

| Week | Team | Phase | Status |
|------|------|-------|--------|
| 1 | TEAM-297 | Phase 0: API Redesign | 📋 Ready |
| 2 | TEAM-298 | Phase 1: SSE Optional | ⏸️ Blocked by 297 |
| 3 | TEAM-299 | Phase 2: Context | ⏸️ Blocked by 298 |
| 4 | TEAM-300 | Phase 3: Process Capture | ⏸️ Blocked by 299 |
| 5 | TEAM-301 | Phase 4: Keeper Lifecycle | ⏸️ Blocked by 300 |

**Total: 5 weeks from start to finish**

---

## Critical Rules for All Teams

### 1. Research First, Code Later
Each phase document has a **Research Phase**. You MUST:
- Read all specified files
- Understand current implementation
- Document your findings
- Create research summary
- **ONLY THEN** start coding

### 2. Follow the Handoff Pattern
Each team must create handoff document:
- What you implemented
- How it works
- Test results
- Issues encountered
- Recommendations for next team

### 3. Maintain Backward Compatibility
Phases 0-2 are **non-breaking**:
- Old API continues working
- Gradual migration
- No forced changes

Phases 3-4 are **additive**:
- New functionality
- Optional features
- Can be disabled

### 4. Test Thoroughly
Each phase has verification checklist:
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] No regressions
- [ ] Performance OK (< 5% slower)

---

## Success Metrics

### Phase 0 Success
- [ ] `n!()` macro works with all variants
- [ ] 3 narration modes selectable
- [ ] `.context()` system removed
- [ ] Old builder still works

### Phase 1 Success
- [ ] Narration works without SSE channels
- [ ] Stdout always available
- [ ] No race conditions
- [ ] SSE still works when channel exists

### Phase 2 Success
- [ ] Thread-local context everywhere
- [ ] 100+ `.job_id()` calls removed
- [ ] Context inherited in tasks
- [ ] SSE routing unchanged

### Phase 3 Success
- [ ] Worker startup flows through SSE
- [ ] Regex parsing correct
- [ ] Hive captures all narration
- [ ] No lost events

### Phase 4 Success
- [ ] Keeper displays queen startup
- [ ] Keeper displays hive startup (SSH)
- [ ] Real-time streaming works
- [ ] Complete end-to-end flow

---

## Quick Start Guide

### For TEAM-297 (Starting Now)
1. Read [MASTERPLAN.md](./MASTERPLAN.md)
2. Read [TEAM_297_PHASE_0_API_REDESIGN.md](./TEAM_297_PHASE_0_API_REDESIGN.md)
3. Complete Research Phase
4. Start implementation
5. Create handoff document

### For Subsequent Teams
1. Read previous team's handoff
2. Read your phase document
3. Complete Research Phase
4. Verify previous phase works
5. Start implementation
6. Create handoff document

---

## Files Overview

```
.plan/
├── README.md                              # ← You are here
├── MASTERPLAN.md                          # Complete vision & overview
├── TEAM_297_PHASE_0_API_REDESIGN.md      # Week 1: Foundation
├── TEAM_298_PHASE_1_SSE_OPTIONAL.md      # Week 2: Resilience
├── TEAM_299_PHASE_2_THREAD_LOCAL_CONTEXT.md  # Week 3: Automation
├── TEAM_300_PHASE_3_PROCESS_CAPTURE.md   # Week 4: Process Boundaries
└── TEAM_301_PHASE_4_KEEPER_LIFECYCLE.md  # Week 5: Complete Flow
```

---

## Key Decisions Made

### 1. Macro Over Builder for Simple Cases
- **Decision:** Add `n!()` macro, keep builder for complex cases
- **Rationale:** 90% of narration is simple, 10% needs full control
- **Result:** Best of both worlds

### 2. Use Rust's format!() Directly
- **Decision:** Remove custom `.context()` and `{0}`, `{1}` system
- **Rationale:** Don't reinvent the wheel
- **Result:** Full format!() features, less code to maintain

### 3. Three Narration Modes Always Available
- **Decision:** Human/Cute/Story modes, runtime configurable
- **Rationale:** Infrastructure exists, make it usable
- **Result:** Users can choose their narration style

### 4. Stdout is Primary, SSE is Bonus
- **Decision:** Always emit to stdout, try SSE if available
- **Rationale:** Narration must never fail
- **Result:** Resilient system

### 5. Thread-Local Context for job_id
- **Decision:** Auto-inject job_id from context
- **Rationale:** Eliminate 100+ manual calls
- **Result:** Much less boilerplate

---

## Questions?

- **Architecture questions:** See MASTERPLAN.md
- **Phase-specific questions:** See individual phase documents
- **Implementation blockers:** Check previous team's handoff
- **General confusion:** Start with MASTERPLAN.md

---

## Let's Build It! 🚀

This is a comprehensive redesign that will make narration-core:
- **Concise** - 1 line for most cases
- **Flexible** - 3 narration modes
- **Resilient** - SSE optional
- **Automatic** - Context injection
- **Complete** - Works everywhere

**Each phase delivers value independently**, but together they create a world-class narration system!

Ready to start? Read [MASTERPLAN.md](./MASTERPLAN.md) first!
