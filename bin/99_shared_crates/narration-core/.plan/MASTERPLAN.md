# Narration V2: Master Implementation Plan

**Version:** 2.0.0  
**Date:** 2025-10-26  
**Status:** READY FOR EXECUTION  
**Total Duration:** 5 weeks (5 phases)

---

## Vision

Redesign narration-core to be:
1. **Concise** - 1 line for 90% of cases (macro-based)
2. **Flexible** - 3 narration modes (human, cute, story)
3. **Resilient** - SSE optional, stdout always works
4. **Automatic** - Thread-local context, no manual job_id
5. **Complete** - Works across process boundaries

---

## Current Problems

### Problem 1: Verbose Builder (328 occurrences!)
```rust
// Takes 4-5 lines for every narration:
NARRATE.action("startup")
    .job_id(&job_id)
    .context(&port)
    .human("Starting on port {}")
    .emit();
```

### Problem 2: Reinventing format!()
```rust
// Custom {0}, {1} replacement instead of Rust's format!():
.context("val1")
.context("val2")
.human("Message {0} and {1}")  // ← Why not just format!()?
```

### Problem 3: Unused Narration Modes
```rust
pub cute: Option<String>,   // ❌ 0 uses
pub story: Option<String>,  // ❌ 0 uses
// Infrastructure exists but unusable!
```

### Problem 4: Fragile SSE
```rust
// Must create channel BEFORE narration or it's lost!
create_job_channel(job_id.clone(), 1000);  // ← Forget this = broken
NARRATE.action("start").job_id(&job_id).emit();
```

### Problem 5: Manual job_id Everywhere
```rust
// Must add .job_id() to every narration (100+ places)
NARRATE.action("step1").job_id(&job_id).emit();
NARRATE.action("step2").job_id(&job_id).emit();
```

### Problem 6: Worker Startup Lost
```rust
// Worker narration during startup goes to void
fn main() {
    NARRATE.action("startup").emit();  // → Lost! (no job context)
}
```

---

## Solution: 5-Phase Redesign

### Phase 0: API Redesign (TEAM-297)
**Goal:** Ultra-concise macro + 3 narration modes  
**Duration:** 1 week  
**Impact:** Foundation for everything else

**Before:**
```rust
NARRATE.action("deploy")
    .context(&name)
    .human("Deploying {}")
    .emit();
```

**After:**
```rust
n!("deploy", "Deploying {}", name);

// Or with all 3 modes:
n!("deploy",
    human: "Deploying {}",
    cute: "🚀 Launching {}!",
    story: "The system whispered: 'Fly, {}'",
    name
);
```

**Deliverables:**
- `n!()` macro with format!() support
- `NarrationMode` enum (Human/Cute/Story)
- Remove `.context()` system
- Runtime mode configuration

### Phase 1: SSE Optional + Privacy Fix (TEAM-298)
**Goal:** Make SSE delivery opportunistic AND fix privacy violation  
**Duration:** 1 week  
**Depends:** Phase 0

**⚠️ CRITICAL ISSUE DISCOVERED**

**PRIVACY VIOLATION:** Current implementation emits ALL narration to global `stderr`, causing:
- Multi-tenant data leaks (User A sees User B's narration)
- Sensitive data exposure (job_id, inference data visible globally)
- Security violation (no isolation between jobs)

**MUST FIX IN PHASE 1:** Change stdout/stderr to be job-scoped only.

**Before:**
```rust
create_job_channel(job_id, 1000);  // ← REQUIRED
NARRATE.action("start").job_id(&job_id).emit();
```

**After:**
```rust
// Works even without channel!
n!("start", "Starting");  // → stdout always works, SSE if available
```

**⚠️ CRITICAL PRIVACY FIX (TEAM-298):**
1. **Remove global stderr** - Privacy violation!
2. **Add keeper mode** - `RBEE_KEEPER_MODE` env var
3. **SSE becomes primary** - Job-scoped, secure
4. **Conditional stderr** - Only in keeper mode (single-user)

**Deliverables:**
- `try_send()` for SSE (failure OK)
- Remove global stderr output (security)
- Add keeper mode flag
- Multi-tenant isolation tests
- SSE as primary output (not bonus)

### Phase 2: Thread-Local Context (TEAM-299)
**Goal:** Auto-inject job_id everywhere  
**Duration:** 1 week  
**Depends:** Phase 1

**Before:**
```rust
NARRATE.action("step1").job_id(&job_id).emit();
NARRATE.action("step2").job_id(&job_id).emit();
NARRATE.action("step3").job_id(&job_id).emit();
```

**After:**
```rust
with_narration_context(ctx.with_job_id(job_id), async {
    n!("step1", "Step 1");  // ← job_id auto-injected!
    n!("step2", "Step 2");
    n!("step3", "Step 3");
}).await
```

**Deliverables:**
- Thread-local context everywhere
- Remove 100+ manual `.job_id()` calls
- Context inheritance in tasks

### Phase 3: Process Capture (TEAM-300)
**Goal:** Capture child stdout → SSE  
**Duration:** 1 week  
**Depends:** Phase 2

**Before:**
```rust
// Worker startup narration lost
let child = Command::new("worker").spawn()?;
// ↑ Worker's stdout goes nowhere!
```

**After:**
```rust
// Hive captures worker stdout
let capture = ProcessNarrationCapture::new(Some(job_id));
let child = capture.spawn(Command::new("worker")).await?;
// → Worker's stdout parsed and emitted via SSE!
```

**Deliverables:**
- `ProcessNarrationCapture` helper
- Regex parsing for narration events
- Hive captures worker startup
- Worker narration flows through SSE

### Phase 4: Keeper Lifecycle (TEAM-301)
**Goal:** Keeper displays daemon startup  
**Duration:** 1 week  
**Depends:** Phase 3

**Before:**
```rust
// Queen startup invisible to user
let child = Command::new("queen-rbee").spawn()?;
```

**After:**
```rust
// Keeper captures and displays
let child = spawn_with_capture("queen-rbee").await?;
// → User sees queen startup in real-time!
```

**Deliverables:**
- Keeper captures queen stdout
- Keeper captures hive stdout (SSH)
- Real-time display to terminal
- Complete narration flow

---

## Dependencies Graph

```
Phase 0: API Redesign (TEAM-297)
    ↓
Phase 1: SSE Optional (TEAM-298)
    ↓
Phase 2: Thread-Local Context (TEAM-299)
    ↓
Phase 3: Process Capture (TEAM-300)
    ↓
Phase 4: Keeper Lifecycle (TEAM-301)
```

**Each phase builds on the previous!**

---

## Success Metrics

### Phase 0 Success
- [ ] `n!()` macro works with format!()
- [ ] All 3 narration modes selectable
- [ ] `.context()` system removed
- [ ] 80% less code for simple cases

### Phase 1 Success
- [ ] Narration works without SSE channels
- [ ] Stdout always available
- [ ] No more race conditions
- [ ] Backward compatible

### Phase 2 Success
- [ ] Thread-local context everywhere
- [ ] 100+ `.job_id()` calls removed
- [ ] No job_id parameters in functions
- [ ] Context inherited in tasks

### Phase 3 Success
- [ ] Worker startup flows through SSE
- [ ] Regex parsing works correctly
- [ ] Hive captures all worker narration
- [ ] No lost narration events

### Phase 4 Success
- [ ] Keeper displays queen startup
- [ ] Keeper displays hive startup (SSH)
- [ ] Real-time streaming works
- [ ] Complete end-to-end flow

---

## Risk Assessment

### Phase 0: Low Risk
- New API parallel to old
- Non-breaking changes
- Can be adopted gradually

### Phase 1: Low Risk
- Makes existing system more resilient
- Backward compatible
- No breaking changes

### Phase 2: Medium Risk
- Touches 100+ files
- Must update many call sites
- Requires careful testing

### Phase 3: High Risk
- New functionality
- Process management complexity
- Regex parsing edge cases

### Phase 4: Medium Risk
- SSH complexity
- Multiple spawn paths
- Integration testing needed

---

## Timeline

| Week | Phase | Team | Deliverable |
|------|-------|------|-------------|
| 1 | Phase 0 | TEAM-297 | API Redesign |
| 2 | Phase 1 | TEAM-298 | SSE Optional |
| 3 | Phase 2 | TEAM-299 | Thread-Local Context |
| 4 | Phase 3 | TEAM-300 | Process Capture |
| 5 | Phase 4 | TEAM-301 | Keeper Lifecycle |

**Total: 5 weeks**

---

## Code Impact

### Lines of Code

| Phase | LOC Added | LOC Removed | Net |
|-------|-----------|-------------|-----|
| Phase 0 | +200 | -300 | -100 |
| Phase 1 | +100 | -50 | +50 |
| Phase 2 | +50 | -150 | -100 |
| Phase 3 | +150 | 0 | +150 |
| Phase 4 | +100 | 0 | +100 |
| **Total** | **+600** | **-500** | **+100** |

**Net result:** Slightly more code, but MUCH better organized and maintainable.

### Files Affected

| Phase | Files Modified | Files Added | Files Removed |
|-------|----------------|-------------|---------------|
| Phase 0 | 10 | 2 | 0 |
| Phase 1 | 5 | 1 | 0 |
| Phase 2 | 46 | 0 | 0 |
| Phase 3 | 3 | 1 | 0 |
| Phase 4 | 2 | 0 | 0 |
| **Total** | **66** | **4** | **0** |

---

## Migration Strategy

### Week 1: Phase 0 (Foundation)
- Add `n!()` macro
- Keep old API working
- Migrate 10 test cases
- Verify both APIs coexist

### Week 2: Phase 1 (Resilience)
- Make SSE optional
- Verify no regressions
- Test without channels
- Document new behavior

### Week 3: Phase 2 (Context)
- Wrap job routers
- Migrate 50% of `.job_id()` calls
- Test thoroughly
- Complete migration

### Week 4: Phase 3 (Process)
- Implement capture
- Update hive spawning
- Test worker startup
- Verify SSE flow

### Week 5: Phase 4 (Keeper)
- Update keeper handlers
- Test SSH capture
- End-to-end testing
- Documentation

---

## Rollback Plan

Each phase is backward compatible:

### Phase 0: Parallel Systems
- Old builder still works
- New macro optional
- Can revert easily

### Phase 1: Additive Only
- New try_send() added
- Old send() still works
- No breaking changes

### Phase 2: Gradual Migration
- Migrate incrementally
- Can stop at any point
- Old pattern still works

### Phase 3: Optional Feature
- Process capture optional
- Old spawning still works
- Can disable easily

### Phase 4: Standalone
- Keeper changes isolated
- Doesn't affect daemons
- Easy to revert

---

## Testing Strategy

### Phase 0 Tests
- Macro with format!() works
- All 3 modes selectable
- Context fallback works
- Performance impact < 5%

### Phase 1 Tests
- Narration without channels
- SSE still works when available
- Stdout always works
- No memory leaks

### Phase 2 Tests
- Context auto-injection
- Task inheritance
- Spawned tasks work
- No job_id leaks

### Phase 3 Tests
- Regex parsing correct
- Worker stdout captured
- SSE delivery works
- Non-narration handled

### Phase 4 Tests
- Queen stdout displayed
- SSH capture works
- Real-time streaming
- No blocking issues

---

## Documentation Deliverables

### Per-Phase Docs
- Research summary (before coding)
- Implementation guide
- Test results
- Handoff document

### Final Docs
- Migration guide
- API reference
- Configuration guide
- Troubleshooting guide

---

## Success Criteria (All Phases)

### Technical
- [ ] All tests pass
- [ ] No performance regression
- [ ] No memory leaks
- [ ] Backward compatible

### User Experience
- [ ] 80% less boilerplate
- [ ] Narration always works
- [ ] 3 modes available
- [ ] Real-time feedback

### Code Quality
- [ ] Simpler API
- [ ] Better organized
- [ ] Well documented
- [ ] Easy to maintain

---

## Next Steps

1. **Review this plan** - Confirm vision and approach
2. **Start Phase 0** - API redesign is foundation
3. **Daily standups** - Each team reports progress
4. **Weekly demos** - Show working features
5. **Final integration** - All phases working together

---

## Contact

**Questions:** Ask in the handoff documents  
**Issues:** Report in research summaries  
**Blockers:** Escalate immediately  

---

## Conclusion

This 5-phase plan will transform narration-core from:
- ❌ Verbose, fragile, limited
- ✅ Concise, resilient, flexible

**Each phase delivers value independently**, but together they create a world-class narration system!

Let's build it! 🚀
