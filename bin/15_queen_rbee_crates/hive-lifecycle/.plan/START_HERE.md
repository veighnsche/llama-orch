# START HERE: Hive Lifecycle Migration

**Welcome to the Hive Lifecycle Migration Project!**

This document is your entry point. Read this first before starting any work.

---

## Quick Overview

**Goal:** Migrate all hive lifecycle logic from `job_router.rs` (1,115 LOC) to the dedicated `hive-lifecycle` crate.

**Why:** `job_router.rs` is too big and mixes routing logic with hive operations. We need clean separation.

**Teams:** 7 teams (TEAM-209 through TEAM-215)

**Duration:** ~2-3 weeks (depending on team velocity)

---

## Team Assignments

| Team | Phase | Task | LOC | Status |
|------|-------|------|-----|--------|
| TEAM-210 | Phase 1 | Foundation | ~150 | 🔴 NOT STARTED |
| TEAM-211 | Phase 2 | Simple Operations | ~100 | 🔴 NOT STARTED |
| TEAM-212 | Phase 3 | Lifecycle Core | ~350 | 🔴 NOT STARTED |
| TEAM-213 | Phase 4 | Install/Uninstall | ~220 | 🔴 NOT STARTED |
| TEAM-214 | Phase 5 | Capabilities | ~100 | 🔴 NOT STARTED |
| TEAM-215 | Phase 6 | Integration | ~50 | 🔴 NOT STARTED |
| TEAM-209 | Phase 7 | Peer Review | N/A | 🔴 NOT STARTED |

---

## Workflow

### Step 1: Find Your Team Number

Look at the table above. Find your team number (TEAM-XXX).

### Step 2: Read Your Phase Document

Open the corresponding phase document:
- TEAM-210 → `01_PHASE_1_FOUNDATION.md`
- TEAM-211 → `02_PHASE_2_SIMPLE_OPERATIONS.md`
- TEAM-212 → `03_PHASE_3_LIFECYCLE_CORE.md`
- TEAM-213 → `04_PHASE_4_INSTALL_UNINSTALL.md`
- TEAM-214 → `05_PHASE_5_CAPABILITIES.md`
- TEAM-215 → `06_PHASE_6_INTEGRATION.md`
- TEAM-209 → `07_PHASE_7_PEER_REVIEW.md`

### Step 3: Read Engineering Rules

**MANDATORY:** Read `/home/vince/Projects/llama-orch/.windsurf/rules/engineering-rules.md`

Key rules:
- ✅ Add TEAM-XXX signatures to all code
- ❌ NO TODO markers (implement or ask for help)
- ❌ NO multiple .md files for one task
- ✅ Max 2 pages per handoff
- ✅ Show actual progress (LOC migrated)

### Step 4: Read Master Plan

Read `00_MASTER_PLAN.md` for overall context and strategy.

### Step 5: Check Dependencies

Before starting, check if your dependencies are complete:

**TEAM-210:** No dependencies (start immediately)

**TEAM-211:** Depends on TEAM-210 (Foundation)
- Wait for TEAM-210 to complete
- Check that types.rs and validation.rs exist

**TEAM-212:** Depends on TEAM-210 (Foundation)
- Wait for TEAM-210 to complete
- Can work in parallel with TEAM-211, 213, 214

**TEAM-213:** Depends on TEAM-210 (Foundation)
- Wait for TEAM-210 to complete
- Can work in parallel with TEAM-211, 212, 214

**TEAM-214:** Depends on TEAM-210 (Foundation)
- Wait for TEAM-210 to complete
- Can work in parallel with TEAM-211, 212, 213

**TEAM-215:** Depends on ALL implementation teams (210-214)
- Wait for TEAM-210, 211, 212, 213, 214 to complete
- This is the integration phase

**TEAM-209:** Depends on TEAM-215 (Integration)
- Wait for TEAM-215 to complete
- This is the final peer review

### Step 6: Implement Your Phase

Follow the instructions in your phase document:
1. Read the source code reference
2. Implement the deliverables
3. Test your code
4. Check acceptance criteria
5. Add TEAM-XXX signatures

### Step 7: Verify Your Work

Before handoff, verify:
- [ ] Code compiles: `cargo check -p queen-rbee-hive-lifecycle`
- [ ] No TODO markers in your code
- [ ] All code has TEAM-XXX signatures
- [ ] All acceptance criteria checked
- [ ] Handoff document ≤2 pages (if needed)

### Step 8: Update Status

Update the status table in this document:
- 🔴 NOT STARTED
- 🟡 IN PROGRESS
- 🟢 COMPLETE

### Step 9: Handoff (if needed)

If your phase has a handoff section, create it:
- Max 2 pages
- Show actual progress (LOC migrated, functions implemented)
- Include code examples
- NO TODO lists for next team

---

## Critical Requirements

### SSE Routing (CRITICAL!)

**ALL narration MUST include `.job_id(job_id)` for SSE routing.**

❌ **WRONG:**
```rust
NARRATE.action("hive_start").human("Starting hive").emit();
```

✅ **CORRECT:**
```rust
NARRATE.action("hive_start").job_id(&job_id).human("Starting hive").emit();
```

**Why:** Without job_id, events are dropped by SSE sink. Users won't see progress.

See MEMORY about SSE routing for details.

### Error Messages

**Preserve exact error messages from original code.**

Users rely on these messages. Don't change them unless absolutely necessary.

### Code Signatures

**Add TEAM-XXX signatures to all new/modified code.**

```rust
// TEAM-210: Created foundation structure
// TEAM-211: Implemented list operation
```

---

## File Locations

### Source Files
- **Original:** `/home/vince/Projects/llama-orch/bin/10_queen_rbee/src/job_router.rs`
- **Target Crate:** `/home/vince/Projects/llama-orch/bin/15_queen_rbee_crates/hive-lifecycle/`

### Reference Files
- **Inspiration:** `/home/vince/Projects/llama-orch/bin/00_rbee_keeper/src/queen_lifecycle.rs`
- **Shared Crate:** `/home/vince/Projects/llama-orch/bin/99_shared_crates/daemon-lifecycle/`
- **Engineering Rules:** `/home/vince/Projects/llama-orch/.windsurf/rules/engineering-rules.md`

### Plan Files
- **Master Plan:** `.plan/00_MASTER_PLAN.md`
- **Phase 1:** `.plan/01_PHASE_1_FOUNDATION.md`
- **Phase 2:** `.plan/02_PHASE_2_SIMPLE_OPERATIONS.md`
- **Phase 3:** `.plan/03_PHASE_3_LIFECYCLE_CORE.md`
- **Phase 4:** `.plan/04_PHASE_4_INSTALL_UNINSTALL.md`
- **Phase 5:** `.plan/05_PHASE_5_CAPABILITIES.md`
- **Phase 6:** `.plan/06_PHASE_6_INTEGRATION.md`
- **Phase 7:** `.plan/07_PHASE_7_PEER_REVIEW.md`

---

## Testing Commands

### Build
```bash
# Check compilation
cargo check -p queen-rbee-hive-lifecycle

# Build everything
cargo build --bin rbee-keeper --bin queen-rbee --bin rbee-hive
```

### Test Operations
```bash
# Test hive operations
./rbee hive list
./rbee hive install
./rbee hive start
./rbee hive status
./rbee hive refresh
./rbee hive stop
```

### Verify LOC Reduction
```bash
# Check job_router.rs size
wc -l bin/10_queen_rbee/src/job_router.rs

# Should be ~350 LOC after migration (was 1,115)
```

---

## Success Metrics

### Before Migration
- `job_router.rs`: 1,115 LOC
- Hive logic mixed with routing logic
- Hard to test hive operations in isolation

### After Migration
- `job_router.rs`: ~350 LOC (routing only)
- `hive-lifecycle`: ~900 LOC (all hive operations)
- Clean separation of concerns
- Testable hive operations

**Target LOC Reduction:** ~65% (1,115 → 350)

---

## Common Issues

### Issue 1: Missing job_id in Narration
**Symptom:** Events don't appear in SSE stream
**Fix:** Add `.job_id(&job_id)` to all narration

### Issue 2: Changed Error Messages
**Symptom:** Users confused by different errors
**Fix:** Copy exact error messages from original

### Issue 3: Import Errors
**Symptom:** Compilation fails with "cannot find X"
**Fix:** Check Cargo.toml dependencies and use statements

### Issue 4: Async/Await Issues
**Symptom:** "cannot await non-future" errors
**Fix:** Ensure functions are marked `async` and use `.await`

---

## Getting Help

### Questions About Architecture
- Read `00_MASTER_PLAN.md`
- Check inspiration files (queen_lifecycle.rs, daemon-lifecycle)

### Questions About Implementation
- Read your phase document
- Check source code reference (job_router.rs lines)
- Look at similar operations in other phases

### Questions About Testing
- Read phase document's "Testing" section
- Run commands from "Testing Commands" section above

### Questions About Rules
- Read `engineering-rules.md`
- Check "Critical Requirements" section above

---

## Phase Dependencies Graph

```
TEAM-210 (Foundation)
    ↓
    ├─→ TEAM-211 (Simple Ops) ─┐
    ├─→ TEAM-212 (Lifecycle)   ├─→ TEAM-215 (Integration)
    ├─→ TEAM-213 (Install)     │        ↓
    └─→ TEAM-214 (Capabilities)┘   TEAM-209 (Peer Review)
```

**Parallel Work:**
- TEAM-211, 212, 213, 214 can work in parallel after TEAM-210 completes
- TEAM-215 must wait for all implementation teams
- TEAM-209 must wait for TEAM-215

---

## Final Checklist

Before considering the migration complete:
- [ ] All 7 teams completed their phases
- [ ] job_router.rs reduced to ~350 LOC
- [ ] All operations work identically
- [ ] SSE routing verified
- [ ] Error messages preserved
- [ ] TEAM-209 peer review APPROVED
- [ ] No regressions found

---

## Next Steps

1. **TEAM-210:** Start immediately with Phase 1 (Foundation)
2. **Other Teams:** Wait for dependencies, then start your phase
3. **TEAM-209:** Wait for TEAM-215, then perform peer review

---

**Good luck! Remember: Read the engineering rules, follow the plan, and communicate clearly.**

**Questions? Check the plan documents. Still stuck? Ask for help.**

---

**Created by:** TEAM-208  
**Date:** 2025-10-22  
**Status:** READY TO START
