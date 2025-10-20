---
trigger: manual
---

# 🐛 MANDATORY DEBUGGING ENGINEERING RULES

> **REQUIRED READING WHEN FIXING BUGS**

**Version:** 1.0 | **Date:** 2025-10-20 | **Status:** MANDATORY  
**Source:** Team 164

## ⚠️ CRITICAL: READ THIS FIRST

**WHY THESE RULES EXIST:**
Teams keep chasing the same bugs because previous teams don't document their investigation. This wastes HOURS of work.

**RULE:** If you investigate a bug, you MUST document it in the code. No exceptions.

---

## 1. Mandatory Bug Documentation Template

**EVERY bug fix MUST include a comment block at the fix location.**

### Template Format

```rust
// ============================================================
// BUG FIX: TEAM-XXX | <SHORT_DESCRIPTION>
// ============================================================
// SUSPICION:
// - <What you initially thought the bug was>
// - <Why you suspected this location/code>
//
// INVESTIGATION:
// - <What you tested/checked>
// - <What you ruled out>
// - <What you discovered>
//
// ROOT CAUSE:
// - <The actual cause of the bug>
// - <Why it happened>
//
// FIX:
// - <What you changed>
// - <Why this solves the problem>
//
// TESTING:
// - <How you verified the fix>
// - <What tests you ran>
// - <What edge cases you checked>
// ============================================================
```

### Example

```rust
// ============================================================
// BUG FIX: TEAM-164 | Queen-rbee crashes on cascade shutdown
// ============================================================
// SUSPICION:
// - Thought it was a race condition in shutdown signal handling
// - Suspected queen_controller.rs::shutdown() was dropping too early
//
// INVESTIGATION:
// - Added logging to shutdown sequence - confirmed signals received
// - Checked tokio runtime shutdown - not the issue
// - Found queen-lifecycle wasn't being awaited properly
// - Discovered Arc<Mutex<QueenState>> was deadlocking
//
// ROOT CAUSE:
// - cascade_shutdown() held mutex lock while calling async shutdown
// - queen_lifecycle::shutdown() also tried to acquire the same lock
// - Deadlock caused timeout, then panic in cleanup
//
// FIX:
// - Released mutex before calling shutdown (scope block)
// - queen_lifecycle::shutdown() now gets clean state access
// - Added timeout with explicit error message
//
// TESTING:
// - Ran xtask e2e cascade-shutdown 20 times - all pass
// - Tested with --keep-alive flag - clean shutdown
// - Verified no leaked processes with ps aux | grep rbee
// ============================================================

pub async fn cascade_shutdown(&self) -> Result<()> {
    let queen_pid = {
        let state = self.state.lock().await;
        state.queen_pid
    }; // Lock released here - CRITICAL for preventing deadlock
    
    self.queen_lifecycle.shutdown(queen_pid).await
}
```

---

## 2. Phases of Bug Documentation

### Phase 1: SUSPICION (Before You Code)

**REQUIRED:** Add a comment when you START investigating:

```rust
// TEAM-XXX: INVESTIGATING - <description>
// SUSPICION: <what you think is wrong>
```

**Why:** If you get stuck and hand off, next team knows where you were looking.

### Phase 2: INVESTIGATION (During Debugging)

**REQUIRED:** Update the comment as you learn:

```rust
// TEAM-XXX: INVESTIGATING - <description>
// SUSPICION: <what you think is wrong>
// INVESTIGATION:
// - Tested X, found Y
// - Ruled out Z
```

**Why:** Your investigation notes help the next team avoid duplicating work.

### Phase 3: FIX IMPLEMENTATION

**REQUIRED:** Document the fix with full template:

```rust
// ============================================================
// BUG FIX: TEAM-XXX | <description>
// ============================================================
// [Full template as shown above]
```

**Why:** Future teams need to understand WHY you fixed it this way.

### Phase 4: TESTING

**REQUIRED:** Document what tests you ran:

```rust
// TESTING:
// - cargo test --bin rbee-keeper -- --nocapture
// - Manual test: ./scripts/test_shutdown.sh
// - Edge case: multiple rapid restarts
// - Result: All pass, no crashes after 50 runs
```

**Why:** If the bug comes back, we know what tests to re-run.

---

## 3. Where to Put the Comment

### ✅ CORRECT: At the Fix Location

```rust
pub fn problematic_function() {
    // ============================================================
    // BUG FIX: TEAM-164 | Fixed race condition in state update
    // [Full documentation here]
    // ============================================================
    
    let state = self.state.lock().await; // THE ACTUAL FIX
}
```

### ❌ WRONG: Only in Commit Message

Commit messages get lost. **Code is the source of truth.**

### ❌ WRONG: Only in Handoff Document

Handoffs get archived. **Code is the source of truth.**

### ❌ WRONG: Only in .docs/

Docs drift. **Code is the source of truth.**

---

## 4. Bug Investigation Without Fix

**SCENARIO:** You investigate a bug but can't fix it before handoff.

**REQUIRED:** Leave an investigation comment:

```rust
// ============================================================
// TEAM-XXX: INVESTIGATION (NO FIX YET) | <description>
// ============================================================
// SUSPICION:
// - <what you thought>
//
// INVESTIGATION:
// - <what you tested>
// - <what you ruled out>
// - <what you're stuck on>
//
// NEXT STEPS:
// - <what the next team should try>
// - <resources/files to check>
// ============================================================
```

**Why:** Next team picks up where you left off instead of starting from zero.

---

## 5. Multiple Teams on Same Bug

**SCENARIO:** You find a TEAM-XXX investigation comment and solve the bug.

**REQUIRED:** Keep the original investigation and add your fix:

```rust
// ============================================================
// TEAM-162: INVESTIGATION (NO FIX YET) | Shutdown crashes
// ============================================================
// SUSPICION:
// - Race condition in signal handling
// INVESTIGATION:
// - Checked shutdown sequence - signals OK
// - Mutex deadlock suspected
// NEXT STEPS:
// - Check queen_lifecycle lock usage
// ============================================================

// ============================================================
// BUG FIX: TEAM-164 | Shutdown crashes - SOLVED
// ============================================================
// BUILDING ON: TEAM-162's investigation above
// ROOT CAUSE:
// - TEAM-162 was right - mutex deadlock
// - Found exact location: cascade_shutdown holding lock too long
// FIX:
// - Released mutex before async call (see code below)
// TESTING:
// - 50 shutdown tests all pass
// ============================================================

pub async fn cascade_shutdown(&self) -> Result<()> {
    // Fix here
}
```

**Why:** Shows the debugging journey and gives credit to investigation work.

---

## 6. Common Mistakes

### ❌ BANNED: Generic Comments

```rust
// TEAM-XXX: Fixed the bug
// TEAM-XXX: This was broken
```

**These are USELESS.** What bug? What was broken? How did you fix it?

### ❌ BANNED: No Testing Documentation

```rust
// BUG FIX: TEAM-XXX | Fixed race condition
// FIX: Added mutex
```

**Missing:** How did you verify it? What tests did you run?

### ❌ BANNED: Removing Investigation Comments

```rust
// TEAM-162: INVESTIGATING...
```

**DON'T DELETE THIS.** Update it or add your fix above it.

---

## 7. Checklist

Before you hand off after fixing a bug:

- [ ] Bug fix comment at the exact fix location
- [ ] All 4 phases documented (Suspicion, Investigation, Root Cause, Fix)
- [ ] Testing section shows what you verified
- [ ] If building on another team's work, kept their investigation comment
- [ ] No generic "fixed the bug" comments
- [ ] Code compiles and tests pass

---

## 8. Why This Matters

**WITHOUT these rules:**
- Team 165 investigates the same bug Team 164 already solved
- Team 166 re-implements a fix that Team 163 tried and failed
- Everyone wastes time chasing the same ghosts

**WITH these rules:**
- Future teams see what you investigated
- Future teams see what you ruled out
- Future teams see why you fixed it this way
- Future teams can verify your fix still works

---

## 9. Consequences

**Violations:**
- ❌ Bug fix without documentation comment = **REJECTED**
- ❌ Generic "fixed bug" comment = **REJECTED**
- ❌ No testing documentation = **REJECTED**

**This is mandatory. This is not optional.**

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│ BUG FIX DOCUMENTATION QUICK REFERENCE                   │
├─────────────────────────────────────────────────────────┤
│ Phase 1: SUSPICION                                      │
│   // TEAM-XXX: INVESTIGATING - <description>           │
│   // SUSPICION: <what you think is wrong>              │
│                                                         │
│ Phase 2: INVESTIGATION                                  │
│   Add: // INVESTIGATION: <what you tested>             │
│                                                         │
│ Phase 3: FIX IMPLEMENTATION                             │
│   Full template with all 4 sections:                   │
│   - SUSPICION                                           │
│   - INVESTIGATION                                       │
│   - ROOT CAUSE                                          │
│   - FIX                                                 │
│   - TESTING                                             │
│                                                         │
│ Location:                                               │
│   ✅ At the exact fix location in code                 │
│   ❌ NOT only in commit/handoff/docs                    │
│                                                         │
│ Multiple Teams:                                         │
│   ✅ Keep original investigation comments              │
│   ✅ Add your fix below with reference                 │
│   ❌ NEVER delete another team's investigation         │
└─────────────────────────────────────────────────────────┘
```

---

**REMEMBER:**

**Code is the source of truth. Document bugs IN THE CODE.**

**Future teams will thank you.**

**Don't make Team 165 chase the same bug you already solved.**