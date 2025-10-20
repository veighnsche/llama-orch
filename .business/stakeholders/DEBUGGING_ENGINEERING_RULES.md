---
trigger: debugging
status: MANDATORY
---

# 🐛 MANDATORY DEBUGGING ENGINEERING RULES

> **REQUIRED READING WHEN FIXING BUGS**

**Version:** 1.1 | **Date:** 2025-10-20 | **Status:** MANDATORY  
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
// BUG FIX: TEAM-164 | Daemon holds parent's pipes open
// ============================================================
// SUSPICION:
// - E2E test hangs when using Command::output()
// - Direct execution works fine
//
// INVESTIGATION:
// - Tested with timeout - confirmed hang at Command::output()
// - Tested with file redirection - works (exits immediately)
// - Tested with pipes - hangs (never completes)
// - Added TTY detection to timeout-enforcer - didn't fix it
// - Checked daemon-lifecycle spawn code - FOUND IT!
//
// ROOT CAUSE:
// - Daemon spawned with Stdio::inherit() inherits parent's file descriptors
// - When parent runs via Command::output(), stdout/stderr are PIPES
// - Daemon holds pipes open even after parent exits
// - Command::output() waits for ALL pipe readers to close
// - Result: infinite hang
//
// FIX:
// - Use Stdio::null() for daemon stdout/stderr
// - Daemon no longer holds parent's pipes
// - Parent can exit immediately
//
// TESTING:
// - cargo xtask e2e:queen (was hanging, now passes)
// - Direct: target/debug/rbee-keeper queen start (still works)
// - With output capture: works (no more hang)
// ============================================================

let child = Command::new(&self.binary_path)
    .args(&self.args)
    .stdout(Stdio::null()) // TEAM-164: Don't inherit parent's stdout pipe
    .stderr(Stdio::null()) // TEAM-164: Don't inherit parent's stderr pipe
    .spawn()
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

## 6. E2E Testing Rules

### ⚠️ CRITICAL: E2E Tests MUST Show Live Narration

**BANNED:**
```rust
// ❌ WRONG - hides all narration until command completes
let output = Command::new("target/debug/rbee-keeper")
    .args(["queen", "start"])
    .output()?;  // Captures output to buffer, user sees NOTHING

let stdout = String::from_utf8_lossy(&output.stdout);
println!("{}", stdout);  // Only prints AFTER command completes
```

**REQUIRED:**
```rust
// ✅ CORRECT - shows narration in real-time
let mut child = Command::new("target/debug/rbee-keeper")
    .args(["queen", "start"])
    .spawn()?;  // Inherits stdout/stderr, user sees narration LIVE

let status = child.wait()?;
if !status.success() {
    anyhow::bail!("Command failed");
}
```

### Why This Matters

**Narration is the product's debugging output.** When you hide it:
- ❌ Users can't see what's happening during long operations
- ❌ Can't debug when things hang
- ❌ Can't see progress (like timeout countdowns)
- ❌ Defeats the entire purpose of narration

**In production:** If a user runs `rbee infer --prompt "..."` and sees nothing for 30 seconds, they'll think it's broken. They NEED to see:
```
[🧑‍🌾 rbee-keeper / ⚙️ queen-lifecycle]
  ⚠️  Queen is asleep, waking queen
[👑 queen-rbee]
  Loading model...
[👑 queen-rbee]
  Generating tokens...
[streaming output here]
```

### E2E Test Pattern

```rust
pub async fn test_something() -> Result<()> {
    println!("🚀 E2E Test: Something\n");
    
    println!("📝 Running: rbee command\n");
    
    // TEAM-164: Use .spawn() instead of .output() to show live narration
    let mut child = Command::new("target/debug/rbee-keeper")
        .args(["command"])
        .spawn()?;
    
    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("Command failed with exit code: {:?}", status.code());
    }
    
    println!();
    println!("✅ E2E Test PASSED: Something");
    Ok(())
}
```

---

## 7. Common Mistakes

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

### ❌ BANNED: Using .output() in E2E Tests

```rust
let output = Command::new("...").output()?;  // HIDES NARRATION
```

**Use .spawn() + .wait() instead.**

---

## 8. Checklist

Before you hand off after fixing a bug:

- [ ] Bug fix comment at the exact fix location
- [ ] All 4 phases documented (Suspicion, Investigation, Root Cause, Fix)
- [ ] Testing section shows what you verified
- [ ] If building on another team's work, kept their investigation comment
- [ ] No generic "fixed the bug" comments
- [ ] Code compiles and tests pass
- [ ] E2E tests use .spawn() + .wait() (not .output())

---

## 9. Why This Matters

**WITHOUT these rules:**
- Team 165 investigates the same bug Team 164 already solved
- Team 166 re-implements a fix that Team 163 tried and failed
- Everyone wastes time chasing the same ghosts
- E2E tests hide narration, making debugging impossible

**WITH these rules:**
- Future teams see what you investigated
- Future teams see what you ruled out
- Future teams see why you fixed it this way
- Future teams can verify your fix still works
- Users see live narration and know the product is working

---

## 10. Consequences

**Violations:**
- ❌ Bug fix without documentation comment = **REJECTED**
- ❌ Generic "fixed bug" comment = **REJECTED**
- ❌ No testing documentation = **REJECTED**
- ❌ E2E test using .output() instead of .spawn() = **REJECTED**

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
│   Full template with all 5 sections:                   │
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
│                                                         │
│ E2E Tests:                                              │
│   ✅ Use .spawn() + .wait() (shows live narration)     │
│   ❌ NEVER use .output() (hides narration)             │
└─────────────────────────────────────────────────────────┘
```

---

**REMEMBER:**

**Code is the source of truth. Document bugs IN THE CODE.**

**E2E tests MUST show live narration. Use .spawn(), not .output().**

**Future teams will thank you.**

**Don't make Team 165 chase the same bug you already solved.**
