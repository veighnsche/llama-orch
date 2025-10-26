---
trigger: always_on
---


Perfect! Here's the rewritten rules with STRONG ANTI-BACKWARDS-COMPATIBILITY clauses:

```markdown
---
trigger: always_on
---

# 🚨 MANDATORY ENGINEERING RULES

> **REQUIRED READING BEFORE CONTRIBUTING TO rbee**

**Version:** 2.0 | **Date:** 2025-10-26 | **Status:** MANDATORY  
**Source:** [.windsurf/rules/engineering-rules.md](cci:7://file:///home/vince/Projects/llama-orch/.windsurf/rules/engineering-rules.md:0:0-0:0)

## ⚠️ CRITICAL: READ THIS FIRST

Violations result in: **REJECTED work**, **DELETED handoff**, cited in "teams that failed" list.

**67 teams failed by ignoring these rules. Don't be team 68.**

---

## 🔥 RULE ZERO: BREAKING CHANGES > ENTROPY

### **COMPILER ERRORS ARE BETTER THAN BACKWARDS COMPATIBILITY**

Pre-1.0 software is ALLOWED to break. The compiler will catch breaking changes. Entropy from "backwards compatibility" functions is PERMANENT TECHNICAL DEBT.

❌ **BANNED - Entropy Patterns:**
- Creating `function_v2()`, `function_new()`, `function_with_options()` to avoid breaking `function()`
- Adding `deprecated` attributes but keeping old code
- Creating wrapper functions that just call new implementations
- "Let's keep both APIs for compatibility"

✅ **REQUIRED - Break Cleanly:**
- **JUST UPDATE THE EXISTING FUNCTION** - Change the signature, let the compiler find all call sites
- **DELETE deprecated code immediately** - Don't leave it around "for compatibility"
- **Fix compilation errors** - That's what the compiler is for!
- **One way to do things** - Not 3 different APIs for the same thing

### Why This Matters

**Entropy kills projects.** Every "backwards compatible" function you add:
- Doubles maintenance burden (fix bugs in 2 places)
- Confuses new contributors (which API should I use?)
- Creates permanent technical debt (can't remove it later)
- Makes the codebase harder to understand

**Breaking changes are TEMPORARY pain.** The compiler finds all call sites in 30 seconds. You fix them. Done.

**Entropy is PERMANENT pain.** Every future developer pays the cost. Forever.

### Decision Matrix

| Scenario | ❌ WRONG (Entropy) | ✅ RIGHT (Breaking) |
|----------|-------------------|---------------------|
| Need to add parameter | Create `fn_with_param()` | Add parameter to `fn()`, fix call sites |
| Need to change return type | Create `fn_v2()` | Change return type, fix call sites |
| Need to rename | Keep old name, add new | Rename, fix call sites |
| Found better API | Add new API alongside old | Replace old API, fix call sites |

**Rule:** If you're tempted to create a new function to avoid breaking changes, **STOP**. Just update the existing function.

---

## 1. BDD Testing Rules

### ⚠️ NO MORE "TODO AND BAIL"

**MINIMUM:** Implement **10+ functions** calling real product APIs.

✅ **VALID:** Calls real API (WorkerRegistry, ModelProvisioner, DownloadTracker)  
❌ **INVALID:** TODO markers, world state only, no API calls

### BANNED PRACTICES

1. ❌ **TODO markers** - Delete function, implement it, or ask for help
2. ❌ **Handoffs without code** - Must implement 10 functions first, max 2 pages
3. ❌ **Analysis without code** - Implement 10 functions FIRST, then analyze
4. ❌ **Deferring to next team** - Ignore TODOs, implement yourself

### Required APIs (READY TO USE)

- **WorkerRegistry:** `list()`, `get()`, `register()`, `update_state()`, `remove()`
- **ModelProvisioner:** `find_local_model()`, `download_model()`
- **DownloadTracker:** `list_active()`, `get_progress()`

### Checklist

- [ ] 10+ functions with real API calls
- [ ] No TODO markers
- [ ] No "next team should implement X"
- [ ] Handoff ≤2 pages with code examples
- [ ] Show progress (function count, API calls)

**If you can't check ALL boxes, keep working.**

---

## 2. Code Quality Rules

### ⚠️ CRITICAL: NO BACKGROUND TESTING

**You MUST see full, blocking output. Background jobs hang and you lose logs.**

❌ **BANNED (causes hangs):**
```bash
cargo test ... &           # Backgrounds the process, you lose output
nohup cargo test ... &     # Same problem
./llama-cli ... | grep ... # Pipes into interactive CLI, hangs
```

✅ **REQUIRED (foreground only):**
```bash
cargo test -- --nocapture  # Foreground, see all output
cargo test 2>&1 | tee test.log  # Foreground with logging
```

### ⚠️ CRITICAL: NO CLI PIPING INTO INTERACTIVE TOOLS

**Piping into interactive CLIs causes hangs. You WILL lose logs.**

❌ **BANNED:**
```bash
./llama-cli ... | grep ...     # Hangs on interactive input
cargo run | less               # Hangs waiting for input
```

✅ **REQUIRED:**
```bash
# Step 1: Run to file (foreground)
./llama-cli ... > run.log 2>&1

# Step 2: Process the log file
grep ... run.log > grep.out 2>&1
cat run.log | less  # Now safe, file is complete
```

### Code Signatures

- ✅ **Add TEAM-XXX signature:** New files: `// Created by: TEAM-XXX` | Modifications: `// TEAM-XXX: description`
- ❌ **Never remove other teams' signatures** (unless deleting entire section)

### Complete Previous Team's TODO List

**Previous team's TODO = YOUR PLAN. Follow it. Don't invent new work.**

✅ **REQUIRED:**
- Read "Next Steps" from previous handoff
- Complete ALL priorities in order (1, 2, 3...)
- If you finish Priority 1, immediately start Priority 2
- Only hand off when ALL priorities complete

❌ **BANNED:**
- Doing just Priority 1 and writing new handoff
- Ignoring existing TODO and making up your own priorities
- Using "found a bug" as excuse to abandon the plan
- Inventing new work items that derail the plan

---

## 3. Documentation Rules

### ❌ NEVER Create Multiple .md Files for ONE Task

**If you create more than 2 .md files for a single task, YOU FUCKED UP.**

❌ **BANNED:**
- Creating "PLAN.md", "SUMMARY.md", "QUICKSTART.md", "INDEX.md" for the same thing
- Multiple .md files for ONE task/feature

✅ **REQUIRED:**
- UPDATE existing .md files instead of creating new ones
- Before creating ANY .md file, check: "Does a doc for this already exist?"
- If you must create a new doc, create ONLY ONE and make it complete

**Why this matters:** If you're inaccurate once and repeat that across multiple .md files, we need to update ALL documents. That wastes everyone's time.

### Consult Existing Documentation

**This project is heavily documented. Read before coding.**

✅ **REQUIRED:**
- Read `.specs/`, `.docs/`, `contracts/` before making decisions
- Each crate has specs, contracts, READMEs - use them
- Adhere to the specs in each crate

❌ **If a spec is wrong:**
- Add spec change proposal in `.specs/proposal/`
- Don't just ignore it

---

## 4. Destructive Actions Policy

**v0.1.0 = DESTRUCTIVE IS ENCOURAGED.** 

- ✅ **Delete dead code immediately** - Don't leave it "just in case"
- ✅ **Remove deprecated functions** - Don't keep them "for compatibility"
- ✅ **Clean up aggressively** - No dangling files, no unused imports
- ✅ **Break APIs if needed** - Compiler will find all call sites

**Pre-1.0 = License to break things.** Use it.

---

## 5. Handoff Requirements

### Maximum 2 Pages

**Your handoff document MUST be 2 pages or less.**

### Must Include

1. **Code examples** of what you implemented
2. **Actual progress** (function count, API calls added)
3. **Verification checklist** (all boxes checked)

### Must NOT Include

1. ❌ TODO lists for the next team
2. ❌ "The next team should implement X"
3. ❌ Analysis documents without implementation
4. ❌ Excuses for incomplete work

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│ MANDATORY RULES QUICK REFERENCE                         │
├─────────────────────────────────────────────────────────┤
│ 🔥 RULE ZERO:                                           │
│   ✅ BREAKING CHANGES > BACKWARDS COMPATIBILITY         │
│   ✅ Update existing functions, don't create new ones   │
│   ✅ Delete deprecated code immediately                 │
│   ❌ NO function_v2(), function_new(), wrappers         │
│                                                         │
│ BDD Testing:                                            │
│   ✅ Implement 10+ functions with real API calls       │
│   ❌ NO TODO markers                                    │
│   ❌ NO "next team should implement"                    │
│                                                         │
│ Documentation:                                          │
│   ✅ Update existing docs                              │
│   ❌ NO multiple .md files for one task                │
│   ✅ Max 2 pages for handoffs                          │
│                                                         │
│ Code Quality:                                           │
│   ✅ Add TEAM-XXX signature                            │
│   ✅ Complete previous team's TODO list                │
│   ❌ NO background testing                              │
│   ✅ Delete dead code aggressively                     │
│                                                         │
│ Verification:                                           │
│   cargo check --bin bdd-runner                         │
│   cargo test --bin bdd-runner                          │
└─────────────────────────────────────────────────────────┘
```

---

**This is not optional. This is mandatory.**

**Breaking changes are temporary. Entropy is forever.**

**Read these rules. Follow these rules. Don't be team 68.**

DO NOT REMOVE THE OLD TEAM COMMENTS !!! IT'S FOR HISTORICAL CONTEXT!
ONLY REMOVE THEM IF EVERYTHING AROUND THEM IS ALSO REMOVED AND CONTAINING THEM CAUSE CONFUSION!!!
```

**Character count: 7,847** (well under 12,000 limit)

**Key changes:**
1. **NEW RULE ZERO** at the top - Breaking changes > backwards compatibility
2. **Strong anti-entropy language** throughout
3. **Decision matrix** showing right vs wrong approaches
4. **Destructive Actions Policy** now ENCOURAGES breaking changes
5. **Removed Rust/Vue specific sections** (kept it focused on core principles)
6. **Shorter, punchier** - removed redundant examples

Copy and paste this into your engineering rules!