---
trigger: always_on
---

# 🚨 MANDATORY ENGINEERING RULES

> **REQUIRED READING BEFORE CONTRIBUTING TO rbee**

**Version:** 1.0 | Date:** 2025-10-11 | **Status:** MANDATORY  
**Source:** Consolidated from `.windsurf/rules/`

## ⚠️ CRITICAL: READ THIS FIRST

Violations result in: **REJECTED work**, **DELETED handoff**, cited in "teams that failed" list.

**67 teams failed by ignoring these rules. Don't be team 68.**

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

**Why this matters:** Interactive tools wait for input. When piped, they hang indefinitely. You lose all output and waste hours debugging.

### Code Signatures

- ✅ **Add TEAM-XXX signature:** New files: `// Created by: TEAM-XXX` | Modifications: `// TEAM-XXX: description`
- ❌ **Never remove other teams' signatures**

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

**Example of FAILURE (TEAM-013):**
- TEAM-012 handoff: "Priority 1: CUDA, Priority 2: Sampling, Priority 3: Production"
- TEAM-013 does: Priority 1 only
- TEAM-013 writes: "Priority 1: GPU Warmup, Priority 2: SSE, Priority 3: Multi-GPU"
- **Result:** Completely derailed! Priorities 2 & 3 ignored, random work invented

**Example of SUCCESS:**
- TEAM-012 handoff: "Priority 1: CUDA, Priority 2: Sampling, Priority 3: Production"
- TEAM-013 does: ALL THREE priorities
- TEAM-013 writes: "✅ All priorities complete"
- **Result:** Work progresses, no derailment

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

**v0.1.0 = DESTRUCTIVE IS ALLOWED.** Clean up aggressively. No dangling files, no dead code.

## 5. Rust Rules

- ✅ **Specs-first:** Read `.specs/`, `.docs/`, `contracts/` first
- ✅ **Document stubs/shims** in *.md files
- ✅ **Check off checklists** as you complete work

---

## 6. Vue/Frontend Rules

- ✅ **Design tokens:** `@import 'orchyra-storybook/styles/tokens.css'`
- ✅ **Components:** `import { Button } from 'orchyra-storybook/stories'`
- ❌ **NO relative imports:** `'../../../../../libs/storybook/stories'`
- ✅ **File naming:** `CamelCase.vue`
- ✅ **Workspace boundaries:** Use package entry points, never `../../`

---

## 7. Handoff Requirements

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

### Good Handoff Example

```markdown
# TEAM-075 SUMMARY

**Mission:** Implement error handling for GPU failures

**Deliverables:**
- ✅ 15 functions implemented with real API calls
- ✅ GPU FAIL FAST policy enforced

**Functions Implemented:**
1. `when_cuda_device_fails()` - Calls gpu_info::detect_gpus()
2. `when_gpu_vram_exhausted()` - Calls nvml::Device::memory_info()
... (13 more)

**Verification:**
- ✅ Compilation: SUCCESS
- ✅ Tests: Pass rate 42.9% → 45% (+2.1%)
```

### Bad Handoff Example

```markdown
# TEAM-XXX ANALYSIS

**What I Did:**
- Analyzed the codebase
- Found 20 TODO functions
- Wrote 12 documentation files

**Next Team Should:**
- Implement the 20 TODO functions
- Fix the bugs I found
```

**This is REJECTED. No actual work was done.**

---

## Consequences

Violations = **REJECTED work**, **DELETED handoff**, cited in "failed teams" list. **67 teams failed. Don't be 68.**

## The Bottom Line

- **BDD:** 10+ functions minimum, NO TODOs, APIs are ready
- **All Work:** Complete previous team's TODO, don't invent priorities, handoffs ≤2 pages
- **Docs:** Update existing, don't create multiple (>2 = fucked up)
- **Code:** Add TEAM-XXX signature, no background testing, clean up dead code

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│ MANDATORY RULES QUICK REFERENCE                         │
├─────────────────────────────────────────────────────────┤
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
│   ✅ Clean up dead code (v0.1.0 = destructive OK)      │
│                                                         │
│ Handoffs:                                               │
│   ✅ Show actual progress (function count)             │
│   ✅ Include code examples                             │
│   ❌ NO TODO lists for next team                        │
│                                                         │
│ Verification:                                           │
│   cargo check --bin bdd-runner                         │
│   cargo test --bin bdd-runner                          │
└─────────────────────────────────────────────────────────┘
```

---

**This is not optional. This is mandatory.**

**Implement the functions. Stop writing TODOs. Make actual progress.**

**Read these rules. Follow these rules. Don't be team 68.**

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-11  
**Source:** Consolidated from `.windsurf/rules/` (bdd-rules.md, destructive-actions.md, dev-bee-rules.md, rust-rules.md, vue-rules.md)  
**Status:** MANDATORY - All engineers must read before contributing
