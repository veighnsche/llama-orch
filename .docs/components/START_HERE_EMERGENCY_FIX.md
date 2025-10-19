# START HERE: Emergency BDD Test Fix

**Date:** 2025-10-19  
**Priority:** 🚨 **CRITICAL**  
**Goal:** Get from 23% to 90%+ test pass rate

---

## 🎯 Your Mission

**Current State:** 69/300 tests passing (23%) - **UNACCEPTABLE**  
**Your Goal:** 270+/300 tests passing (90%+) - **REQUIRED FOR v0.1.0**

You are part of a **6-team emergency response** to fix the BDD test suite.

---

## 📋 Team Assignments

### TEAM-117: Fix Ambiguous Steps
**Lead Focus:** Remove duplicate step definitions  
**Time:** 4 hours  
**Difficulty:** ⭐⭐ Medium  
**Files:** `test-harness/bdd/src/steps/*.rs`

### TEAM-118: Missing Steps Batch 1
**Lead Focus:** Implement steps 1-18  
**Time:** 4 hours  
**Difficulty:** ⭐⭐⭐ Medium-Hard  
**Files:** `error_handling.rs`, `worker_registration.rs`, `lifecycle.rs`

### TEAM-119: Missing Steps Batch 2
**Lead Focus:** Implement steps 19-36  
**Time:** 4 hours  
**Difficulty:** ⭐⭐⭐ Medium-Hard  
**Files:** `worker_preflight.rs`, `authentication.rs`, `secrets.rs`

### TEAM-120: Missing Steps Batch 3
**Lead Focus:** Implement steps 37-54  
**Time:** 4 hours  
**Difficulty:** ⭐⭐⭐ Medium-Hard  
**Files:** `error_handling.rs`, `audit_logging.rs`, `deadline_propagation.rs`

### TEAM-121: Missing Steps Batch 4 + Timeouts
**Lead Focus:** Implement steps 55-71 + fix timeout handling  
**Time:** 4 hours  
**Difficulty:** ⭐⭐⭐⭐ Hard  
**Files:** `integration_scenarios.rs`, `world.rs`, multiple files

### TEAM-122: Fix Panics + Final Integration
**Lead Focus:** Fix all panics, verify 90%+ pass rate  
**Time:** 4 hours  
**Difficulty:** ⭐⭐⭐⭐⭐ Very Hard  
**Files:** All step files, final verification

---

## 🚀 Getting Started

### Step 1: Read Your Assignment (5 min)

**Your team number:** `TEAM-1XX` (117-122)

**Read your detailed assignment:**
```bash
# Open the master plan
cat .docs/components/EMERGENCY_FIX_MASTER_PLAN.md

# Find your team section (TEAM-117 through TEAM-122)
# Read your specific tasks
```

### Step 2: Create Your Branch (2 min)

```bash
# Create your team branch
git checkout -b fix/team-XXX-description

# Examples:
# git checkout -b fix/team-117-ambiguous-steps
# git checkout -b fix/team-118-missing-batch-1
# git checkout -b fix/team-119-missing-batch-2
# git checkout -b fix/team-120-missing-batch-3
# git checkout -b fix/team-121-missing-batch-4-timeouts
# git checkout -b fix/team-122-panics-final
```

### Step 3: Get Your Task List (5 min)

**Each team has a specific task list in the master plan.**

Find your section and copy your tasks to a checklist:

```bash
# Create your team's working document
cat > .docs/components/TEAM_XXX_TASKS.md << 'EOF'
# TEAM-XXX Tasks

## My Tasks
- [ ] Task 1
- [ ] Task 2
...

## Progress
- Started: [TIME]
- Completed: [TIME]
- Status: IN PROGRESS
EOF
```

### Step 4: Run Current Tests (5 min)

**See the current state:**

```bash
cd /home/vince/Projects/llama-orch

# Run BDD tests
cargo xtask bdd:test

# Check results
cat test-harness/bdd/.test-logs/bdd-results-*.txt
```

**Expected output:** 69/300 passing (23%)

---

## 📚 Resources

### Key Files to Understand

1. **Test Infrastructure:**
   - `test-harness/bdd/src/steps/` - All step definitions
   - `test-harness/bdd/src/steps/world.rs` - Shared test state
   - `test-harness/bdd/tests/features/` - Feature files (scenarios)

2. **Step Definition Pattern:**
```rust
#[given(expr = "worker has {int} slots total")]
pub async fn given_worker_slots(world: &mut World, slots: usize) {
    // Implementation here
    world.worker_slots = slots;
    tracing::info!("✅ Worker configured with {} slots", slots);
}
```

3. **Error Handling Pattern:**
```rust
#[when(expr = "I send request to {string}")]
pub async fn when_send_request(world: &mut World, url: String) -> Result<(), String> {
    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;
    
    world.last_response = Some(response);
    Ok(())
}
```

### Finding Duplicate Steps (TEAM-117)

```bash
# Find all step definitions
grep -r "#\[given\|#\[when\|#\[then" test-harness/bdd/src/steps/ > /tmp/all_steps.txt

# Find duplicates
sort /tmp/all_steps.txt | uniq -d
```

### Finding Unimplemented Steps (TEAM-118-121)

```bash
# Your specific steps are listed in EMERGENCY_FIX_MASTER_PLAN.md
# under your team section

# To verify a step is missing:
cargo test --test cucumber 2>&1 | grep "Step doesn't match"
```

### Finding Panics (TEAM-122)

```bash
# Run tests and capture panics
cargo test --test cucumber 2>&1 | grep -A 5 "panicked"

# Common panic locations:
# - unwrap() on None
# - expect() with bad data
# - index out of bounds
# - failed assertions
```

---

## 🔧 Implementation Guidelines

### DO ✅

1. **Follow existing patterns** - Look at similar steps in the same file
2. **Add proper logging** - Use `tracing::info!("✅ Step completed")`
3. **Handle errors gracefully** - Return `Result<(), String>` when needed
4. **Test your changes** - Run tests after each implementation
5. **Keep it simple** - Real implementation, not perfect implementation
6. **Update world state** - Store results in `world` for later steps
7. **Add comments** - Explain non-obvious logic

### DON'T ❌

1. **Don't add TODO markers** - Implement it properly or skip it
2. **Don't use unwrap()** - Use `?` or `expect()` with context
3. **Don't modify feature files** - Unless absolutely necessary (TEAM-117 only)
4. **Don't add new scenarios** - Fix existing ones only
5. **Don't optimize** - Working > perfect
6. **Don't merge to main** - Push to your branch only
7. **Don't block others** - Work on your assigned files only

---

## 🎯 Success Criteria

### For Each Team

**TEAM-117:**
- ✅ Zero "ambiguous step" errors
- ✅ All tests compile
- ✅ No lost functionality

**TEAM-118, 119, 120, 121:**
- ✅ All assigned steps implemented
- ✅ No TODO markers in your code
- ✅ Tests compile and run
- ✅ Your steps pass when services available

**TEAM-122:**
- ✅ Zero panics in test runs
- ✅ 270+/300 tests passing (90%+)
- ✅ Completion report written
- ✅ All branches merged

### Overall Success

**REQUIRED:** 270+/300 tests passing (90%+)  
**ACCEPTABLE:** 260-269/300 (87-90%)  
**UNACCEPTABLE:** <260/300 (<87%)

---

## 📊 Progress Tracking

### Report Your Progress

**Every 2 hours, update your status:**

```bash
# Update your team document
echo "## Progress Update $(date)" >> .docs/components/TEAM_XXX_TASKS.md
echo "- Completed: X/Y tasks" >> .docs/components/TEAM_XXX_TASKS.md
echo "- Blockers: [NONE or describe]" >> .docs/components/TEAM_XXX_TASKS.md
echo "- ETA: X hours remaining" >> .docs/components/TEAM_XXX_TASKS.md
```

### When You're Done

```bash
# 1. Run tests
cargo xtask bdd:test

# 2. Verify your improvements
cat test-harness/bdd/.test-logs/bdd-results-*.txt

# 3. Commit your work
git add .
git commit -m "TEAM-XXX: [Brief description of what you fixed]"

# 4. Push your branch
git push origin fix/team-XXX-description

# 5. Create completion report
cat > .docs/components/TEAM_XXX_COMPLETE.md << 'EOF'
# TEAM-XXX Completion Report

## Summary
- Tasks assigned: X
- Tasks completed: X
- Time taken: X hours

## Changes Made
- File 1: [description]
- File 2: [description]

## Test Results
- Before: X/300 passing
- After: X/300 passing
- Improvement: +X scenarios

## Blockers Encountered
- [NONE or list]

## Recommendations
- [Any suggestions for next steps]

Status: ✅ COMPLETE
EOF
```

---

## 🆘 Getting Help

### If You're Blocked

1. **Check existing implementations** - Look at similar steps
2. **Read the master plan** - Your task might have hints
3. **Check test output** - Error messages are helpful
4. **Ask TEAM-122** - They're the integration team

### Common Issues

**Issue:** "I don't know how to implement this step"  
**Solution:** Look for similar steps in the same file, copy the pattern

**Issue:** "Tests won't compile"  
**Solution:** Check for missing imports, typos in function names

**Issue:** "Step passes but scenario still fails"  
**Solution:** Check if later steps in the scenario are also unimplemented

**Issue:** "I'm running out of time"  
**Solution:** Implement what you can, document what's left, ask TEAM-122 for help

---

## 🎓 Quick Reference

### Test a Single Scenario

```bash
# Run specific feature
cargo test --test cucumber -- --tags @your-tag

# Run specific scenario (by name)
cargo test --test cucumber -- "Scenario name"
```

### Check Compilation

```bash
# Fast check
cargo check --package test-harness-bdd

# Full build
cargo build --package test-harness-bdd
```

### View Test Output

```bash
# Latest results
cat test-harness/bdd/.test-logs/bdd-results-*.txt | tail -20

# Failures only
cat test-harness/bdd/.test-logs/failures-*.txt

# Full log
less test-harness/bdd/.test-logs/test-output-*.log
```

---

## 📅 Timeline

### Day 1 (Today)
- **Hour 0-4:** TEAM-117, 118, 119 work
- **Hour 4-8:** TEAM-120, 121 work
- **End of Day:** 5 teams complete

### Day 2 (Tomorrow)
- **Hour 0-3:** TEAM-122 fixes panics
- **Hour 3-4:** TEAM-122 final verification
- **End of Day:** 90%+ pass rate achieved ✅

---

## 🎯 Final Checklist

Before you start:
- [ ] I know my team number (117-122)
- [ ] I've read my section in EMERGENCY_FIX_MASTER_PLAN.md
- [ ] I've created my branch
- [ ] I've run the current tests to see the baseline
- [ ] I understand my success criteria

Before you finish:
- [ ] All my assigned tasks are complete
- [ ] Tests compile without errors
- [ ] I've run the test suite
- [ ] I've committed and pushed my branch
- [ ] I've created my completion report
- [ ] I've verified my improvements

---

## 💪 Let's Do This!

**You are part of the solution.**

The test suite is broken. We're fixing it. Together.

**Your work matters.** Every step you implement gets us closer to 90%.

**No excuses. No shortcuts. Just results.**

---

**Status:** 🚀 **READY TO START**  
**Your Team:** TEAM-1XX (find your number above)  
**Your Mission:** Fix your assigned tasks  
**Your Deadline:** 4 hours  
**Your Goal:** Help achieve 90%+ pass rate

**LET'S GO! 🔥**

---

## Quick Start Commands

```bash
# 1. Find your team number (117-122)
echo "I am TEAM-XXX"

# 2. Create your branch
git checkout -b fix/team-XXX-description

# 3. Read your tasks
cat .docs/components/EMERGENCY_FIX_MASTER_PLAN.md | grep -A 50 "TEAM-XXX"

# 4. Start working!
cd test-harness/bdd/src/steps

# 5. When done, run tests
cd /home/vince/Projects/llama-orch
cargo xtask bdd:test
```

**NOW GO FIX THOSE TESTS! 💪**
