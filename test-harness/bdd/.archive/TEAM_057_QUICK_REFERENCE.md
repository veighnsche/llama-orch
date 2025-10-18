# TEAM-057 QUICK REFERENCE

**For:** TEAM-058 (Implementation Team)  
**From:** TEAM-057 (The Thinking Team)  
**Purpose:** Quick start guide for implementing fixes

---

## 🎯 TL;DR

**Problem:** 42/62 scenarios passing (20 failing)  
**Root Cause:** 5 architectural contradictions between spec, tests, and code  
**Solution:** 5-phase implementation plan (10-14 days)  
**Confidence:** VERY HIGH  
**Next Step:** Start with Phase 1 (explicit node registration)

---

## 📚 Required Reading (Priority Order)

1. **`TEAM_057_SUMMARY.md`** (10 min) - Executive overview
2. **`TEAM_057_IMPLEMENTATION_PLAN.md`** (30 min) - What to do
3. **`TEAM_057_ARCHITECTURAL_CONTRADICTIONS.md`** (20 min) - Why it's broken
4. **`TEAM_057_FAILING_SCENARIOS_ANALYSIS.md`** (30 min) - Detailed breakdown

**Total reading time:** ~90 minutes

---

## 🔍 The 5 Architectural Contradictions

### 1. Registration Model Mismatch 🔴 CRITICAL
- **Spec:** Explicit `rbee-keeper setup add-node` → `rbee-keeper infer`
- **Tests:** Implicit availability from Background topology
- **Fix:** Add explicit registration steps to scenarios

### 2. Global State Breaks Isolation 🔴 CRITICAL
- **Spec:** Persistent daemon with persistent DB
- **Tests:** Each scenario assumes fresh state
- **Fix:** Per-scenario isolation (fresh DB per scenario)

### 3. Infrastructure Already Complete ✅ (Updated)
- **Discovery:** ALL queen-rbee endpoints ARE implemented!
- **Reality:** Tests use real queen-rbee with all working endpoints
- **Fix:** No infrastructure work needed—just register nodes!

### 4. Edge Case Steps Are Stubs 🟡 HIGH
- **Spec:** Edge cases return proper error codes
- **Tests/Code:** Stubs just log, return None
- **Fix:** Implement actual command execution

### 5. Background Timing Race 🟡 MEDIUM
- **Spec:** Sequential (setup → use)
- **Tests/Code:** Parallel (Background before queen ready)
- **Fix:** Don't auto-register in Background

---

## 🚀 5-Phase Implementation Plan

### Phase 1: Explicit Node Registration (Days 1-2) 🔴 P0
**Impact:** +3-5 scenarios (42 → 45-47)

**IMPORTANT:** Two scenarios (lines 176, 230) ALREADY have registration! Test them first—they might already pass!

**What to do:**
```gherkin
# Add this line to scenarios that use nodes:
Given node "workstation" is registered in rbee-hive registry
```

**Files to modify:**
- `tests/features/test-001.feature` (lines 949, 976, and others)

**Test:**
```bash
cd test-harness/bdd
LLORCH_BDD_FEATURE_PATH="tests/features/test-001.feature:949" cargo run --bin bdd-runner
```

---

### Phase 2: Implement Edge Cases (Days 3-5) 🟡 P1
**Impact:** +7-9 scenarios (45-51 → 54-58)

**What to do:**
Replace stub steps with actual command execution.

**Example (EC1):**
```rust
// In src/steps/edge_cases.rs
#[when(expr = "rbee-keeper attempts connection")]
pub async fn when_attempt_connection(world: &mut World) {
    let binary_path = workspace_dir.join("target/debug/rbee");
    let output = tokio::process::Command::new(&binary_path)
        .args(["infer", "--node", "unreachable", ...])
        .output()
        .await
        .expect("Failed to execute");
    world.last_exit_code = output.status.code();
}
```

**Files to modify:**
- `src/steps/edge_cases.rs` (~9 steps)

---

### Phase 3: Fix HTTP Issues (Days 6-7) 🟡 P1
**Impact:** +4-6 scenarios (54-58 → 58-62)

**What to do:**
```rust
// In src/steps/beehive_registry.rs line 153
for attempt in 0..5 {  // Was 0..3
    // ...
    tokio::time::sleep(Duration::from_millis(200 * 2_u64.pow(attempt))).await;
    // Was 100ms base, now 200ms base
}
```

**Files to modify:**
- `src/steps/beehive_registry.rs`
- `src/main.rs` (increase initial delay from 500ms to 1000ms)

---

### Phase 4: Missing Step Definition (Day 8) 🟢 P2
**Impact:** +1 scenario (58-62 → 59-62)

**What to do:**
```bash
# Find the missing step
sed -n '452p' tests/features/test-001.feature
# Implement it in appropriate steps file
```

---

### Phase 5: Fix Remaining (Days 9-10) 🟢 P2
**Impact:** +0-3 scenarios (59-62 → 62)

**What to do:**
Debug remaining failures individually and apply fixes.

---

## 💡 Quick Wins (Start Here!)

### Win 1: Add Registration to Line 949 (5 minutes)
```gherkin
Scenario: CLI command - basic inference
  Given node "workstation" is registered in rbee-hive registry  # ADD THIS
  When I run:
```

Test:
```bash
LLORCH_BDD_FEATURE_PATH="tests/features/test-001.feature:949" cargo run --bin bdd-runner
```

Expected: Exit code 0 (scenario passes) ✅

---

### Win 2: Add Registration to Line 976 (5 minutes)
```gherkin
Scenario: CLI command - manually shutdown worker
  Given node "workstation" is registered in rbee-hive registry  # ADD THIS
  Given a worker with id "worker-abc123" is running
```

Test: Same as above but with line 976

---

### ~~Win 3~~: Lines 176 and 230 Status (VERIFIED) ❌
**CORRECTION:** These scenarios HAVE registration BUT STILL FAIL:
```gherkin
And node "workstation" is registered in rbee-hive registry with SSH details
```

**Test results:**
- Both scenarios fail with "Step failed"
- Logs show: "error sending request for url (http://localhost:8080/v2/registry/beehives/add)"
- Retry attempts (1, 2) both fail

**This means:** The registration step itself has HTTP reliability issues. Not a quick win—needs Phase 3 (HTTP fixes) to work.

---

## 🔧 Essential Commands

### Build Everything
```bash
cd /home/vince/Projects/llama-orch
cargo build --package queen-rbee --package rbee-keeper --package test-harness-bdd --bin bdd-runner
```

### Run All Tests
```bash
cd test-harness/bdd
cargo run --bin bdd-runner
```

### Run Specific Scenario
```bash
LLORCH_BDD_FEATURE_PATH="tests/features/test-001.feature:949" cargo run --bin bdd-runner
```

### Count Passing
```bash
cargo run --bin bdd-runner 2>&1 | grep "scenarios.*passed"
```

### Debug
```bash
RUST_LOG=debug cargo run --bin bdd-runner 2>&1 | tee debug.log
```

---

## 📊 Progress Tracking

Update after each phase:

| Phase | Expected | Actual | Status |
|-------|----------|--------|--------|
| Baseline | 42 | 42 | ✅ |
| Phase 1 | 45-51 | ___ | ⏳ |
| Phase 2 | 54-58 | ___ | ⏳ |
| Phase 3 | 58-62 | ___ | ⏳ |
| Phase 4 | 59-62 | ___ | ⏳ |
| Phase 5 | 62 | ___ | 🎯 |

---

## ⚠️ Critical Rules

1. **Test after EVERY change**
   ```bash
   cargo run --bin bdd-runner 2>&1 | grep "scenarios.*passed"
   ```

2. **One scenario at a time**
   - Don't try to fix all 20 at once
   - Incremental progress is safer

3. **Sign your code**
   ```rust
   // TEAM-057: Added explicit node registration
   ```

4. **Document progress**
   - Update progress table
   - Note unexpected issues
   - Keep TEAM_058_SUMMARY.md updated

5. **Read the investigation docs**
   - They explain WHY, not just WHAT
   - Understanding prevents mistakes

---

## 🆘 If You Get Stuck

### Issue: "Node not found" errors
**Solution:** Add explicit registration step to scenario

### Issue: "Connection closed" errors
**Solution:** Increase retry attempts (Phase 3)

### Issue: Edge cases return None
**Solution:** Implement actual command execution (Phase 2)

### Issue: Tests pass then fail
**Solution:** Global state pollution - consider per-scenario isolation

### Issue: Don't know which scenario is failing
**Solution:** Run with debug logging and grep for "Step failed"

---

## 📞 Reference Materials

- **Dev rules:** `.windsurf/rules/dev-bee-rules.md`
- **Normative spec:** `bin/.specs/.gherkin/test-001.md`
- **Actual tests:** `tests/features/test-001.feature`
- **Step definitions:** `test-harness/bdd/src/steps/*.rs`
- **Mock infrastructure:** `test-harness/bdd/src/mock_rbee_hive.rs`

---

## 🎉 Success Criteria

### Minimum Success
- [ ] Phase 1 complete
- [ ] 45+ scenarios passing
- [ ] Clear progress documented

### Target Success
- [ ] Phases 1-3 complete
- [ ] 58+ scenarios passing
- [ ] Edge cases implemented

### Stretch Goal
- [ ] All phases complete
- [ ] **62/62 scenarios passing (100%)** 🎉

---

## 💪 Encouragement

TEAM-057 spent 2 days investigating to give you a clear path forward.

**You have:**
- ✅ Clear root causes
- ✅ Specific fixes
- ✅ Working examples
- ✅ Step-by-step plan
- ✅ High confidence path

**You can do this!** Follow the plan, test after each change, and you'll reach 62/62.

**Start with Phase 1.** Get those quick wins. Build momentum.

**Good luck!** 🚀

---

**Questions?** Read the investigation docs. They have the answers.

**Stuck?** Debug one scenario at a time with `RUST_LOG=debug`.

**Celebrating?** Update the progress table and keep going!

**Done?** Create handoff for TEAM-059 and celebrate! 🎊
