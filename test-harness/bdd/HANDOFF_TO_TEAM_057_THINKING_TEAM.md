# HANDOFF TO TEAM-057: THE THINKING TEAM 🧠

**From:** TEAM-056  
**Date:** 2025-10-10T21:13:00+02:00  
**Status:** 🔴 42/62 SCENARIOS PASSING - ARCHITECTURAL INVESTIGATION REQUIRED  
**Priority:** P0 - CRITICAL - Deep architectural analysis needed to unblock development

---

## 🎯 YOUR MISSION: ARCHITECTURAL INVESTIGATION & REDESIGN

You are **THE THINKING TEAM** - specialized in deep architectural analysis, root cause investigation, and systematic problem-solving.

**Your mission:**
1. **Investigate root architectural flaws** causing 42/62 stagnation
2. **Identify contradictions** between spec, tests, and implementation
3. **Propose comprehensive multi-phase redesign**
4. **Start with spec**, cascade through tests, then code

**CRITICAL:** Think deeply. Investigate thoroughly. Propose drastic changes if needed. We are stuck for a reason - surface fixes won't work.

---

## 🔴 THE PROBLEM: DEVELOPMENT STAGNATION

**Current:** 42/62 scenarios passing (67.7%)  
**Pattern:** Multiple teams tried fixes, no improvement  
**Symptom:** 20 scenarios consistently failing  
**Root Cause:** Architectural debt from 50+ incremental AI teams

**Your job:** Find and fix architectural contradictions.

---

## 🧠 INVESTIGATION METHODOLOGY

### Phase 0: Deep Analysis (Days 1-3) - NO CODING YET

#### Required Reading (6+ hours):
1. `/home/vince/Projects/llama-orch/bin/.specs/.gherkin/test-001.md` - NORMATIVE SPEC
2. `/home/vince/Projects/llama-orch/test-harness/bdd/tests/features/test-001.feature` - ACTUAL TESTS
3. `test-harness/bdd/TEAM_055_SUMMARY.md` - HTTP retry work
4. `test-harness/bdd/TEAM_056_SUMMARY.md` - Root cause analysis
5. `test-harness/bdd/HANDOFF_TO_TEAM_056.md` - Previous attempts

#### Create These Documents:

**1. ARCHITECTURAL_CONTRADICTIONS.md**
- Spec vs Tests contradictions
- Tests vs Implementation contradictions  
- Implementation vs Reality contradictions

**2. FAILING_SCENARIOS_ANALYSIS.md**
For EACH of 20 failing scenarios:
- What it tests
- Preconditions needed vs actual
- Root cause (not symptom)
- Architectural fix required

**3. DEPENDENCY_GRAPH.md**
- What depends on what
- Timing issues
- State sharing vs isolation

---

## 🔍 SPECIFIC BUGS TO INVESTIGATE

### Bug 1: The Background Problem ⚠️ CRITICAL

**Symptom:** Scenarios using nodes fail - nodes not registered

**Investigation:**
- Should Background auto-register nodes from topology?
- Or should scenarios explicitly register?
- What does spec say (lines 35-106)?
- Timing issue - queen-rbee not ready during Background?

**Root Cause Hypothesis:**
Spec shows **two-phase setup**:
1. User runs `rbee-keeper setup add-node` (register)
2. User runs `rbee-keeper infer` (use registered node)

Tests have **single Background** that defines topology WITHOUT registration.

**This is a FUNDAMENTAL ARCHITECTURAL MISMATCH.**

**Files:**
- `test-harness/bdd/src/steps/background.rs`
- `test-harness/bdd/src/steps/beehive_registry.rs`
- `bin/.specs/.gherkin/test-001.md` (lines 35-106)

**Your Task:** Determine correct approach:
- Option A: Background registers (timing prevents this)
- Option B: Scenarios register explicitly (verbose but clear)
- Option C: Redesign spec for auto-discovery

### Bug 2: Mock Infrastructure Problem

**Symptom:** Inference fails despite mock servers existing

**Investigation:**
- Mock rbee-hive on 9200 responding?
- Mock worker on 8001 responding?
- Does queen-rbee find mocks correctly?
- Disconnect between expectations and reality?

**Root Cause Hypothesis:**
Mock exists but queen-rbee expects **real SSH** to start rbee-hive. With `MOCK_SSH=true`, SSH succeeds but queen-rbee tries `workstation.home.arpa:9200` instead of `localhost:9200`.

**Files:**
- `test-harness/bdd/src/mock_rbee_hive.rs`
- `bin/queen-rbee/src/http/inference.rs`

**Your Task:** Trace actual connection flow, identify disconnect.

### Bug 3: Edge Case Problem

**Symptom:** EC1-EC9 return `None` instead of exit code 1

**Investigation:**
- Should execute real commands or mock?
- What does spec say about error handling?
- Are step definitions stubs or implementations?

**Root Cause Hypothesis:**
Step definitions are **stubs** that just log. Either:
- Intentional placeholders
- Unintentional - thought they were implemented

**Files:**
- `test-harness/bdd/src/steps/edge_cases.rs`
- `test-harness/bdd/tests/features/test-001.feature` (lines 633-745)

**Your Task:** Determine intent, implement accordingly.

### Bug 4: Step Definition Problem

**Symptom:** "Then" steps just log without verifying

**Example:**
```rust
#[then(expr = "tokens are streamed to stdout")]
pub async fn then_tokens_streamed_stdout(world: &mut World) {
    tracing::debug!("Tokens should be streamed to stdout");
    // NO VERIFICATION!
}
```

**Root Cause:** Stubs created early, never implemented. Creates **false positives**.

**Files:** All in `test-harness/bdd/src/steps/`

**Your Task:** Audit ALL step definitions, implement verification.

### Bug 5: Global State Problem

**Symptom:** Tests interfere via shared queen-rbee instance

**Investigation:**
- Global queen-rbee causing state pollution?
- Should each scenario have own queen-rbee?
- Database reset between scenarios?
- Race conditions?

**Root Cause Hypothesis:**
Global queen-rbee uses **single database** persisting across scenarios. Node registered in one scenario stays for all. Creates **test interdependencies**.

**Files:**
- `test-harness/bdd/src/steps/global_queen.rs`
- `test-harness/bdd/src/main.rs`

**Your Task:** Investigate state management, propose robust solution.

---

## 🚨 CRITICAL ARCHITECTURAL QUESTIONS

Answer these in your investigation:

### Q1: Registration Model
Should nodes be:
- **A) Explicitly registered** (matches spec)
- **B) Auto-discovered** (simpler)
- **C) Implicitly available** (current test assumption)

**Your answer:** _____  
**Rationale:** _____  
**Impact:** _____

### Q2: Test Isolation
Should scenarios:
- **A) Share state** (current)
- **B) Be isolated** (safer)
- **C) Be ordered** (fragile)

**Your answer:** _____  
**Rationale:** _____  
**Impact:** _____

### Q3: Mock Strategy
Should mocks:
- **A) Simulate real behavior** (complex)
- **B) Return canned responses** (current)
- **C) Use real components** (slow)

**Your answer:** _____  
**Rationale:** _____  
**Impact:** _____

### Q4: Step Definition Philosophy
Should step definitions:
- **A) Verify behavior** (strict)
- **B) Document intent** (current)
- **C) Mix both** (flexible)

**Your answer:** _____  
**Rationale:** _____  
**Impact:** _____

### Q5: Background Scope
Should Background:
- **A) Set minimal state** (current)
- **B) Set complete state** (comprehensive)
- **C) Do nothing** (explicit)

**Your answer:** _____  
**Rationale:** _____  
**Impact:** _____

---

## 📋 INVESTIGATION CHECKLIST

### Phase 0: Analysis (Days 1-3)
- [ ] Read all 5 required documents
- [ ] Create ARCHITECTURAL_CONTRADICTIONS.md
- [ ] Create FAILING_SCENARIOS_ANALYSIS.md
- [ ] Create dependency graph
- [ ] Investigate all 5 bugs
- [ ] Answer all 5 critical questions
- [ ] Write TEAM_057_INVESTIGATION_REPORT.md

### Phase 1: Spec Redesign (Days 4-5)
- [ ] Propose changes to `bin/.specs/.gherkin/test-001.md`
- [ ] Resolve spec contradictions
- [ ] Define clear preconditions
- [ ] Document correct architecture

### Phase 2: Test Redesign (Days 6-7)
- [ ] Update `test-harness/bdd/tests/features/test-001.feature`
- [ ] Align with corrected spec
- [ ] Add explicit preconditions
- [ ] Remove implicit assumptions

### Phase 3: Step Redesign (Days 8-10)
- [ ] Audit `test-harness/bdd/src/steps/*`
- [ ] Identify stubs vs implementations
- [ ] Implement verification logic
- [ ] Fix timing and state issues

### Phase 4: Code Fixes (Days 11-14)
- [ ] Fix queen-rbee to match spec
- [ ] Fix mock infrastructure
- [ ] Fix CLI commands
- [ ] Ensure proper error handling

### Phase 5: Verification (Days 15-16)
- [ ] Run full test suite
- [ ] Verify 62/62 passing
- [ ] Document changes
- [ ] Create handoff for TEAM-058

---

## 🛠️ INVESTIGATION TOOLS

### Manual Testing
```bash
RUST_LOG=debug ./target/debug/queen-rbee --port 8080 --database /tmp/test.db 2>&1 | tee queen.log
./target/debug/rbee setup add-node --name workstation ...
./target/debug/rbee infer --node workstation ...
```

### Database Inspection
```bash
sqlite3 /tmp/test.db "SELECT * FROM beehives;"
```

### Network Inspection
```bash
curl http://localhost:8080/health
curl http://localhost:9200/v1/health
curl http://localhost:8001/v1/ready
```

### Test Debugging
```bash
RUST_LOG=debug cargo run --bin bdd-runner 2>&1 | tee test_debug.log
LLORCH_BDD_FEATURE_PATH="tests/features/test-001.feature:949" cargo run --bin bdd-runner
```

---

## 💡 THINKING TEAM PRINCIPLES

1. **Question Everything** - Don't assume anything is correct
2. **Seek Root Causes** - Don't fix symptoms
3. **Think Systematically** - Consider ripple effects
4. **Document Thoroughly** - Write everything down
5. **Propose Boldly** - Don't fear drastic changes

---

## 🚨 CRITICAL WARNINGS

1. **DON'T RUSH TO CODE** - Understand problem first
2. **DON'T MAKE SURFACE FIXES** - Fix architecture
3. **DON'T IGNORE CONTRADICTIONS** - Resolve them
4. **DON'T PRESERVE BAD PATTERNS** - Propose better ones
5. **DON'T WORK IN ISOLATION** - Document reasoning

---

## 🎯 EXPECTED DELIVERABLES

1. **TEAM_057_INVESTIGATION_REPORT.md** - Complete findings
2. **ARCHITECTURAL_CONTRADICTIONS.md** - All contradictions
3. **FAILING_SCENARIOS_ANALYSIS.md** - All 20 scenarios analyzed
4. **SPEC_REDESIGN_PROPOSAL.md** - Proposed spec changes
5. **IMPLEMENTATION_PLAN.md** - Multi-phase plan
6. **Updated spec** - `bin/.specs/.gherkin/test-001.md`
7. **Updated tests** - `test-harness/bdd/tests/features/test-001.feature`
8. **Updated steps** - `test-harness/bdd/src/steps/*`
9. **Fixed code** - All implementation files
10. **62/62 passing** - Complete success

---

**TEAM-056 signing off.**

**Your mission:** Unblock development through deep architectural investigation and systematic redesign.

**Timeline:** 16 days for complete investigation and fix

**Confidence:** High - with proper investigation, the path will be clear

**Remember:** You are the THINKING TEAM. Think first, code later. 🧠

---

**END OF HANDOFF**
