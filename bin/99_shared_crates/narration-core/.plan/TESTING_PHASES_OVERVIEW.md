# Narration-Core Testing Phases Overview

**Last Updated:** October 26, 2025  
**Status:** In Progress

---

## Phase Structure

### ✅ COMPLETE

**TEAM-302: Phase 1 - Test Harness & Job Integration**
- Status: ✅ COMPLETE
- Deliverables: Test harness, SSE utilities, 11 integration tests
- Files: `tests/harness/`, `tests/job_server_*.rs`
- Tests: 18 passing (11 integration + 7 utility)

**TEAM-303: Phase 2 - E2E Integration (Initial)**
- Status: ✅ COMPLETE (with technical debt)
- Deliverables: Lightweight E2E tests, 3 test binaries
- Files: `tests/e2e_job_client_integration.rs`, `tests/bin/fake_*.rs`
- Tests: 10 passing (5 E2E + 5 utility)
- Technical Debt: See TEAM-303-TECHNICAL_DEBT.md

---

### 🚨 CRITICAL FIXES (Must Do Before Production)

**TEAM-304: Fix [DONE] Signal Architecture**
- Status: 🚨 CRITICAL - MUST FIX
- Priority: P0 (Blocking)
- Duration: 4-5 hours
- Mission: Move [DONE] signal from narration-core to job-server
- Files: `job-server/src/lib.rs`, test files
- Blocks: All future work
- Document: `.plan/TEAM_304_FIX_DONE_SIGNAL.md`

**TEAM-305: Fix Circular Dependency**
- Status: 🚨 CRITICAL - TECHNICAL DEBT
- Priority: P0 (Blocking)
- Duration: 2-3 hours
- Mission: Extract job-registry-interface to break circular dependency
- Files: New crate `job-registry-interface/`, test binaries
- Blocks: Real JobRegistry in tests
- Document: `.plan/TEAM_305_FIX_CIRCULAR_DEPENDENCY.md`

**TEAM-308: Fix All Broken Tests**
- Status: 🔧 CLEANUP REQUIRED
- Priority: P1 (High)
- Duration: 3-4 hours
- Mission: Fix all tests after TEAM-304 and TEAM-305 changes
- Files: All test files
- Blocks: Production readiness
- Document: `.plan/TEAM_308_FIX_ALL_TESTS.md`

---

### 📋 FUTURE WORK (After Critical Fixes)

**TEAM-306: Context Propagation & Performance**
- Status: READY (After TEAM-304, TEAM-305, TEAM-308)
- Priority: P2 (Medium)
- Duration: 1 week
- Mission: Test context propagation, performance benchmarks
- Files: New test files for context and performance
- Document: `.plan/TEAM_306_CONTEXT_PROPAGATION.md`

**TEAM-307: Failure Scenarios & BDD**
- Status: READY (After TEAM-306)
- Priority: P2 (Medium)
- Duration: 1 week
- Mission: Failure scenario tests, BDD updates
- Files: New failure test files, BDD features
- Document: `.plan/TEAM_307_FAILURE_SCENARIOS.md`

---

## Critical Path

```
TEAM-302 ✅
    ↓
TEAM-303 ✅ (with debt)
    ↓
┌───────────────────────────────────┐
│  CRITICAL FIXES (Must Do First)  │
├───────────────────────────────────┤
│  TEAM-304: Fix [DONE] Signal     │ ← P0
│  TEAM-305: Fix Circular Dep      │ ← P0
│  TEAM-308: Fix All Tests         │ ← P1
└───────────────────────────────────┘
    ↓
TEAM-306: Context & Performance
    ↓
TEAM-307: Failures & BDD
    ↓
Production Ready ✅
```

---

## Technical Debt Summary

### 1. [DONE] Signal Misplaced (TEAM-304)
- **Problem:** narration-core emits [DONE], should be job-server
- **Impact:** Architectural violation, production issues
- **Fix:** Move [DONE] to job-server
- **Document:** `.plan/DONE_SIGNAL_INVESTIGATION.md`

### 2. Circular Dependency (TEAM-305)
- **Problem:** job-server ↔ narration-core circular dependency
- **Impact:** Test binaries use HashMap instead of real JobRegistry
- **Fix:** Extract job-registry-interface
- **Document:** `.plan/TEAM_303_TECHNICAL_DEBT.md`

### 3. Broken Tests (TEAM-308)
- **Problem:** Tests fail after TEAM-304 and TEAM-305 changes
- **Impact:** Cannot merge, cannot deploy
- **Fix:** Update all tests for new architecture

---

## Test Coverage

### Current (After TEAM-302 + TEAM-303)
- **Total Tests:** 28 tests
- **Pass Rate:** 100% (but with technical debt)
- **Production Coverage:** 85%
- **Missing:** Job lifecycle integration

### After Critical Fixes (TEAM-304 + TEAM-305 + TEAM-308)
- **Total Tests:** 28+ tests (some may be added)
- **Pass Rate:** 100% (no technical debt)
- **Production Coverage:** 95%
- **Includes:** Job lifecycle integration

### After Full Implementation (TEAM-306 + TEAM-307)
- **Total Tests:** 50+ tests
- **Pass Rate:** 100%
- **Production Coverage:** 98%+
- **Includes:** Context, performance, failures

---

## Files by Phase

### TEAM-302 (Complete)
```
tests/harness/mod.rs
tests/harness/sse_utils.rs
tests/harness/README.md
tests/job_server_basic.rs
tests/job_server_concurrent.rs
```

### TEAM-303 (Complete with debt)
```
tests/e2e_job_client_integration.rs
tests/bin/fake_queen.rs
tests/bin/fake_hive.rs
tests/bin/fake_worker.rs
tests/e2e_real_processes.rs
```

### TEAM-304 (To Do)
```
job-server/src/lib.rs (modify execute_and_stream)
job-server/tests/done_signal_tests.rs (new)
narration-core/tests/*.rs (remove n!("done", "[DONE]"))
job-client/src/lib.rs (add [ERROR] handling)
```

### TEAM-305 (To Do)
```
job-registry-interface/ (new crate)
job-server/src/lib.rs (implement trait)
narration-core/tests/bin/*.rs (use real JobRegistry)
```

### TEAM-308 (To Do)
```
All test files (fix for new architecture)
tests/integration.rs (delete or fix)
```

---

## Running Tests

### Current Tests (TEAM-302 + TEAM-303)
```bash
# Basic tests
cargo test -p observability-narration-core --test job_server_basic --features axum

# Concurrent tests
cargo test -p observability-narration-core --test job_server_concurrent --features axum

# E2E tests
cargo test -p observability-narration-core --test e2e_job_client_integration --features axum

# Real process E2E (requires build first)
cargo build --bin fake-queen-rbee --bin fake-rbee-hive --bin fake-worker --features axum
cargo test -p observability-narration-core --test e2e_real_processes --features axum -- --ignored
```

### After Critical Fixes
```bash
# All tests should pass without --ignored
cargo test -p observability-narration-core --features axum
cargo test -p job-server
cargo test -p job-client
```

---

## Priority Order

1. **TEAM-304** (P0) - Fix [DONE] signal (4-5 hours)
2. **TEAM-305** (P0) - Fix circular dependency (2-3 hours)
3. **TEAM-308** (P1) - Fix all tests (3-4 hours)
4. **TEAM-306** (P2) - Context & performance (1 week)
5. **TEAM-307** (P2) - Failures & BDD (1 week)

**Total Critical Path:** ~10-12 hours to production ready

---

## Success Criteria

### Production Ready Checklist
- [ ] TEAM-304 complete ([DONE] signal fixed)
- [ ] TEAM-305 complete (circular dependency fixed)
- [ ] TEAM-308 complete (all tests passing)
- [ ] 100% test pass rate
- [ ] 95%+ production coverage
- [ ] No technical debt
- [ ] Documentation complete
- [ ] CI/CD passing

### Full Testing Complete
- [ ] TEAM-306 complete (context & performance)
- [ ] TEAM-307 complete (failures & BDD)
- [ ] 98%+ production coverage
- [ ] Performance benchmarks established
- [ ] Failure scenarios tested
- [ ] BDD features updated

---

## Documents

### Planning
- `TESTING_PHASES_OVERVIEW.md` (this file)
- `TESTING_PHASE_1_QUICKSTART.md`

### Technical Debt
- `TEAM_303_TECHNICAL_DEBT.md`
- `DONE_SIGNAL_INVESTIGATION.md`

### Phase Documents
- `TEAM_302_PHASE_1_TEST_HARNESS.md`
- `TEAM_303_PHASE_2_MULTI_SERVICE_E2E.md`
- `TEAM_304_FIX_DONE_SIGNAL.md`
- `TEAM_305_FIX_CIRCULAR_DEPENDENCY.md`
- `TEAM_306_CONTEXT_PROPAGATION.md`
- `TEAM_307_FAILURE_SCENARIOS.md`
- `TEAM_308_FIX_ALL_TESTS.md`

### Handoffs
- `TEAM_302_HANDOFF.md`
- `TEAM_303_HANDOFF.md`
- `TEAM_303_ROBUST_E2E_HANDOFF.md`

---

**Next Action:** Start TEAM-304 to fix [DONE] signal architecture
