# Testing Audit Summary

**Date:** October 26, 2025  
**Scope:** narration-core test suite  
**Auditor:** TEAM-302

---

## Executive Summary

The narration-core test suite has **147 unit and integration tests** covering core functionality, but is **critically lacking E2E and multi-service integration tests**. Only **3 E2E tests** exist, and they only cover Axum middleware, not the full service stack.

**Critical Gap:** No tests for the complete Keeper → Queen → Hive → Worker flow with narration propagation.

---

## Current Test Inventory

### ✅ Well-Covered Areas (88 tests)

1. **Unit Tests (50 tests)**
   - n!() macro functionality (22 tests)
   - Edge cases and error handling (26 tests)
   - Format consistency (2 tests)

2. **Core Integration (38 tests)**
   - Thread-local context (15 tests)
   - SSE optional/fallback (14 tests)
   - Process capture (13 tests - TEAM-300)
   - SSE channel lifecycle (9 tests)

**Status:** These tests are relevant and passing. Keep as-is.

---

### ⚠️ Needs Review/Update (56 tests)

3. **Job Isolation (19 tests)**
   - Status: May need updates for TEAM-299 context changes
   - Action: Review and verify with new context system

4. **Privacy Isolation (10 tests)**
   - Status: Still relevant but may need expansion
   - Action: Add tests for process capture secret leakage

5. **Integration.rs (14 tests)**
   - Status: Generic integration tests, may be outdated
   - Action: Review each test, remove/update as needed

6. **Narration Edge Cases (26 tests)**
   - Status: Comprehensive but isolated
   - Action: Add multi-service edge case tests

---

### ❌ Critically Missing (0 tests)

7. **Multi-Service E2E**
   - Keeper → Queen → Hive flows: **0 tests**
   - Full stack narration propagation: **0 tests**
   - Cross-process communication: **0 tests**

8. **Job-Server/Client Integration**
   - Job creation → SSE streaming: **0 dedicated tests**
   - Concurrent job isolation: **0 tests**
   - Stream cleanup: **0 tests**

9. **Performance/Load**
   - High-frequency narration: **0 tests**
   - Concurrent streams (>100): **0 tests**
   - Memory profiling: **0 tests**

10. **Failure Scenarios**
    - Network failures: **0 tests**
    - Service crashes: **0 tests**
    - Timeout handling: **0 tests**

---

## BDD Test Status

### Current Features (4 files)

1. **cute_mode.feature** - ⚠️ Outdated
   - Uses old builder API, needs n!() macro update
   
2. **levels.feature** - ✅ Still relevant
   - Log level filtering tests work with new system
   
3. **story_mode.feature** - ⚠️ Outdated  
   - Uses old builder API, needs n!() macro update
   
4. **worker_orcd_integration.feature** - ⚠️ Outdated
   - Integration patterns have changed significantly

### Required Updates

All features need updating for:
- TEAM-297: n!() macro instead of Narration::new()
- TEAM-298: Optional SSE with stdout fallback
- TEAM-299: Thread-local context injection
- TEAM-300: ProcessNarrationCapture patterns

### New Features Needed

5. **multi_service_flow.feature** - 🆕 Required
   - Test Keeper → Queen → Hive flows
   
6. **job_server_integration.feature** - 🆕 Required
   - Test job-server/client patterns
   
7. **process_capture.feature** - 🆕 Required
   - Test worker stdout capture end-to-end

---

## Critical Findings

### 1. E2E Test Gap

**Current:** 3 E2E tests (Axum middleware only)  
**Required:** 40+ E2E tests covering full stack  
**Risk:** Production failures not caught by tests  
**Priority:** 🔴 CRITICAL

### 2. No Multi-Service Testing

**Gap:** Zero tests for service-to-service narration flow  
**Impact:** Cannot verify end-to-end functionality  
**Example Missing Test:**
```rust
// This test does not exist!
#[tokio::test]
async fn test_keeper_to_worker_narration_full_stack() {
    // Keeper submits → Queen routes → Hive spawns → Worker emits
    // Verify narration flows back through SSE to keeper
}
```
**Priority:** 🔴 CRITICAL

### 3. Job-Server Integration Untested

**Gap:** job-server and job-client are used in production but lack dedicated integration tests  
**Risk:** Channel leaks, job isolation failures undetected  
**Priority:** 🔴 CRITICAL

### 4. Process Capture E2E Missing

**Gap:** TEAM-300 implemented ProcessNarrationCapture but only has unit tests  
**Missing:** End-to-end test where worker stdout → captured → SSE → client  
**Priority:** 🟡 HIGH

### 5. No Performance Baselines

**Gap:** No performance tests to detect regressions  
**Risk:** Performance degradation unnoticed  
**Priority:** 🟢 MEDIUM

---

## Recommended Actions

### Immediate (Week 1)

1. **Create Test Harness Infrastructure**
   - Build `NarrationTestHarness` for multi-service tests
   - Implement fake service framework
   - Add SSE stream testing utilities

2. **Job-Server/Client Integration Tests**
   - Test job creation → SSE streaming
   - Test concurrent job isolation
   - Test stream cleanup

**Estimated:** 40 new tests

### Short-Term (Week 2)

3. **Multi-Service E2E Tests**
   - Keeper → Queen flow (10 tests)
   - Queen → Hive flow (10 tests)
   - Full stack E2E (10 tests)
   - Process capture E2E (10 tests)

**Estimated:** 40 new tests

### Medium-Term (Week 3)

4. **Context Propagation Tests**
   - Thread-local context across services (15 tests)
   - Correlation ID end-to-end (10 tests)

5. **Performance Tests**
   - High-frequency narration (5 tests)
   - Concurrent streams (5 tests)
   - Memory profiling (5 tests)

**Estimated:** 40 new tests

### Long-Term (Week 4)

6. **Failure Scenario Tests**
   - Network failures (10 tests)
   - Service crashes (10 tests)
   - Timeout handling (10 tests)

7. **BDD Feature Updates**
   - Update 4 existing features
   - Add 3 new features

**Estimated:** 30 new tests + 50 BDD scenarios

---

## Test Architecture Proposal

### Fake Binary Framework

**Purpose:** Enable realistic multi-process testing without full binary overhead

**Components:**
```rust
// fake_binaries/mod.rs
pub struct FakeQueen {
    port: u16,
    process: Child,
}

impl FakeQueen {
    pub async fn start() -> Self {
        // Spawns lightweight HTTP server
        // Emits narration via n!() macro
        // Forwards to configured hive
    }
    
    pub fn url(&self) -> String {
        format!("http://localhost:{}", self.port)
    }
}
```

**Benefits:**
- Realistic process boundaries
- Actual HTTP communication
- Real SSE streaming
- Narration context propagation
- Process capture testing

**Implementation:**
- Each fake binary is a small async main()
- Accepts configuration via env vars
- Emits structured narration
- Can simulate failures

### Test Harness

**Purpose:** Unified testing infrastructure for all test types

**API:**
```rust
let harness = NarrationTestHarness::start().await;

// Submit operation
let job_id = harness.submit_job(operation).await;

// Get SSE stream
let mut stream = harness.get_sse_stream(&job_id);

// Verify narration
stream.assert_next("action", "Message").await;

// Spawn fake service
let queen = harness.spawn_fake_queen().await;
```

**Features:**
- Job-server integration
- SSE stream management
- Fake service orchestration
- Assertion helpers
- Automatic cleanup

---

## Success Metrics

### Coverage Goals

| Category | Current | Target | Delta |
|----------|---------|--------|-------|
| Unit Tests | 50 | 60 | +10 |
| Integration Tests | 38 | 80 | +42 |
| E2E Tests | 3 | 40 | +37 |
| Performance Tests | 0 | 15 | +15 |
| Failure Tests | 0 | 20 | +20 |
| BDD Scenarios | 15 | 50 | +35 |
| **Total** | **106** | **265** | **+159** |

### Quality Metrics

- ✅ All existing tests continue passing
- ✅ 80% E2E coverage for multi-service flows
- ✅ 100% job-server/client integration coverage
- ✅ Performance baselines established
- ✅ Failure scenarios documented
- ✅ BDD features updated and passing

---

## Timeline

- **Week 1:** Foundation + Job Integration (40 tests)
- **Week 2:** Multi-Service E2E (40 tests)
- **Week 3:** Context + Performance (40 tests)
- **Week 4:** Failures + BDD (30 tests + 35 scenarios)

**Total Duration:** 4 weeks  
**Total New Tests:** ~150 integration/e2e tests  
**Total New BDD Scenarios:** ~35 scenarios  
**Overall Increase:** From 147 tests → ~300+ tests

---

## Risk Assessment

### High Risk (No Tests)

- ❌ Multi-service narration propagation
- ❌ Job-server channel management
- ❌ Process capture end-to-end
- ❌ Service crash scenarios

**Mitigation:** Prioritize Week 1-2 deliverables

### Medium Risk (Limited Tests)

- ⚠️ Context propagation across async boundaries
- ⚠️ High-frequency narration performance
- ⚠️ Memory leaks under load

**Mitigation:** Week 3 performance testing

### Low Risk (Well Tested)

- ✅ n!() macro functionality
- ✅ SSE optional fallback
- ✅ Basic process capture
- ✅ Thread-local context

**Status:** Continue monitoring

---

## Dependencies

### Internal
- job-server crate
- job-client crate
- operations-contract crate
- Test infrastructure (tokio, serial_test, etc.)

### External
- None (all testing can be done in-process or with fake binaries)

### Blockers
- None identified

---

## Conclusion

The narration-core test suite has solid unit and core integration coverage but **critically lacks E2E and multi-service integration tests**. 

**Immediate action required:**
1. Build test harness infrastructure
2. Implement fake binary framework
3. Add job-server/client integration tests
4. Create multi-service E2E tests

**Expected outcome:** Comprehensive test coverage (300+ tests) providing confidence in production deployment of the narration system.

**Next steps:** Begin Phase 1 implementation (Week 1: Foundation).

---

**Document Status:** ✅ COMPLETE  
**Approval Required:** Yes (for 4-week testing sprint)  
**Resources Required:** 1 engineer, 4 weeks
