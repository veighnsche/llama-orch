# Testing Implementation Roadmap

**Timeline:** 4 weeks  
**Teams:** TEAM-302 through TEAM-305  
**Goal:** Comprehensive testing coverage for narration-core

---

## Overview

Transform narration-core from 147 tests with limited E2E coverage to a comprehensive test suite with 180+ tests, full multi-service E2E verification, performance baselines, and updated BDD features.

---

## Week-by-Week Breakdown

### Week 1: TEAM-302 - Test Harness & Job Integration

**Phase:** Foundation  
**Duration:** 5 days  
**Focus:** Infrastructure + job-server/client testing

#### Deliverables
- Test harness infrastructure
- SSE testing utilities
- Job-server integration tests
- Job-client integration tests

#### Metrics
- **Tests Added:** 8
- **Code Added:** ~600 LOC
- **Status:** Ready to implement

#### Key Files
- `tests/harness/mod.rs`
- `tests/harness/sse_utils.rs`
- `tests/integration/job_server_basic.rs`
- `tests/integration/job_server_concurrent.rs`
- `job-client/tests/integration_tests.rs`

**Document:** `.plan/TEAM_302_PHASE_1_TEST_HARNESS.md`

---

### Week 2: TEAM-303 - Multi-Service E2E Tests

**Phase:** E2E Testing  
**Duration:** 5 days  
**Focus:** Cross-service narration flows

#### Deliverables
- Fake binary framework (queen, hive, worker)
- Keeper → Queen E2E tests
- Queen → Hive E2E tests
- Full stack E2E tests
- Process capture E2E tests

#### Metrics
- **Tests Added:** 7
- **Code Added:** ~800 LOC
- **Status:** Blocked on TEAM-302

#### Key Files
- `tests/fake_binaries/fake_queen.rs`
- `tests/fake_binaries/fake_hive.rs`
- `tests/fake_binaries/fake_worker.rs`
- `tests/e2e/keeper_queen.rs`
- `tests/e2e/queen_hive.rs`
- `tests/e2e/full_stack.rs`
- `tests/e2e/process_capture_e2e.rs`

**Document:** `.plan/TEAM_303_PHASE_2_MULTI_SERVICE_E2E.md`

---

### Week 3: TEAM-304 - Context Propagation & Performance

**Phase:** Quality Assurance  
**Duration:** 5 days  
**Focus:** Context integrity + performance baselines

#### Deliverables
- Context propagation tests
- High-frequency narration tests
- Concurrent stream tests
- Memory usage tests
- Performance benchmarks

#### Metrics
- **Tests Added:** 12 + 3 benchmarks
- **Code Added:** ~650 LOC
- **Status:** Blocked on TEAM-303

#### Key Files
- `tests/e2e/context_propagation.rs`
- `tests/performance/high_frequency.rs`
- `tests/performance/concurrent_streams.rs`
- `tests/performance/memory_usage.rs`
- `tests/performance/benchmarks.rs`

**Document:** `.plan/TEAM_304_PHASE_3_CONTEXT_PERFORMANCE.md`

---

### Week 4: TEAM-305 - Failure Scenarios & BDD Updates

**Phase:** Robustness + Documentation  
**Duration:** 5 days  
**Focus:** Failure handling + BDD features

#### Deliverables
- Network failure tests
- Service crash tests
- Timeout handling tests
- Updated BDD features (cute, story mode)
- New BDD features (multi-service, process capture)
- Updated step definitions

#### Metrics
- **Tests Added:** 10 + 15 BDD scenarios
- **Code Added:** ~370 LOC
- **Features:** 4 updated/created
- **Status:** Blocked on TEAM-304

#### Key Files
- `tests/failure/network_failures.rs`
- `tests/failure/service_crashes.rs`
- `tests/failure/timeout_handling.rs`
- `bdd/features/multi_service_flow.feature`
- `bdd/features/process_capture.feature`
- `bdd/src/steps/service_steps.rs`

**Document:** `.plan/TEAM_305_PHASE_4_FAILURES_BDD.md`

---

## Cumulative Progress

### Test Count Growth

| Week | Team | New Tests | Cumulative |
|------|------|-----------|------------|
| Start | - | 0 | 147 |
| Week 1 | TEAM-302 | 8 | 155 |
| Week 2 | TEAM-303 | 7 | 162 |
| Week 3 | TEAM-304 | 12 | 174 |
| Week 4 | TEAM-305 | 10 | 184 |
| **Total** | **4 teams** | **37** | **184** |

*Plus 50 BDD scenarios*

### Code Added

| Week | Team | LOC Added |
|------|------|-----------|
| Week 1 | TEAM-302 | 600 |
| Week 2 | TEAM-303 | 800 |
| Week 3 | TEAM-304 | 650 |
| Week 4 | TEAM-305 | 370 |
| **Total** | **4 teams** | **2,420** |

---

## Test Coverage Goals

### Before

- **Unit Tests:** 50
- **Integration Tests:** 38
- **E2E Tests:** 3
- **Performance Tests:** 0
- **Failure Tests:** 0
- **BDD Scenarios:** 15 (outdated)

**Total:** 106 tests + 15 scenarios

### After

- **Unit Tests:** 50
- **Integration Tests:** 46 (38 + 8 new)
- **E2E Tests:** 40 (3 + 37 new)
- **Performance Tests:** 15 (new)
- **Failure Tests:** 10 (new)
- **BDD Scenarios:** 50 (updated/new)

**Total:** 161 tests + 50 scenarios

### Improvement

- **+73% more tests** (106 → 184)
- **+1,233% E2E coverage** (3 → 40)
- **Performance baselines established**
- **Failure scenarios covered**
- **BDD features modernized**

---

## Key Infrastructure

### Test Harness (TEAM-302)

```rust
let harness = NarrationTestHarness::start().await;
let job_id = harness.submit_job(operation).await;
let mut stream = harness.get_sse_stream(&job_id);
stream.assert_next("action", "message").await;
```

### Fake Binaries (TEAM-303)

```rust
// Simulated services for E2E testing
let queen = FakeQueen::start(8500).await;
let hive = FakeHive::start_with_worker(9000, "fake_worker").await;
let worker = FakeWorker spawned by hive with process capture;
```

### Performance Benchmarks (TEAM-304)

```rust
// Established baselines
- 1000+ events/sec throughput
- 100 concurrent streams
- <1ms latency
- <1MB memory per job
```

### BDD Features (TEAM-305)

```gherkin
Scenario: Full stack narration flow
  Given fake services are running
  When keeper submits operation
  Then narration propagates through all services
```

---

## Dependencies

### Between Teams

```
TEAM-302 (Foundation)
    ↓
TEAM-303 (E2E) ← Uses test harness
    ↓
TEAM-304 (Performance) ← Uses E2E tests
    ↓
TEAM-305 (Completion) ← Uses all infrastructure
```

### External Dependencies

- `job-server` crate
- `job-client` crate
- `operations-contract` crate
- `tokio` (async runtime)
- `axum` (HTTP server)
- `cucumber` (BDD framework)

---

## Risk Management

### High Risk Items

1. **Fake Binary Complexity** (TEAM-303)
   - Mitigation: Start simple, iterate
   - Fallback: Mock HTTP servers instead

2. **Timing-Sensitive Tests** (TEAM-303, TEAM-304)
   - Mitigation: Use timeouts, synchronization
   - Fallback: Mark tests as `#[ignore]` if flaky

3. **Performance Variability** (TEAM-304)
   - Mitigation: Run on isolated hardware
   - Fallback: Establish ranges, not fixed values

### Medium Risk Items

1. **Port Conflicts** (TEAM-303)
   - Mitigation: Use port 0 for auto-assignment
   
2. **Memory Leak Detection** (TEAM-304)
   - Mitigation: Long-running tests marked `#[ignore]`

3. **BDD Step Complexity** (TEAM-305)
   - Mitigation: Reuse existing infrastructure

---

## Success Criteria

### Must Have ✅

- [ ] Test harness operational
- [ ] 8+ job-server/client tests
- [ ] 7+ E2E tests
- [ ] 12+ context/performance tests
- [ ] 10+ failure scenario tests
- [ ] 50 BDD scenarios updated

### Should Have ✅

- [ ] Performance baselines documented
- [ ] Fake binary framework working
- [ ] All existing tests still passing
- [ ] Documentation complete

### Nice to Have

- [ ] Test coverage metrics
- [ ] CI/CD integration
- [ ] Automated performance regression detection

---

## Running the Tests

### All Tests
```bash
cargo test -p observability-narration-core
```

### By Phase
```bash
# Phase 1: Job integration
cargo test -p observability-narration-core --test 'integration/*'

# Phase 2: E2E tests
cargo test -p observability-narration-core --test 'e2e/*'

# Phase 3: Performance (release mode)
cargo test -p observability-narration-core --test 'performance/*' --release

# Phase 4: Failure scenarios
cargo test -p observability-narration-core --test 'failure/*'
```

### BDD Tests
```bash
cargo test -p observability-narration-core-bdd
```

### Specific Test
```bash
cargo test -p observability-narration-core test_full_stack_narration_flow -- --nocapture
```

---

## Timeline

| Week | Phase | Team | Status |
|------|-------|------|--------|
| 1 | Foundation | TEAM-302 | Ready |
| 2 | E2E | TEAM-303 | Blocked on 302 |
| 3 | Performance | TEAM-304 | Blocked on 303 |
| 4 | Completion | TEAM-305 | Blocked on 304 |

**Total Duration:** 4 weeks (20 working days)

---

## Team Handoffs

Each team creates a handoff document:

- **TEAM-302:** `.plan/TEAM_302_HANDOFF.md`
- **TEAM-303:** `.plan/TEAM_303_HANDOFF.md`
- **TEAM-304:** `.plan/TEAM_304_HANDOFF.md`
- **TEAM-305:** `.plan/TEAM_305_FINAL_HANDOFF.md`

Handoffs include:
1. What was built
2. Test results
3. Known issues
4. Next steps

---

## Maintenance Plan

### Weekly
- Review failing tests
- Update for API changes
- Add tests for new features

### Monthly
- Review BDD scenarios
- Update fake binaries
- Check performance baselines

### Quarterly
- Full test suite audit
- Performance regression analysis
- Coverage metrics review

---

## Resources

### Documentation
- Comprehensive Testing Plan: `COMPREHENSIVE_TESTING_PLAN.md`
- Testing Audit: `TESTING_AUDIT_SUMMARY.md`
- Phase 1 Quick Start: `TESTING_PHASE_1_QUICKSTART.md`

### Phase Documents
- TEAM-302: `TEAM_302_PHASE_1_TEST_HARNESS.md`
- TEAM-303: `TEAM_303_PHASE_2_MULTI_SERVICE_E2E.md`
- TEAM-304: `TEAM_304_PHASE_3_CONTEXT_PERFORMANCE.md`
- TEAM-305: `TEAM_305_PHASE_4_FAILURES_BDD.md`

### Implementation Guides
Each phase document contains:
- Day-by-day breakdown
- Code examples
- Verification steps
- Troubleshooting

---

## Conclusion

This 4-week testing implementation roadmap transforms narration-core from having basic test coverage to having comprehensive, production-ready test infrastructure.

**Key Achievements:**
- ✅ 37 new integration/E2E tests
- ✅ Test harness infrastructure
- ✅ Fake binary framework
- ✅ Performance baselines
- ✅ Failure scenario coverage
- ✅ Modernized BDD features

**Result:** World-class testing for a production-grade narration system.

---

**Implementation Status:** Ready to begin with TEAM-302  
**Approval Required:** Yes (4-week sprint)  
**Resources Required:** 1 engineer per week per team
