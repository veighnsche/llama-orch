# Testing Gaps Master Checklist - Part 4: E2E Flows + Test Infrastructure

**Date:** Oct 22, 2025  
**Source:** Phase 5 Team Investigations  
**Status:** COMPREHENSIVE TEST PLAN

---

## 12. End-to-End: Inference Flow (NOT YET IMPLEMENTED)

**Source:** TEAM-242

### 12.1 Happy Path E2E Tests

#### Full Inference Flow
- [ ] Test complete flow: Keeper → Queen → Hive → Worker → Tokens → Display
- [ ] Test all 15 steps execute correctly
- [ ] Test narration flows through all layers
- [ ] Test tokens stream in order
- [ ] Test [DONE] marker received
- [ ] Test ✅ Complete displayed

**Priority:** N/A (not implemented)  
**Complexity:** Very High  
**Estimated Effort:** 7-10 days

#### Narration Flow
- [ ] Test narration from all components
- [ ] Test job_id propagates through all layers
- [ ] Test narration routes to SSE correctly
- [ ] Test client receives all narration events

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days

### 12.2 Error Scenario Tests

#### Hive Not Running
- [ ] Test error detected
- [ ] Test error narration emitted
- [ ] Test error propagated to keeper
- [ ] Test ❌ Failed displayed

**Priority:** N/A (not implemented)  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Worker Not Available
- [ ] Test no workers for model
- [ ] Test error narration emitted
- [ ] Test error propagated to keeper

**Priority:** N/A (not implemented)  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Model Not Found
- [ ] Test model doesn't exist
- [ ] Test error narration emitted
- [ ] Test error propagated to keeper

**Priority:** N/A (not implemented)  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Model Load Failure
- [ ] Test model load fails (corrupted, VRAM)
- [ ] Test error narration emitted
- [ ] Test error propagated to keeper

**Priority:** N/A (not implemented)  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### VRAM Exhaustion
- [ ] Test VRAM exhausted
- [ ] Test error narration emitted
- [ ] Test error propagated to keeper

**Priority:** N/A (not implemented)  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Network Timeout
- [ ] Test keeper timeout (30s)
- [ ] Test connection closed
- [ ] Test error displayed
- [ ] Test server continues (orphaned)

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3 days

#### Client Disconnect
- [ ] Test Ctrl+C during inference
- [ ] Test connection closed
- [ ] Test worker continues (orphaned)
- [ ] Test no cleanup signal

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3 days

### 12.3 State Management Tests

#### Distributed State
- [ ] Test job state (queen)
- [ ] Test hive state (queen + hive)
- [ ] Test worker state (hive + worker)
- [ ] Test model state (worker)
- [ ] Test state consistency across components

**Priority:** N/A (not implemented)  
**Complexity:** Very High  
**Estimated Effort:** 5-7 days

### 12.4 Timeout Propagation Tests

#### Layered Timeouts
- [ ] Test keeper timeout (30s)
- [ ] Test queen timeout (none - issue)
- [ ] Test hive timeout (none - issue)
- [ ] Test worker timeout (none - issue)
- [ ] Test timeout propagation (only keeper has timeout)

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days

### 12.5 Resource Cleanup Tests

#### Normal Completion
- [ ] Test worker frees inference context
- [ ] Test hive removes job
- [ ] Test queen removes job
- [ ] Test all SSE channels removed

**Priority:** N/A (not implemented)  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### Error Completion
- [ ] Test cleanup on error
- [ ] Test all resources freed
- [ ] Test no memory leaks

**Priority:** N/A (not implemented)  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### Client Disconnect
- [ ] Test cleanup on disconnect
- [ ] Test cancel signal propagates (not implemented)
- [ ] Test orphaned operations (issue)

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days

### 12.6 Edge Case Tests

#### Multiple Concurrent Requests
- [ ] Test 10 concurrent inference requests
- [ ] Test all requests complete
- [ ] Test no interference
- [ ] Test tokens don't mix

**Priority:** N/A (not implemented)  
**Complexity:** Very High  
**Estimated Effort:** 5-7 days

#### Worker Crash Mid-Inference
- [ ] Test worker crashes while generating
- [ ] Test hive detects crash
- [ ] Test error propagated
- [ ] Test no retry (issue)

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days

#### Hive Crash Mid-Operation
- [ ] Test hive crashes while proxying
- [ ] Test queen detects crash
- [ ] Test error propagated
- [ ] Test no retry (issue)

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days

#### Queen Restart
- [ ] Test queen restarts mid-operation
- [ ] Test all jobs lost (no persistence)
- [ ] Test hives orphaned
- [ ] Test workers orphaned

**Priority:** N/A (not implemented)  
**Complexity:** Very High  
**Estimated Effort:** 5-7 days

#### Network Partitions
- [ ] Test partition between queen and hive
- [ ] Test partition between hive and worker
- [ ] Test timeout behavior
- [ ] Test error propagation

**Priority:** N/A (not implemented)  
**Complexity:** Very High  
**Estimated Effort:** 5-7 days

---

## 13. Test Infrastructure & Tooling

### 13.1 BDD Test Framework

#### Framework Setup
- [ ] Review existing BDD framework (bdd-runner)
- [ ] Document how to write BDD tests
- [ ] Document how to run BDD tests
- [ ] Create BDD test templates
- [ ] Create BDD test examples

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 3-5 days

#### Feature Coverage
- [ ] Write BDD features for all shared crates
- [ ] Write BDD features for all binaries
- [ ] Write BDD features for all integration flows
- [ ] Write BDD features for all E2E flows

**Priority:** HIGH  
**Complexity:** Very High  
**Estimated Effort:** 20-30 days

### 13.2 Integration Test Helpers

#### Test Utilities
- [ ] Create mock HTTP server
- [ ] Create mock SSE client
- [ ] Create mock narration sink
- [ ] Create test job registry
- [ ] Create test hive registry
- [ ] Create test worker registry

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 5-7 days

#### Test Fixtures
- [ ] Create test config files
- [ ] Create test model files
- [ ] Create test binary paths
- [ ] Create test SSH keys
- [ ] Create test capabilities

**Priority:** MEDIUM  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

### 13.3 E2E Test Environment

#### Environment Setup
- [ ] Create E2E test script
- [ ] Create test isolation (temp dirs, ports)
- [ ] Create cleanup script
- [ ] Create CI/CD integration

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 5-7 days

#### Test Data
- [ ] Create test models (small, for speed)
- [ ] Create test prompts
- [ ] Create test configurations
- [ ] Create test hive definitions

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1-2 days

### 13.4 Performance Test Framework

#### Load Testing
- [ ] Create load test framework
- [ ] Test 100 concurrent requests
- [ ] Test 1000 concurrent requests
- [ ] Measure latency (p50, p95, p99)
- [ ] Measure throughput (requests/sec)

**Priority:** MEDIUM  
**Complexity:** High  
**Estimated Effort:** 5-7 days

#### Stress Testing
- [ ] Test memory usage under load
- [ ] Test CPU usage under load
- [ ] Test network bandwidth under load
- [ ] Test resource exhaustion scenarios

**Priority:** LOW  
**Complexity:** High  
**Estimated Effort:** 5-7 days

### 13.5 Chaos Testing

#### Failure Injection
- [ ] Create network partition simulator
- [ ] Create process kill simulator
- [ ] Create timeout simulator
- [ ] Create resource exhaustion simulator

**Priority:** LOW  
**Complexity:** Very High  
**Estimated Effort:** 7-10 days

#### Chaos Scenarios
- [ ] Test random component failures
- [ ] Test random network failures
- [ ] Test random timeouts
- [ ] Test system recovery

**Priority:** LOW  
**Complexity:** Very High  
**Estimated Effort:** 7-10 days

### 13.6 Test Documentation

#### Test Plans
- [ ] Document test strategy
- [ ] Document test priorities
- [ ] Document test coverage goals
- [ ] Document test execution process

**Priority:** HIGH  
**Complexity:** Low  
**Estimated Effort:** 2-3 days

#### Test Reports
- [ ] Create test report template
- [ ] Document test results format
- [ ] Create coverage report template
- [ ] Create CI/CD report integration

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1-2 days

---

## 14. Test Priorities & Roadmap

### 14.1 Priority 1: Critical Path (Weeks 1-4)

**Focus:** Implemented features that are user-facing

#### Week 1: Shared Crates Core
- [ ] Narration core (SSE, job_id, formatting)
- [ ] Job registry (concurrent access, memory leaks)
- [ ] Daemon lifecycle (spawn, Stdio::null())

**Estimated Effort:** 10-15 days  
**Team Size:** 2-3 developers

#### Week 2: Keeper ↔ Queen Integration
- [ ] Happy path flows (HiveList, HiveStart, HiveStop)
- [ ] Error propagation (HTTP, SSE, timeouts)
- [ ] Resource cleanup (normal, error, disconnect)

**Estimated Effort:** 10-15 days  
**Team Size:** 2-3 developers

#### Week 3: Queen ↔ Hive Integration
- [ ] Hive lifecycle (spawn, health, capabilities)
- [ ] Heartbeat flow (send, receive, staleness)
- [ ] SSH integration (test connection)

**Estimated Effort:** 10-15 days  
**Team Size:** 2-3 developers

#### Week 4: Test Infrastructure
- [ ] BDD framework setup
- [ ] Integration test helpers
- [ ] E2E test environment

**Estimated Effort:** 10-15 days  
**Team Size:** 2-3 developers

### 14.2 Priority 2: Edge Cases (Weeks 5-8)

**Focus:** Error scenarios and edge cases

#### Week 5-6: Error Scenarios
- [ ] All timeout scenarios
- [ ] All network failure scenarios
- [ ] All crash scenarios
- [ ] All disconnect scenarios

**Estimated Effort:** 15-20 days  
**Team Size:** 2-3 developers

#### Week 7-8: Concurrent & Load
- [ ] Concurrent operation tests
- [ ] Load testing (100-1000 requests)
- [ ] Memory leak detection
- [ ] Resource exhaustion

**Estimated Effort:** 15-20 days  
**Team Size:** 2-3 developers

### 14.3 Priority 3: Future Features (Weeks 9-12)

**Focus:** Not yet implemented features

#### Week 9-10: Worker Operations
- [ ] Worker lifecycle tests
- [ ] Model provisioning tests
- [ ] Inference coordination tests

**Estimated Effort:** 15-20 days  
**Team Size:** 2-3 developers  
**Note:** Only after features implemented

#### Week 11-12: E2E Inference
- [ ] Full inference flow tests
- [ ] Token streaming tests
- [ ] Distributed state tests

**Estimated Effort:** 15-20 days  
**Team Size:** 2-3 developers  
**Note:** Only after features implemented

### 14.4 Priority 4: Advanced Testing (Weeks 13-16)

**Focus:** Performance and chaos testing

#### Week 13-14: Performance
- [ ] Load testing (high scale)
- [ ] Stress testing
- [ ] Performance profiling
- [ ] Optimization

**Estimated Effort:** 15-20 days  
**Team Size:** 1-2 developers

#### Week 15-16: Chaos
- [ ] Failure injection
- [ ] Chaos scenarios
- [ ] Recovery testing
- [ ] Resilience validation

**Estimated Effort:** 15-20 days  
**Team Size:** 1-2 developers

---

## 15. Summary & Metrics

### 15.1 Total Test Count

**By Category:**
- Shared Crates: ~150 tests
- Binaries: ~120 tests
- Integration Flows: ~100 tests
- E2E Flows: ~50 tests (not implemented)
- Test Infrastructure: ~30 tasks

**Total: ~450 tests + 30 infrastructure tasks**

### 15.2 Total Effort Estimate

**By Priority:**
- Priority 1 (Critical Path): 40-60 days
- Priority 2 (Edge Cases): 30-40 days
- Priority 3 (Future Features): 30-40 days (when implemented)
- Priority 4 (Advanced): 30-40 days

**Total: 130-180 days (with 1 developer)**

**With Team of 3:**
- Priority 1: 4 weeks
- Priority 2: 4 weeks
- Priority 3: 4 weeks (when implemented)
- Priority 4: 4 weeks

**Total: 16 weeks (4 months) with 3 developers**

### 15.3 Test Coverage Goals

**Current Coverage:** ~10-20% (basic unit tests only)

**Target Coverage:**
- Unit Tests: 80%+ (all shared crates, core logic)
- Integration Tests: 70%+ (all component boundaries)
- E2E Tests: 50%+ (critical user flows)
- Performance Tests: 100% (all operations benchmarked)

### 15.4 Critical Gaps (Must Fix First)

1. **SSE Channel Lifecycle** - Memory leaks, race conditions
2. **Concurrent Access** - Job registry, config files, registries
3. **Timeout Propagation** - Only keeper has timeout
4. **Resource Cleanup** - No cleanup on disconnect
5. **Error Propagation** - Some errors not propagated
6. **Stdio::null()** - Critical for E2E tests
7. **job_id Propagation** - Must include in all narration

### 15.5 Test Automation

**CI/CD Integration:**
- [ ] Run unit tests on every commit
- [ ] Run integration tests on every PR
- [ ] Run E2E tests nightly
- [ ] Run performance tests weekly
- [ ] Generate coverage reports
- [ ] Block merge if tests fail

**Test Execution:**
- [ ] Parallel test execution (where possible)
- [ ] Test isolation (no shared state)
- [ ] Fast feedback (<5 min for unit tests)
- [ ] Comprehensive feedback (<30 min for integration)

---

## 16. Recommendations

### 16.1 Immediate Actions (Week 1)

1. **Set up BDD framework** - Document and create templates
2. **Create test helpers** - Mock servers, test fixtures
3. **Write critical path tests** - Narration, job registry, keeper↔queen
4. **Fix critical bugs** - Memory leaks, race conditions

### 16.2 Short-Term Actions (Weeks 2-4)

1. **Complete Priority 1 tests** - All critical path flows
2. **Set up CI/CD** - Automated test execution
3. **Generate coverage reports** - Track progress
4. **Document test strategy** - Team alignment

### 16.3 Medium-Term Actions (Weeks 5-12)

1. **Complete Priority 2 tests** - All edge cases
2. **Implement missing features** - Worker operations, inference
3. **Write tests for new features** - As they're implemented
4. **Optimize test execution** - Parallel, fast feedback

### 16.4 Long-Term Actions (Weeks 13-16)

1. **Performance testing** - Load, stress, profiling
2. **Chaos testing** - Failure injection, recovery
3. **Continuous improvement** - Refactor, optimize
4. **Maintain coverage** - Keep tests up to date

---

**END OF TESTING GAPS MASTER CHECKLIST**

**Total Documents:** 4 parts  
**Total Tests:** ~450 tests  
**Total Effort:** 130-180 days (1 developer) or 16 weeks (3 developers)  
**Priority:** Start with Part 1 (Shared Crates) and Part 3 (Keeper↔Queen Integration)
