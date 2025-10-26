# TEAM-307: Comprehensive BDD Test Suite

**Date:** October 26, 2025  
**Status:** 🚧 IN PROGRESS  
**Team:** TEAM-307

---

## Mission

Create a comprehensive BDD test suite that tests **ALL behaviors** (not just failures) of the narration system, including:
- Happy path scenarios
- Context propagation
- Multi-service flows
- Process capture
- Failure scenarios
- Edge cases

**Goal:** Production-ready BDD coverage for all narration behaviors.

---

## Scope

### What We're Testing (ALL BEHAVIORS)

#### 1. Core Narration Behaviors ✅
- Basic narration emission
- Field validation (actor, action, target, human)
- Narration modes (Human, Cute, Story)
- Levels (INFO, WARN, ERROR, FATAL)

#### 2. Context Propagation 🆕
- job_id auto-injection
- correlation_id propagation
- actor inheritance
- Context across async boundaries
- Context isolation

#### 3. SSE Streaming 🆕
- Job-scoped SSE channels
- Stream lifecycle
- [DONE], [ERROR], [CANCELLED] signals
- Backpressure handling

#### 4. Process Capture 🆕
- Worker stdout capture
- Narration parsing from stdout
- Re-emission with job_id
- Mixed output handling

#### 5. Multi-Service Flow 🆕
- Keeper → Queen → Hive → Worker
- job_id propagation across services
- correlation_id end-to-end
- Narration aggregation

#### 6. Failure Scenarios 🆕
- Network failures
- Service crashes
- Timeouts
- Stream disconnections
- Recovery mechanisms

---

## Feature Files to Create/Update

### Core Features (Update Existing)

1. **cute_mode.feature** - Update for n!() macro
2. **story_mode.feature** - Update for n!() macro
3. **levels.feature** - Update for n!() macro

### New Features (Comprehensive Behaviors)

4. **context_propagation.feature** 🆕 - Context injection and propagation
5. **sse_streaming.feature** 🆕 - SSE channel lifecycle and signals
6. **process_capture.feature** 🆕 - Worker stdout capture
7. **multi_service_flow.feature** 🆕 - End-to-end multi-service scenarios
8. **job_lifecycle.feature** 🆕 - Job creation, execution, completion
9. **failure_scenarios.feature** 🆕 - Network, crash, timeout failures
10. **edge_cases.feature** 🆕 - Boundary conditions and edge cases

---

## Implementation Plan

### Phase 1: Core Behaviors (Day 1)

**Update existing features:**
- ✅ cute_mode.feature - Add n!() macro scenarios
- ✅ story_mode.feature - Add n!() macro scenarios
- ✅ levels.feature - Add n!() macro scenarios

**Create:**
- 🆕 context_propagation.feature (15 scenarios)
- 🆕 job_lifecycle.feature (10 scenarios)

### Phase 2: Streaming & Capture (Day 2)

**Create:**
- 🆕 sse_streaming.feature (12 scenarios)
- 🆕 process_capture.feature (8 scenarios)

### Phase 3: Multi-Service (Day 3)

**Create:**
- 🆕 multi_service_flow.feature (15 scenarios)
- 🆕 edge_cases.feature (10 scenarios)

### Phase 4: Failure Scenarios (Day 4)

**Create:**
- 🆕 failure_scenarios.feature (20 scenarios)

### Phase 5: Step Definitions & Verification (Day 5)

**Implement:**
- Update all step definitions
- Run all scenarios
- Fix any failures
- Document results

---

## Feature Breakdown

### 1. context_propagation.feature

**Scenarios:**
1. job_id auto-injection
2. correlation_id auto-injection
3. actor auto-injection
4. All fields together
5. Context in same task
6. Context across await points
7. Context in spawned tasks (manual)
8. Context NOT inherited by tokio::spawn
9. Context isolation between jobs
10. Nested contexts
11. Context with tokio::select!
12. Context with tokio::timeout
13. Context across channels
14. Context in futures::join_all
15. Deep nesting (5 levels)

### 2. sse_streaming.feature

**Scenarios:**
1. Create SSE channel for job
2. Emit narration to SSE
3. Receive narration from SSE
4. [DONE] signal on completion
5. [ERROR] signal on failure
6. [CANCELLED] signal on cancellation
7. Multiple events in order
8. Concurrent jobs isolated
9. Stream cleanup on drop
10. Backpressure handling
11. Late subscriber (no events)
12. Early subscriber (all events)

### 3. process_capture.feature

**Scenarios:**
1. Capture worker stdout
2. Parse narration format
3. Re-emit with job_id
4. Mixed narration and logs
5. Worker crash captured
6. Multiple workers isolated
7. Long-running worker
8. Worker with no narration

### 4. multi_service_flow.feature

**Scenarios:**
1. Keeper → Queen → Hive → Worker
2. job_id propagates end-to-end
3. correlation_id propagates end-to-end
4. Narration from all services
5. Service-specific actors
6. Operation forwarding
7. Error propagation
8. Timeout propagation
9. Cancellation propagation
10. Multiple concurrent operations
11. Service restart recovery
12. Partial service failure
13. Network partition handling
14. Load balancing (multiple hives)
15. Worker pool management

### 5. job_lifecycle.feature

**Scenarios:**
1. Create job
2. Submit operation
3. Execute job
4. Stream results
5. Complete job
6. Job with timeout
7. Cancel job
8. Job failure
9. Job retry
10. Job cleanup

### 6. failure_scenarios.feature

**Scenarios:**
1. Network connection refused
2. Network timeout
3. SSE stream disconnect
4. Service crash during operation
5. Worker crash during execution
6. Timeout during execution
7. Channel full (backpressure)
8. Channel closed
9. Invalid operation
10. Missing job_id
11. Duplicate job_id
12. Corrupted SSE data
13. Out of memory
14. Disk full
15. Permission denied
16. Resource exhaustion
17. Deadlock detection
18. Race condition handling
19. Partial failure recovery
20. Cascading failure prevention

### 7. edge_cases.feature

**Scenarios:**
1. Empty narration message
2. Very long message (10KB)
3. Unicode characters
4. Special characters
5. Null bytes
6. Binary data
7. Extremely high frequency (1000/sec)
8. Zero events
9. Single event
10. Million events

---

## Step Definitions Structure

```
bdd/src/steps/
├── narration_steps.rs      - Core narration (updated)
├── context_steps.rs         - Context propagation (new)
├── sse_steps.rs            - SSE streaming (new)
├── process_steps.rs        - Process capture (new)
├── service_steps.rs        - Multi-service (new)
├── job_steps.rs            - Job lifecycle (new)
├── failure_steps.rs        - Failure scenarios (new)
└── common.rs               - Shared utilities (new)
```

---

## Success Criteria

### Coverage

- ✅ 100+ BDD scenarios
- ✅ All happy paths tested
- ✅ All failure scenarios tested
- ✅ All edge cases tested

### Quality

- ✅ All scenarios pass
- ✅ Clear, readable Gherkin
- ✅ Maintainable step definitions
- ✅ Fast execution (< 2 minutes total)

### Documentation

- ✅ Each feature documented
- ✅ Step definitions documented
- ✅ Examples provided
- ✅ Troubleshooting guide

---

## Deliverables

### Feature Files

- 3 updated features (cute, story, levels)
- 7 new features (context, sse, process, multi-service, job, failure, edge)
- **Total: 10 feature files**

### Scenarios

- ~100 scenarios total
- ~50 happy path
- ~30 failure scenarios
- ~20 edge cases

### Step Definitions

- ~500 LOC of step definitions
- ~200 LOC of test utilities
- **Total: ~700 LOC**

### Documentation

- Feature documentation
- Step definition guide
- Troubleshooting guide
- **Total: ~3 docs**

---

## Timeline

- **Day 1:** Core behaviors + context propagation (25 scenarios)
- **Day 2:** SSE streaming + process capture (20 scenarios)
- **Day 3:** Multi-service + edge cases (25 scenarios)
- **Day 4:** Failure scenarios (20 scenarios)
- **Day 5:** Step definitions + verification (all passing)

**Total: 5 days, ~90 scenarios**

---

## Next Steps

1. Create context_propagation.feature
2. Create sse_streaming.feature
3. Create process_capture.feature
4. Create multi_service_flow.feature
5. Create job_lifecycle.feature
6. Create failure_scenarios.feature
7. Create edge_cases.feature
8. Update existing features
9. Implement step definitions
10. Run and verify all tests

---

**TEAM-307 Mission: Comprehensive BDD coverage for production readiness**
