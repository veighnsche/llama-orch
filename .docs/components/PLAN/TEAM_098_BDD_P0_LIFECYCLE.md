# TEAM-098: BDD P0 Lifecycle Tests

**Phase:** 1 - BDD Test Development  
**Duration:** 5-7 days  
**Priority:** P0 - Critical  
**Status:** 🔴 NOT STARTED

---

## Mission

Write comprehensive BDD tests for P0 lifecycle features:
1. Worker PID Tracking & Force-Kill
2. Error Handling (no unwrap/expect, structured errors)

**Deliverable:** 30 BDD scenarios covering lifecycle and error handling

---

## Assignments

### 1. Worker Lifecycle & PID Tracking (15 scenarios)
**File:** Expand `test-harness/bdd/tests/features/110-rbee-hive-lifecycle.feature`

**Scenarios to Add:**
- [ ] LIFE-001: Store worker PID on spawn
- [ ] LIFE-002: Track PID across worker lifecycle
- [ ] LIFE-003: Force-kill worker after graceful timeout (10s)
- [ ] LIFE-004: Force-kill hung worker (SIGTERM → SIGKILL)
- [ ] LIFE-005: Process liveness check (not just HTTP)
- [ ] LIFE-006: Ready timeout - kill if stuck in Loading > 30s
- [ ] LIFE-007: Parallel worker shutdown (all workers concurrently)
- [ ] LIFE-008: Shutdown timeout enforcement (30s total)
- [ ] LIFE-009: Shutdown progress metrics logged
- [ ] LIFE-010: PID cleanup on worker removal
- [ ] LIFE-011: Detect worker crash via PID (process not found)
- [ ] LIFE-012: Zombie process cleanup
- [ ] LIFE-013: Multiple workers force-killed in parallel
- [ ] LIFE-014: Force-kill audit logging
- [ ] LIFE-015: Graceful shutdown preferred over force-kill

**Step Definitions Required:**
```rust
// test-harness/bdd/src/steps/lifecycle.rs
#[when("worker is spawned")]
#[then("worker PID is stored in registry")]
#[when("worker does not respond within {int}s")]
#[then("rbee-hive force-kills worker process")]
#[then("force-kill event is logged")]
#[then("all workers shutdown concurrently")]
#[then("shutdown completes in < {int}s")]
```

---

### 2. Error Handling Tests (15 scenarios)
**File:** `test-harness/bdd/tests/features/320-error-handling.feature`

**Scenarios to Write:**
- [ ] ERR-001: No unwrap() in production code paths
- [ ] ERR-002: Structured error responses (JSON format)
- [ ] ERR-003: Error correlation IDs included
- [ ] ERR-004: Correlation IDs logged for debugging
- [ ] ERR-005: Graceful degradation (DB unavailable)
- [ ] ERR-006: Safe error messages (no sensitive data)
- [ ] ERR-007: Error messages don't contain raw tokens
- [ ] ERR-008: Error messages don't contain file paths
- [ ] ERR-009: Error messages don't contain internal IPs
- [ ] ERR-010: Error recovery for non-fatal errors
- [ ] ERR-011: Panic-free operation under load
- [ ] ERR-012: Error response includes error_code field
- [ ] ERR-013: Error response includes details object
- [ ] ERR-014: HTTP status codes match error types
- [ ] ERR-015: Error audit logging

**Step Definitions Required:**
```rust
// test-harness/bdd/src/steps/errors.rs
#[when("error occurs during worker spawn")]
#[then("response is JSON with error structure")]
#[then("response includes correlation_id")]
#[then("correlation_id is logged")]
#[then("error message does NOT contain {string}")]
#[then("system does NOT panic")]
#[then("error response includes error_code")]
```

---

## Deliverables

### Feature Files
- [ ] `110-rbee-hive-lifecycle.feature` (expand with 15 scenarios)
- [ ] `320-error-handling.feature` (15 scenarios)

### Step Definitions
- [ ] `src/steps/lifecycle.rs` (PID tracking, force-kill)
- [ ] `src/steps/errors.rs` (error handling)

### Documentation
- [ ] Update test coverage metrics
- [ ] Create handoff document (≤ 2 pages)

---

## Acceptance Criteria

- [ ] 30 scenarios total (15 lifecycle + 15 error handling)
- [ ] All scenarios use Given-When-Then
- [ ] Tags: @p0, @lifecycle, @error
- [ ] Step definitions use real code from `/bin/`
- [ ] Tests run without compilation errors
- [ ] Tests fail against current code (expected)

---

## Testing Commands

```bash
# Run lifecycle tests
LLORCH_BDD_FEATURE_PATH=test-harness/bdd/tests/features/110-rbee-hive-lifecycle.feature \
  cargo run --bin bdd-runner

# Run error handling tests
LLORCH_BDD_FEATURE_PATH=test-harness/bdd/tests/features/320-error-handling.feature \
  cargo run --bin bdd-runner

# Run with tags
cargo run --bin bdd-runner -- --tags @lifecycle
cargo run --bin bdd-runner -- --tags @error
```

---

## Progress Tracking

### Day 1-3: Lifecycle Tests
- [ ] Expand 110-rbee-hive-lifecycle.feature (15 scenarios)
- [ ] Implement lifecycle step definitions
- [ ] Run tests (expect failures)

### Day 4-5: Error Handling Tests
- [ ] Write 320-error-handling.feature (15 scenarios)
- [ ] Implement error step definitions
- [ ] Run tests (expect failures)

### Day 6-7: Integration & Handoff
- [ ] Run all tests together
- [ ] Generate coverage report
- [ ] Create handoff document

---

## Checklist

**Feature Files:**
- [ ] 110-rbee-hive-lifecycle.feature expanded (15 scenarios) ❌ TODO
- [ ] 320-error-handling.feature (15 scenarios) ❌ TODO

**Step Definitions:**
- [ ] lifecycle.rs ❌ TODO
- [ ] errors.rs ❌ TODO

**Documentation:**
- [ ] Coverage documented ❌ TODO
- [ ] Handoff created ❌ TODO

**Completion:** 0/30 scenarios (0%)

---

**Created by:** TEAM-096 | 2025-10-18  
**Assigned to:** TEAM-098  
**Previous Team:** TEAM-097 (Security Tests)  
**Next Team:** TEAM-099 (Operations Tests)
