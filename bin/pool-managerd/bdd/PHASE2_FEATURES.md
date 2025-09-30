# Phase 2 BDD Feature Files - Complete ✓

**Date**: 2025-09-30  
**Status**: Feature files created (specification phase)  
**Implementation**: Pending (to be done in parallel with step definitions)

---

## Summary

Created **6 feature files** for Phase 2 covering **54 scenarios** and **39 behaviors** across drain/reload lifecycle and supervision/backoff/circuit breaker functionality.

---

## Feature Files Created

### Lifecycle Management (25 behaviors)

#### 1. `lifecycle/drain.feature` (10 scenarios)
**Spec**: OC-POOL-3010, OC-POOL-3031

- ✓ Drain request sets draining flag (B-DRAIN-001)
- ✓ Draining pool refuses new lease allocations (B-DRAIN-002)
- ✓ Draining pool allows existing leases to complete (B-DRAIN-003)
- ✓ Drain waits for active_leases to reach 0 (B-DRAIN-004)
- ✓ Drain force-stops after deadline expires (B-DRAIN-005)
- ✓ Drain stops engine process after leases drain (B-DRAIN-006)
- ✓ Drain updates registry health to not ready (B-DRAIN-007)
- ✓ Drain with no active leases completes immediately
- ✓ Drain emits metrics for drain duration
- ✓ Drain with inflight requests logs warning

#### 2. `lifecycle/reload.feature` (14 scenarios)
**Spec**: OC-POOL-3038

- ✓ Reload drains pool first (B-RELOAD-001)
- ✓ Reload stages new model via model-provisioner (B-RELOAD-002)
- ✓ Reload stops old engine process (B-RELOAD-003)
- ✓ Reload starts new engine with new model (B-RELOAD-004)
- ✓ Reload waits for new engine health check (B-RELOAD-005)
- ✓ Reload sets ready=true on success (B-RELOAD-006)
- ✓ Reload rolls back on failure (atomic) (B-RELOAD-007)
- ✓ Reload updates engine_version in registry (B-RELOAD-008)
- ✓ Reload preserves pool_id and device_mask (B-RELOAD-009)
- ✓ Reload with same model version is idempotent
- ✓ Reload failure preserves original state
- ✓ Reload emits metrics for reload duration
- ✓ Reload failure emits failure metrics
- ✓ Reload with drain timeout fails gracefully

### Supervision & Recovery (14 behaviors)

#### 3. `supervision/crash_detection.feature` (9 scenarios)
**Spec**: OC-POOL-3010

- ✓ Supervisor detects when engine process exits (B-SUPER-001)
- ✓ Supervisor detects when health check fails (B-SUPER-002)
- ✓ Supervisor detects driver/CUDA errors (B-SUPER-003)
- ✓ Supervisor transitions pool to unready on crash (B-SUPER-004)
- ✓ Supervisor captures exit signals
- ✓ Supervisor distinguishes graceful vs crash exit
- ✓ Supervisor logs crash context
- ✓ Supervisor increments crash counter
- ✓ Supervisor detects OOM kills

#### 4. `supervision/backoff.feature` (9 scenarios)
**Spec**: OC-POOL-3011

- ✓ First restart has initial_ms delay (B-SUPER-010)
- ✓ Subsequent restarts double delay (exponential) (B-SUPER-011)
- ✓ Backoff delay is capped at max_ms (B-SUPER-012)
- ✓ Backoff includes jitter to prevent thundering herd (B-SUPER-013)
- ✓ Backoff resets after stable run period (B-SUPER-014)
- ✓ Backoff policy logs delay decisions
- ✓ Backoff respects minimum delay
- ✓ Backoff delay increases per crash type
- ✓ Backoff emits metrics

#### 5. `supervision/circuit_breaker.feature` (11 scenarios)
**Spec**: OC-POOL-3011

- ✓ Circuit opens after N consecutive failures (B-SUPER-020)
- ✓ Open circuit prevents restart attempts (B-SUPER-021)
- ✓ Circuit transitions to half-open after timeout (B-SUPER-022)
- ✓ Half-open allows single test restart (B-SUPER-023)
- ✓ Circuit closes after successful test restart (B-SUPER-024)
- ✓ Circuit reopens if test restart fails (B-SUPER-025)
- ✓ Circuit breaker logs state transitions
- ✓ Circuit breaker emits metrics
- ✓ Circuit breaker respects manual reset
- ✓ Circuit breaker failure threshold is configurable
- ✓ Circuit breaker distinguishes error types

#### 6. `supervision/restart_storm.feature` (10 scenarios)
**Spec**: OC-POOL-3011

- ✓ Restart counter increments on each restart (B-SUPER-030)
- ✓ Restart counter resets after stable period (B-SUPER-031)
- ✓ Restart storms are logged with restart_count (B-SUPER-032)
- ✓ Maximum restart rate is enforced (B-SUPER-033)
- ✓ Restart rate limit uses sliding window
- ✓ Restart storm triggers circuit breaker
- ✓ Restart storm emits critical alert
- ✓ Restart storm metrics are tracked
- ✓ Restart storm distinguishes crash types
- ✓ Restart storm respects manual override

---

## Spec Coverage

| Requirement | Feature File | Scenarios | Status |
|-------------|--------------|-----------|--------|
| **OC-POOL-3010** | crash_detection.feature, drain.feature | 13 | 📝 Spec |
| **OC-POOL-3011** | backoff.feature, circuit_breaker.feature, restart_storm.feature | 30 | 📝 Spec |
| **OC-POOL-3031** | drain.feature | 10 | 📝 Spec |
| **OC-POOL-3038** | reload.feature | 14 | 📝 Spec |

---

## Implementation Roadmap

### Step 1: Drain Implementation
**Files to create/update**:
- `src/lifecycle/drain.rs` - DrainRequest, execute_drain()
- `src/lifecycle/mod.rs` - Export drain module
- Registry updates for draining flag

**Step definitions needed**:
- `src/steps/drain.rs` - ~15 step definitions
- Given/When/Then for drain scenarios

### Step 2: Reload Implementation
**Files to create/update**:
- `src/lifecycle/reload.rs` - ReloadRequest, execute_reload()
- Integration with model-provisioner
- Atomic rollback logic

**Step definitions needed**:
- `src/steps/reload.rs` - ~20 step definitions
- Model staging mocks
- Rollback verification steps

### Step 3: Crash Detection
**Files to create/update**:
- `src/lifecycle/supervision.rs` - CrashDetector, process monitoring
- Signal handling
- Health check polling

**Step definitions needed**:
- `src/steps/supervision.rs` - ~15 step definitions
- Process exit simulation
- Health check failure mocks

### Step 4: Backoff Policy
**Files to update**:
- `src/lifecycle/supervision.rs` - BackoffPolicy with exponential logic
- Jitter calculation
- Stable run detection

**Step definitions needed**:
- `src/steps/backoff.rs` - ~12 step definitions
- Time advancement mocks
- Delay calculation verification

### Step 5: Circuit Breaker
**Files to update**:
- `src/lifecycle/supervision.rs` - CircuitBreaker state machine
- Threshold tracking
- Half-open test logic

**Step definitions needed**:
- `src/steps/circuit_breaker.rs` - ~15 step definitions
- State transition verification
- Timeout simulation

### Step 6: Restart Storm Prevention
**Files to update**:
- `src/lifecycle/supervision.rs` - RestartRateLimiter
- Sliding window tracking
- Storm detection

**Step definitions needed**:
- `src/steps/restart_storm.rs` - ~12 step definitions
- Rate limit verification
- Window advancement

---

## Parallel Development Strategy

### Week 1: Drain & Reload
1. **Day 1-2**: Implement drain logic + drain step definitions
2. **Day 3-4**: Implement reload logic + reload step definitions
3. **Day 5**: Integration testing, fix issues

### Week 2: Supervision Foundation
1. **Day 1-2**: Implement crash detection + crash step definitions
2. **Day 3-4**: Implement backoff policy + backoff step definitions
3. **Day 5**: Integration testing, verify exponential backoff

### Week 3: Circuit Breaker & Storm Prevention
1. **Day 1-3**: Implement circuit breaker + circuit breaker step definitions
2. **Day 4-5**: Implement restart storm prevention + storm step definitions

### Week 4: Integration & Refinement
1. **Day 1-2**: End-to-end testing of all Phase 2 features
2. **Day 3-4**: Metrics emission, logging verification
3. **Day 5**: Documentation, proof bundles

---

## Testing Strategy

### Unit Tests
Each implementation module should have unit tests:
- `drain.rs` - Drain logic, deadline handling
- `reload.rs` - Atomic rollback, model staging
- `supervision.rs` - Backoff calculation, circuit breaker state machine

### BDD Tests
Run features as implementation progresses:
```bash
# Test drain
LLORCH_BDD_FEATURE_PATH=tests/features/lifecycle/drain.feature \
  cargo run -p pool-managerd-bdd --bin bdd-runner

# Test reload
LLORCH_BDD_FEATURE_PATH=tests/features/lifecycle/reload.feature \
  cargo run -p pool-managerd-bdd --bin bdd-runner

# Test supervision
LLORCH_BDD_FEATURE_PATH=tests/features/supervision/ \
  cargo run -p pool-managerd-bdd --bin bdd-runner
```

### Integration Tests
Create E2E tests in `tests/`:
- `tests/drain_reload_e2e.rs` - Full drain/reload cycle
- `tests/supervision_e2e.rs` - Crash recovery with backoff

---

## Dependencies Required

### New Crates
```toml
[dependencies]
# For process monitoring
nix = "0.27"  # Signal handling, process management

# For time/duration
tokio = { version = "1", features = ["time"] }

# For metrics (if not already present)
prometheus = "0.13"
```

### Existing Dependencies
- `anyhow` - Error handling
- `serde_json` - Handoff files
- `tracing` - Logging

---

## Metrics to Implement

### Drain Metrics
- `pool_drain_duration_ms` (histogram)
- `pool_drain_total` (counter, labels: pool_id, outcome)
- `pool_drain_force_stop_total` (counter)

### Reload Metrics
- `pool_reload_duration_ms` (histogram)
- `pool_reload_success_total` (counter)
- `pool_reload_failure_total` (counter, labels: reason)

### Supervision Metrics
- `engine_crash_total` (counter, labels: pool_id, reason)
- `backoff_delay_ms` (histogram)
- `restart_scheduled_total` (counter)
- `circuit_breaker_open_total` (counter)
- `circuit_breaker_state` (gauge, 0=closed, 1=open, 2=half-open)
- `restart_storm_total` (counter)
- `restart_rate` (gauge)

---

## Next Actions

1. **Choose starting point**: Drain or Crash Detection
2. **Implement core logic** in `src/lifecycle/`
3. **Create step definitions** in `src/steps/`
4. **Run BDD tests** to verify behavior
5. **Iterate** until all scenarios pass

---

## Files Created

```
bin/pool-managerd/bdd/tests/features/
├── lifecycle/
│   ├── drain.feature           # NEW - 10 scenarios
│   └── reload.feature          # NEW - 14 scenarios
└── supervision/
    ├── crash_detection.feature # NEW - 9 scenarios
    ├── backoff.feature         # NEW - 9 scenarios
    ├── circuit_breaker.feature # NEW - 11 scenarios
    └── restart_storm.feature   # NEW - 10 scenarios
```

---

## Conclusion

Phase 2 feature files provide **complete behavioral specifications** for drain/reload and supervision functionality. These serve as:

1. **Living documentation** of expected behavior
2. **Acceptance criteria** for implementation
3. **Test cases** that will verify correctness
4. **Spec compliance** proof for OC-POOL-3010, 3011, 3031, 3038

Ready to begin parallel implementation + step definition development following the **Option C** approach.
