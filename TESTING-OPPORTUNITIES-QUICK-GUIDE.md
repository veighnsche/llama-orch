# Testing Opportunities - Quick Guide

**TL;DR:** 175+ additional tests identified, organized by priority and effort

---

## Quick Stats

| Metric | Value |
|--------|-------|
| **Total Additional Tests** | 175+ |
| **Total Effort** | 60-90 days (1 dev) or 20-30 days (3 devs) |
| **Coverage Gain** | ~70% → ~85% |
| **Value** | 90-140 days manual testing saved |

---

## By Priority (Start Here)

### 🔴 CRITICAL (Do First - 20-30 days)

#### 1. Graceful Shutdown (8 tests)
- **File:** `hive-lifecycle/src/stop.rs`
- **Why:** Prevents zombie processes
- **Effort:** 3-5 days
- **Key Tests:**
  - SIGTERM success (5s timeout)
  - SIGTERM → SIGKILL fallback
  - Idempotent shutdown

#### 2. Capabilities Cache (12 tests)
- **File:** `hive-lifecycle/src/start.rs`
- **Why:** Performance critical
- **Effort:** 5-7 days
- **Key Tests:**
  - Cache hit/miss/refresh
  - Staleness detection (>24h)
  - Concurrent access

#### 3. Error Propagation (35 tests)
- **File:** `job_router.rs`
- **Why:** User experience critical
- **Effort:** 10-15 days
- **Key Tests:**
  - Helpful error messages
  - Actionable advice
  - Consistent formatting

**Subtotal: 55 tests, 20-30 days**

---

### 🟡 HIGH (Do Second - 20-30 days)

#### 4. Job Router Operations (25 tests)
- **File:** `job_router.rs`
- **Why:** Core routing logic
- **Effort:** 8-12 days
- **Key Tests:**
  - Status operation
  - SSH test operation
  - Hive list/get/status

#### 5. Hive Registry Edge Cases (20 tests)
- **File:** `hive-registry/src/lib.rs`
- **Why:** State management
- **Effort:** 7-10 days
- **Key Tests:**
  - Staleness detection
  - Worker aggregation
  - Concurrent updates

#### 6. Job Registry Edge Cases (20 tests)
- **File:** `job-registry/src/lib.rs`
- **Why:** Job lifecycle
- **Effort:** 7-10 days
- **Key Tests:**
  - Large payloads (1MB)
  - Stream cancellation
  - Concurrent jobs

**Subtotal: 65 tests, 20-30 days**

---

### 🟢 MEDIUM (Do Third - 15-20 days)

#### 7. Narration Routing (15 tests)
- **File:** `narration-core/src/lib.rs`
- **Why:** SSE isolation
- **Effort:** 5-7 days
- **Key Tests:**
  - Job ID propagation
  - Channel isolation
  - Concurrent channels (10)

#### 8. Integration Flows (40 tests)
- **File:** Multiple (keeper, queen, hive)
- **Why:** End-to-end functionality
- **Effort:** 10-15 days
- **Key Tests:**
  - Keeper → Queen flow
  - Queen → Hive flow
  - Full E2E flow

**Subtotal: 55 tests, 15-20 days**

---

## By Component

```
hive-lifecycle/
  ├── graceful_shutdown_tests.rs        (8 tests)   ← CRITICAL
  └── capabilities_cache_tests.rs       (12 tests)  ← CRITICAL

job_router.rs
  ├── error_propagation_tests.rs        (35 tests)  ← CRITICAL
  └── job_router_operations_tests.rs    (25 tests)  ← HIGH

hive-registry/
  └── hive_registry_edge_cases_tests.rs (20 tests)  ← HIGH

job-registry/
  └── job_registry_edge_cases_tests.rs  (20 tests)  ← HIGH

narration-core/
  └── narration_routing_tests.rs        (15 tests)  ← MEDIUM

integration/
  └── integration_flow_tests.rs         (40 tests)  ← MEDIUM
```

---

## Implementation Phases

### Phase 2A: Critical (Week 1-2)
```
55 tests, 20-30 days, 2-3 developers
├── Graceful Shutdown (8)
├── Capabilities Cache (12)
└── Error Propagation (35)
```

### Phase 2B: High Priority (Week 3-4)
```
65 tests, 20-30 days, 2-3 developers
├── Job Router Operations (25)
├── Hive Registry Edge Cases (20)
└── Job Registry Edge Cases (20)
```

### Phase 2C: Medium Priority (Week 5-6)
```
55 tests, 15-20 days, 2-3 developers
├── Narration Routing (15)
└── Integration Flows (40)
```

---

## Quick Test Examples

### Graceful Shutdown
```rust
#[tokio::test]
async fn test_sigterm_timeout_sigkill_fallback() {
    // If SIGTERM doesn't work, SIGKILL should be sent
    // Verify: Process stops within 5s + SIGKILL delay
}
```

### Capabilities Cache
```rust
#[tokio::test]
async fn test_cache_staleness_24h() {
    // Cache older than 24h should be refreshed
    // Verify: Fresh capabilities fetched
}
```

### Error Propagation
```rust
#[tokio::test]
async fn test_hive_not_found_helpful_error() {
    // Error should list available hives
    // Verify: Message includes available options
}
```

### Integration Flow
```rust
#[tokio::test]
async fn test_full_e2e_keeper_queen_hive() {
    // keeper → queen → hive → queen → keeper
    // Verify: Data flows correctly, no errors
}
```

---

## Success Criteria

### Coverage
- [ ] 85%+ overall coverage
- [ ] 95%+ for graceful shutdown
- [ ] 90%+ for capabilities cache
- [ ] 85%+ for error propagation

### Quality
- [ ] 99%+ test pass rate
- [ ] <30 second runtime
- [ ] 0 flaky tests
- [ ] All tests documented

### Value
- [ ] 90-140 days manual testing saved
- [ ] 80% fewer user-facing errors
- [ ] 90% fewer support tickets

---

## Getting Started

### Step 1: Assign Team
- 2-3 developers
- 1 QA engineer (optional)
- 1 tech lead (architecture)

### Step 2: Create Stubs
```bash
# Create test files
touch bin/15_queen_rbee_crates/hive-lifecycle/tests/graceful_shutdown_tests.rs
touch bin/15_queen_rbee_crates/hive-lifecycle/tests/capabilities_cache_tests.rs
# ... etc
```

### Step 3: Implement Tests
- Follow TEAM-244 patterns
- Use NUC-friendly scale (5-10 concurrent, 100 max)
- Add TEAM-XXX signatures
- No TODO markers

### Step 4: Verify
```bash
cargo test --workspace
cargo test --workspace -- --nocapture
```

### Step 5: Integrate
- Add to CI/CD pipeline
- Set up coverage reporting
- Monitor metrics

---

## Key Patterns (From TEAM-244)

### Pattern 1: Error Testing
```rust
#[tokio::test]
async fn test_operation_error() {
    let result = operation().await;
    assert!(result.is_err());
    let error = result.unwrap_err().to_string();
    assert!(error.contains("helpful message"));
}
```

### Pattern 2: Concurrent Testing
```rust
#[tokio::test]
async fn test_concurrent() {
    let handles: Vec<_> = (0..10)
        .map(|i| tokio::spawn(operation(i)))
        .collect();
    
    for handle in handles {
        assert!(handle.await.unwrap().is_ok());
    }
}
```

### Pattern 3: Timeout Testing
```rust
#[tokio::test]
async fn test_timeout() {
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        operation()
    ).await;
    assert!(result.is_err());
}
```

---

## Resources

### Documentation
- `TEAM-244-SUMMARY.md` - Full implementation details
- `TEAM-244-ADDITIONAL-OPPORTUNITIES.md` - Detailed analysis
- `TESTING-ROADMAP-PHASE-2.md` - Implementation plan
- `TESTING-OPPORTUNITIES-SUMMARY.md` - Complete summary

### Guides
- `bin/.plan/TESTING_ENGINEER_GUIDE.md` - Complete guide
- `bin/.plan/TESTING_QUICK_START.md` - Quick start
- `bin/.plan/TESTING_PRIORITIES_VISUAL.md` - Visual reference

### Examples
- `bin/99_shared_crates/daemon-lifecycle/tests/stdio_null_tests.rs`
- `bin/99_shared_crates/narration-core/tests/sse_channel_lifecycle_tests.rs`
- `bin/99_shared_crates/job-registry/tests/concurrent_access_tests.rs`

---

## FAQ

### Q: How long will Phase 2 take?
**A:** 20-30 days with 3 developers, or 60-90 days with 1 developer

### Q: What's the ROI?
**A:** 90-140 days of manual testing saved, 80% fewer user-facing errors

### Q: Do I need to do all 175 tests?
**A:** No, start with Phase 2A (55 tests) for immediate impact

### Q: Can I parallelize the work?
**A:** Yes, each phase can be done by separate teams

### Q: What if tests fail?
**A:** That's the point! Tests should catch bugs before production

---

## Next Actions

1. **Review** this guide (5 min)
2. **Assign** Phase 2A team (2-3 developers)
3. **Create** test file stubs (1 hour)
4. **Implement** tests (20-30 days)
5. **Verify** all pass (1 hour)
6. **Integrate** into CI/CD (2 hours)
7. **Repeat** for Phase 2B & 2C

---

## Summary

**175+ additional tests identified**
- 55 CRITICAL tests (20-30 days)
- 65 HIGH priority tests (20-30 days)
- 55 MEDIUM priority tests (15-20 days)

**Coverage: ~70% → ~85%**
**Value: 90-140 days manual testing saved**

**Ready? Let's build a bulletproof test suite!**

---

**Prepared by:** TEAM-244  
**Date:** Oct 22, 2025  
**Status:** ✅ Ready for Implementation
