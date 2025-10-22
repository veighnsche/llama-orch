# Testing Roadmap - Phase 2 (After TEAM-244)

**Date:** Oct 22, 2025  
**Status:** Planning  
**Scope:** 175+ additional tests to reach 85% coverage

---

## Overview

TEAM-244 completed **Priority 1 & 2 tests (125 tests)**. This roadmap outlines **Phase 2: Priority 3+ tests (175+ tests)** to achieve comprehensive coverage.

---

## Phase 2 Breakdown

### Phase 2A: Critical User-Facing (20-30 days)

**Focus:** Operations users interact with directly

#### 1. Graceful Shutdown Tests (8 tests)
**Component:** `hive-lifecycle/src/stop.rs`  
**Why:** Users need reliable hive shutdown  
**Effort:** 3-5 days

```rust
#[tokio::test]
async fn test_sigterm_success() {
    // SIGTERM should stop hive within 5s
}

#[tokio::test]
async fn test_sigterm_timeout_sigkill_fallback() {
    // If SIGTERM fails, SIGKILL should be sent
}

#[tokio::test]
async fn test_stop_is_idempotent() {
    // Stopping already-stopped hive should succeed
}
```

#### 2. Capabilities Cache Tests (12 tests)
**Component:** `hive-lifecycle/src/start.rs`  
**Why:** Performance critical, affects scheduling  
**Effort:** 5-7 days

```rust
#[tokio::test]
async fn test_cache_hit_returns_cached() {
    // Should return cached capabilities
}

#[tokio::test]
async fn test_cache_miss_fetches_fresh() {
    // Should fetch fresh capabilities
}

#[tokio::test]
async fn test_cache_staleness_24h() {
    // Should detect stale cache (>24h)
}
```

#### 3. Error Propagation Tests (35 tests)
**Component:** `job_router.rs` (all operations)  
**Why:** User experience critical  
**Effort:** 10-15 days

```rust
#[tokio::test]
async fn test_hive_not_found_helpful_error() {
    // Should list available hives
}

#[tokio::test]
async fn test_binary_not_found_suggests_build() {
    // Should suggest: cargo build --bin rbee-hive
}

#[tokio::test]
async fn test_network_error_with_retry_advice() {
    // Should suggest retry or check network
}
```

**Subtotal: 55 tests, 20-30 days**

---

### Phase 2B: Core Functionality (20-30 days)

**Focus:** Internal operations that must work reliably

#### 4. Job Router Operations Tests (25 tests)
**Component:** `job_router.rs` (lines 132-371)  
**Why:** Core routing logic  
**Effort:** 8-12 days

```rust
#[tokio::test]
async fn test_status_operation_with_active_hives() {
    // Should return live status
}

#[tokio::test]
async fn test_ssh_test_operation_success() {
    // Should test SSH connection
}

#[tokio::test]
async fn test_hive_list_operation() {
    // Should list all hives
}
```

#### 5. Hive Registry Edge Cases (20 tests)
**Component:** `hive-registry/src/lib.rs`  
**Why:** State management critical  
**Effort:** 7-10 days

```rust
#[tokio::test]
async fn test_staleness_detection_30s() {
    // Should mark stale after 30s
}

#[tokio::test]
async fn test_concurrent_updates_same_hive() {
    // Should handle concurrent updates
}

#[tokio::test]
async fn test_worker_aggregation() {
    // Should aggregate worker states
}
```

#### 6. Job Registry Edge Cases (20 tests)
**Component:** `job-registry/src/lib.rs`  
**Why:** Job lifecycle management  
**Effort:** 7-10 days

```rust
#[tokio::test]
async fn test_large_payload_1mb() {
    // Should handle 1MB payloads
}

#[tokio::test]
async fn test_stream_cancellation_cleanup() {
    // Should clean up on disconnect
}

#[tokio::test]
async fn test_concurrent_job_creation() {
    // Should handle concurrent jobs
}
```

**Subtotal: 65 tests, 20-30 days**

---

### Phase 2C: Integration & Isolation (15-20 days)

**Focus:** Cross-component communication

#### 7. Narration Routing Tests (15 tests)
**Component:** `narration-core/src/lib.rs`  
**Why:** SSE isolation critical  
**Effort:** 5-7 days

```rust
#[tokio::test]
async fn test_job_id_propagation_to_channel() {
    // Narration with job_id should reach correct channel
}

#[tokio::test]
async fn test_narration_without_job_id_dropped() {
    // Narration without job_id should be dropped (fail-fast)
}

#[tokio::test]
async fn test_10_concurrent_channels_isolated() {
    // 10 channels should not interfere
}
```

#### 8. Integration Flow Tests (40 tests)
**Component:** Multiple (keeper, queen, hive)  
**Why:** End-to-end functionality  
**Effort:** 10-15 days

```rust
#[tokio::test]
async fn test_keeper_queen_job_creation() {
    // POST /v1/jobs should create job
}

#[tokio::test]
async fn test_queen_hive_start_operation() {
    // Should start hive and stream results
}

#[tokio::test]
async fn test_hive_queen_heartbeat_flow() {
    // Hive should send heartbeat every 15s
}

#[tokio::test]
async fn test_full_e2e_flow() {
    // keeper → queen → hive → queen → keeper
}
```

**Subtotal: 55 tests, 15-20 days**

---

## Implementation Timeline

### Week 1-2: Phase 2A (Critical User-Facing)
- Graceful Shutdown (8 tests)
- Capabilities Cache (12 tests)
- Error Propagation (35 tests)
- **Total: 55 tests**
- **Team:** 2-3 developers
- **Effort:** 20-30 days

### Week 3-4: Phase 2B (Core Functionality)
- Job Router Operations (25 tests)
- Hive Registry Edge Cases (20 tests)
- Job Registry Edge Cases (20 tests)
- **Total: 65 tests**
- **Team:** 2-3 developers
- **Effort:** 20-30 days

### Week 5-6: Phase 2C (Integration & Isolation)
- Narration Routing (15 tests)
- Integration Flows (40 tests)
- **Total: 55 tests**
- **Team:** 2-3 developers
- **Effort:** 15-20 days

---

## Testing Patterns (Follow TEAM-244)

### Pattern 1: Error Testing
```rust
#[tokio::test]
async fn test_operation_with_helpful_error() {
    // Setup: Create condition that causes error
    // Execute: Call operation
    // Assert: Error message is helpful
    
    let result = operation().await;
    assert!(result.is_err());
    let error = result.unwrap_err().to_string();
    assert!(error.contains("helpful message"));
    assert!(error.contains("how to fix"));
}
```

### Pattern 2: Concurrent Testing
```rust
#[tokio::test]
async fn test_concurrent_operations() {
    // Setup: Create 10 concurrent tasks
    // Execute: Run operations in parallel
    // Assert: No data corruption or race conditions
    
    let mut handles = vec![];
    for i in 0..10 {
        let handle = tokio::spawn(async move {
            operation(i).await
        });
        handles.push(handle);
    }
    
    for handle in handles {
        assert!(handle.await.unwrap().is_ok());
    }
}
```

### Pattern 3: Integration Testing
```rust
#[tokio::test]
async fn test_full_flow() {
    // Setup: Initialize all components
    // Execute: Run full flow (keeper → queen → hive)
    // Assert: Data flows correctly, no errors
    
    let keeper = setup_keeper();
    let queen = setup_queen();
    let hive = setup_hive();
    
    let result = keeper.create_job().await;
    assert!(result.is_ok());
    
    let stream = queen.execute_job(result.job_id).await;
    assert!(stream.is_ok());
}
```

---

## Success Criteria

### Coverage Goals
- [ ] Graceful Shutdown: 95%+ coverage
- [ ] Capabilities Cache: 90%+ coverage
- [ ] Error Propagation: 85%+ coverage
- [ ] Job Router: 80%+ coverage
- [ ] Hive Registry: 85%+ coverage
- [ ] Job Registry: 85%+ coverage
- [ ] Narration Routing: 90%+ coverage
- [ ] Integration Flows: 75%+ coverage

### Quality Goals
- [ ] All tests pass consistently
- [ ] No flaky tests (>99% pass rate)
- [ ] Tests run in <30 seconds
- [ ] Tests are maintainable (clear names, good docs)
- [ ] Tests follow TEAM-244 patterns

### Documentation Goals
- [ ] Each test file has clear header comment
- [ ] Each test has descriptive name
- [ ] Each test has "Why Critical" comment
- [ ] Each test has TEAM-XXX signature

---

## Resource Requirements

### Team Size
- **Ideal:** 3 developers
- **Minimum:** 2 developers
- **Maximum:** 4 developers (diminishing returns)

### Timeline
- **Phase 2A:** 20-30 days (1 developer)
- **Phase 2B:** 20-30 days (1 developer)
- **Phase 2C:** 15-20 days (1 developer)
- **Total:** 55-80 days (1 developer) or 20-30 days (3 developers)

### Tools
- Rust 1.70+
- Tokio async runtime
- Tempfile for fixtures
- Standard test framework

---

## Risk Mitigation

### Risk 1: Tests Take Too Long
**Mitigation:** Use NUC-friendly scale (5-10 concurrent, 100 max items)

### Risk 2: Tests Are Flaky
**Mitigation:** Use proper timeouts (±50ms tolerance), avoid system dependencies

### Risk 3: Tests Are Hard to Maintain
**Mitigation:** Follow TEAM-244 patterns, use clear naming, add comments

### Risk 4: Tests Don't Catch Bugs
**Mitigation:** Focus on error paths, edge cases, concurrent scenarios

---

## Deliverables

### Per Phase
- [ ] Test files (all tests passing)
- [ ] Documentation (README, comments)
- [ ] Coverage reports
- [ ] Performance benchmarks

### Final Deliverable
- [ ] 175+ tests implemented
- [ ] 85%+ coverage achieved
- [ ] All tests passing
- [ ] CI/CD integration complete
- [ ] Documentation complete

---

## Success Metrics

### Before Phase 2
- Coverage: ~15% (TEAM-TESTING + TEAM-244)
- Tests: 197 (72 + 125)
- Manual testing: 150-200 days

### After Phase 2
- Coverage: ~85%
- Tests: 372 (197 + 175)
- Manual testing: 40-60 days
- **Savings: 90-140 days**

---

## Next Actions

1. **Assign Phase 2A team** (2-3 developers)
2. **Create test file stubs** (8 files)
3. **Implement tests** (following TEAM-244 patterns)
4. **Run tests locally** (verify all pass)
5. **Integrate into CI/CD** (run on every commit)
6. **Monitor coverage** (track improvement)
7. **Repeat for Phase 2B & 2C**

---

## References

- `TEAM-244-SUMMARY.md` - TEAM-244 implementation details
- `TEAM-244-QUICK-REFERENCE.md` - Quick commands
- `TEAM-244-ADDITIONAL-OPPORTUNITIES.md` - Detailed opportunities
- `bin/.plan/TESTING_ENGINEER_GUIDE.md` - Testing guide
- `bin/.plan/TESTING_QUICK_START.md` - Quick start

---

**Ready to start Phase 2? Assign teams and begin implementation!**
