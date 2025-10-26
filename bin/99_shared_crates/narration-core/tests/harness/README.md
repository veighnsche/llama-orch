# Test Harness Documentation

**Created by:** TEAM-302  
**Purpose:** E2E testing infrastructure for narration-core

---

## Overview

The narration-core test harness provides infrastructure for E2E testing of narration flows across service boundaries.

## Components

### NarrationTestHarness

Main test harness for multi-service testing.

**Usage:**
```rust
let harness = NarrationTestHarness::start().await;
let job_id = harness.submit_job(operation).await;
let mut stream = harness.get_sse_stream(&job_id);
```

**Methods:**
- `start()` - Create new test harness with in-memory job registry
- `submit_job(operation)` - Create job and SSE channel
- `get_sse_stream(job_id)` - Get SSE stream tester for job
- `base_url()` - Get base URL for HTTP requests
- `registry()` - Get job registry reference

### SSEStreamTester

Helper for testing SSE streams.

**Usage:**
```rust
stream.assert_next("action", "message").await;
let events = stream.collect_until_done().await;
```

**Methods:**
- `next_event()` - Wait for next event (5 second timeout)
- `assert_next(action, message)` - Assert next event matches criteria
- `collect_until_done()` - Collect all events until [DONE]
- `assert_no_more_events()` - Assert no events received (100ms timeout)

### SSE Utilities

Utility functions in `sse_utils` module.

**Functions:**
- `collect_events_until_done(rx, timeout)` - Collect string events
- `assert_sequence(events, expected)` - Assert event sequence
- `assert_event_contains(event, actor, action, message)` - Assert event fields
- `assert_contains_actions(events, actions)` - Assert actions present

---

## Usage Examples

### Basic Test

```rust
#[tokio::test]
async fn test_narration_flow() {
    let harness = NarrationTestHarness::start().await;
    let job_id = harness.submit_job(operation).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, async {
        n!("test", "Test message");
    }).await;
    
    let mut stream = harness.get_sse_stream(&job_id);
    stream.assert_next("test", "Test message").await;
}
```

### Concurrent Jobs

```rust
#[tokio::test]
async fn test_concurrent_jobs() {
    let harness = NarrationTestHarness::start().await;
    
    let job1 = harness.submit_job(op1).await;
    let job2 = harness.submit_job(op2).await;
    
    // Each job has isolated SSE channel
    let mut stream1 = harness.get_sse_stream(&job1);
    let mut stream2 = harness.get_sse_stream(&job2);
}
```

### High-Frequency Events

```rust
#[tokio::test]
async fn test_high_frequency() {
    let harness = NarrationTestHarness::start().await;
    let job_id = harness.submit_job(operation).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, async {
        for i in 0..100 {
            n!("rapid", "Event {}", i);
        }
    }).await;
    
    let mut stream = harness.get_sse_stream(&job_id);
    let events = stream.collect_until_done().await;
    assert_eq!(events.len(), 100);
}
```

### Job Isolation

```rust
#[tokio::test]
async fn test_isolation() {
    let harness = NarrationTestHarness::start().await;
    let job_id = harness.submit_job(operation).await;
    
    // Emit to job
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, async {
        n!("test", "Message");
    }).await;
    
    let mut stream = harness.get_sse_stream(&job_id);
    stream.assert_next("test", "Message").await;
    
    // Verify no cross-contamination
    stream.assert_no_more_events().await;
}
```

---

## Design Principles

### 1. Isolation

Each test gets its own harness instance with isolated job registry and SSE channels.

### 2. Simplicity

Test harness provides high-level API that hides complexity of job creation and SSE channel management.

### 3. Fast Execution

Tests use NUC-friendly limits:
- 10 concurrent jobs (not 100)
- 100 events per test (not 1000)
- 5 second timeouts (not 30)

### 4. Clear Assertions

Assertion methods provide clear error messages when tests fail.

---

## Common Patterns

### Pattern 1: Basic Flow

1. Create harness
2. Submit job
3. Get SSE stream
4. Emit narration with job_id
5. Assert events received

### Pattern 2: Concurrent Jobs

1. Create harness
2. Submit multiple jobs
3. Emit to each job concurrently
4. Verify isolation (no cross-contamination)

### Pattern 3: High-Frequency

1. Create harness
2. Submit job
3. Emit many events rapidly
4. Collect and verify all events

---

## Troubleshooting

### Test Hangs

**Symptom:** Test never completes  
**Cause:** Waiting for event that never arrives  
**Solution:** Check that job_id is set in narration context

### Events Not Received

**Symptom:** `assert_next()` panics with "stream ended"  
**Cause:** SSE channel not created or job_id mismatch  
**Solution:** Verify `submit_job()` was called and job_id matches

### Cross-Contamination

**Symptom:** Job receives events from other jobs  
**Cause:** Bug in SSE channel isolation  
**Solution:** This should never happen - file bug report

### Timeout Errors

**Symptom:** Test fails with timeout  
**Cause:** Event not emitted or channel blocked  
**Solution:** Check that narration is actually emitted with correct job_id

---

## Implementation Notes

### Job Registry

Uses in-memory `JobRegistry<String>` from `job-server` crate.

### SSE Channels

Created via `observability_narration_core::output::sse_sink::create_job_channel()`.

### Timeouts

- `next_event()`: 5 seconds
- `assert_no_more_events()`: 100ms

### Channel Capacity

Default: 1000 events per job (sufficient for tests).

---

## Future Enhancements

### Phase 2 (TEAM-303)

- Fake binary framework for multi-service E2E
- HTTP server integration
- Process capture testing

### Phase 3 (TEAM-304)

- Performance benchmarks
- Memory profiling
- Load testing

### Phase 4 (TEAM-305)

- Failure scenario testing
- Network failure simulation
- Service crash recovery

---

## Related Documentation

- **Testing Plan:** `.plan/COMPREHENSIVE_TESTING_PLAN.md`
- **Phase 1 Guide:** `.plan/TESTING_PHASE_1_QUICKSTART.md`
- **TEAM-302 Plan:** `.plan/TEAM_302_PHASE_1_TEST_HARNESS.md`

---

**Status:** ✅ COMPLETE  
**Tests Added:** 11 integration tests  
**Code Added:** ~600 LOC  
**Team:** TEAM-302
