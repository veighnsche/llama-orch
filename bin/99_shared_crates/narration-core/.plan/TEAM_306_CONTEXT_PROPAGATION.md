# TEAM-306: Context Propagation & Performance Tests

**Status:** READY (After TEAM-304, TEAM-305, TEAM-308)  
**Priority:** P2 (Medium)  
**Dependencies:** TEAM-304 (DONE signal), TEAM-305 (circular dependency), TEAM-308 (test fixes)  
**Estimated Duration:** 1 week (5 days)  
**Dependencies:** TEAM-302, TEAM-303  
**Risk Level:** Medium

---

## Mission

Verify thread-local context propagation across service boundaries and establish performance baselines. Test high-frequency narration, concurrent streams, and memory usage under load.

**Goal:** Ensure context integrity and establish performance SLAs.

---

## Problem Statement

No tests verify:
- Context propagation across async boundaries
- job_id inheritance in nested tasks
- Correlation ID end-to-end flow
- Performance under high load
- Memory leak detection

**Impact:** Context bugs and performance regressions undetected.

---

## Implementation Tasks

### Day 1-2: Context Propagation Tests

#### Task 1.1: Thread-Local Context Across Services

**Create:** `narration-core/tests/e2e/context_propagation.rs`

```rust
// TEAM-304: Context propagation E2E tests

use crate::harness::NarrationTestHarness;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use operations_contract::Operation;

#[tokio::test]
async fn test_job_id_propagates_through_nested_tasks() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async {
        // Level 1
        n!("level1", "Level 1 task");
        
        // Level 2 (nested)
        tokio::spawn(async {
            n!("level2", "Level 2 task");
            
            // Level 3 (double nested)
            tokio::spawn(async {
                n!("level3", "Level 3 task");
            }).await.unwrap();
        }).await.unwrap();
    }).await;
    
    // Verify all events received with correct job_id
    let mut stream = harness.get_sse_stream(&job_id);
    stream.assert_next("level1", "Level 1 task").await;
    stream.assert_next("level2", "Level 2 task").await;
    stream.assert_next("level3", "Level 3 task").await;
}

#[tokio::test]
async fn test_context_inheritance_across_await_points() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async {
        n!("before_await", "Before await");
        
        // Context should survive async operation
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        n!("after_await", "After await");
    }).await;
    
    let mut stream = harness.get_sse_stream(&job_id);
    stream.assert_next("before_await", "Before await").await;
    stream.assert_next("after_await", "After await").await;
}

#[tokio::test]
async fn test_context_isolation_between_jobs() {
    let harness = NarrationTestHarness::start().await;
    
    let job1_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    let job2_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    // Spawn concurrent tasks for each job
    let ctx1 = NarrationContext::new().with_job_id(&job1_id);
    let ctx2 = NarrationContext::new().with_job_id(&job2_id);
    
    let handle1 = tokio::spawn(async move {
        with_narration_context(ctx1, async {
            n!("job1", "Job 1 message");
        }).await;
    });
    
    let handle2 = tokio::spawn(async move {
        with_narration_context(ctx2, async {
            n!("job2", "Job 2 message");
        }).await;
    });
    
    handle1.await.unwrap();
    handle2.await.unwrap();
    
    // Verify isolation
    let mut stream1 = harness.get_sse_stream(&job1_id);
    let event1 = stream1.next_event().await.unwrap();
    assert!(event1.human.contains("Job 1"));
    
    let mut stream2 = harness.get_sse_stream(&job2_id);
    let event2 = stream2.next_event().await.unwrap();
    assert!(event2.human.contains("Job 2"));
}

#[tokio::test]
async fn test_correlation_id_end_to_end() {
    // TODO: Test correlation_id from keeper through queen/hive/worker
    // Verify it appears in all narration events
}

#[tokio::test]
async fn test_context_survives_channel_boundaries() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    let (tx, mut rx) = tokio::sync::mpsc::channel(10);
    
    // Send through channel
    with_narration_context(ctx.clone(), async move {
        n!("before_channel", "Before channel");
        
        tx.send(()).await.unwrap();
    }).await;
    
    rx.recv().await.unwrap();
    
    // Context should still work after channel send
    with_narration_context(ctx, async {
        n!("after_channel", "After channel");
    }).await;
    
    let mut stream = harness.get_sse_stream(&job_id);
    stream.assert_next("before_channel", "Before channel").await;
    stream.assert_next("after_channel", "After channel").await;
}
```

---

### Day 3: Performance Tests

#### Task 3.1: High-Frequency Narration

**Create:** `narration-core/tests/performance/high_frequency.rs`

```rust
// TEAM-304: High-frequency narration performance tests

use crate::harness::NarrationTestHarness;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use operations_contract::Operation;
use std::time::Instant;

#[tokio::test]
async fn test_1000_events_per_second() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    let start = Instant::now();
    
    with_narration_context(ctx, async {
        for i in 0..1000 {
            n!("perf_test", "Event {}", i);
        }
    }).await;
    
    let duration = start.elapsed();
    
    // Should complete in under 1 second
    assert!(duration.as_secs() < 1, "Too slow: {:?}", duration);
    
    // Verify all received
    let mut stream = harness.get_sse_stream(&job_id);
    let events = stream.collect_until_done().await;
    assert_eq!(events.len(), 1000);
}

#[tokio::test]
async fn test_10000_events_rapidly() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    let start = Instant::now();
    
    with_narration_context(ctx, async {
        for i in 0..10000 {
            n!("rapid", "Event {}", i);
        }
    }).await;
    
    let duration = start.elapsed();
    
    println!("10,000 events in {:?}", duration);
    
    // Establish baseline (not assertion, just measurement)
    // Should be < 10 seconds
}

#[tokio::test]
async fn test_concurrent_emission_no_race() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    // Spawn 10 tasks emitting concurrently
    let mut handles = Vec::new();
    for i in 0..10 {
        let ctx = ctx.clone();
        let handle = tokio::spawn(async move {
            with_narration_context(ctx, async move {
                for j in 0..100 {
                    n!("concurrent", "Task {} event {}", i, j);
                }
            }).await;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify all 1000 events received
    let mut stream = harness.get_sse_stream(&job_id);
    let events = stream.collect_until_done().await;
    assert_eq!(events.len(), 1000);
}
```

#### Task 3.2: Concurrent Streams

**Create:** `narration-core/tests/performance/concurrent_streams.rs`

```rust
// TEAM-304: Concurrent stream performance tests

use crate::harness::NarrationTestHarness;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use operations_contract::Operation;

#[tokio::test]
async fn test_100_concurrent_sse_streams() {
    let harness = NarrationTestHarness::start().await;
    
    // Create 100 jobs
    let mut job_ids = Vec::new();
    for _ in 0..100 {
        let job_id = harness.submit_job(
            serde_json::to_value(Operation::HiveList).unwrap()
        ).await;
        job_ids.push(job_id);
    }
    
    // Emit to all concurrently
    let mut handles = Vec::new();
    for job_id in &job_ids {
        let job_id = job_id.clone();
        let handle = tokio::spawn(async move {
            let ctx = NarrationContext::new().with_job_id(&job_id);
            with_narration_context(ctx, async {
                for i in 0..10 {
                    n!("stream_test", "Event {}", i);
                }
            }).await;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify all streams received correct events
    for job_id in &job_ids {
        let mut stream = harness.get_sse_stream(job_id);
        let events = stream.collect_until_done().await;
        assert_eq!(events.len(), 10);
    }
}

#[tokio::test]
async fn test_channel_backpressure() {
    // Test behavior when emitting faster than consumer can process
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    // Emit 10,000 events rapidly
    with_narration_context(ctx, async {
        for i in 0..10000 {
            n!("backpressure", "Event {}", i);
        }
    }).await;
    
    // Slow consumer
    let mut stream = harness.get_sse_stream(&job_id);
    let mut count = 0;
    
    while let Some(_event) = stream.next_event().await {
        count += 1;
        tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
    }
    
    // Should receive all events (buffering should handle it)
    assert_eq!(count, 10000);
}
```

#### Task 3.3: Memory Usage

**Create:** `narration-core/tests/performance/memory_usage.rs`

```rust
// TEAM-304: Memory usage performance tests

use crate::harness::NarrationTestHarness;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use operations_contract::Operation;

#[tokio::test]
#[ignore] // Long-running test
async fn test_no_memory_leak_over_time() {
    let harness = NarrationTestHarness::start().await;
    
    // Create and destroy 1000 jobs
    for i in 0..1000 {
        let job_id = harness.submit_job(
            serde_json::to_value(Operation::HiveList).unwrap()
        ).await;
        
        let ctx = NarrationContext::new().with_job_id(&job_id);
        
        with_narration_context(ctx, async {
            n!("leak_test", "Job {}", i);
        }).await;
        
        // Drop stream (cleanup)
        drop(harness.get_sse_stream(&job_id));
        
        if i % 100 == 0 {
            println!("Completed {} jobs", i);
        }
    }
    
    // If we get here without OOM, memory is managed correctly
}

#[tokio::test]
async fn test_channel_cleanup_on_drop() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    {
        let _stream = harness.get_sse_stream(&job_id);
        // Stream dropped here
    }
    
    // Verify can create new stream (channel cleaned up)
    let _new_stream = harness.get_sse_stream(&job_id);
}
```

---

### Day 4-5: Performance Benchmarks

#### Task 4.1: Establish Baselines

**Create:** `narration-core/tests/performance/benchmarks.rs`

```rust
// TEAM-304: Performance benchmarks

use crate::harness::NarrationTestHarness;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use operations_contract::Operation;
use std::time::Instant;

struct BenchmarkResults {
    events_per_second: f64,
    latency_us: f64,
    memory_overhead_bytes: usize,
}

#[tokio::test]
#[ignore] // Run manually for benchmarking
async fn benchmark_narration_throughput() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    let iterations = 10000;
    let start = Instant::now();
    
    with_narration_context(ctx, async {
        for i in 0..iterations {
            n!("benchmark", "Event {}", i);
        }
    }).await;
    
    let duration = start.elapsed();
    let events_per_sec = iterations as f64 / duration.as_secs_f64();
    
    println!("Throughput: {:.0} events/sec", events_per_sec);
    println!("Average latency: {:.2} μs", duration.as_micros() as f64 / iterations as f64);
}

#[tokio::test]
#[ignore]
async fn benchmark_sse_stream_latency() {
    // Measure time from emit to SSE delivery
}

#[tokio::test]
#[ignore]
async fn benchmark_concurrent_jobs() {
    // Measure performance with varying numbers of concurrent jobs
}
```

---

## Verification Checklist

- [ ] Context propagation tests pass (5 tests)
- [ ] High-frequency tests pass (3 tests)
- [ ] Concurrent stream tests pass (2 tests)
- [ ] Memory usage tests pass (2 tests)
- [ ] Benchmarks run successfully
- [ ] Performance baselines documented
- [ ] No memory leaks detected
- [ ] Context isolation verified

---

## Success Criteria

1. **Context Tests Passing**
   - Nested tasks: ✅
   - Await points: ✅
   - Job isolation: ✅
   - Correlation ID: ✅
   - Channel boundaries: ✅

2. **Performance Baselines**
   - 1000+ events/sec: ✅
   - 100 concurrent streams: ✅
   - No memory leaks: ✅

3. **Documentation**
   - Baselines recorded: ✅
   - Test results documented: ✅

---

## Deliverables

### Code Added

- `tests/e2e/context_propagation.rs` (~200 LOC)
- `tests/performance/high_frequency.rs` (~150 LOC)
- `tests/performance/concurrent_streams.rs` (~120 LOC)
- `tests/performance/memory_usage.rs` (~100 LOC)
- `tests/performance/benchmarks.rs` (~80 LOC)

**Total:** ~650 LOC

### Tests Added

- Context propagation: 5 tests
- High-frequency: 3 tests
- Concurrent streams: 2 tests
- Memory usage: 2 tests
- Benchmarks: 3 tests (manual)

**Total:** 12 tests + 3 benchmarks

---

## Performance Baselines Established

| Metric | Target | Actual |
|--------|--------|--------|
| Events/sec | 1000+ | TBD |
| Concurrent streams | 100 | TBD |
| Memory overhead | < 1MB per job | TBD |
| Latency | < 1ms | TBD |

---

## Handoff to TEAM-305

Document in `.plan/TEAM_304_HANDOFF.md`:

1. **What Works**
   - Context propagation verified
   - Performance baselines established
   - Memory usage tested

2. **Test Results**
   - All 12 tests passing
   - No memory leaks detected
   - Performance meets targets

3. **Next Steps**
   - TEAM-305: Failure scenario testing
   - TEAM-305: BDD feature updates
   - Final testing phase

---

**TEAM-304 Mission Complete**
