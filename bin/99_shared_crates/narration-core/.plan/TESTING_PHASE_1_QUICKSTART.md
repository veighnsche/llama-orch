# Testing Phase 1: Quick Start Guide

**Duration:** Week 1  
**Goal:** Build foundation for E2E testing  
**Deliverables:** Test harness + Job integration tests

---

## Day 1-2: Test Harness Infrastructure

### Step 1: Create Test Harness Module

**Create:** `narration-core/tests/harness/mod.rs`

```rust
//! Test harness for multi-service narration testing

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use job_server::JobRegistry;
use observability_narration_core::sse_sink::NarrationEvent;

pub struct NarrationTestHarness {
    registry: Arc<JobRegistry<String>>,
    base_url: String,
}

impl NarrationTestHarness {
    pub async fn start() -> Self {
        // TODO: Initialize job registry
        // TODO: Start mock HTTP server
        // TODO: Setup SSE channels
        
        Self {
            registry: Arc::new(JobRegistry::new()),
            base_url: "http://localhost:8765".to_string(),
        }
    }
    
    pub async fn submit_job(&self, operation: serde_json::Value) -> String {
        // TODO: Create job via registry
        // TODO: Return job_id
        unimplemented!()
    }
    
    pub fn get_sse_stream(&self, job_id: &str) -> SSEStreamTester {
        // TODO: Get receiver for job_id channel
        unimplemented!()
    }
}

pub struct SSEStreamTester {
    receiver: mpsc::Receiver<NarrationEvent>,
}

impl SSEStreamTester {
    pub async fn next_event(&mut self) -> Option<NarrationEvent> {
        self.receiver.recv().await
    }
    
    pub async fn assert_next(&mut self, action: &str, message: &str) {
        let event = self.next_event().await
            .expect("Expected narration event but stream ended");
        
        assert_eq!(event.action, action);
        assert!(event.human.contains(message));
    }
}
```

### Step 2: Add Test Utilities

**Create:** `narration-core/tests/harness/sse_utils.rs`

```rust
//! SSE testing utilities

use tokio::time::{timeout, Duration};

pub async fn collect_events_until_done(
    rx: &mut tokio::sync::mpsc::Receiver<String>,
    timeout_secs: u64,
) -> Vec<String> {
    let mut events = Vec::new();
    
    loop {
        match timeout(Duration::from_secs(timeout_secs), rx.recv()).await {
            Ok(Some(line)) => {
                if line.contains("[DONE]") {
                    break;
                }
                events.push(line);
            }
            Ok(None) => break, // Channel closed
            Err(_) => break,   // Timeout
        }
    }
    
    events
}

pub fn assert_sequence(events: &[String], expected: &[&str]) {
    assert_eq!(
        events.len(),
        expected.len(),
        "Event count mismatch. Got: {:?}, Expected: {:?}",
        events,
        expected
    );
    
    for (event, expected_substr) in events.iter().zip(expected.iter()) {
        assert!(
            event.contains(expected_substr),
            "Event '{}' doesn't contain '{}'",
            event,
            expected_substr
        );
    }
}
```

---

## Day 3: Job-Server Integration Tests

### Test 1: Basic Job Creation

**Create:** `narration-core/tests/integration/job_server_integration.rs`

```rust
use crate::harness::NarrationTestHarness;
use operations_contract::Operation;

#[tokio::test]
async fn test_job_creation_with_narration() {
    let harness = NarrationTestHarness::start().await;
    
    // Create job
    let operation = serde_json::to_value(Operation::HiveList).unwrap();
    let job_id = harness.submit_job(operation).await;
    
    assert!(!job_id.is_empty());
    
    // Verify SSE channel created
    let mut stream = harness.get_sse_stream(&job_id);
    
    // Emit test narration (simulating service)
    use observability_narration_core::{n, with_narration_context, NarrationContext};
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, async {
        n!("test_action", "Test message");
    }).await;
    
    // Verify received via SSE
    stream.assert_next("test_action", "Test message").await;
}
```

### Test 2: Concurrent Jobs

```rust
#[tokio::test]
async fn test_multiple_concurrent_jobs() {
    let harness = NarrationTestHarness::start().await;
    
    // Create 10 jobs
    let mut job_ids = Vec::new();
    for i in 0..10 {
        let op = serde_json::to_value(Operation::HiveList).unwrap();
        let job_id = harness.submit_job(op).await;
        job_ids.push(job_id);
    }
    
    // Emit narration to each job
    for (i, job_id) in job_ids.iter().enumerate() {
        let ctx = NarrationContext::new().with_job_id(job_id);
        let msg = format!("Job {} message", i);
        
        with_narration_context(ctx, async move {
            n!("job_test", "{}", msg);
        }).await;
    }
    
    // Verify isolation: each job only receives its own message
    for (i, job_id) in job_ids.iter().enumerate() {
        let mut stream = harness.get_sse_stream(job_id);
        let event = stream.next_event().await.unwrap();
        
        assert!(event.human.contains(&format!("Job {} message", i)));
        
        // Verify no cross-contamination
        // (timeout means no other events received)
        let no_more = tokio::time::timeout(
            tokio::time::Duration::from_millis(100),
            stream.next_event()
        ).await;
        assert!(no_more.is_err(), "Job {} received unexpected event", i);
    }
}
```

### Test 3: Stream Cleanup

```rust
#[tokio::test]
async fn test_sse_stream_cleanup() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    {
        // Create stream in limited scope
        let mut stream = harness.get_sse_stream(&job_id);
        
        // Emit event
        let ctx = NarrationContext::new().with_job_id(&job_id);
        with_narration_context(ctx, async {
            n!("test", "Before drop");
        }).await;
        
        stream.assert_next("test", "Before drop").await;
        
        // Stream dropped here
    }
    
    // Verify channel is cleaned up
    // New stream should not receive old events
    let mut new_stream = harness.get_sse_stream(&job_id);
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, async {
        n!("test", "After drop");
    }).await;
    
    // Should only receive new event
    new_stream.assert_next("test", "After drop").await;
}
```

---

## Day 4: Job-Client Integration

### Test 1: Submit and Stream

**Create:** `job-client/tests/integration_tests.rs`

```rust
use job_client::JobClient;
use operations_contract::Operation;
use tokio::net::TcpListener;
use axum::{Router, routing::post, Json};

async fn spawn_test_server() -> String {
    // Start minimal job server for testing
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    
    let app = Router::new()
        .route("/v1/jobs", post(create_job_handler));
    
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    
    format!("http://{}", addr)
}

async fn create_job_handler(
    Json(payload): Json<serde_json::Value>
) -> Json<serde_json::Value> {
    // Return job_id and sse_url
    Json(serde_json::json!({
        "job_id": "test-job-123",
        "sse_url": "/v1/jobs/test-job-123/stream"
    }))
}

#[tokio::test]
async fn test_submit_and_stream() {
    let server_url = spawn_test_server().await;
    let client = JobClient::new(&server_url);
    
    let operation = Operation::HiveList;
    let mut lines = Vec::new();
    
    let result = client.submit_and_stream(operation, |line| {
        lines.push(line.to_string());
        Ok(())
    }).await;
    
    assert!(result.is_ok());
    assert!(!lines.is_empty());
}
```

---

## Day 5: Verification and Documentation

### Verification Checklist

```bash
# Run all new tests
cargo test -p observability-narration-core test_job_creation
cargo test -p observability-narration-core test_multiple_concurrent
cargo test -p observability-narration-core test_sse_stream_cleanup
cargo test -p job-client test_submit_and_stream

# Verify no regressions
cargo test -p observability-narration-core
cargo test -p job-server
cargo test -p job-client
```

### Update Documentation

1. Add README to `tests/harness/`
2. Document test harness API
3. Add examples for common test patterns
4. Update COMPREHENSIVE_TESTING_PLAN.md with completion status

---

## Expected Outcomes

### Tests Added

- ✅ 3 job-server integration tests
- ✅ 1 job-client integration test
- ✅ Test harness infrastructure
- ✅ SSE testing utilities

**Total:** ~4 new tests + infrastructure

### Code Added

- `tests/harness/mod.rs` (~150 LOC)
- `tests/harness/sse_utils.rs` (~50 LOC)
- `tests/integration/job_server_integration.rs` (~200 LOC)
- `job-client/tests/integration_tests.rs` (~100 LOC)

**Total:** ~500 LOC of test infrastructure and tests

---

## Common Issues and Solutions

### Issue 1: Port Conflicts

**Problem:** Test servers bind to same port  
**Solution:** Use port 0 for automatic assignment

```rust
let listener = TcpListener::bind("127.0.0.1:0").await?;
let addr = listener.local_addr()?;
```

### Issue 2: Test Flakiness

**Problem:** Race conditions in async tests  
**Solution:** Use timeouts and proper synchronization

```rust
tokio::time::timeout(Duration::from_secs(5), async_operation).await?
```

### Issue 3: Channel Cleanup

**Problem:** Channels not cleaned up between tests  
**Solution:** Use `#[serial]` attribute or unique job IDs

```rust
#[tokio::test]
#[serial]
async fn test_with_cleanup() {
    // Test code
}
```

---

## Next Steps

After completing Phase 1:

1. **Review:** All tests passing
2. **Document:** Update comprehensive plan
3. **Proceed:** Move to Phase 2 (Multi-Service E2E)

**Phase 2 Preview:**
- Fake binary framework
- Keeper → Queen flows
- Queen → Hive flows
- Full stack E2E tests

---

## Quick Reference

### Run Specific Test
```bash
cargo test -p observability-narration-core test_job_creation -- --nocapture
```

### Run All Integration Tests
```bash
cargo test -p observability-narration-core --test 'integration/*'
```

### Debug Test Output
```bash
RUST_LOG=debug cargo test test_name -- --nocapture
```

### Check Test Coverage
```bash
cargo tarpaulin --workspace --exclude-files target/*
```

---

**Phase 1 Status:** Ready to implement  
**Estimated Time:** 5 days  
**Complexity:** Medium  
**Blockers:** None
