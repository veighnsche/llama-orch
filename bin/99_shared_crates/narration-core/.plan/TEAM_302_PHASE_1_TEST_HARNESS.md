# TEAM-302: Phase 1 - Test Harness & Job Integration

**Status:** READY  
**Estimated Duration:** 1 week (5 days)  
**Dependencies:** TEAM-300, TEAM-301 complete  
**Risk Level:** Medium

---

## Mission

Build foundational test infrastructure for narration-core E2E testing. Create test harness for multi-service scenarios and implement comprehensive job-server/job-client integration tests.

**Goal:** Enable realistic testing of narration flow across service boundaries.

---

## Problem Statement

Current test suite has 147 tests but only 3 E2E tests. No infrastructure exists for testing:
- Multi-service narration flows
- Job-server channel management
- SSE stream behavior
- Concurrent job isolation

**Impact:** Production issues not caught by tests.

---

## Implementation Tasks

### Day 1: Test Harness Core

#### Task 1.1: Create Harness Module Structure

**Create:** `narration-core/tests/harness/mod.rs`

```rust
// TEAM-302: Test harness for multi-service narration testing

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use job_server::JobRegistry;
use observability_narration_core::sse_sink::NarrationEvent;

pub mod sse_utils;

/// Test harness for multi-service narration testing
pub struct NarrationTestHarness {
    registry: Arc<JobRegistry<String>>,
    base_url: String,
    job_streams: HashMap<String, mpsc::Receiver<NarrationEvent>>,
}

impl NarrationTestHarness {
    /// Start test harness with in-memory job registry
    pub async fn start() -> Self {
        let registry = Arc::new(JobRegistry::new());
        
        Self {
            registry,
            base_url: "http://localhost:8765".to_string(),
            job_streams: HashMap::new(),
        }
    }
    
    /// Submit operation and create job
    pub async fn submit_job(&self, operation: serde_json::Value) -> String {
        // Create job via registry
        let job_id = self.registry.create_job();
        
        // Store operation payload
        self.registry.set_payload(&job_id, operation);
        
        // Create SSE channel for this job
        observability_narration_core::sse_sink::create_job_channel(
            job_id.clone(),
            1000
        );
        
        job_id
    }
    
    /// Get SSE stream tester for job
    pub fn get_sse_stream(&self, job_id: &str) -> SSEStreamTester {
        let rx = observability_narration_core::sse_sink::take_job_receiver(job_id)
            .expect("Failed to get job receiver");
        
        SSEStreamTester::new(rx)
    }
    
    /// Get base URL for HTTP requests
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

/// Helper for testing SSE streams
pub struct SSEStreamTester {
    receiver: mpsc::Receiver<NarrationEvent>,
}

impl SSEStreamTester {
    pub fn new(receiver: mpsc::Receiver<NarrationEvent>) -> Self {
        Self { receiver }
    }
    
    /// Wait for next event (with timeout)
    pub async fn next_event(&mut self) -> Option<NarrationEvent> {
        tokio::time::timeout(
            tokio::time::Duration::from_secs(5),
            self.receiver.recv()
        )
        .await
        .ok()
        .flatten()
    }
    
    /// Assert next event matches criteria
    pub async fn assert_next(&mut self, action: &str, message_contains: &str) {
        let event = self.next_event().await
            .expect("Expected narration event but stream ended");
        
        assert_eq!(event.action, action, "Action mismatch");
        assert!(
            event.human.contains(message_contains),
            "Message '{}' doesn't contain '{}'",
            event.human,
            message_contains
        );
    }
    
    /// Collect all events until [DONE] or timeout
    pub async fn collect_until_done(&mut self) -> Vec<NarrationEvent> {
        let mut events = Vec::new();
        
        while let Some(event) = self.next_event().await {
            if event.human.contains("[DONE]") {
                break;
            }
            events.push(event);
        }
        
        events
    }
}
```

#### Task 1.2: Create SSE Utilities

**Create:** `narration-core/tests/harness/sse_utils.rs`

```rust
// TEAM-302: SSE testing utilities

use tokio::time::{timeout, Duration};

/// Collect events from string channel until [DONE] marker
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

/// Assert event sequence matches expected pattern
pub fn assert_sequence(events: &[String], expected: &[&str]) {
    assert_eq!(
        events.len(),
        expected.len(),
        "Event count mismatch. Got {} events, expected {}",
        events.len(),
        expected.len()
    );
    
    for (i, (event, expected_substr)) in events.iter().zip(expected.iter()).enumerate() {
        assert!(
            event.contains(expected_substr),
            "Event {} '{}' doesn't contain '{}'",
            i,
            event,
            expected_substr
        );
    }
}

/// Assert narration event contains expected fields
pub fn assert_event_contains(
    event: &observability_narration_core::NarrationEvent,
    actor: Option<&str>,
    action: Option<&str>,
    message: Option<&str>,
) {
    if let Some(expected_actor) = actor {
        assert_eq!(event.actor, expected_actor, "Actor mismatch");
    }
    
    if let Some(expected_action) = action {
        assert_eq!(event.action, expected_action, "Action mismatch");
    }
    
    if let Some(expected_msg) = message {
        assert!(
            event.human.contains(expected_msg),
            "Message '{}' doesn't contain '{}'",
            event.human,
            expected_msg
        );
    }
}
```

---

### Day 2: Job-Server Integration Tests

#### Task 2.1: Basic Job Creation Test

**Create:** `narration-core/tests/integration/job_server_basic.rs`

```rust
// TEAM-302: Job-server basic integration tests

use crate::harness::NarrationTestHarness;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use operations_contract::Operation;

#[tokio::test]
async fn test_job_creation_with_narration() {
    // Start test harness
    let harness = NarrationTestHarness::start().await;
    
    // Create job
    let operation = serde_json::to_value(Operation::HiveList).unwrap();
    let job_id = harness.submit_job(operation).await;
    
    assert!(!job_id.is_empty());
    
    // Get SSE stream
    let mut stream = harness.get_sse_stream(&job_id);
    
    // Emit test narration (simulating service)
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, async {
        n!("test_action", "Test message from service");
    }).await;
    
    // Verify received via SSE
    stream.assert_next("test_action", "Test message").await;
}

#[tokio::test]
async fn test_job_narration_isolation() {
    let harness = NarrationTestHarness::start().await;
    
    // Create two jobs
    let job1_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    let job2_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    // Emit to job 1
    let ctx1 = NarrationContext::new().with_job_id(&job1_id);
    with_narration_context(ctx1, async {
        n!("job1", "Message for job 1");
    }).await;
    
    // Emit to job 2
    let ctx2 = NarrationContext::new().with_job_id(&job2_id);
    with_narration_context(ctx2, async {
        n!("job2", "Message for job 2");
    }).await;
    
    // Verify isolation
    let mut stream1 = harness.get_sse_stream(&job1_id);
    stream1.assert_next("job1", "Message for job 1").await;
    
    let mut stream2 = harness.get_sse_stream(&job2_id);
    stream2.assert_next("job2", "Message for job 2").await;
}

#[tokio::test]
async fn test_sse_channel_cleanup() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    {
        // Create stream in limited scope
        let mut stream = harness.get_sse_stream(&job_id);
        
        let ctx = NarrationContext::new().with_job_id(&job_id);
        with_narration_context(ctx, async {
            n!("test", "Before drop");
        }).await;
        
        stream.assert_next("test", "Before drop").await;
        
        // Stream dropped here
    }
    
    // Verify channel still works with new receiver
    let mut new_stream = harness.get_sse_stream(&job_id);
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, async {
        n!("test", "After drop");
    }).await;
    
    new_stream.assert_next("test", "After drop").await;
}
```

---

### Day 3: Concurrent Job Tests

#### Task 3.1: Concurrent Job Isolation

**Create:** `narration-core/tests/integration/job_server_concurrent.rs`

```rust
// TEAM-302: Concurrent job testing

use crate::harness::NarrationTestHarness;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use operations_contract::Operation;

#[tokio::test]
async fn test_10_concurrent_jobs() {
    let harness = NarrationTestHarness::start().await;
    
    // Create 10 jobs
    let mut job_ids = Vec::new();
    for _ in 0..10 {
        let op = serde_json::to_value(Operation::HiveList).unwrap();
        let job_id = harness.submit_job(op).await;
        job_ids.push(job_id);
    }
    
    // Emit narration to each job concurrently
    let mut handles = Vec::new();
    for (i, job_id) in job_ids.iter().enumerate() {
        let job_id = job_id.clone();
        let handle = tokio::spawn(async move {
            let ctx = NarrationContext::new().with_job_id(&job_id);
            let msg = format!("Job {} message", i);
            
            with_narration_context(ctx, async move {
                n!("job_test", "{}", msg);
            }).await;
        });
        handles.push(handle);
    }
    
    // Wait for all emissions
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify each job received only its own message
    for (i, job_id) in job_ids.iter().enumerate() {
        let mut stream = harness.get_sse_stream(job_id);
        let event = stream.next_event().await
            .expect(&format!("Job {} didn't receive event", i));
        
        assert!(
            event.human.contains(&format!("Job {} message", i)),
            "Job {} received wrong message: {}",
            i,
            event.human
        );
    }
}

#[tokio::test]
async fn test_high_frequency_narration() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    // Emit 100 events rapidly
    with_narration_context(ctx, async {
        for i in 0..100 {
            n!("rapid_test", "Event {}", i);
        }
    }).await;
    
    // Verify all received
    let mut stream = harness.get_sse_stream(&job_id);
    let events = stream.collect_until_done().await;
    
    assert_eq!(events.len(), 100, "Not all events received");
}

#[tokio::test]
async fn test_job_context_in_nested_tasks() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(Operation::HiveList).unwrap()
    ).await;
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async {
        // Outer task
        n!("outer", "Outer task");
        
        // Nested task (should inherit context)
        tokio::spawn(async {
            n!("inner", "Inner task");
        }).await.unwrap();
    }).await;
    
    // Verify both events received
    let mut stream = harness.get_sse_stream(&job_id);
    stream.assert_next("outer", "Outer task").await;
    stream.assert_next("inner", "Inner task").await;
}
```

---

### Day 4: Job-Client Integration

#### Task 4.1: Job-Client Submit and Stream

**Create:** `job-client/tests/integration_tests.rs`

```rust
// TEAM-302: Job-client integration tests

use job_client::JobClient;
use operations_contract::Operation;
use tokio::net::TcpListener;
use axum::{Router, routing::{get, post}, Json, extract::Path};
use std::sync::Arc;
use tokio::sync::Mutex;

struct TestState {
    jobs: Arc<Mutex<Vec<String>>>,
}

async fn create_job_handler(
    axum::extract::State(state): axum::extract::State<Arc<TestState>>,
    Json(payload): Json<serde_json::Value>
) -> Json<serde_json::Value> {
    let job_id = format!("test-job-{}", uuid::Uuid::new_v4());
    state.jobs.lock().await.push(job_id.clone());
    
    Json(serde_json::json!({
        "job_id": job_id,
        "sse_url": format!("/v1/jobs/{}/stream", job_id)
    }))
}

async fn stream_job_handler(
    Path(job_id): Path<String>,
) -> impl axum::response::IntoResponse {
    use axum::response::sse::{Event, Sse};
    use futures::stream;
    
    let events = vec![
        "data: Test event 1\n\n",
        "data: Test event 2\n\n",
        "data: [DONE]\n\n",
    ];
    
    let stream = stream::iter(events.into_iter().map(|e| {
        Ok::<_, std::io::Error>(Event::default().data(e))
    }));
    
    Sse::new(stream)
}

async fn spawn_test_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    
    let state = Arc::new(TestState {
        jobs: Arc::new(Mutex::new(Vec::new())),
    });
    
    let app = Router::new()
        .route("/v1/jobs", post(create_job_handler))
        .route("/v1/jobs/:job_id/stream", get(stream_job_handler))
        .with_state(state);
    
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    
    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    format!("http://{}", addr)
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
    assert_eq!(lines.len(), 3); // 2 events + [DONE]
    assert!(lines[0].contains("Test event 1"));
    assert!(lines[1].contains("Test event 2"));
    assert!(lines[2].contains("[DONE]"));
}

#[tokio::test]
async fn test_submit_only_returns_job_id() {
    let server_url = spawn_test_server().await;
    let client = JobClient::new(&server_url);
    
    let operation = Operation::HiveList;
    let job_id = client.submit(operation).await.unwrap();
    
    assert!(!job_id.is_empty());
    assert!(job_id.starts_with("test-job-"));
}
```

---

### Day 5: Documentation and Verification

#### Task 5.1: Create Harness README

**Create:** `narration-core/tests/harness/README.md`

```markdown
# Test Harness Documentation

## Overview

The narration-core test harness provides infrastructure for E2E testing of narration flows across service boundaries.

## Components

### NarrationTestHarness

Main test harness for multi-service testing.

```rust
let harness = NarrationTestHarness::start().await;
let job_id = harness.submit_job(operation).await;
let mut stream = harness.get_sse_stream(&job_id);
```

### SSEStreamTester

Helper for testing SSE streams.

```rust
stream.assert_next("action", "message").await;
let events = stream.collect_until_done().await;
```

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
}
```

## Created by TEAM-302
```

---

## Verification Checklist

- [ ] Test harness module compiles
- [ ] SSE utilities module compiles
- [ ] Basic job creation test passes
- [ ] Job isolation test passes
- [ ] Channel cleanup test passes
- [ ] Concurrent jobs test passes (10 jobs)
- [ ] High-frequency test passes (100 events)
- [ ] Nested context test passes
- [ ] Job-client integration tests pass
- [ ] Documentation complete

---

## Success Criteria

1. **Test Infrastructure Complete**
   - NarrationTestHarness working
   - SSEStreamTester working
   - Utilities module complete

2. **Job-Server Tests Passing**
   - Basic creation: ✅
   - Isolation: ✅
   - Cleanup: ✅
   - Concurrent: ✅

3. **Job-Client Tests Passing**
   - Submit and stream: ✅
   - Submit only: ✅

4. **Documentation**
   - Harness README: ✅
   - Examples: ✅

---

## Deliverables

### Code Added

- `tests/harness/mod.rs` (~150 LOC)
- `tests/harness/sse_utils.rs` (~80 LOC)
- `tests/harness/README.md` (documentation)
- `tests/integration/job_server_basic.rs` (~100 LOC)
- `tests/integration/job_server_concurrent.rs` (~120 LOC)
- `job-client/tests/integration_tests.rs` (~150 LOC)

**Total:** ~600 LOC

### Tests Added

- Job-server basic: 3 tests
- Job-server concurrent: 3 tests
- Job-client: 2 tests

**Total:** 8 tests

---

## Handoff to TEAM-303

Document in `.plan/TEAM_302_HANDOFF.md`:

1. **What Works**
   - Test harness infrastructure
   - Job creation and isolation
   - SSE stream testing
   - Job-client integration

2. **Test Results**
   - All 8 tests passing
   - Concurrent jobs working
   - Channel cleanup verified

3. **Next Steps**
   - TEAM-303: Multi-service E2E tests
   - Use harness for fake service testing
   - Build on established patterns

---

## Known Limitations

1. **In-Memory Only:** Test harness uses in-memory job registry, not real HTTP server
2. **No Persistent State:** State resets between tests
3. **Limited Concurrency:** Tested up to 100 concurrent events, not stress-tested beyond

---

**TEAM-302 Mission Complete**
