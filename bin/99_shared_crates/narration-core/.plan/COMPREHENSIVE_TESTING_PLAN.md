# Comprehensive Testing Plan for Narration System

**Status:** PLANNING  
**Scope:** narration-core, job-server, job-client integration  
**Goal:** Achieve comprehensive E2E and integration test coverage

---

## Current State Analysis

### Existing Tests (147 tests across 11 files)

#### Unit Tests (Still Relevant ✅)
1. **process_capture_integration_tests.rs** (13 tests)
   - Status: ✅ RELEVANT - Tests TEAM-300 process capture
   - Coverage: Parsing, spawning, output streaming
   
2. **macro_tests.rs** (22 tests)
   - Status: ✅ RELEVANT - Tests n!() macro API
   - Coverage: Format strings, mode selection, argument handling

3. **narration_edge_cases_tests.rs** (26 tests)
   - Status: ✅ RELEVANT - Edge cases and error handling
   - Coverage: Long strings, special chars, null handling

4. **format_consistency.rs** (2 tests)
   - Status: ✅ RELEVANT - Output format validation
   - Coverage: JSON structure, field presence

#### Integration Tests (Partially Relevant)

5. **thread_local_context_tests.rs** (15 tests)
   - Status: ✅ RELEVANT - Tests TEAM-299 context injection
   - Coverage: Context propagation, async boundaries

6. **sse_optional_tests.rs** (14 tests)
   - Status: ✅ RELEVANT - Tests TEAM-298 fallback behavior
   - Coverage: SSE failure, stdout fallback

7. **sse_channel_lifecycle_tests.rs** (9 tests)
   - Status: ✅ RELEVANT - Channel creation/cleanup
   - Coverage: Memory leaks, race conditions

8. **narration_job_isolation_tests.rs** (19 tests)
   - Status: ⚠️ NEEDS REVIEW - Job isolation
   - Coverage: Job-specific channels, cross-contamination

9. **privacy_isolation_tests.rs** (10 tests)
   - Status: ⚠️ NEEDS REVIEW - Privacy concerns
   - Coverage: Secret redaction, data isolation

10. **integration.rs** (14 tests)
    - Status: ⚠️ NEEDS UPDATE - Old integration patterns
    - Coverage: Various integration scenarios

#### E2E Tests (Limited ❌)

11. **e2e_axum_integration.rs** (3 tests)
    - Status: ❌ INSUFFICIENT - Only Axum middleware
    - Coverage: Correlation ID middleware, single handler
    - Missing: Multi-service flows, job-server/client, SSE streaming

---

## BDD Tests (Needs Redesign)

### Current Features

1. **cute_mode.feature**
   - Status: ⚠️ OUTDATED - Needs update for TEAM-297 changes
   
2. **levels.feature**
   - Status: ✅ RELEVANT - Log level filtering
   
3. **story_mode.feature**
   - Status: ⚠️ OUTDATED - Needs update for TEAM-297 changes
   
4. **worker_orcd_integration.feature**
   - Status: ⚠️ OUTDATED - Worker integration patterns changed

### BDD Redesign Required

All BDD features need updating for:
- n!() macro instead of builder API
- Thread-local context injection
- Optional SSE delivery
- Process capture patterns

---

## Major Testing Gaps

### 1. Multi-Service E2E Tests ❌ MISSING

No tests for:
- Keeper → Queen → Hive → Worker flow
- SSE streaming across service boundaries
- Narration event propagation through entire stack
- Correlation ID tracking end-to-end

### 2. Job-Server/Job-Client Integration ❌ MISSING

No tests for:
- Job creation → SSE stream → narration delivery
- Multiple concurrent jobs
- Job isolation with narration channels
- Stream cleanup on client disconnect

### 3. Process Capture E2E ❌ MISSING

Limited tests for:
- Worker stdout → Hive capture → SSE delivery
- Narration parsing in realistic scenarios
- Mixed narration/non-narration output
- Process failure scenarios

### 4. Performance/Load Tests ❌ MISSING

No tests for:
- High-frequency narration (>1000/sec)
- Many concurrent SSE streams (>100)
- Memory usage under load
- Channel backpressure handling

### 5. Cross-Process Tests ❌ MISSING

No tests for:
- Actual binary-to-binary communication
- Daemon lifecycle with narration
- SSH-based narration (keeper → remote hive)

---

## Comprehensive Testing Plan

### Phase 1: Foundation (Week 1)

#### 1.1 Test Harness Infrastructure

**Create:** `narration-core/tests/harness/mod.rs`

```rust
/// Test harness for multi-service narration testing
pub struct NarrationTestHarness {
    /// In-memory job registry
    registry: Arc<JobRegistry<String>>,
    /// HTTP server for job endpoints
    server: TestServer,
    /// SSE stream receivers
    streams: HashMap<String, SSEStreamReceiver>,
}

impl NarrationTestHarness {
    /// Start test harness with mock services
    pub async fn start() -> Self;
    
    /// Submit operation and get SSE stream
    pub async fn submit_job(&self, op: Operation) -> TestJobHandle;
    
    /// Create fake service that emits narration
    pub fn spawn_fake_service(&self, name: &str) -> FakeService;
    
    /// Verify narration event received
    pub async fn assert_narration(&self, job_id: &str, predicate: impl Fn(&NarrationEvent) -> bool);
}
```

#### 1.2 Fake Binary Framework

**Create:** `narration-core/tests/fake_binaries/`

```
fake_binaries/
├── mod.rs              # Common infrastructure
├── fake_queen.rs       # Simulated queen-rbee
├── fake_hive.rs        # Simulated rbee-hive  
├── fake_worker.rs      # Simulated worker
└── fake_keeper.rs      # Simulated keeper
```

**Each fake binary:**
- Accepts commands via stdin/args
- Emits narration to stdout
- Supports job_id context
- Can simulate failures

**Example:**
```rust
// fake_worker.rs
#[tokio::main]
async fn main() {
    let job_id = std::env::var("JOB_ID").unwrap();
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async {
        n!("startup", "Worker starting");
        tokio::time::sleep(Duration::from_millis(100)).await;
        n!("ready", "Worker ready");
    }).await;
}
```

#### 1.3 SSE Test Utilities

**Create:** `narration-core/tests/sse_test_utils.rs`

```rust
/// Helper for testing SSE streams
pub struct SSEStreamTester {
    receiver: mpsc::Receiver<String>,
}

impl SSEStreamTester {
    /// Create from job_id
    pub fn from_job_id(job_id: &str) -> Self;
    
    /// Wait for next event (with timeout)
    pub async fn next_event(&mut self, timeout: Duration) -> Result<NarrationEvent>;
    
    /// Collect all events until [DONE]
    pub async fn collect_until_done(&mut self) -> Vec<NarrationEvent>;
    
    /// Assert event sequence
    pub fn assert_sequence(&self, events: &[&str]) -> Result<()>;
}
```

---

### Phase 2: Job-Server/Client Integration (Week 1)

#### 2.1 Job Creation and Streaming

**Create:** `narration-core/tests/job_server_integration.rs`

```rust
#[tokio::test]
async fn test_job_creation_with_narration() {
    let harness = NarrationTestHarness::start().await;
    
    // Create job
    let job_id = harness.submit_job(Operation::HiveList).await;
    
    // Get SSE stream
    let mut stream = harness.get_sse_stream(&job_id);
    
    // Emit narration from "fake service"
    harness.emit_narration(&job_id, "action", "Message");
    
    // Verify received via SSE
    let event = stream.next_event(Duration::from_secs(1)).await.unwrap();
    assert_eq!(event.action, "action");
    assert_eq!(event.human, "Message");
}

#[tokio::test]
async fn test_multiple_concurrent_jobs() {
    // Test job isolation with 10 concurrent jobs
}

#[tokio::test]
async fn test_sse_stream_cleanup_on_disconnect() {
    // Test channel cleanup when client disconnects
}

#[tokio::test]
async fn test_job_narration_after_completion() {
    // Test that [DONE] stops stream properly
}
```

#### 2.2 Job-Client Integration

**Create:** `job-client/tests/integration_tests.rs`

```rust
#[tokio::test]
async fn test_submit_and_stream_with_narration() {
    let server = spawn_test_server().await;
    let client = JobClient::new(server.url());
    
    let operation = Operation::HiveList;
    let mut lines = Vec::new();
    
    client.submit_and_stream(operation, |line| {
        lines.push(line.to_string());
        Ok(())
    }).await.unwrap();
    
    assert!(lines.len() > 0);
    assert!(lines.last().unwrap().contains("[DONE]"));
}
```

---

### Phase 3: Multi-Service E2E (Week 2)

#### 3.1 Keeper → Queen Flow

**Create:** `narration-core/tests/e2e_keeper_queen.rs`

```rust
#[tokio::test]
async fn test_keeper_to_queen_narration_flow() {
    // 1. Start fake queen
    let queen = FakeQueen::start().await;
    
    // 2. Keeper submits operation
    let client = JobClient::new(queen.url());
    let mut events = Vec::new();
    
    client.submit_and_stream(Operation::HiveList, |line| {
        events.push(line.to_string());
        Ok(())
    }).await.unwrap();
    
    // 3. Verify queen's narration was streamed
    assert!(events.iter().any(|e| e.contains("queen_router")));
}
```

#### 3.2 Queen → Hive Flow

**Create:** `narration-core/tests/e2e_queen_hive.rs`

```rust
#[tokio::test]
async fn test_queen_forwards_to_hive_with_narration() {
    // 1. Start fake hive
    let hive = FakeHive::start().await;
    
    // 2. Start fake queen (configured to forward to hive)
    let queen = FakeQueen::start_with_hive(hive.url()).await;
    
    // 3. Submit worker operation
    let client = JobClient::new(queen.url());
    let mut events = Vec::new();
    
    client.submit_and_stream(Operation::WorkerSpawn(req), |line| {
        events.push(line.to_string());
        Ok(())
    }).await.unwrap();
    
    // 4. Verify both queen and hive narration received
    assert!(events.iter().any(|e| e.contains("queen")));
    assert!(events.iter().any(|e| e.contains("hive")));
}
```

#### 3.3 Full Stack E2E

**Create:** `narration-core/tests/e2e_full_stack.rs`

```rust
#[tokio::test]
async fn test_full_stack_narration_flow() {
    // Keeper → Queen → Hive → Worker with end-to-end narration
    
    // 1. Start all fake services
    let worker = FakeWorker::start().await;
    let hive = FakeHive::start_with_worker(worker.port()).await;
    let queen = FakeQueen::start_with_hive(hive.url()).await;
    
    // 2. Keeper submits worker spawn
    let client = JobClient::new(queen.url());
    let mut events = Vec::new();
    
    client.submit_and_stream(Operation::WorkerSpawn(req), |line| {
        events.push(line.to_string());
        Ok(())
    }).await.unwrap();
    
    // 3. Verify narration from all layers
    assert_narration_sequence(&events, vec![
        ("queen", "forward"),
        ("hive", "spawn"),
        ("worker", "startup"),  // Via process capture!
        ("worker", "ready"),
        ("hive", "complete"),
    ]);
}
```

---

### Phase 4: Process Capture E2E (Week 2)

#### 4.1 Worker Stdout Capture

**Create:** `narration-core/tests/e2e_process_capture.rs`

```rust
#[tokio::test]
async fn test_worker_narration_captured_and_streamed() {
    let harness = NarrationTestHarness::start().await;
    
    // 1. Fake hive spawns fake worker with process capture
    let job_id = harness.submit_job(Operation::WorkerSpawn(req)).await;
    
    // 2. Fake worker emits narration to stdout
    let fake_worker = FakeWorker::spawn_with_capture(&job_id);
    
    // 3. Hive captures stdout via ProcessNarrationCapture
    // 4. Re-emits with job_id
    // 5. Flows through SSE to client
    
    let mut stream = harness.get_sse_stream(&job_id);
    
    // Verify worker narration received
    let events = stream.collect_until_done().await;
    assert!(events.iter().any(|e| e.actor == "proc-cap" && e.human.contains("worker")));
}

#[tokio::test]
async fn test_mixed_narration_and_regular_output() {
    // Test that ProcessNarrationCapture handles both narration
    // format and regular stdout correctly
}

#[tokio::test]
async fn test_worker_crash_narration() {
    // Test that worker crash messages are captured
}
```

---

### Phase 5: Context Propagation E2E (Week 3)

#### 5.1 Thread-Local Context Across Services

**Create:** `narration-core/tests/e2e_context_propagation.rs`

```rust
#[tokio::test]
async fn test_job_id_propagates_through_services() {
    // Test that job_id set in queen propagates to hive via HTTP headers
}

#[tokio::test]
async fn test_correlation_id_end_to_end() {
    // Test correlation_id from keeper → queen → hive → worker
}

#[tokio::test]
async fn test_nested_context_propagation() {
    // Test context inheritance in nested async tasks
}
```

---

### Phase 6: Performance and Load Testing (Week 3)

#### 6.1 High-Frequency Narration

**Create:** `narration-core/tests/performance_tests.rs`

```rust
#[tokio::test]
async fn test_high_frequency_narration() {
    // Emit 10,000 narration events rapidly
    // Verify no events lost
    // Measure throughput
}

#[tokio::test]
async fn test_many_concurrent_sse_streams() {
    // Create 100 concurrent SSE streams
    // Emit to all simultaneously
    // Verify isolation and delivery
}

#[tokio::test]
async fn test_channel_backpressure() {
    // Emit faster than consumer can process
    // Verify backpressure handling
}

#[tokio::test]
async fn test_memory_usage_under_load() {
    // Long-running test with many jobs
    // Monitor memory growth
}
```

---

### Phase 7: Failure Scenarios (Week 4)

#### 7.1 Network Failures

**Create:** `narration-core/tests/failure_scenarios.rs`

```rust
#[tokio::test]
async fn test_sse_stream_disconnect() {
    // Client disconnects mid-stream
    // Verify cleanup and no hanging channels
}

#[tokio::test]
async fn test_service_crash_during_narration() {
    // Hive crashes while streaming
    // Verify graceful failure and recovery
}

#[tokio::test]
async fn test_network_timeout_handling() {
    // Simulate network delays
    // Verify timeout behavior
}
```

---

### Phase 8: BDD Redesign (Week 4)

#### 8.1 Update Existing Features

**Update:** `bdd/features/cute_mode.feature`

```gherkin
Feature: Cute Mode Narration
  
  Background:
    Given the narration mode is set to "cute"
    Given a test harness is started
  
  Scenario: Simple cute narration
    When I emit narration with n!("action", "Message")
    Then the cute field should be populated
    And the cute field should contain emoji
  
  Scenario: Cute mode propagates through SSE
    Given a job is created with ID "job-123"
    When I emit cute narration in job context
    Then the SSE stream should contain cute narration
```

#### 8.2 New Feature: Multi-Service Flow

**Create:** `bdd/features/multi_service_flow.feature`

```gherkin
Feature: Multi-Service Narration Flow
  
  Scenario: Keeper to Queen to Hive
    Given fake queen is running on port 8500
    And fake hive is running on port 9000
    And queen is configured to forward to hive
    When keeper submits WorkerSpawn operation
    Then narration from queen should be received
    And narration from hive should be received
    And narration should maintain job_id context
  
  Scenario: Process capture in multi-service flow
    Given fake hive with process capture enabled
    When worker is spawned
    Then worker stdout should be captured
    And worker narration should flow through SSE
```

#### 8.3 New Feature: Job Server Integration

**Create:** `bdd/features/job_server_integration.feature`

```gherkin
Feature: Job Server Integration
  
  Scenario: Job creation with narration channel
    When a job is created via POST /v1/jobs
    Then a narration channel should be created
    And narration should route to the correct channel
  
  Scenario: Multiple concurrent jobs
    Given 10 jobs are created
    When each job emits narration
    Then narration should be isolated per job
    And no cross-contamination should occur
```

---

## Test Organization

### Directory Structure

```
narration-core/
├── tests/
│   ├── harness/
│   │   ├── mod.rs                    # Test harness infrastructure
│   │   ├── fake_services.rs          # Fake service implementations
│   │   └── sse_utils.rs              # SSE testing utilities
│   │
│   ├── fake_binaries/
│   │   ├── fake_queen.rs             # Simulated queen binary
│   │   ├── fake_hive.rs              # Simulated hive binary
│   │   ├── fake_worker.rs            # Simulated worker binary
│   │   └── mod.rs
│   │
│   ├── unit/                         # Move existing unit tests here
│   │   ├── macro_tests.rs
│   │   ├── narration_edge_cases.rs
│   │   └── format_consistency.rs
│   │
│   ├── integration/                  # Integration tests
│   │   ├── job_server_integration.rs
│   │   ├── job_client_integration.rs
│   │   ├── sse_lifecycle.rs
│   │   ├── context_propagation.rs
│   │   └── process_capture.rs
│   │
│   ├── e2e/                          # End-to-end tests
│   │   ├── keeper_queen.rs
│   │   ├── queen_hive.rs
│   │   ├── full_stack.rs
│   │   └── process_capture_e2e.rs
│   │
│   ├── performance/                  # Performance tests
│   │   ├── high_frequency.rs
│   │   ├── concurrent_streams.rs
│   │   └── memory_usage.rs
│   │
│   └── failure/                      # Failure scenario tests
│       ├── network_failures.rs
│       ├── service_crashes.rs
│       └── timeout_handling.rs
│
└── bdd/
    ├── features/
    │   ├── cute_mode.feature         # Updated
    │   ├── story_mode.feature        # Updated
    │   ├── multi_service_flow.feature    # New
    │   ├── job_server_integration.feature # New
    │   └── process_capture.feature       # New
    │
    └── src/
        ├── steps/
        │   ├── narration_steps.rs
        │   ├── service_steps.rs
        │   ├── sse_steps.rs
        │   └── job_steps.rs
        └── world.rs
```

---

## Test Metrics and Goals

### Coverage Goals

| Category | Current | Target | Priority |
|----------|---------|--------|----------|
| Unit Tests | 88 tests | 100 tests | Medium |
| Integration Tests | 30 tests | 80 tests | **High** |
| E2E Tests | 3 tests | 40 tests | **Critical** |
| BDD Scenarios | 15 scenarios | 50 scenarios | High |
| Performance Tests | 0 tests | 15 tests | Medium |
| Failure Tests | 0 tests | 20 tests | High |

### Success Criteria

- ✅ **80% E2E coverage** for multi-service flows
- ✅ **100% job-server/client integration** coverage
- ✅ **Process capture E2E** tests passing
- ✅ **Performance benchmarks** established
- ✅ **All BDD features** updated and passing
- ✅ **Failure scenarios** tested and documented

---

## Implementation Timeline

### Week 1: Foundation
- Day 1-2: Test harness infrastructure
- Day 3-4: Fake binary framework
- Day 5: Job-server/client integration tests

### Week 2: E2E Tests
- Day 1-2: Keeper → Queen flows
- Day 3-4: Queen → Hive flows
- Day 5: Full stack E2E + process capture

### Week 3: Performance & Context
- Day 1-2: Context propagation tests
- Day 3-4: Performance tests
- Day 5: Load testing

### Week 4: Failures & BDD
- Day 1-2: Failure scenario tests
- Day 3-4: BDD feature updates
- Day 5: Documentation and cleanup

---

## Running the Tests

### All Tests
```bash
cargo test -p observability-narration-core
```

### By Category
```bash
# Unit tests only
cargo test -p observability-narration-core --test 'unit/*'

# Integration tests
cargo test -p observability-narration-core --test 'integration/*'

# E2E tests (slower)
cargo test -p observability-narration-core --test 'e2e/*'

# Performance tests
cargo test -p observability-narration-core --test 'performance/*' --release

# BDD tests
cargo test -p observability-narration-core-bdd
```

### Specific Test
```bash
cargo test -p observability-narration-core test_full_stack_narration_flow
```

---

## Maintenance

### Test Review Schedule

- **Weekly:** Review failing tests
- **Monthly:** Update BDD scenarios for new features
- **Quarterly:** Performance benchmarks review
- **Yearly:** Full test suite audit

### Test Ownership

- **Unit Tests:** Core narration team
- **Integration Tests:** Service integration team
- **E2E Tests:** Platform team
- **BDD Tests:** Product team + QA
- **Performance Tests:** Performance engineering team

---

## Conclusion

This comprehensive testing plan transforms the narration system from having limited E2E coverage to having a robust, multi-layered test suite covering:

1. ✅ **147 existing tests** reviewed and categorized
2. 🆕 **200+ new tests** planned across all categories
3. 🆕 **Fake binary framework** for realistic testing
4. 🆕 **Test harness** for multi-service scenarios
5. 🆕 **Performance benchmarks** for load testing
6. 🆕 **Failure scenario** coverage
7. 🆕 **Updated BDD features** for all new functionality

**Total: ~350+ tests covering unit, integration, E2E, performance, and failure scenarios.**

This achieves world-class test coverage for a production-ready narration system.
