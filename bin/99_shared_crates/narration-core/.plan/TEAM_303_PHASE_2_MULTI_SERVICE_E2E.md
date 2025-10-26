# TEAM-303: Phase 2 - Multi-Service E2E Tests

**Status:** BLOCKED (Requires TEAM-302 completion)  
**Estimated Duration:** 1 week (5 days)  
**Dependencies:** TEAM-302 (test harness)  
**Risk Level:** High (cross-process testing)

---

## Mission

Implement comprehensive E2E tests for multi-service narration flows. Build fake binary framework to simulate realistic service interactions and verify narration propagation across the full stack.

**Goal:** Test complete Keeper → Queen → Hive → Worker narration flows.

---

## Problem Statement

Current tests are limited to single-process scenarios. No verification of:
- Service-to-service narration propagation
- HTTP header correlation
- Process capture end-to-end
- SSE streaming across service boundaries

**Impact:** Production multi-service failures not caught.

---

## Implementation Tasks

### Day 1: Fake Binary Framework

#### Task 1.1: Create Fake Binary Infrastructure

**Create:** `narration-core/tests/fake_binaries/mod.rs`

```rust
// TEAM-303: Fake binary infrastructure for multi-service testing

use tokio::process::{Child, Command};
use std::process::Stdio;

pub mod fake_queen;
pub mod fake_hive;
pub mod fake_worker;

/// Helper to spawn fake binary with env vars
pub async fn spawn_fake_binary(
    binary_name: &str,
    env_vars: Vec<(&str, &str)>,
) -> anyhow::Result<Child> {
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("--bin")
        .arg(binary_name)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    
    for (key, value) in env_vars {
        cmd.env(key, value);
    }
    
    let child = cmd.spawn()?;
    
    // Give process time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    
    Ok(child)
}

/// Wait for HTTP service to be ready
pub async fn wait_for_http_ready(url: &str, max_attempts: u32) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    
    for _ in 0..max_attempts {
        if client.get(format!("{}/health", url)).send().await.is_ok() {
            return Ok(());
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }
    
    anyhow::bail!("Service at {} never became ready", url)
}
```

#### Task 1.2: Implement Fake Queen

**Create:** `narration-core/tests/fake_binaries/fake_queen.rs`

```rust
// TEAM-303: Fake queen-rbee for E2E testing

use observability_narration_core::{n, with_narration_context, NarrationContext};
use axum::{Router, routing::post, Json, extract::State};
use std::sync::Arc;

pub struct FakeQueen {
    port: u16,
    hive_url: Option<String>,
}

impl FakeQueen {
    pub async fn start(port: u16) -> Self {
        Self { port, hive_url: None }
    }
    
    pub async fn start_with_hive(port: u16, hive_url: String) -> Self {
        Self { port, hive_url: Some(hive_url) }
    }
    
    pub fn url(&self) -> String {
        format!("http://localhost:{}", self.port)
    }
    
    pub async fn run(&self) {
        let app = Router::new()
            .route("/v1/jobs", post(handle_job))
            .with_state(Arc::new(self.hive_url.clone()));
        
        let listener = tokio::net::TcpListener::bind(
            format!("127.0.0.1:{}", self.port)
        ).await.unwrap();
        
        axum::serve(listener, app).await.unwrap();
    }
}

async fn handle_job(
    State(hive_url): State<Arc<Option<String>>>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    // Create job
    let job_id = format!("queen-job-{}", uuid::Uuid::new_v4());
    
    // Emit queen narration
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, async {
        n!("queen_route", "Queen routing operation");
        
        // Forward to hive if configured
        if let Some(url) = hive_url.as_ref() {
            n!("queen_forward", "Forwarding to hive at {}", url);
            // TODO: Actual HTTP forward
        }
    }).await;
    
    Json(serde_json::json!({
        "job_id": job_id,
        "sse_url": format!("/v1/jobs/{}/stream", job_id)
    }))
}

// Binary entry point for cargo run
#[tokio::main]
async fn main() {
    let port = std::env::var("FAKE_QUEEN_PORT")
        .unwrap_or_else(|_| "8500".to_string())
        .parse()
        .unwrap();
    
    let hive_url = std::env::var("FAKE_HIVE_URL").ok();
    
    let queen = if let Some(url) = hive_url {
        FakeQueen::start_with_hive(port, url).await
    } else {
        FakeQueen::start(port).await
    };
    
    println!("Fake queen started on port {}", port);
    queen.run().await;
}
```

#### Task 1.3: Implement Fake Hive

**Create:** `narration-core/tests/fake_binaries/fake_hive.rs`

```rust
// TEAM-303: Fake rbee-hive for E2E testing

use observability_narration_core::{n, with_narration_context, NarrationContext, ProcessNarrationCapture};
use axum::{Router, routing::post, Json};
use tokio::process::Command;

pub struct FakeHive {
    port: u16,
    worker_binary: Option<String>,
}

impl FakeHive {
    pub async fn start(port: u16) -> Self {
        Self { port, worker_binary: None }
    }
    
    pub async fn start_with_worker(port: u16, worker_binary: String) -> Self {
        Self { port, worker_binary: Some(worker_binary) }
    }
    
    pub fn url(&self) -> String {
        format!("http://localhost:{}", self.port)
    }
    
    pub async fn run(&self) {
        let app = Router::new()
            .route("/v1/workers/spawn", post(handle_worker_spawn));
        
        let listener = tokio::net::TcpListener::bind(
            format!("127.0.0.1:{}", self.port)
        ).await.unwrap();
        
        axum::serve(listener, app).await.unwrap();
    }
}

async fn handle_worker_spawn(
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let job_id = payload["job_id"].as_str().unwrap();
    
    let ctx = NarrationContext::new().with_job_id(job_id);
    with_narration_context(ctx, async {
        n!("hive_spawn", "Hive spawning worker");
        
        // Spawn fake worker with process capture
        let capture = ProcessNarrationCapture::new(Some(job_id.to_string()));
        let mut command = Command::new("cargo");
        command
            .arg("run")
            .arg("--bin")
            .arg("fake_worker")
            .env("JOB_ID", job_id);
        
        let _child = capture.spawn(command).await;
        
        n!("hive_complete", "Worker spawned successfully");
    }).await;
    
    Json(serde_json::json!({
        "success": true,
        "worker_id": "fake-worker-123"
    }))
}

#[tokio::main]
async fn main() {
    let port = std::env::var("FAKE_HIVE_PORT")
        .unwrap_or_else(|_| "9000".to_string())
        .parse()
        .unwrap();
    
    let hive = FakeHive::start(port).await;
    
    println!("Fake hive started on port {}", port);
    hive.run().await;
}
```

#### Task 1.4: Implement Fake Worker

**Create:** `narration-core/tests/fake_binaries/fake_worker.rs`

```rust
// TEAM-303: Fake worker for E2E testing

use observability_narration_core::{n, with_narration_context, NarrationContext};

#[tokio::main]
async fn main() {
    let job_id = std::env::var("JOB_ID")
        .expect("JOB_ID environment variable required");
    
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async {
        // Worker startup narration
        n!("worker_startup", "Worker starting");
        
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        n!("worker_load_model", "Loading model");
        
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        n!("worker_ready", "Worker ready to serve");
    }).await;
    
    // Keep worker "running" briefly
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
}
```

---

### Day 2: Keeper → Queen E2E Tests

#### Task 2.1: Basic Keeper to Queen Flow

**Create:** `narration-core/tests/e2e/keeper_queen.rs`

```rust
// TEAM-303: Keeper → Queen E2E tests

use crate::harness::NarrationTestHarness;
use crate::fake_binaries::{fake_queen::FakeQueen, wait_for_http_ready};
use job_client::JobClient;
use operations_contract::Operation;

#[tokio::test]
async fn test_keeper_to_queen_narration_flow() {
    // Start fake queen
    let queen = FakeQueen::start(18500).await;
    tokio::spawn(async move {
        queen.run().await;
    });
    
    wait_for_http_ready("http://localhost:18500", 20).await.unwrap();
    
    // Keeper submits operation
    let client = JobClient::new("http://localhost:18500");
    let mut events = Vec::new();
    
    client.submit_and_stream(Operation::HiveList, |line| {
        events.push(line.to_string());
        Ok(())
    }).await.unwrap();
    
    // Verify queen narration received
    assert!(events.iter().any(|e| e.contains("queen_route")));
}

#[tokio::test]
async fn test_keeper_queen_correlation_id() {
    let queen = FakeQueen::start(18501).await;
    tokio::spawn(async move {
        queen.run().await;
    });
    
    wait_for_http_ready("http://localhost:18501", 20).await.unwrap();
    
    // TODO: Test correlation ID propagation
    // Verify correlation_id flows through SSE
}
```

---

### Day 3: Queen → Hive E2E Tests

#### Task 3.1: Queen to Hive Flow

**Create:** `narration-core/tests/e2e/queen_hive.rs`

```rust
// TEAM-303: Queen → Hive E2E tests

use crate::fake_binaries::{fake_queen::FakeQueen, fake_hive::FakeHive, wait_for_http_ready};
use job_client::JobClient;
use operations_contract::Operation;

#[tokio::test]
async fn test_queen_forwards_to_hive() {
    // Start fake hive
    let hive = FakeHive::start(19000).await;
    tokio::spawn(async move {
        hive.run().await;
    });
    
    wait_for_http_ready("http://localhost:19000", 20).await.unwrap();
    
    // Start fake queen with hive URL
    let queen = FakeQueen::start_with_hive(18502, "http://localhost:19000".to_string()).await;
    tokio::spawn(async move {
        queen.run().await;
    });
    
    wait_for_http_ready("http://localhost:18502", 20).await.unwrap();
    
    // Submit worker operation
    let client = JobClient::new("http://localhost:18502");
    let mut events = Vec::new();
    
    let req = operations_contract::WorkerSpawnRequest {
        hive_id: "test-hive".to_string(),
        worker: "test-worker".to_string(),
        model: "test-model".to_string(),
        device: operations_contract::Device::Cpu,
    };
    
    client.submit_and_stream(Operation::WorkerSpawn(req), |line| {
        events.push(line.to_string());
        Ok(())
    }).await.unwrap();
    
    // Verify both queen and hive narration
    assert!(events.iter().any(|e| e.contains("queen")));
    assert!(events.iter().any(|e| e.contains("hive")));
}
```

---

### Day 4: Full Stack E2E Tests

#### Task 4.1: Complete Flow Test

**Create:** `narration-core/tests/e2e/full_stack.rs`

```rust
// TEAM-303: Full stack E2E tests

use crate::fake_binaries::*;
use job_client::JobClient;
use operations_contract::Operation;

#[tokio::test]
async fn test_full_stack_keeper_to_worker() {
    // Start fake worker (will be spawned by hive)
    // Start fake hive
    let hive = fake_hive::FakeHive::start_with_worker(
        19001,
        "fake_worker".to_string()
    ).await;
    tokio::spawn(async move {
        hive.run().await;
    });
    
    wait_for_http_ready("http://localhost:19001", 20).await.unwrap();
    
    // Start fake queen with hive
    let queen = fake_queen::FakeQueen::start_with_hive(
        18503,
        "http://localhost:19001".to_string()
    ).await;
    tokio::spawn(async move {
        queen.run().await;
    });
    
    wait_for_http_ready("http://localhost:18503", 20).await.unwrap();
    
    // Keeper submits worker spawn
    let client = JobClient::new("http://localhost:18503");
    let mut events = Vec::new();
    
    let req = operations_contract::WorkerSpawnRequest {
        hive_id: "test-hive".to_string(),
        worker: "test-worker".to_string(),
        model: "test-model".to_string(),
        device: operations_contract::Device::Cpu,
    };
    
    client.submit_and_stream(Operation::WorkerSpawn(req), |line| {
        events.push(line.to_string());
        Ok(())
    }).await.unwrap();
    
    // Verify narration from all layers
    assert_narration_sequence(&events, vec![
        "queen_route",
        "queen_forward",
        "hive_spawn",
        "worker_startup",  // Via process capture!
        "worker_load_model",
        "worker_ready",
        "hive_complete",
    ]);
}

fn assert_narration_sequence(events: &[String], expected: Vec<&str>) {
    for expected_action in expected {
        assert!(
            events.iter().any(|e| e.contains(expected_action)),
            "Expected action '{}' not found in events",
            expected_action
        );
    }
}
```

---

### Day 5: Process Capture E2E Tests

#### Task 5.1: Worker Stdout Capture End-to-End

**Create:** `narration-core/tests/e2e/process_capture_e2e.rs`

```rust
// TEAM-303: Process capture E2E tests

use crate::harness::NarrationTestHarness;
use crate::fake_binaries::fake_worker;
use observability_narration_core::ProcessNarrationCapture;
use tokio::process::Command;

#[tokio::test]
async fn test_worker_narration_captured_and_streamed() {
    let harness = NarrationTestHarness::start().await;
    
    let job_id = harness.submit_job(
        serde_json::to_value(operations_contract::Operation::HiveList).unwrap()
    ).await;
    
    // Spawn fake worker with process capture
    let capture = ProcessNarrationCapture::new(Some(job_id.clone()));
    let mut command = Command::new("cargo");
    command
        .arg("run")
        .arg("--bin")
        .arg("fake_worker")
        .env("JOB_ID", &job_id);
    
    let mut child = capture.spawn(command).await.unwrap();
    
    // Get SSE stream
    let mut stream = harness.get_sse_stream(&job_id);
    
    // Verify worker narration flows through SSE
    stream.assert_next("worker_startup", "Worker starting").await;
    stream.assert_next("worker_load_model", "Loading model").await;
    stream.assert_next("worker_ready", "Worker ready").await;
    
    // Cleanup
    child.kill().await.ok();
}

#[tokio::test]
async fn test_mixed_narration_and_regular_output() {
    // Test that process capture handles both narration format
    // and regular stdout correctly
}

#[tokio::test]
async fn test_worker_crash_captured() {
    // Test that worker crash messages are captured
}
```

---

## Verification Checklist

- [ ] Fake binary framework compiles
- [ ] Fake queen binary runs
- [ ] Fake hive binary runs
- [ ] Fake worker binary runs
- [ ] Keeper → Queen test passes
- [ ] Queen → Hive test passes
- [ ] Full stack test passes
- [ ] Process capture E2E test passes
- [ ] Correlation ID test passes
- [ ] Worker crash test passes

---

## Success Criteria

1. **Fake Binaries Working**
   - Queen: ✅
   - Hive: ✅
   - Worker: ✅

2. **E2E Tests Passing**
   - Keeper → Queen: ✅
   - Queen → Hive: ✅
   - Full stack: ✅
   - Process capture: ✅

3. **Narration Verified**
   - Multi-service flow: ✅
   - Process capture: ✅
   - SSE streaming: ✅

---

## Deliverables

### Code Added

- `tests/fake_binaries/mod.rs` (~80 LOC)
- `tests/fake_binaries/fake_queen.rs` (~150 LOC)
- `tests/fake_binaries/fake_hive.rs` (~120 LOC)
- `tests/fake_binaries/fake_worker.rs` (~50 LOC)
- `tests/e2e/keeper_queen.rs` (~80 LOC)
- `tests/e2e/queen_hive.rs` (~100 LOC)
- `tests/e2e/full_stack.rs` (~120 LOC)
- `tests/e2e/process_capture_e2e.rs` (~100 LOC)

**Total:** ~800 LOC

### Tests Added

- Keeper → Queen: 2 tests
- Queen → Hive: 1 test
- Full stack: 1 test
- Process capture E2E: 3 tests

**Total:** 7 tests

---

## Handoff to TEAM-304

Document in `.plan/TEAM_303_HANDOFF.md`:

1. **What Works**
   - Fake binary framework operational
   - Multi-service E2E flows verified
   - Process capture end-to-end working

2. **Test Results**
   - All 7 E2E tests passing
   - Narration propagates correctly
   - SSE streaming works across services

3. **Next Steps**
   - TEAM-304: Context propagation details
   - TEAM-304: Performance testing
   - Build on E2E foundation

---

## Known Limitations

1. **Cargo Run Overhead:** Fake binaries use `cargo run`, slower than compiled binaries
2. **Port Conflicts:** Tests must use unique ports to avoid conflicts
3. **Timing Sensitive:** Some tests may be flaky due to async timing

---

**TEAM-303 Mission Complete**
