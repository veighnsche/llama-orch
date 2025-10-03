# FT-023: Integration Test Framework

**Team**: Foundation-Alpha  
**Sprint**: Sprint 4 - Integration + Gate 1  
**Size**: M (2 days)  
**Days**: 43 - 44  
**Spec Ref**: M0-W-1820

---

## Story Description

Establish integration test framework for end-to-end testing of HTTP → FFI → CUDA → FFI → HTTP flow. This validates the complete worker pipeline before Gate 1.

---

## Acceptance Criteria

- [ ] Test framework supports starting/stopping worker process
- [ ] Test fixtures provide mock model loading
- [ ] Helper functions for HTTP requests (execute, health, cancel)
- [ ] SSE stream parsing and validation
- [ ] Test isolation (each test gets clean worker instance)
- [ ] Timeout handling for long-running tests
- [ ] Test output includes logs and VRAM usage
- [ ] CI integration with CUDA feature flag

---

## Dependencies

### Upstream (Blocks This Story)
- FT-022: KV cache management (Expected completion: Day 42)
- FT-012: FFI integration tests (Expected completion: Day 25)

### Downstream (This Story Blocks)
- FT-024: HTTP-FFI-CUDA integration test needs framework
- FT-025: Gate 1 validation tests need framework

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/integration/framework.rs` - Test framework
- `bin/worker-orcd/tests/integration/helpers.rs` - Helper functions
- `bin/worker-orcd/tests/integration/fixtures.rs` - Test fixtures
- `bin/worker-orcd/tests/integration/mod.rs` - Module exports

### Key Interfaces
```rust
// framework.rs
use std::process::{Child, Command};
use std::time::Duration;

pub struct WorkerTestHarness {
    process: Child,
    port: u16,
    worker_id: String,
}

impl WorkerTestHarness {
    pub async fn start(model_path: &str, gpu_device: i32) -> Result<Self, TestError> {
        let port = find_free_port();
        let worker_id = uuid::Uuid::new_v4().to_string();
        
        let process = Command::new("target/debug/worker-orcd")
            .args(&[
                "--worker-id", &worker_id,
                "--model", model_path,
                "--gpu-device", &gpu_device.to_string(),
                "--port", &port.to_string(),
            ])
            .spawn()?;
        
        let harness = Self { process, port, worker_id };
        harness.wait_for_ready(Duration::from_secs(30)).await?;
        
        Ok(harness)
    }
    
    pub async fn execute(&self, req: ExecuteRequest) -> Result<SseStream, TestError>;
    pub async fn health(&self) -> Result<HealthResponse, TestError>;
    pub async fn cancel(&self, job_id: &str) -> Result<(), TestError>;
    
    pub fn base_url(&self) -> String {
        format!("http://localhost:{}", self.port)
    }
}

impl Drop for WorkerTestHarness {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

// helpers.rs
pub async fn collect_sse_events(stream: SseStream) -> Vec<InferenceEvent>;
pub fn assert_event_order(events: &[InferenceEvent]) -> Result<(), String>;
pub fn extract_tokens(events: &[InferenceEvent]) -> Vec<String>;
```

### Implementation Notes
- Worker process spawned per test (isolation)
- Automatic cleanup via Drop trait
- SSE parsing with timeout protection
- Helper assertions for common patterns
- Mock model support for fast tests

---

## Testing Strategy

### Unit Tests
- Test framework starts/stops worker
- Test SSE stream parsing
- Test event order validation
- Test timeout handling

### Integration Tests
- Test framework with real worker
- Test multiple sequential tests
- Test cleanup on test failure

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Framework documented with examples
- [ ] Helper functions tested
- [ ] CI integration verified
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §12.3 Integration Tests (M0-W-1820)
- Related Stories: FT-024 (integration test), FT-025 (gate 1)

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
