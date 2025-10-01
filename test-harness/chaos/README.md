# chaos

**Chaos engineering tests for resilience and fault tolerance**

`test-harness/chaos` — Chaos tests that inject failures to verify system resilience.

---

## What This Test Suite Does

chaos provides **fault injection testing** for llama-orch:

- **Network failures** — Simulate connection drops, timeouts
- **Process crashes** — Kill engines, orchestrator, pool-manager
- **Resource exhaustion** — OOM, disk full, CPU saturation
- **Partial failures** — Some replicas fail, others succeed
- **Recovery validation** — Verify graceful degradation and recovery

**Purpose**: Ensure system handles failures gracefully

---

## Chaos Scenarios

### Network Failures

- **Connection timeout** — Simulate slow/unresponsive engines
- **Connection refused** — Engine not listening
- **Intermittent failures** — Random connection drops
- **Partial network partition** — Some nodes unreachable

### Process Crashes

- **Engine crash** — SIGKILL during inference
- **Orchestrator crash** — SIGKILL during dispatch
- **Pool-manager crash** — SIGKILL during provisioning
- **Graceful shutdown** — SIGTERM with cleanup

### Resource Exhaustion

- **Out of memory** — Simulate OOM conditions
- **Disk full** — No space for models/logs
- **CPU saturation** — 100% CPU usage
- **GPU unavailable** — No GPU devices

### Partial Failures

- **Some replicas fail** — 1/3 replicas crash
- **Rolling failures** — Sequential replica failures
- **Cascading failures** — Failure triggers more failures

---

## Running Tests

### All Chaos Tests

```bash
# Run all chaos tests
cargo test -p test-harness-chaos -- --nocapture
```

### Specific Scenario

```bash
# Network failures
cargo test -p test-harness-chaos -- test_network_timeout --nocapture

# Process crashes
cargo test -p test-harness-chaos -- test_engine_crash --nocapture

# Resource exhaustion
cargo test -p test-harness-chaos -- test_oom --nocapture
```

---

## Test Examples

### Network Timeout

```rust
#[tokio::test]
async fn test_network_timeout() {
    let orchestrator = start_orchestrator().await;
    
    // Inject network delay
    inject_network_delay(Duration::from_secs(30)).await;
    
    // Enqueue job
    let result = orchestrator.enqueue(job).await;
    
    // Should timeout gracefully
    assert!(matches!(result, Err(Error::Timeout)));
    
    // System should recover
    clear_network_delay().await;
    let result = orchestrator.enqueue(job).await;
    assert!(result.is_ok());
}
```

### Engine Crash

```rust
#[tokio::test]
async fn test_engine_crash() {
    let orchestrator = start_orchestrator().await;
    let engine = start_engine().await;
    
    // Start job
    let job_id = orchestrator.enqueue(job).await?;
    
    // Kill engine mid-inference
    engine.kill().await;
    
    // Job should fail gracefully
    let status = orchestrator.get_status(&job_id).await?;
    assert_eq!(status.state, JobState::Failed);
    
    // Orchestrator should remain healthy
    let health = orchestrator.health().await?;
    assert_eq!(health.state, HealthState::Healthy);
}
```

### Partial Replica Failure

```rust
#[tokio::test]
async fn test_partial_replica_failure() {
    let orchestrator = start_orchestrator().await;
    let replicas = start_replicas(3).await;
    
    // Kill 1/3 replicas
    replicas[0].kill().await;
    
    // Jobs should still succeed on remaining replicas
    let job_id = orchestrator.enqueue(job).await?;
    let status = orchestrator.get_status(&job_id).await?;
    assert_eq!(status.state, JobState::Completed);
    
    // Pool should be degraded but operational
    let pool_status = orchestrator.get_pool_status("default").await?;
    assert_eq!(pool_status.state, PoolState::Degraded);
}
```

---

## Chaos Injection

### Network Delay

```rust
// Inject 5s delay on all network calls
inject_network_delay(Duration::from_secs(5)).await;

// Clear delay
clear_network_delay().await;
```

### Process Kill

```rust
// SIGKILL (immediate)
process.kill().await;

// SIGTERM (graceful)
process.terminate().await;
```

### Resource Limit

```rust
// Limit memory to 100MB
set_memory_limit(100 * 1024 * 1024).await;

// Remove limit
clear_memory_limit().await;
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p test-harness-chaos -- --nocapture
```

---

## Dependencies

### Internal

- `orchestrator-core` — Orchestrator logic
- `pool-managerd` — Pool manager
- `worker-adapters-mock` — Mock adapter

### External

- `tokio` — Async runtime
- `tokio::time` — Time manipulation

---

## Specifications

Implements requirements from:
- ORCH-3050 (Chaos testing)
- ORCH-3051 (Fault tolerance)

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
