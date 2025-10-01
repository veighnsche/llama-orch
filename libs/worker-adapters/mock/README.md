# mock

**Mock adapter for testing**

`libs/worker-adapters/mock` — WorkerAdapter implementation for testing without a real engine.

---

## What This Adapter Does

mock provides **testing infrastructure** for llama-orch:

- **No real engine** — Simulates engine behavior without external dependencies
- **Configurable responses** — Control output tokens, timing, errors
- **Deterministic** — Reproducible behavior for tests
- **Fast** — No network calls or GPU operations
- **Test-friendly** — Easy to configure for different scenarios

**Engine**: None (in-memory simulation)

---

## Usage

### Create Adapter

```rust
use worker_adapters_mock::MockAdapter;

// Default configuration
let adapter = MockAdapter::new();

// Custom configuration
let adapter = MockAdapter::builder()
    .with_tokens(vec!["Hello", " world", "!"])
    .with_delay_ms(10)
    .build();
```

### Submit Task

```rust
use worker_adapters_adapter_api::{WorkerAdapter, TaskRequest};

let task = TaskRequest {
    job_id: "job-123".to_string(),
    model: "mock-model".to_string(),
    prompt: "Hello, world!".to_string(),
    max_tokens: 100,
    temperature: Some(0.7),
    seed: Some(42),
    session_id: None,
};

let mut stream = adapter.submit(task).await?;

while let Some(event) = stream.receiver.recv().await {
    match event {
        TokenEvent::Started { engine_version } => {
            println!("Started: {}", engine_version);
        }
        TokenEvent::Token { text, index } => {
            print!("{}", text);
        }
        TokenEvent::End { metrics } => {
            println!("\nDone: {} tokens", metrics.tokens_generated);
        }
        TokenEvent::Error { error } => {
            eprintln!("Error: {}", error);
        }
    }
}
```

---

## Configuration

### MockAdapterBuilder

```rust
use worker_adapters_mock::MockAdapter;

let adapter = MockAdapter::builder()
    // Set tokens to emit
    .with_tokens(vec!["Hello", " there", "!"])
    
    // Set delay between tokens (ms)
    .with_delay_ms(10)
    
    // Set engine version
    .with_engine_version("mock-1.0.0")
    
    // Simulate error after N tokens
    .with_error_after(5)
    
    // Set health state
    .with_health_state(HealthState::Healthy)
    
    .build();
```

---

## Test Scenarios

### Success Case

```rust
#[tokio::test]
async fn test_successful_generation() {
    let adapter = MockAdapter::builder()
        .with_tokens(vec!["Hello", " world"])
        .build();
    
    let task = TaskRequest { /* ... */ };
    let mut stream = adapter.submit(task).await.unwrap();
    
    // Verify tokens
    assert_eq!(stream.next().await, Some(TokenEvent::Started { /* ... */ }));
    assert_eq!(stream.next().await, Some(TokenEvent::Token { text: "Hello", index: 0 }));
    assert_eq!(stream.next().await, Some(TokenEvent::Token { text: " world", index: 1 }));
    assert_eq!(stream.next().await, Some(TokenEvent::End { /* ... */ }));
}
```

### Error Case

```rust
#[tokio::test]
async fn test_error_during_generation() {
    let adapter = MockAdapter::builder()
        .with_tokens(vec!["Hello"])
        .with_error_after(1)
        .build();
    
    let task = TaskRequest { /* ... */ };
    let mut stream = adapter.submit(task).await.unwrap();
    
    // Verify error
    assert_eq!(stream.next().await, Some(TokenEvent::Started { /* ... */ }));
    assert_eq!(stream.next().await, Some(TokenEvent::Token { text: "Hello", index: 0 }));
    assert!(matches!(stream.next().await, Some(TokenEvent::Error { .. })));
}
```

### Slow Response

```rust
#[tokio::test]
async fn test_slow_generation() {
    let adapter = MockAdapter::builder()
        .with_tokens(vec!["Slow", " response"])
        .with_delay_ms(100) // 100ms between tokens
        .build();
    
    let start = Instant::now();
    let task = TaskRequest { /* ... */ };
    let mut stream = adapter.submit(task).await.unwrap();
    
    // Consume all tokens
    while let Some(_) = stream.next().await {}
    
    // Verify timing
    assert!(start.elapsed() >= Duration::from_millis(200));
}
```

---

## Health Check

```rust
let adapter = MockAdapter::builder()
    .with_health_state(HealthState::Healthy)
    .build();

let health = adapter.health().await?;
assert_eq!(health.state, HealthState::Healthy);
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p worker-adapters-mock -- --nocapture

# Run specific test
cargo test -p worker-adapters-mock -- test_mock_adapter --nocapture
```

---

## Dependencies

### Internal

- `worker-adapters-adapter-api` — WorkerAdapter trait

### External

- `tokio` — Async runtime
- `async-trait` — Async trait support

---

## Use Cases

### Unit Tests

Test orchestrator logic without real engines:

```rust
#[tokio::test]
async fn test_orchestrator_dispatch() {
    let mock_adapter = MockAdapter::builder()
        .with_tokens(vec!["Test", " response"])
        .build();
    
    let orchestrator = Orchestrator::new();
    orchestrator.register_adapter("mock", mock_adapter);
    
    // Test dispatch logic
}
```

### Integration Tests

Test adapter integration without external dependencies:

```rust
#[tokio::test]
async fn test_adapter_host() {
    let adapter_host = AdapterHost::new();
    adapter_host.register("mock", MockAdapter::new());
    
    // Test adapter host logic
}
```

### BDD Tests

Use in Cucumber scenarios:

```gherkin
Given a mock adapter with tokens "Hello world"
When I submit a task
Then I should receive 2 tokens
```

---

## Specifications

Implements requirements from:
- ORCH-3054 (Adapter registry)
- ORCH-3055 (Adapter dispatch)
- ORCH-3056 (Adapter lifecycle)
- ORCH-3057 (Health checks)
- ORCH-3058 (Error handling)

See `.specs/00_llama-orch.md` for full requirements.

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
