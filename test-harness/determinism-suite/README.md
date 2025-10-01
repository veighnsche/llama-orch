# determinism-suite

**Determinism verification tests for reproducible inference**

`test-harness/determinism-suite` — Tests that verify identical prompts + seeds produce identical outputs.

---

## What This Test Suite Does

determinism-suite provides **reproducibility testing** for llama-orch:

- **Seed verification** — Same seed produces same tokens
- **Token-by-token comparison** — Exact token sequence matching
- **Metadata validation** — Engine version, model, parameters
- **Cross-run consistency** — Multiple runs produce identical results
- **Proof bundle generation** — Record seeds and outputs

**Purpose**: Guarantee deterministic inference for multi-agent workflows

---

## Determinism Requirements

### Same Input → Same Output

Given:
- Same prompt
- Same seed
- Same model
- Same engine version
- Same parameters (temperature, top_p, etc.)

Then:
- Token sequence must be identical
- Token IDs must be identical
- Logprobs must be identical (if available)

---

## Test Scenarios

### Basic Determinism

```rust
#[tokio::test]
async fn test_determinism_basic() {
    let orchestrator = start_orchestrator().await;
    
    let request = EnqueueRequest {
        prompt: "Hello, world!".to_string(),
        model: "llama-3.1-8b-instruct".to_string(),
        max_tokens: 100,
        seed: Some(42),
        temperature: Some(0.7),
        ..Default::default()
    };
    
    // Run 1
    let tokens1 = orchestrator.enqueue_and_collect(request.clone()).await?;
    
    // Run 2
    let tokens2 = orchestrator.enqueue_and_collect(request.clone()).await?;
    
    // Must be identical
    assert_eq!(tokens1, tokens2);
}
```

### Cross-Session Determinism

```rust
#[tokio::test]
async fn test_determinism_cross_session() {
    let orchestrator = start_orchestrator().await;
    
    // Session 1
    let tokens1 = run_with_seed(42).await?;
    
    // Restart orchestrator
    orchestrator.stop().await;
    let orchestrator = start_orchestrator().await;
    
    // Session 2
    let tokens2 = run_with_seed(42).await?;
    
    // Must be identical across sessions
    assert_eq!(tokens1, tokens2);
}
```

### Multi-Run Consistency

```rust
#[tokio::test]
async fn test_determinism_multi_run() {
    let orchestrator = start_orchestrator().await;
    
    let request = EnqueueRequest {
        prompt: "Hello, world!".to_string(),
        seed: Some(42),
        ..Default::default()
    };
    
    // Run 10 times
    let mut results = Vec::new();
    for _ in 0..10 {
        let tokens = orchestrator.enqueue_and_collect(request.clone()).await?;
        results.push(tokens);
    }
    
    // All runs must be identical
    for i in 1..results.len() {
        assert_eq!(results[0], results[i], "Run {} differs from run 0", i);
    }
}
```

---

## Running Tests

### All Determinism Tests

```bash
# Run all tests
cargo test -p test-harness-determinism-suite -- --nocapture
```

### Specific Test

```bash
# Basic determinism
cargo test -p test-harness-determinism-suite -- test_determinism_basic --nocapture

# Cross-session
cargo test -p test-harness-determinism-suite -- test_determinism_cross_session --nocapture

# Multi-run
cargo test -p test-harness-determinism-suite -- test_determinism_multi_run --nocapture
```

---

## Proof Bundles

Tests generate proof bundles with:

- **Seeds** — RNG seeds used
- **Outputs** — Token sequences
- **Metadata** — Engine version, model, parameters
- **Timestamps** — Run timestamps

Example proof bundle:

```json
{
  "run_id": "det-001",
  "seed": 42,
  "prompt": "Hello, world!",
  "model": "llama-3.1-8b-instruct",
  "engine_version": "b1234-cuda",
  "tokens": ["Hello", " there", "!"],
  "token_ids": [12345, 12346, 12347],
  "timestamp": "2025-10-01T00:00:00Z"
}
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p test-harness-determinism-suite -- --nocapture
```

---

## Dependencies

### Internal

- `orchestrator-core` — Orchestrator logic
- `proof-bundle` — Test artifact generation

### External

- `tokio` — Async runtime
- `serde` — Serialization

---

## Specifications

Implements requirements from:
- ORCH-3050 (Determinism testing)
- ORCH-3051 (Reproducibility)

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
