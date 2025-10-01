# e2e-haiku

**End-to-end tests with real models (TinyLlama for fast smoke tests)**

`test-harness/e2e-haiku` — E2E tests using small, fast models for smoke testing the full stack.

---

## What This Test Suite Does

e2e-haiku provides **end-to-end smoke tests** for llama-orch:

- **Real models** — TinyLlama-1.1B (fast, fits modest VRAM)
- **Full stack** — Orchestrator → Pool Manager → Engine → Adapter
- **Real inference** — Actual token generation, not mocks
- **Determinism** — Verify reproducible outputs
- **Quick feedback** — Fast enough for CI/dev loop

**Purpose**: Smoke test the full system with real models

---

## Test Model

### TinyLlama-1.1B-Chat-v1.0

- **Size**: ~600MB (Q4_K_M quantized)
- **VRAM**: ~1GB
- **Speed**: ~50 tokens/sec on modest GPU
- **Purpose**: Fast smoke tests, not production quality

**Why TinyLlama?**
- Small enough for CI runners
- Fast enough for quick feedback
- Real model, real inference
- Deterministic with seeds

---

## Test Scenarios

### Basic Inference

```rust
#[tokio::test]
async fn test_e2e_basic_inference() {
    // Start full stack
    let orchestrator = start_orchestrator().await;
    let pool_manager = start_pool_manager().await;
    let engine = provision_tinyllama().await;
    
    // Enqueue job
    let request = EnqueueRequest {
        prompt: "Hello, world!".to_string(),
        model: "TinyLlama-1.1B-Chat-v1.0".to_string(),
        max_tokens: 10,
        seed: Some(42),
        ..Default::default()
    };
    
    let job_id = orchestrator.enqueue(request).await?;
    
    // Wait for completion
    let status = orchestrator.wait_for_completion(&job_id).await?;
    assert_eq!(status.state, JobState::Completed);
    assert!(status.tokens_generated.unwrap() > 0);
}
```

### Determinism Verification

```rust
#[tokio::test]
async fn test_e2e_determinism() {
    let orchestrator = start_orchestrator().await;
    
    let request = EnqueueRequest {
        prompt: "Write a haiku about coding".to_string(),
        model: "TinyLlama-1.1B-Chat-v1.0".to_string(),
        max_tokens: 20,
        seed: Some(42),
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

### Multi-Job Concurrency

```rust
#[tokio::test]
async fn test_e2e_concurrent_jobs() {
    let orchestrator = start_orchestrator().await;
    
    // Enqueue 5 jobs concurrently
    let mut handles = Vec::new();
    for i in 0..5 {
        let orch = orchestrator.clone();
        let handle = tokio::spawn(async move {
            let request = EnqueueRequest {
                prompt: format!("Job {}", i),
                seed: Some(42 + i),
                ..Default::default()
            };
            orch.enqueue_and_wait(request).await
        });
        handles.push(handle);
    }
    
    // All should complete successfully
    for handle in handles {
        let result = handle.await?;
        assert!(result.is_ok());
    }
}
```

---

## Running Tests

### All E2E Tests

```bash
# Run all tests (requires TinyLlama model)
cargo test -p test-harness-e2e-haiku -- --nocapture
```

### Specific Test

```bash
# Basic inference
cargo test -p test-harness-e2e-haiku -- test_e2e_basic_inference --nocapture

# Determinism
cargo test -p test-harness-e2e-haiku -- test_e2e_determinism --nocapture

# Concurrency
cargo test -p test-harness-e2e-haiku -- test_e2e_concurrent_jobs --nocapture
```

---

## Setup

### Download TinyLlama

```bash
# Download model to workspace test-models directory
cd ../../.test-models/tinyllama
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

**Note**: Test models are stored in `.test-models/` at the workspace root. See `.docs/testing/TEST_MODELS.md` for details.

### Configure

```yaml
# config.yaml
pools:
  - id: haiku
    engine: llamacpp
    replicas: 1
    port: 8081
    model:
      id: local:/.test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    flags:
      - --parallel
      - "1"
      - --no-cont-batching
      - --metrics
```

---

## CI Integration

### GitHub Actions

```yaml
- name: Cache test models
  uses: actions/cache@v3
  with:
    path: .test-models
    key: test-models-${{ hashFiles('.docs/testing/TEST_MODELS.md') }}

- name: Download TinyLlama
  run: |
    mkdir -p .test-models/tinyllama
    if [ ! -f .test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf ]; then
      wget -O .test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
        https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    fi

- name: Run E2E tests
  run: cargo test -p test-harness-e2e-haiku -- --nocapture
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p test-harness-e2e-haiku -- --nocapture
```

---

## Dependencies

### Internal

- `orchestrator-core` — Orchestrator logic
- `pool-managerd` — Pool manager
- `provisioners-engine-provisioner` — Engine provisioning
- `worker-adapters-llamacpp-http` — llama.cpp adapter

### External

- `tokio` — Async runtime
- `reqwest` — HTTP client

---

## Specifications

Implements requirements from:
- ORCH-3050 (E2E testing)
- ORCH-3051 (Smoke tests)

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
