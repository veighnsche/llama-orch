# worker-compute BDD Tests

Behavior-Driven Development tests for compute backend trait.

## Running Tests

```bash
# Run all BDD tests
cd bin/worker-crates/worker-compute/bdd
cargo run --bin bdd-runner

# Run specific feature
cargo run --bin bdd-runner -- tests/features/compute_backend.feature
```

## Features

- **Compute Backend Initialization** — Verify device initialization
  - Device ID validation
  - Multi-device support
  - Error handling for invalid devices

- **Model Loading** — Verify model loading behavior
  - Path validation
  - Format validation (GGUF)
  - Memory usage reporting

- **Inference Execution** — Verify inference behavior
  - Parameter validation (temperature, max_tokens)
  - Token generation
  - Max tokens enforcement

## Critical Behaviors

These tests verify **critical compute backend contract behaviors** that must work correctly across all implementations (CUDA, Metal, ROCm):

1. **Device initialization** affects worker startup
2. **Model loading** affects memory allocation
3. **Inference execution** affects generation quality

**Consequence of undertesting**: Worker crashes, OOM errors, incorrect inference.
