# worker-common BDD Tests

Behavior-Driven Development tests for common worker types and utilities.

## Running Tests

```bash
# Run all BDD tests
cargo run --bin bdd-runner

# Run specific feature
cargo run --bin bdd-runner -- tests/features/sampling_config.feature
```

## Features

- **Sampling Configuration** — Verify sampling parameter behavior
  - Greedy vs advanced sampling detection
  - Parameter validation
  - Sampling mode classification

- **Error Handling** — Verify error classification
  - Retriability detection
  - HTTP status code mapping
  - Error code stability

- **Ready Callback** — Verify worker readiness protocol
  - Memory usage reporting
  - Memory architecture reporting (vram-only vs unified)
  - Callback payload validation

## Critical Behaviors

These tests verify **critical worker contract behaviors** that must work correctly across all worker implementations (worker-orcd, worker-aarmd, etc.):

1. **Sampling configuration** affects inference quality
2. **Error classification** affects orchestrator retry logic
3. **Ready callbacks** affect pool manager scheduling

**Consequence of undertesting**: Production failures, incorrect retries, scheduling bugs.
