# bdd

**Standalone BDD test runner for Cucumber features**

`test-harness/bdd` — Binary `bdd-runner` that executes Cucumber feature files across the monorepo.

---

## What This Tool Does

bdd provides **centralized BDD test execution** for llama-orch:

- **Standalone runner** — Binary that runs all Cucumber features
- **Feature targeting** — Run specific features or directories
- **World setup** — Shared test context and fixtures
- **Proof bundles** — Test artifact generation
- **Parallel execution** — Run scenarios concurrently

**Used by**: All BDD test suites in the monorepo

---

## Usage

### Run All Features

```bash
# Run all BDD tests
cargo run -p test-harness-bdd --bin bdd-runner
```

### Run Specific Feature

```bash
# Target specific feature file
LLORCH_BDD_FEATURE_PATH=tests/features/enqueue.feature \
  cargo run -p test-harness-bdd --bin bdd-runner

# Target directory
LLORCH_BDD_FEATURE_PATH=tests/features/orchestrator/ \
  cargo run -p test-harness-bdd --bin bdd-runner
```

### With Verbose Output

```bash
# Enable debug logging
RUST_LOG=debug cargo run -p test-harness-bdd --bin bdd-runner
```

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLORCH_BDD_FEATURE_PATH` | Target specific feature/directory | All features |
| `RUST_LOG` | Logging level | `info` |
| `LLORCH_PROOF_DIR` | Proof bundle output directory | `.proof_bundle` |
| `LLORCH_RUN_ID` | Test run identifier | Auto-generated |

---

## World Context

The BDD runner provides a shared `World` context for all scenarios:

```rust
use test_harness_bdd::World;

#[given("a running orchestrator")]
async fn given_orchestrator(world: &mut World) {
    world.orchestrator = Some(start_orchestrator().await?);
}

#[when("I enqueue a job")]
async fn when_enqueue(world: &mut World) {
    let response = world.orchestrator
        .as_ref()
        .unwrap()
        .enqueue(job)
        .await?;
    world.job_id = Some(response.job_id);
}

#[then("the job should be queued")]
async fn then_queued(world: &mut World) {
    let status = world.orchestrator
        .as_ref()
        .unwrap()
        .get_status(world.job_id.as_ref().unwrap())
        .await?;
    assert_eq!(status.state, JobState::Queued);
}
```

---

## Feature Organization

Features are organized by component:

```
tests/features/
├── orchestrator/
│   ├── enqueue.feature
│   ├── dispatch.feature
│   └── session.feature
├── pool-manager/
│   ├── provision.feature
│   └── health.feature
└── adapters/
    ├── llamacpp.feature
    └── vllm.feature
```

---

## Testing

### Unit Tests

```bash
# Run unit tests
cargo test -p test-harness-bdd -- --nocapture
```

### Integration Tests

```bash
# Run BDD scenarios
cargo run -p test-harness-bdd --bin bdd-runner
```

---

## Dependencies

### Internal

- `orchestrator-core` — Orchestrator logic
- `pool-managerd` — Pool manager
- `worker-adapters-mock` — Mock adapter for tests

### External

- `cucumber` — BDD test framework
- `tokio` — Async runtime
- `async-trait` — Async trait support

---

## Specifications

Implements requirements from:
- ORCH-3050 (BDD test harness)
- ORCH-3051 (Test execution)

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
