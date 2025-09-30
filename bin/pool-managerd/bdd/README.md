# pool-managerd-bdd

**BDD test suite for pool-managerd**

`bin/pool-managerd/bdd` — Cucumber-based behavior-driven development tests for the pool-managerd daemon.

---

## What This Crate Does

This is the **BDD test harness** for pool-managerd. It provides:

- **Cucumber/Gherkin scenarios** testing pool-managerd behavior
- **Step definitions** in Rust using the `cucumber` crate
- **Integration tests** covering pool lifecycle, engine provisioning, health monitoring
- **Proof bundle output** for test artifacts and traceability

**Tests**:
- Pool preload and readiness
- Engine provisioning (download, compile, start)
- Handoff file detection
- Health monitoring and restart/backoff
- GPU discovery and device masks
- Node registration with orchestratord
- VRAM/RAM utilization tracking

---

## Running Tests

### All Scenarios

```bash
# Run all BDD tests
cargo test -p pool-managerd-bdd -- --nocapture

# Or use the BDD runner binary
cargo run -p pool-managerd-bdd --bin bdd-runner
```

### Specific Feature

```bash
# Set environment variable to target specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/preload.feature \
cargo test -p pool-managerd-bdd -- --nocapture
```

### Check for Undefined Steps

```bash
# Verify all steps are implemented
cargo test -p pool-managerd-bdd --lib -- features_have_no_undefined_or_ambiguous_steps
```

---

## Feature Files

Located in `tests/features/`:

- `preload.feature` — Pool preload lifecycle, readiness transitions
- `provisioning.feature` — Engine download, compilation, startup
- `handoff.feature` — Handoff file detection and processing
- `health.feature` — Health checks, restart logic, backoff
- `gpu_discovery.feature` — GPU detection, device masks, VRAM tracking
- `registration.feature` — Node registration with orchestratord
- `observability.feature` — Metrics emission, structured logs

---

## Step Definitions

Located in `src/steps/`:

- `background.rs` — Setup and teardown (pool-managerd instance, mock engines)
- `preload.rs` — Pool preload and readiness steps
- `provisioning.rs` — Engine provisioning steps
- `handoff.rs` — Handoff file watching steps
- `health.rs` — Health monitoring steps
- `gpu.rs` — GPU discovery steps
- `registration.rs` — Node registration steps
- `assertions.rs` — Common assertion helpers

---

## World State

The `World` struct maintains test state:

```rust
pub struct World {
    pub pool_managerd_url: String,
    pub last_response: Option<Response>,
    pub pool_registry: Option<Registry>,
    pub mock_engines: Vec<MockEngine>,
    pub handoff_files: Vec<PathBuf>,
}
```

---

## Example Scenario

```gherkin
Feature: Pool Preload

  Scenario: Successfully preload a pool
    Given pool-managerd is running
    And GPU 0 is available
    When I preload pool "pool-0" with model "llama-3.1-8b-instruct"
    Then the pool state should be "provisioning"
    And the engine should download the model
    And the pool state should transition to "ready"
    And the handoff file should exist
```

---

## Proof Bundle Output

BDD tests emit proof bundles to `.proof_bundle/bdd/<run_id>/`:

- `scenarios.ndjson` — Scenario results
- `steps.ndjson` — Step execution details
- `metadata.json` — Test run metadata
- `seeds.txt` — Random seeds for reproducibility

Set `LLORCH_RUN_ID` to control the output directory.

---

## Dependencies

### Parent Crate

- `pool-managerd` — The binary being tested

### Test Infrastructure

- `cucumber` — BDD framework
- `tokio` — Async runtime for tests
- `reqwest` — HTTP client for API calls
- `serde_json` — JSON parsing
- `proof-bundle` — Test artifact output

---

## Writing New Tests

### 1. Add Feature File

Create `tests/features/my_feature.feature`:

```gherkin
Feature: My New Feature

  Scenario: Test something
    Given pool-managerd is running
    When I do something
    Then something should happen
```

### 2. Implement Steps

Add to `src/steps/my_steps.rs`:

```rust
use cucumber::{given, when, then};
use crate::world::World;

#[when("I do something")]
async fn when_i_do_something(world: &mut World) {
    // Implementation
}

#[then("something should happen")]
async fn then_something_happens(world: &mut World) {
    // Assertions
}
```

### 3. Run Tests

```bash
cargo test -p pool-managerd-bdd -- --nocapture
```

---

## Specifications

Tests verify requirements from:
- OC-POOL-3001, OC-POOL-3002, OC-POOL-3003
- OC-POOL-3010, OC-POOL-3011, OC-POOL-3012
- OC-POOL-3020, OC-POOL-3021, OC-POOL-3030

See `.specs/30-pool-managerd.md` for full requirements.

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
