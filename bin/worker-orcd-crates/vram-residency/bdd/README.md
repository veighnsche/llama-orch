# vram-residency BDD Tests

**Behavior-Driven Development tests for VRAM residency**

This directory contains BDD tests using Cucumber/Gherkin for the `vram-residency` crate.

---

## Running Tests

### Run all BDD tests

```bash
cd bin/worker-orcd-crates/vram-residency/bdd
cargo test
```

### Run specific feature

```bash
cargo test --test seal_model
cargo test --test verify_seal
cargo test --test vram_policy
```

### Run BDD runner directly

```bash
cargo run --bin bdd-runner
```

---

## Test Mode

Tests automatically detect GPU availability:

- **🎮 GPU Mode**: If NVIDIA GPU detected, tests use real CUDA
- **💻 Mock Mode**: If no GPU, tests use mock VRAM allocator

Same tests work in both modes!

---

## Feature Files

### `seal_model.feature`
Tests for sealing models in VRAM:
- Valid seal operations
- Invalid shard ID rejection
- Insufficient VRAM handling
- Capacity limits

### `verify_seal.feature`
Tests for seal verification:
- Valid seal verification
- Tampered digest detection
- Forged signature detection

### `vram_policy.feature`
Tests for VRAM-only policy enforcement:
- Policy initialization
- UMA/zero-copy/pinned memory detection
- Policy violation handling

---

## Test Structure

```
bdd/
├── Cargo.toml              # BDD test dependencies
├── src/
│   ├── main.rs             # BDD runner entry point
│   └── steps/
│       ├── mod.rs          # Step module exports
│       ├── world.rs        # BDD World state
│       ├── seal_model.rs   # Seal operation steps
│       ├── verify_seal.rs  # Verification steps
│       └── assertions.rs   # Assertion steps
└── tests/
    └── features/
        ├── seal_model.feature
        ├── verify_seal.feature
        └── vram_policy.feature
```

---

## Writing New Tests

### 1. Add Feature File

Create a new `.feature` file in `tests/features/`:

```gherkin
Feature: My New Feature
  As a user
  I want to do something
  So that I achieve a goal

  Scenario: My scenario
    Given some precondition
    When I perform an action
    Then I expect a result
```

### 2. Implement Steps

Add step definitions in `src/steps/`:

```rust
use cucumber::{given, when, then};
use super::world::BddWorld;

#[given("some precondition")]
async fn given_precondition(world: &mut BddWorld) {
    // Setup code
}

#[when("I perform an action")]
async fn when_action(world: &mut BddWorld) {
    // Action code
}

#[then("I expect a result")]
async fn then_result(world: &mut BddWorld) {
    // Assertion code
}
```

### 3. Register Steps

Add new step module to `src/steps/mod.rs`:

```rust
pub mod my_new_steps;
```

---

## BDD World State

The `BddWorld` struct maintains test state:

```rust
pub struct BddWorld {
    pub manager: Option<VramManager>,
    pub gpu_info: Option<GpuInfo>,
    pub shards: HashMap<String, SealedShard>,
    pub last_result: Option<Result<(), VramError>>,
    // ... other fields
}
```

---

## Assertions

Common assertion patterns:

```gherkin
Then the seal should succeed
Then the seal should fail
Then the seal should fail with "InvalidInput"
Then the verification should succeed
Then the verification should fail with "SealVerificationFailed"
Then an audit event "VramSealed" should be emitted
```

---

## Environment Variables

- `LLORCH_BDD_FEATURE_PATH`: Override feature file directory
- `LLORCH_RUN_ID`: Test run identifier (for proof bundles)
- `LLORCH_PROOF_DIR`: Proof bundle output directory

---

## Integration with CI/CD

BDD tests run in CI/CD pipelines:

```yaml
- name: Run BDD tests
  run: |
    cd bin/worker-orcd-crates/vram-residency/bdd
    cargo test
```

Tests automatically fall back to mock mode on CPU-only runners.

---

## Debugging

Enable verbose output:

```bash
cargo test -- --nocapture
```

Run single scenario:

```bash
cargo test --test seal_model -- "Successfully seal model"
```

---

## Status

- ✅ BDD framework installed
- ✅ 3 feature files created
- ✅ Step definitions implemented
- ⬜ Audit logging integration (pending)
- ⬜ Policy enforcement steps (pending CUDA integration)

---

## References

- [Cucumber Book](https://cucumber.io/docs/guides/)
- [cucumber-rs Documentation](https://cucumber-rs.github.io/cucumber/)
- [Gherkin Reference](https://cucumber.io/docs/gherkin/reference/)
