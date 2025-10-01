# catalog-core-bdd

**BDD test suite for catalog-core**

`libs/catalog-core/bdd` — Cucumber-based behavior-driven development tests for the catalog-core model registry library.

---

## What This Crate Does

This is the **BDD test harness** for catalog-core. It provides:

- **Cucumber/Gherkin scenarios** testing catalog behavior
- **Step definitions** in Rust using the `cucumber` crate
- **Unit-level BDD tests** for catalog operations and lifecycle
- **Proof bundle output** for test artifacts

**Tests**:
- Model registration
- SHA-256 verification
- Lifecycle state transitions (Pending → Active → Retired)
- Catalog queries (by ID, name, state)
- Filesystem storage and persistence

---

## Running Tests

### All Scenarios

```bash
# Run all BDD tests
cargo test -p catalog-core-bdd -- --nocapture

# Or use the BDD runner binary (if available)
cargo run -p catalog-core-bdd --bin bdd-runner
```

### Specific Feature

```bash
# Set environment variable to target specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/registration.feature \
cargo test -p catalog-core-bdd -- --nocapture
```

---

## Feature Files

Located in `tests/features/`:

- `registration.feature` — Model registration with metadata
- `verification.feature` — SHA-256 digest verification
- `lifecycle.feature` — State transitions (Pending, Active, Retired, Failed)
- `queries.feature` — Find models by ID, name, state
- `persistence.feature` — Filesystem storage and reload

---

## Example Scenario

```gherkin
Feature: Model Verification

  Scenario: Successfully verify a model
    Given a catalog with model "llama-3.1-8b-instruct" in state "Pending"
    And the model file exists at "/tmp/models/llama-3.1-8b-instruct.gguf"
    When I verify the model with expected digest "abc123def456..."
    Then the verification should succeed
    And the model state should be "Active"
    And the verified_at timestamp should be set
```

---

## Step Definitions

Located in `src/steps/`:

- `catalog.rs` — Catalog creation and manipulation steps
- `registration.rs` — Model registration steps
- `verification.rs` — Verification steps
- `lifecycle.rs` — State transition steps
- `assertions.rs` — Catalog state assertions

---

## Testing

```bash
# Run all tests
cargo test -p catalog-core-bdd -- --nocapture

# Check for undefined steps
cargo test -p catalog-core-bdd --lib -- features_have_no_undefined_or_ambiguous_steps
```

---

## Dependencies

### Parent Crate

- `catalog-core` — The library being tested

### Test Infrastructure

- `cucumber` — BDD framework
- `tokio` — Async runtime for tests
- `tempfile` — Temporary directories for test catalogs
- `proof-bundle` — Test artifact output

---

## Specifications

Tests verify requirements from:
- ORCH-3004, ORCH-3005, ORCH-3008, ORCH-3010, ORCH-3011
- ORCH-3016, ORCH-3017, ORCH-3027, ORCH-3028
- ORCH-3044, ORCH-3045

See `.specs/00_llama-orch.md` for full requirements.

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
