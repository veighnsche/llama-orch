# model-provisioner BDD Tests

**Cucumber/BDD tests for model-provisioner**

`libs/provisioners/model-provisioner/bdd` — Behavior-driven tests for model resolution and catalog scenarios.

---

## What This Test Suite Does

BDD tests for **model-provisioner** verify:

- **Model resolution** — Resolve local file paths
- **Digest verification** — SHA256 checksum validation
- **Catalog integration** — Register and update models
- **Handoff files** — Correct metadata emission
- **Error handling** — Missing files, invalid digests

---

## Running Tests

### All Features

```bash
# Run all BDD tests
cargo test -p model-provisioner-bdd -- --nocapture
```

### Specific Feature

```bash
# Run specific feature file
LLORCH_BDD_FEATURE_PATH=tests/features/resolve.feature \
  cargo test -p model-provisioner-bdd -- --nocapture
```

---

## Test Scenarios

### Model Resolution

```gherkin
Feature: Model Resolution

  Scenario: Resolve local model file
    Given a local model file at "/models/llama-3.1-8b.gguf"
    When I resolve the model reference
    Then the resolved path should be "/models/llama-3.1-8b.gguf"
    And the model ID should be normalized
```

### Digest Verification

```gherkin
Feature: Digest Verification

  Scenario: Verify model digest
    Given a model file with known digest
    When I verify the digest
    Then the verification should succeed
    And the result should be recorded in catalog
```

### Catalog Integration

```gherkin
Feature: Catalog Integration

  Scenario: Register model in catalog
    Given a resolved model
    When I register it in the catalog
    Then the catalog should contain the model
    And the lifecycle state should be Active
```

---

## Dependencies

### Internal

- `provisioners-model-provisioner` — Library under test
- `catalog-core` — Catalog integration

### External

- `cucumber` — BDD test framework
- `tokio` — Async runtime

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
