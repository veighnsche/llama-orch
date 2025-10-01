# engine-provisioner BDD Tests

**Cucumber/BDD tests for engine-provisioner**

`libs/provisioners/engine-provisioner/bdd` — Behavior-driven tests for engine provisioning scenarios.

---

## What This Test Suite Does

BDD tests for **engine-provisioner** verify:

- **Process lifecycle** — Start, stop, restart engines
- **Handoff files** — Correct metadata emission
- **Health checks** — Readiness wait and validation
- **Error handling** — Graceful failures and error messages
- **Configuration** — Flag normalization and validation

---

## Running Tests

### All Features

```bash
# Run all BDD tests
cargo test -p engine-provisioner-bdd -- --nocapture
```

### Specific Feature

```bash
# Run specific feature file
LLORCH_BDD_FEATURE_PATH=tests/features/provision.feature \
  cargo test -p engine-provisioner-bdd -- --nocapture
```

---

## Test Scenarios

### Provision Engine

```gherkin
Feature: Engine Provisioning

  Scenario: Provision llama.cpp engine
    Given a pool config for llamacpp
    When I provision the engine
    Then the engine process should be running
    And a handoff file should exist
    And the handoff should contain engine metadata
```

### Health Checks

```gherkin
Feature: Health Checks

  Scenario: Wait for engine readiness
    Given an engine is starting
    When I wait for readiness
    Then the engine should respond to health checks
    And the status should be 200 OK
```

### Graceful Shutdown

```gherkin
Feature: Graceful Shutdown

  Scenario: Stop engine gracefully
    Given a running engine
    When I stop the engine
    Then it should receive SIGTERM
    And it should stop within 5 seconds
```

---

## Dependencies

### Internal

- `provisioners-engine-provisioner` — Library under test

### External

- `cucumber` — BDD test framework
- `tokio` — Async runtime

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
