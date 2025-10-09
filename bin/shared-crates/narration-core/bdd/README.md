# narration-core BDD Tests

**Cucumber/BDD tests for narration-core**

`libs/observability/narration-core/bdd` — Behavior-driven tests for structured observability scenarios.

---

## What This Test Suite Does

BDD tests for **narration-core** verify:

- **Core narration** — Actor/action/target taxonomy
- **Auto-injection** — Correlation ID propagation
- **Secret redaction** — Authorization header masking
- **Test capture** — Assertion helpers for BDD
- **OpenTelemetry** — Integration with tracing
- **HTTP headers** — X-Correlation-ID propagation
- **Field taxonomy** — Required and optional fields
- **Feature flags** — Conditional behavior

**Status**: ✅ 100% Coverage (200+ scenarios)

---

## Running Tests

### All Features

```bash
# Run all BDD tests
cargo test -p observability-narration-core-bdd -- --nocapture

# Or use bdd-runner
cargo run -p observability-narration-core-bdd --bin bdd-runner
```

### Specific Feature

```bash
# Run specific feature file
LLORCH_BDD_FEATURE_PATH=tests/features/core_narration.feature \
  cargo test -p observability-narration-core-bdd -- --nocapture
```

---

## Test Features

### 1. Core Narration

```gherkin
Feature: Core Narration

  Scenario: Emit basic narration event
    Given an actor "rbees-orcd"
    And an action "enqueue"
    And a target "job-123"
    When I emit narration
    Then the event should contain actor
    And the event should contain action
    And the event should contain target
```

### 2. Secret Redaction

```gherkin
Feature: Secret Redaction

  Scenario: Redact authorization header
    Given a narration event with authorization "Bearer secret-token"
    When the event is logged
    Then the authorization should be "[REDACTED]"
```

### 3. Test Capture

```gherkin
Feature: Test Capture

  Scenario: Capture narration in tests
    Given a capture adapter is installed
    When I emit narration
    Then I can assert the event was emitted
    And I can assert field values
```

---

## Coverage

- **8 features** with 200+ scenarios
- **100% behavior coverage**
- All critical paths tested
- Edge cases covered

See `BEHAVIORS.md` for complete catalog.

---

## Dependencies

### Internal

- `observability-narration-core` — Library under test

### External

- `cucumber` — BDD test framework
- `tokio` — Async runtime
- `tracing` — Structured logging

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Complete (100% coverage)
- **Maintainers**: @llama-orch-maintainers
