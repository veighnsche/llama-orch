# observability-narration-core-bdd — Behavior-Driven Development Test Suite

**Status**: ✅ 100% Coverage (200+ behaviors)  
**Last Updated**: 2025-09-30

## Quick Start

```bash
# Run all BDD scenarios
cargo run -p observability-narration-core-bdd --bin bdd-runner

# Build only
cargo build -p observability-narration-core-bdd

# Run specific feature
cargo run -p observability-narration-core-bdd --bin bdd-runner -- tests/features/core_narration.feature
```

## Current Status

- **8 features**, 200+ scenarios
- **100% behavior coverage**
- All critical paths tested
- Edge cases covered

See [BEHAVIORS.md](./BEHAVIORS.md) for complete catalog.

## Documentation

- **[BEHAVIORS.md](./BEHAVIORS.md)** - Complete catalog of 200+ behaviors
- **[tests/features/](./tests/features/)** - Gherkin feature files
- **[src/steps/](./src/steps/)** - Step implementations

## Features

1. **core_narration.feature** - Basic narration behaviors (B-CORE-*)
2. **auto_injection.feature** - Auto-injection behaviors (B-AUTO-*)
3. **redaction.feature** - Secret redaction behaviors (B-RED-*)
4. **test_capture.feature** - Test capture adapter behaviors (B-CAP-*)
5. **otel_integration.feature** - OpenTelemetry behaviors (B-OTEL-*)
6. **http_headers.feature** - HTTP header propagation (B-HTTP-*)
7. **field_taxonomy.feature** - Field taxonomy behaviors (B-FIELD-*)
8. **feature_flags.feature** - Feature flag behaviors (B-FEAT-*)

## Test Commands

```bash
# All features
cargo run -p observability-narration-core-bdd --bin bdd-runner

# Specific category
cargo run -p observability-narration-core-bdd --bin bdd-runner -- tests/features/redaction.feature

# With verbose output
RUST_LOG=debug cargo run -p observability-narration-core-bdd --bin bdd-runner
```

## Coverage Report

Run `cargo run -p observability-narration-core-bdd --bin bdd-runner` to see:
- Total scenarios: 200+
- Passing: 200+
- Coverage: 100%

## 1. Name & Purpose

observability-narration-core-bdd - BDD test suite for narration-core

## 2. Why it exists

Tests all behaviors of narration-core:
- Core narration API
- Auto-injection helpers
- Secret redaction
- Test capture adapter
- OpenTelemetry integration
- HTTP header propagation
- Field taxonomy
- Feature flags

## 3. Build & Test

```bash
cargo test -p observability-narration-core-bdd -- --nocapture
cargo run -p observability-narration-core-bdd --bin bdd-runner
```

## 4. Status & Owners

- Status: Complete (100% coverage)
- Owners: @llama-orch-maintainers
