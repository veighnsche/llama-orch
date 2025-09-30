# orchestratord-bdd — Behavior-Driven Development Test Suite

**Status**: ✅ Core Complete (78% passing)  
**Last Updated**: 2025-09-30

## Quick Start

```bash
# Run all BDD scenarios
cargo run -p orchestratord-bdd --bin bdd-runner

# Build only
cargo build -p orchestratord-bdd

# Check for undefined steps
cargo test -p orchestratord-bdd --lib -- features_have_no_undefined_or_ambiguous_steps
```

## Current Status

- **18 features**, 41 scenarios
- **84/108 steps passing** (78%)
- **Core features**: 100% passing
- **New features**: Need step implementations

See [COMPLETION_REPORT.md](./COMPLETION_REPORT.md) for details.

## Documentation

- **[BEHAVIORS.md](./BEHAVIORS.md)** - Complete catalog of 200+ behaviors
- **[FEATURE_MAPPING.md](./FEATURE_MAPPING.md)** - Features → Scenarios → Steps mapping
- **[COMPLETION_REPORT.md](./COMPLETION_REPORT.md)** - Current status and results
- **[NEXT_STEPS.md](./NEXT_STEPS.md)** - Path to 100%
- **[POOL_MANAGERD_INTEGRATION.md](../POOL_MANAGERD_INTEGRATION.md)** - Daemon integration guide

## 1. Name & Purpose

orchestratord-bdd (core)

## 2. Why it exists (Spec traceability)

- ORCH-3004 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3004)
- ORCH-3005 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3005)
- ORCH-3008 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3008)
- ORCH-3010 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3010)
- ORCH-3011 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3011)
- ORCH-3016 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3016)
- ORCH-3017 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3017)
- ORCH-3027 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3027)
- ORCH-3028 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3028)
- ORCH-3044 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3044)
- ORCH-3045 — [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md#orch-3045)


## 3. Public API surface

- OpenAPI: [contracts/openapi/control.yaml](../../../contracts/openapi/control.yaml)
- OpenAPI: [contracts/openapi/data.yaml](../../../contracts/openapi/data.yaml)
- OpenAPI operations: 16
  - examples: cancelTask, createArtifact, createCatalogModel, createTask, deleteCatalogModel


## 4. How it fits

- Part of the core orchestrator. Upstream: adapters, Downstream: workers.

```mermaid
flowchart LR
  callers[Clients] --> orch[Orchestrator]
  orch --> adapters[Worker Adapters]
  adapters --> engines[Engines]
```

## 5. Build & Test

- Workspace fmt/clippy: `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features
-- -D warnings`
- Tests for this crate: `cargo test -p orchestratord-bdd -- --nocapture`
- Provider verify: `cargo test -p orchestratord --test provider_verify -- --nocapture`


## 6. Contracts

- OpenAPI:
  - [contracts/openapi/control.yaml](../../../contracts/openapi/control.yaml)
  - [contracts/openapi/data.yaml](../../../contracts/openapi/data.yaml)


## 7. Config & Env

- See deployment configs and environment variables used by the daemons.

## 8. Metrics & Logs

- Emits queue depth, latency percentiles, and engine/version labels.

## 9. Runbook (Dev)

- Regenerate artifacts: `cargo xtask regen-openapi && cargo xtask regen-schema`
- Rebuild docs: `cargo run -p tools-readme-index --quiet`


## 10. Status & Owners

- Status: alpha
- Owners: @llama-orch-maintainers

## 11. Changelog pointers

- None

## 12. Footnotes

- Spec: [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md)
- Requirements: [requirements/00_llama-orch.yaml](../../../requirements/00_llama-orch.yaml)

### Additional Details
- Data/control plane routes, SSE framing details, backpressure headers, provider verify entry
points.


## What this crate is not

- Not a general-purpose inference server; focuses on orchestration.
