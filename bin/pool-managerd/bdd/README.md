# pool-managerd-bdd — pool-managerd BDD harness

## 1. Name & Purpose

pool-managerd-bdd (BDD harness for pool-managerd)

## 2. Why it exists (Spec traceability)

- OC-POOL-3001 — [.specs/30-pool-managerd.md](../../../.specs/30-pool-managerd.md#oc-pool-3001)
- OC-POOL-3002 — [.specs/30-pool-managerd.md](../../../.specs/30-pool-managerd.md#oc-pool-3002)
- OC-POOL-3003 — [.specs/30-pool-managerd.md](../../../.specs/30-pool-managerd.md#oc-pool-3003)
- OC-POOL-3010 — [.specs/30-pool-managerd.md](../../../.specs/30-pool-managerd.md#oc-pool-3010)
- OC-POOL-3011 — [.specs/30-pool-managerd.md](../../../.specs/30-pool-managerd.md#oc-pool-3011)
- OC-POOL-3012 — [.specs/30-pool-managerd.md](../../../.specs/30-pool-managerd.md#oc-pool-3012)
- OC-POOL-3020 — [.specs/30-pool-managerd.md](../../../.specs/30-pool-managerd.md#oc-pool-3020)
- OC-POOL-3021 — [.specs/30-pool-managerd.md](../../../.specs/30-pool-managerd.md#oc-pool-3021)
- OC-POOL-3030 — [.specs/30-pool-managerd.md](../../../.specs/30-pool-managerd.md#oc-pool-3030)

## 3. Public API surface

- HTTP API for pool management (health, readiness, drain, reload)

## 4. How it fits

- Pool manager daemon that supervises engine lifecycle and placement.

```mermaid
flowchart LR
  orchestratord --> pool-managerd
  pool-managerd --> engines[Engine Processes]
```

## 5. Build & Test

- Workspace fmt/clippy: `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features -- -D warnings`
- Tests for this crate: `cargo test -p pool-managerd-bdd -- --nocapture`
- BDD runner: `cargo run -p pool-managerd-bdd --bin bdd-runner`

## 6. Contracts

- None (internal daemon)

## 7. Config & Env

- `LLORCH_BDD_FEATURE_PATH`: Override features directory for targeted BDD runs

## 8. Metrics & Logs

- Emits preload outcomes, VRAM/RAM utilization, driver_reset events, and restart counters.

## 9. Runbook (Dev)

- Run all features: `cargo run -p pool-managerd-bdd --bin bdd-runner`
- Run specific feature: `LLORCH_BDD_FEATURE_PATH=tests/features/preload.feature cargo run -p pool-managerd-bdd --bin bdd-runner`

## 10. Status & Owners

- Status: alpha
- Owners: @llama-orch-maintainers

## 11. Changelog pointers

- None

## 12. Footnotes

- Spec: [.specs/30-pool-managerd.md](../../../.specs/30-pool-managerd.md)
- Requirements: [requirements/30-pool-managerd.yaml](../../../requirements/30-pool-managerd.yaml)

### Additional Details
- BDD scenarios for preload, readiness, restart/backoff, device masks, and observability.

## What this crate is not

- Not a general-purpose process supervisor; focuses on engine pool management.
