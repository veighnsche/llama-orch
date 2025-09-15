# orchestrator-core — orchestrator-core (core)

## 1. Name & Purpose

orchestrator-core (core)

## 2. Why it exists (Spec traceability)

- ORCH-3004 — [.specs/orchestrator-spec.md](../.specs/orchestrator-spec.md#orch-3004)
- ORCH-3005 — [.specs/orchestrator-spec.md](../.specs/orchestrator-spec.md#orch-3005)
- ORCH-3008 — [.specs/orchestrator-spec.md](../.specs/orchestrator-spec.md#orch-3008)
- ORCH-3010 — [.specs/orchestrator-spec.md](../.specs/orchestrator-spec.md#orch-3010)
- ORCH-3011 — [.specs/orchestrator-spec.md](../.specs/orchestrator-spec.md#orch-3011)
- ORCH-3016 — [.specs/orchestrator-spec.md](../.specs/orchestrator-spec.md#orch-3016)
- ORCH-3017 — [.specs/orchestrator-spec.md](../.specs/orchestrator-spec.md#orch-3017)
- ORCH-3027 — [.specs/orchestrator-spec.md](../.specs/orchestrator-spec.md#orch-3027)
- ORCH-3028 — [.specs/orchestrator-spec.md](../.specs/orchestrator-spec.md#orch-3028)
- ORCH-3044 — [.specs/orchestrator-spec.md](../.specs/orchestrator-spec.md#orch-3044)
- ORCH-3045 — [.specs/orchestrator-spec.md](../.specs/orchestrator-spec.md#orch-3045)

## 3. Public API surface

- Rust crate API (internal)

## 4. How it fits

- Part of the core orchestrator. Upstream: adapters, Downstream: workers.

## 5. Build & Test

- Workspace fmt/clippy: `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features
-- -D warnings`
- Tests for this crate: `cargo test -p orchestrator-core -- --nocapture`

## 6. Contracts

- None

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

- Spec: [.specs/orchestrator-spec.md](../.specs/orchestrator-spec.md)
- Requirements: [requirements/index.yaml](../requirements/index.yaml)

### Additional Details

- Queue invariants and property tests overview (fairness, capacity, rejection policies).
- Capacity policies and bounded FIFO behavior.

## What this crate is not

- Not a general-purpose inference server; focuses on orchestration.
