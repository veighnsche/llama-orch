# pool-managerd-bdd — pool-managerd-bdd (core)

## 1. Name & Purpose

pool-managerd-bdd (core)

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

- Rust crate API (internal)

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
- Tests for this crate: `cargo test -p pool-managerd-bdd -- --nocapture`


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

- Spec: [.specs/00_llama-orch.md](../../../.specs/00_llama-orch.md)
- Requirements: [requirements/00_llama-orch.yaml](../../../requirements/00_llama-orch.yaml)

### Additional Details
- Preload/Ready lifecycle, NVIDIA-only guardrails, restart/backoff behavior.


## What this crate is not

- Not a general-purpose inference server; focuses on orchestration.
