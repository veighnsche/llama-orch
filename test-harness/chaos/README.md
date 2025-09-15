# test-harness-chaos — test-harness-chaos (test-harness)

## 1. Name & Purpose

test-harness-chaos (test-harness)

## 2. Why it exists (Spec traceability)

- ORCH-3050 — [.specs/orchestrator-spec.md](../../.specs/orchestrator-spec.md#orch-3050)
- ORCH-3051 — [.specs/orchestrator-spec.md](../../.specs/orchestrator-spec.md#orch-3051)


## 3. Public API surface

- Rust crate API (internal)

## 4. How it fits

- Provides test scaffolding for validation suites.

```mermaid
flowchart LR
  crates[Crates] --> harness[Test Harness]
  harness --> results[Reports]
```

## 5. Build & Test

- Workspace fmt/clippy: `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features
-- -D warnings`
- Tests for this crate: `cargo test -p test-harness-chaos -- --nocapture`


## 6. Contracts

- None


## 7. Config & Env

- Not applicable.

## 8. Metrics & Logs

- Minimal logs.

## 9. Runbook (Dev)

- Regenerate artifacts: `cargo xtask regen-openapi && cargo xtask regen-schema`
- Rebuild docs: `cargo run -p tools-readme-index --quiet`


## 10. Status & Owners

- Status: alpha
- Owners: @llama-orch-maintainers

## 11. Changelog pointers

- None

## 12. Footnotes

- Spec: [.specs/orchestrator-spec.md](../../.specs/orchestrator-spec.md)
- Requirements: [requirements/index.yaml](../../requirements/index.yaml)

### Additional Details
- Which tests are ignored vs required; how to run real-model Haiku; determinism suite scope.


## What this crate is not

- Not a production service.
