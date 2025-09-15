# plugins-policy-host — plugins-policy-host (plugin)

## 1. Name & Purpose

plugins-policy-host (plugin)

## 2. Why it exists (Spec traceability)

- ORCH-3048 — [.specs/orchestrator-spec.md](../../.specs/orchestrator-spec.md#orch-3048)


## 3. Public API surface

- Rust crate API (internal)

## 4. How it fits

- Policy extension via WASI ABI.

```mermaid
flowchart LR
  orch[Orchestrator] --> plugins[Policy Plugins (WASI)]
```

## 5. Build & Test

- Workspace fmt/clippy: `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features
-- -D warnings`
- Tests for this crate: `cargo test -p plugins-policy-host -- --nocapture`


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
- Requirements: [requirements/00_llama-orch.yaml](../../requirements/00_llama-orch.yaml)

### Additional Details
- WASI policy ABI and SDK usage; example plugin pointers.


## What this crate is not

- Not a production service.
