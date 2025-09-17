# Project Guide — Getting Started

Welcome! This guide shows you how to come up to speed with the home profile quickly.

## 1. Read These First

1. `.docs/HOME_PROFILE.md` — high-level behaviour and promises.
2. `.docs/HOME_PROFILE_TARGET.md` — the reference hardware we test on.
3. `.docs/workflow.md` — stage map and developer checklist.
4. `.specs/00_llama-orch.md` — normative requirements (ORCH-IDs).
5. `.specs/20-orchestratord.md` — HTTP behaviour for handlers.

## 2. Contracts & Regeneration

- OpenAPI specs live in `contracts/openapi/{data.yaml,control.yaml}`.
- Config schema lives in `contracts/config-schema/src/lib.rs` (run `cargo xtask regen-schema`).
- Metrics contract is `.specs/metrics/otel-prom.md` + `ci/metrics.lint.json`.
- After editing specs, run:
  ```bash
  cargo xtask regen-openapi
  cargo xtask regen-schema
  cargo run -p tools-spec-extract --quiet
  ```

## 3. Tests You Must Know

| Suite | Location | Purpose |
|-------|----------|---------|
| Provider verify | `orchestratord/tests/provider_verify.rs` | Verifies handlers against pact + OpenAPI |
| BDD | `test-harness/bdd/tests/features/` | End-to-end journeys (admission, streaming, catalog, artifacts, budgets) |
| Determinism | `test-harness/determinism-suite/` | Byte-exact stream checks per engine |
| Metrics lint | `test-harness/metrics-contract/` | Ensures `/metrics` matches the contract |
| Haiku | `test-harness/e2e-haiku/` | Anti-cheat gate on real GPU |

Always run `cargo xtask dev:loop` before pushing; it formats, regenerates contracts, runs tests, and checks links.

## 4. TODO Discipline

- `TODO.md` at repo root is the single source of planning truth.
- After completing work, add a bullet describing what changed and link to specs/tests touched.
- Archive with `ci/scripts/archive_todo.sh` once the plan is complete; the script stores entries under `.docs/DONE/`.

## 5. Helpful Commands

```bash
cargo test --workspace --all-features -- --nocapture
cargo clippy --all-targets --all-features -- -D warnings
bash ci/scripts/check_links.sh
```

## 6. Where to Ask Questions

- Specs unclear? Draft a proposal under `.specs/proposals/` referencing requirement IDs.
- Need to document an investigation? Commit a markdown file under `.docs/` (or near the component) and link from `TODO.md`.

Welcome to the home profile. Keep the docs updated and the GPUs busy.
