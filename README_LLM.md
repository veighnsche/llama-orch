# README for LLM Developers — Decision Rules and Workflow

This document defines how an LLM developer must make decisions and contribute to this repository. Keep it short, decisive, and traceable.

## Golden Rules

- No backwards compatibility pre‑v1.0.0
  - Do not preserve pre‑1.0 behavior for compatibility. Do not write shims, adapters, or keep dead code for BC. Break and rebuild if it’s better. Remove unused paths aggressively.

- Spec is the source of truth
  - For every technical decision, consult `.specs/` first. If something is ambiguous, propose a spec update before writing code.

- Proposal required for spec changes
  - Do not change a spec without a proposal and review. Keep proposals minimal (problem, change, impact, new/changed requirement IDs, migration/rollback). Once accepted, update the spec and proceed.

- Order of work: Spec → Contract → Tests → Code
  - Spec: author/adjust normative text in `.specs/` using RFC‑2119 language with stable requirement IDs.
  - Contract: update formal interfaces (e.g., `contracts/openapi/`, `contracts/config-schema/`, ABIs) to reflect the spec.
  - Tests: write proofs (unit/integration/property/CDC) that reference the requirement IDs.
  - Code: implement the behavior to satisfy the tests and spec.

- Always update the TODO tracker with progress
  - After each meaningful change, update the current TODO tracker (e.g., `TODO.md` or the active TODO in `.docs/`). Keep it factual: what changed, where, and why.

- Finish what you start
  - Do not leave items half‑done. If blocked, explicitly note the blocker in the TODO and open a proposal/issue.

## Traceability and Quality Gates

- Requirement IDs everywhere
  - Reference requirement IDs (e.g., `ORCH-…` umbrella and `OC-…` component-specific) in code comments near key logic and in test names/docs.

- Idempotent regeneration
  - Regeneration tools MUST be diff‑clean on a second run. Favor determinism and reproducibility over cleverness.

- Contracts are single source of truth for APIs
  - Update OpenAPI (`contracts/openapi/`), config schema, and metrics per `.specs/metrics/otel-prom.md` before or with code. Generated clients/servers and CDC tests should pass before merging. Metrics labels MUST include `engine` and engine‑specific version labels (e.g., `engine_version`, `trtllm_version`).

- No premature optimization
  - Optimize only after correctness and contract proofs are in place. Remove dead code early.

## Minimal Spec/Proposal Workflow

1) Write/adjust spec in `.specs/` with RFC‑2119 terms and stable IDs.
2) If a change is material, open a small proposal in the PR description and commit message. Include: problem, change summary, impacted areas, new/changed IDs, migration/rollback.
3) Update contracts (OpenAPI/config schema/ABIs) to match the spec.
4) Add/adjust tests that prove the requirements.
5) Implement code. Keep diffs tight and focused.
6) Update the TODO tracker with progress and links to proofs.

## PR Checklist (run before merge)

- Specs/Contracts
  - Spec updated or explicitly confirmed unchanged; requirement IDs present where applicable.
  - OpenAPI/config schema updated when surfaces change; examples compile.

- Proofs
  - Requirements regen is clean: `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
  - Links are valid: `bash ci/scripts/check_links.sh`
  - Workspace is healthy: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings`
  - Tests pass: `cargo test --workspace --all-features -- --nocapture`

- Hygiene
  - No BC shims or dead code left behind.
  - Commits and PR description reference relevant requirement IDs.
  - TODO tracker updated with what changed.

## Workflow alignment (SPEC→SHIP v2)

- Guiding principles (see `.docs/workflow.md`)
  - Spec is law with stable IDs (`ORCH-XXXX`); contracts are versioned artifacts.
  - Contract‑first; TDD; determinism by default; fail fast; short sessions.
  - Real‑model proof: the Haiku E2E test MUST pass; prefer a GPU worker over LAN; mocks cannot satisfy release gates.

- Stages and gates (brief)
  - Stage 0 — Contract freeze: OpenAPI + config schema regenerated; CI fails on diffs.
  - Stage 1 — CDC + snapshots: Pact + insta green before provider code.
  - Stage 2 — Provider verify: orchestrator passes pact verification.
  - Stage 3 — Properties: core invariants via proptest.
  - Stage 4 — Determinism: two replicas per engine; byte‑exact streams.
  - Stage 5 — Observability: metrics exactly per `.specs/metrics/otel-prom.md`.
  - Stage 6 — Real‑model E2E (Haiku): pass within budget; metrics delta observed.

- Developer loop (deterministic)
  - `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo xtask regen-openapi && cargo xtask regen-schema`
  - `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
  - `cargo test --workspace --all-features -- --nocapture`
  - `bash ci/scripts/check_links.sh`

- Environment conventions
  - `TZ=Europe/Amsterdam` for E2E‑Haiku; `REQUIRE_REAL_LLAMA=1` to enforce a real Worker.

- Definition of Done (per requirement)
  - Contract coverage in OpenAPI/Schema; consumer pact exists; provider verification passes; unit/property tests cover edges; observability proves it (metrics/logs); `requirements/index.yaml` links req → tests → code.

## Non‑Goals and Anti‑Patterns

- Do not invent behavior that is not in the spec; propose first.
- Do not keep unused flags/ENV/feature gates “just in case”.
- Do not bypass contracts/tests to “unblock” code.
- Do not rely on hidden state or nondeterministic outputs.

## When in Doubt

- Prefer smaller, well‑scoped proposals and PRs.
- Ask the spec for guidance; if it’s silent, propose the minimum change to make it explicit.
- Choose clarity and determinism over compatibility until v1.0.0.
