# Refinement Plan — Single Source of Truth (SSOT)

This document is the single source of truth for refinement work across code, tests, specs, contracts, CI, and docs. All work items below are contract-first and proof-driven. Any PR touching a task MUST paste the task’s Proof output into the PR description.

Quality gates required for every task before merge:

- Green: cargo fmt --all -- --check
- Green: cargo clippy --all-targets --all-features -- -D warnings
- Green: cargo test --workspace --all-features -- --nocapture (unit + integration + harnesses)
- Green: BDD (0 undefined/ambiguous) once BDD is introduced
- Green: trybuild (when introduced)
- Green: Schema validations (contracts-config-schema)
- Green: OpenAPI validation and regen (cargo xtask regen-openapi)
- Green: Requirements extraction regen (cargo run -p tools-spec-extract --quiet)
- Green: Metrics linter tests (test-harness/metrics-contract)
- Docs link checker: bash ci/scripts/check_links.sh
- Diff-clean after all regenerations: git diff --exit-code

---

## P0 — Blockers

### 1) Make fmt/clippy gates green (fix nested test; apply rustfmt)

- Why: Current fmt check shows diffs; clippy -D warnings fails due to an inner test item and lints.
- Files/dirs to touch:
  - `test-harness/metrics-contract/tests/metrics_lint.rs`
  - `orchestratord/tests/provider_verify.rs`
  - `test-harness/e2e-haiku/src/lib.rs`
- Acceptance Criteria:
  - No nested test functions (each `#[test]` is top-level).
  - cargo fmt produces no diffs (fmt check green).
  - cargo clippy with -D warnings green across workspace.
- Proof:
  - cargo fmt --all -- --check
  - cargo clippy --all-targets --all-features -- -D warnings
  - cargo test --workspace --all-features -- --nocapture

### 2) SPEC link path drift fix confirmation (.specs vs specs) and deterministic regen

- Why: The requirements extractor previously targeted `specs/orchestrator-spec.md`; repository uses `.specs/orchestrator-spec.md`.
- Files/dirs to touch:
  - `tools/spec-extract/src/main.rs`
  - `COMPLIANCE.md`
  - `requirements/index.yaml`
- Acceptance Criteria:
  - Spec-extract reads `.specs/orchestrator-spec.md` and generates deterministic output.
  - COMPLIANCE links target `.specs/orchestrator-spec.md#...` anchors.
  - Second regen run produces no diff (idempotent).
- Proof:
  - cargo run -p tools-spec-extract --quiet
  - git diff --exit-code
  - cargo run -p tools-spec-extract --quiet && git diff --exit-code

### 3) OpenAPI ↔ generated types ↔ client drift guard (data + control)

- Why: Prevent contract drift between `contracts/openapi/*.yaml`, `contracts/api-types`, and `tools/openapi-client`.
- Files/dirs to touch:
  - `contracts/openapi/data.yaml`
  - `contracts/openapi/control.yaml`
  - `contracts/api-types/src/generated.rs`
  - `contracts/api-types/src/generated_control.rs`
  - `tools/openapi-client/src/generated.rs`
  - `orchestratord/src/main.rs` (routes wired)
  - `orchestratord/tests/provider_verify.rs`
  - `cli/consumer-tests/tests/*.rs`
- Acceptance Criteria:
  - All OpenAPI docs parse successfully; types/clients regenerate deterministically.
  - Provider verify passes for known endpoints/statuses; pacts (if present) align with OpenAPI.
  - No unknown paths/statuses; error envelopes match declared schemas.
- Proof:
  - cargo xtask regen-openapi
  - cargo test -p orchestratord --test provider_verify -- --nocapture
  - cargo test -p cli-consumer-tests -- --nocapture
  - git diff --exit-code

---

## P1 — High Priority

### 4) SSE contract: end-to-end shape assertions and snapshot policy

- Why: SSE framing is normative (ORCH-2002). Ensure consistent parsing and minimal snapshots without flakiness.
- Files/dirs to touch:
  - `test-harness/determinism-suite/src/` (helpers)
  - `test-harness/determinism-suite/tests/byte_exact.rs`
  - `cli/consumer-tests/tests/snapshot_transcript.rs`
- Acceptance Criteria:
  - SSE parser covers `started`, `token`, `metrics`, `end`, `error`.
  - Snapshots are stable and minimal; no flaky fields.
  - Determinism test remains ignored by default but runnable locally; parser unit tests are not ignored.
- Proof:
  - cargo test -p test-harness-determinism-suite -- --nocapture
  - cargo test -p cli-consumer-tests -- --nocapture

### 5) Property tests for queue invariants (enable when queue exists)

- Why: Ensure FIFO within priority class, fairness, and capacity policy correctness (reject/drop-lru/shed-low-priority).
- Files/dirs to touch:
  - `orchestrator-core/tests/props_queue.rs`
  - `orchestrator-core/src/` (when queue implementation is added)
- Acceptance Criteria:
  - Replace placeholders with a minimal model or real queue once implemented.
  - Properties execute and pass in CI (or are gated behind a feature until ready).
- Proof:
  - cargo test -p orchestrator-core -- --nocapture

### 6) Control plane provider verification scaffolding

- Why: Extend provider verification to `control.yaml` so pacts/tests cover drain/reload/health/replicasets.
- Files/dirs to touch:
  - `orchestratord/tests/provider_verify.rs`
  - `contracts/openapi/control.yaml`
  - `contracts/pacts/` (to be created for control interactions)
- Acceptance Criteria:
  - Add tests mirroring control plane shapes; support template matching and status checks.
  - Deterministic pass locally.
- Proof:
  - cargo test -p orchestratord --test provider_verify -- --nocapture

### 7) Metrics contract wiring plan and placeholders

- Why: `ci/metrics.lint.json` defines names/labels; tests currently validate schema only. Prepare emission sites and guard label sets.
- Files/dirs to touch:
  - `ci/metrics.lint.json`
  - `test-harness/metrics-contract/tests/metrics_lint.rs`
  - Emission sites (to be created when implementing runtime)
- Acceptance Criteria:
  - Keep linter tests compiling and green.
  - Document metric emission sites and label budgets to avoid unbounded cardinality.
- Proof:
  - cargo test -p test-harness-metrics-contract -- --nocapture
  - bash ci/scripts/check_links.sh

---

## P2 — Medium Priority

### 8) Remove unwrap/expect in non-test code; propagate errors

- Why: Hard failures hinder resilience; prefer Result with context.
- Files/dirs to touch:
  - `xtask/src/main.rs`
  - `tools/spec-extract/src/main.rs`
- Acceptance Criteria:
  - Replace `unwrap/expect` in non-test code with error propagation using anyhow/context.
  - clippy -D warnings remains green.
- Proof:
  - cargo clippy --all-targets --all-features -- -D warnings

### 9) Atomic write with EXDEV-safe fallback in generators

- Why: `fs::write` is non-atomic and rename can fail across devices (EXDEV). Ensure durability and determinism.
- Files/dirs to touch:
  - `xtask/src/main.rs` (write_if_changed)
  - `contracts/config-schema/src/lib.rs` (emit_schema_json)
- Acceptance Criteria:
  - Writes occur via tmp file + fsync + rename; fallback to copy for EXDEV.
  - Regeneration remains deterministic; second run produces no diff.
- Proof:
  - cargo xtask regen-openapi && cargo xtask regen-schema && cargo run -p tools-spec-extract --quiet
  - git diff --exit-code

### 10) Introduce trybuild tests for compile-time guarantees

- Why: Enforce compile-time API invariants (e.g., exhaustive ErrorKind matching in providers/clients).
- Files/dirs to touch:
  - `tools/openapi-client/tests/` (to be created)
  - `contracts/api-types/tests/` (to be created)
- Acceptance Criteria:
  - At least one trybuild suite runs and passes (or is feature-gated until ready).
- Proof:
  - cargo test -p tools-openapi-client -- --nocapture

### 11) BDD scaffolding (Cucumber) for OrchQueue flows

- Why: Behavior-driven spec traceability for admission, backpressure, cancel, sessions.
- Files/dirs to touch:
  - `test-harness/bdd/features/*.feature` (to be created)
  - `test-harness/bdd/src/steps.rs` (to be created)
- Acceptance Criteria:
  - Features compile and run locally (0 undefined/ambiguous steps).
- Proof:
  - cargo test -p test-harness-bdd -- --nocapture (to be created)

---

## P3 — Nice-to-haves

### 12) Implement xtask wrappers for CI convenience

- Why: Wrap common CI flows (`ci:haiku:*`, `ci:determinism`) with real logic instead of stubs.
- Files/dirs to touch:
  - `xtask/src/main.rs`
- Acceptance Criteria:
  - xtask commands delegate to workspace tests/tools; outputs are deterministic and diff-clean.
- Proof:
  - cargo xtask ci:haiku:cpu
  - cargo xtask ci:determinism

### 13) Docs and onboarding improvements

- Why: Make regeneration and validation flows obvious for contributors.
- Files/dirs to touch:
  - `COMPLIANCE.md` (auto-generated by spec-extract)
  - `.docs/PROJECT_GUIDE.md`
  - `README.md` (to be created)
- Acceptance Criteria:
  - README includes quickstart with quality gates and regeneration instructions.
- Proof:
  - bash ci/scripts/check_links.sh

---

## Cross-References & Traceability

### SPEC references and anchors

- `.specs/orchestrator-spec.md#316-api-contracts-determinism` (SSE contract)
- `.specs/orchestrator-spec.md#32-queues-admission-under-load` (backpressure)
- `.specs/orchestrator-spec.md#310-config-lifecycle` (config validation)
- `.specs/orchestrator-spec.md#39-observability-telemetry` (logging and metrics)

### Contract gaps

- BDD not present; to be created under `test-harness/bdd/`.
- trybuild not present; to be added under affected crates.

### Traceability Map (sample)

- ORCH-2002 → Code: `orchestratord/src/main.rs` (SSE route stubs); Tests: `cli/consumer-tests/tests/stub_wiremock.rs`, `test-harness/determinism-suite/tests/byte_exact.rs`; Contracts: `contracts/openapi/data.yaml` (SSE as text/event-stream)
- ORCH-2007 → Code: `orchestratord/src/main.rs` (admission/backpressure); Tests: `cli/consumer-tests/tests/stub_wiremock.rs`; Contracts: `contracts/openapi/data.yaml` (429 headers)
- ORCH-3030 → Code: `contracts/config-schema/src/lib.rs`; Tests: `contracts/config-schema/tests/validate_examples.rs`; Contracts: `contracts/schemas/config.schema.json`
- ORCH-2101/2102/2103/2104 → Code: `orchestratord/src/main.rs` (routes wired); Tests: extend `orchestratord/tests/provider_verify.rs`; Contracts: `contracts/openapi/control.yaml`
- ORCH-3028 → Code: emission sites (to be added); Tests: `test-harness/metrics-contract/tests/metrics_lint.rs`; Contracts: `ci/metrics.lint.json`

---

## Execution Checklist (run in order)

1) Make fmt/clippy gates green
2) Confirm SPEC link path fix and deterministic regen
3) Guard OpenAPI ↔ types ↔ client drift (data + control)
4) Strengthen SSE contract assertions and snapshots
5) Enable property tests for queue invariants when implementation lands
6) Extend provider verification to control plane
7) Prepare metrics wiring plan; keep lints green
8) Remove unwrap/expect in non-test code
9) Implement EXDEV-safe atomic writes for generators
10) Add trybuild tests for compile-time invariants
11) Introduce BDD scaffolding and ensure 0 ambiguous steps
12) Implement xtask CI wrappers
13) Improve docs/README and contributor flow
