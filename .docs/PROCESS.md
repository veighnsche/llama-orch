# Full Process for Spec Changes (Spec → Contracts → Tests → Code)

This document is a practical, end‑to‑end checklist for making and landing a SPEC change safely across the workspace. It covers discovery, implementation, regeneration, verification, and roll‑back considerations.

Scope and audiences:
- SPEC authors proposing changes in `.specs/` or requirements in `requirements/`.
- Engineers implementing the change across contracts (OpenAPI + config schema), code, and tests.
- Reviewers verifying adherence and CI health.

---

## Primary Sources of Truth

- SPEC documents: `.specs/**` (e.g., `.specs/00_llama-orch.md`, proposals/*, metrics/*)
- Requirements index: `requirements/*.yaml`
- OpenAPI contracts (control/data): `contracts/openapi/control.yaml`, `contracts/openapi/data.yaml`
- Config schema types: `contracts/config-schema/src/**/*.rs` → emits `contracts/schemas/config.schema.json`
- Metrics contract: CI linter `ci/metrics.lint.json` + tests

Generated artifacts and helpers:
- API types (data/control): `contracts/api-types/src/generated*.rs`
- Minimal client: `tools/openapi-client/src/generated.rs`
- Spec extract index: `tools/spec-extract` (feeds docs/testing)
- Xtask entrypoints: `cargo xtask regen-openapi`, `cargo xtask regen-schema`, `cargo xtask spec-extract`

Tests that enforce contracts:
- Provider/OpenAPI checks: `orchestratord/tests/provider_verify.rs`
- Metrics lint: `orchestratord/src/main.rs` metrics check + `ci/metrics.lint.json`
- Inline unit tests: colocated `#[cfg(test)]` (e.g., `orchestratord/src/http/*.rs`, `orchestratord/src/metrics.rs`)
- Integration tests: `orchestratord/tests/*.rs`, test harness crates under `test-harness/*`

---

## Typical Change Surfaces (“Blast Radius”)

- OpenAPI additions/renames/removals
  - Update `contracts/openapi/*.yaml`.
  - Regenerate API types and client (xtask), update handlers and tests.
  - Provider tests will fail until handlers match.

- Config schema shape changes
  - Update Rust types under `contracts/config-schema/src/*`.
  - Regenerate JSON schema (xtask), commit the new `contracts/schemas/config.schema.json`.

- Metrics names/labels/semantics
  - Update definitions in `orchestratord/src/metrics.rs`.
  - Update `ci/metrics.lint.json` labels and required metrics.
  - Fix references in code paths emitting metrics and in tests.

- Error taxonomy and codes
  - For data plane: derive from OpenAPI enums; regen types.
  - Update handler mappings in `orchestratord/src/http/data.rs` and related SSE/error mapping.
  - Update pact/provider tests if shape or headers change.

- SSE streaming shape (events, fields)
  - Update `orchestratord/src/http/data.rs` streaming transcript.
  - Align with OpenAPI examples and provider tests that assert budgets/SSE shape.

- Control plane semantics (drain/reload/health/capabilities)
  - Update handlers in `orchestratord/src/http/control.rs`.
  - Ensure `pool-managerd` interfaces and state reflect the changes.

- Documentation and testing inventory
  - Update `.docs/testing/bdd-mapping.md` and `.docs/testing/infile-test-inventory.md` to reflect new/changed behaviors.

---

## Step‑By‑Step Process

1) Proposal and Alignment
- Create or update SPEC doc in `.specs/**` with requirement IDs (e.g., ORCH‑####, OC‑CORE‑####).
- If applicable, add/update a requirements YAML under `requirements/`.
- Include a concise change summary and an example request/response or config snippet.

2) Contracts First
- OpenAPI: modify `contracts/openapi/*.yaml` (paths, schemas, headers, enums, examples).
- Config schema: modify Rust types in `contracts/config-schema/src/*`.
- Metrics: adjust `ci/metrics.lint.json` if adding/renaming metrics or label sets.

3) Regenerate Artifacts (deterministic)
- Run: `cargo xtask regen-openapi` (validates OAPI, writes generated API types and client)
- Run: `cargo xtask regen-schema` (emits `contracts/schemas/config.schema.json`)
- Optional: `cargo xtask spec-extract` to refresh spec index for docs/testing.

4) Implement Code Changes
- Data plane handlers: `orchestratord/src/http/data.rs` (admission, SSE, errors, budgets, backpressure headers).
- Control plane handlers: `orchestratord/src/http/control.rs` (drain/reload/health/capabilities).
- Metrics emission points: `orchestratord/src/metrics.rs` and any callers.
- Pool/registry changes: `pool-managerd/src/*.rs`.
- Worker adapter interfaces if SSE/error contracts shift: `worker-adapters/adapter-api` + mocks under `worker-adapters/mock`.

5) Update and/or Add Tests
- Provider tests: `orchestratord/tests/provider_verify.rs` (statuses, headers, SSE metrics fields, shapes).
- Metrics tests: lint in `orchestratord/src/main.rs` and unit tests in `orchestratord/src/metrics.rs`.
- Inline unit tests in affected handlers.
- Integration tests and BDD mappings if behavior changes.

6) Verify End‑to‑End
- Fast loop: `cargo xtask dev:loop` (fmt, clippy, regen, workspace tests, link check).
- Or targeted:
  - `cargo test -p orchestratord -- --nocapture`
  - `cargo test --workspace`
- If metrics names/labels changed, ensure linter and tests agree.

7) Documentation & Tracking
- Update `.docs/testing/bdd-mapping.md` and `.docs/testing/infile-test-inventory.md` to reflect new or modified behaviors.
- Update `TODO.md` with any follow‑ups and Phase items.
- If you add/rename requirement IDs, search and update references across code and docs (see Queries below).

8) PR Hygiene
- Include SPEC diff summary, impacted surfaces, and before/after examples.
- List regenerated files explicitly.
- Link to passing CI and any pact/provider outputs.
- Include rollback note if the change is risky.

---

## Queries to Identify Impact

Run these at repo root to find references quickly.

- OpenAPI path and handler boundaries:
```
rg -n "pub\s+async\s+fn\s+.*->\s+Response|axum::(extract|response)" orchestratord/src
```

- Metrics names and emit points:
```
rg -n "metrics::|register_.*_with_registry|_TOTAL|_RATIO|_BYTES|_MS|QUEUE_DEPTH" orchestratord/src
```

- Error taxonomy mentions (adapter mapping, envelopes):
```
rg -n "ErrorEnvelope|ErrorKind|AdapterErr|WorkerError|code\":\s*\"[A-Z_]+\"" orchestratord/src
```

- Requirement IDs referenced in code/tests/docs:
```
rg -n "\b(ORCH|OC-CORE|OC-POOL)-\d{3,4}\b" --hidden
```

- Generated types affected by OpenAPI:
```
rg -n "contracts/api-types/src/(generated|generated_control).rs"
```

---

## Command Reference

- Validate and regenerate contracts:
  - `cargo xtask regen-openapi`
  - `cargo xtask regen-schema`
  - `cargo xtask spec-extract`

- Full developer loop:
  - `cargo xtask dev:loop`

- Targeted tests:
  - `cargo test -p orchestratord -- --nocapture`
  - `cargo test --workspace`

---

## Acceptance Checklist for a SPEC Change PR

- __Specs updated__: `.specs/**` and/or `requirements/*.yaml` with requirement IDs.
- __Contracts updated__: OpenAPI and/or config schema sources edited.
- __Artifacts regenerated__: API types, client, config schema JSON.
- __Code updated__: handlers, metrics, registry/placement, adapters (as needed).
- __Tests passing__: provider tests, metrics lint, inline tests, workspace tests.
- __Docs updated__: BDD mapping and in‑file test inventory reflect new/changed behavior.
- __CI green__: clippy, tests, link checker, any pact verifications.

---

## Rollback and Compatibility Notes

- If changing metrics label sets, consider temporary dual‑write or keep old labels for a deprecation window; update `ci/metrics.lint.json` accordingly.
- For OpenAPI breaking changes, consider versioned endpoints or default responses while clients catch up.
- For config schema changes, keep fields optional initially and enforce later.

---

## Example: Metric Rename Workflow

1. Edit `orchestratord/src/metrics.rs` names/labels.
2. Update `ci/metrics.lint.json` required metric and labels.
3. Adjust emits in calling sites (e.g., `admission.rs`, SSE in `data.rs`).
4. Update tests: `orchestratord/tests/admission_metrics.rs`, `orchestratord/src/metrics.rs` unit tests.
5. Run `cargo xtask dev:loop` and fix regressions.

## Example: Add Field to AdmissionResponse

1. Update `contracts/openapi/data.yaml` schema for `AdmissionResponse`.
2. `cargo xtask regen-openapi` (regenerates `contracts/api-types/*` and client).
3. Update `orchestratord/src/http/data.rs` to populate the field.
4. Update provider test assertions if shape changes.
5. Run tests.

---

Maintainers can extend this document with crate‑specific pitfalls or additional commands as the system evolves.
