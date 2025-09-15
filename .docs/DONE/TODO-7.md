# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Blockers (in order)

- [x] S0.1 — Review `contracts/openapi/data.yaml` for Stage 0
  - Verify `Engine` enum present and referenced by public resources (e.g., `TaskRequest`, typed errors) per OrchQueue v1.
  - Verify `x-req-id` on all relevant endpoints (`/v1/tasks`, `/v1/tasks/{id}/stream`, `/v1/tasks/{id}/cancel`, `/v1/sessions/*`).
  - Confirm title describes OrchQueue v1 and SSE event names/fields are declared.
- [x] S0.2 — Review `contracts/openapi/control.yaml` for Stage 0
  - Verify `Engine` enum present where applicable and `x-req-id` annotations on control endpoints.
  - Confirm pool drain/reload/health and replicasets surfaces match SPEC.
- [x] S0.3 — Generate API types and server stubs
  - Run: `cargo xtask regen-openapi` to emit/update `contracts/api-types` and stubs in `orchestratord/`.
  - Result: regen-openapi OK; generated files unchanged (`contracts/api-types/src/generated.rs`, `tools/openapi-client/src/generated.rs`).
- [x] S0.4 — Emit JSON Schema from Rust config types
  - Run: `cargo xtask regen-schema` to regenerate `config.schema.json`.
  - Result: regen-schema OK; output unchanged (`contracts/schemas/config.schema.json`).
- [x] S0.5 — Regenerate requirements index from SPEC
  - Run: `cargo run -p tools-spec-extract --quiet` then `git diff --name-only`.
  - Result: requirements index OK; all requirement YAMLs unchanged.
- [ ] S0.6 — Verify workspace health (fmt/lint optional here) and record results
  - Optional quick check: `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings`
  - Record any follow-ups if warnings/errors appear.
  - Current known/handled: addressed lints in BDD harness (see Progress Log 2025-09-15).
  - Current outcome: `cargo fmt --all -- --check` reported formatting differences across several files (e.g., `test-harness/bdd/src/steps/scheduling.rs`, `.../security.rs`, tests). Next action: run `cargo fmt --all` to apply formatting. Clippy not yet executed.

## Progress Log (what changed)

- 2025-09-15: Initialized BDD harness (step registry + core feature files); tests pass.
- 2025-09-15: Expanded BDD step registry to cover lifecycle, WFQ quotas, deadlines/SSE metrics, preemption, error taxonomy, security, pool-manager lifecycle, config-schema, determinism, and metrics/observability. Added matching Gherkin features under `test-harness/bdd/tests/features/`.
- 2025-09-15: Modularized `tools/readme-index` into a library with clear modules (`types`, `workspace`, `openapi_utils`, `render`, `root_readme`, `path_utils`, `io_utils`) and a minimal binary entrypoint `src/bin.rs` that calls `tools_readme_index::run()`.
- 2025-09-15: Cleaned unused imports in the new modules; adjusted Cargo configuration to use explicit bin (`autobins = false`).
- 2025-09-15: Ran `cargo test --workspace --all-features` — all crates green; BDD coverage test (`features_have_no_undefined_or_ambiguous_steps`) passes.
- 2025-09-15: Stage 0 S0.1 and S0.2 reviewed — OpenAPI data/control have `Engine` enums, SSE events, and `x-req-id` annotations; ready for regeneration.
- 2025-09-15: Stage 0 S0.3 — Ran `cargo xtask regen-openapi`: validated OpenAPI; generated files unchanged; OK.
- 2025-09-15: Stage 0 S0.4 — Ran `cargo xtask regen-schema`: schema emission OK; `contracts/schemas/config.schema.json` unchanged.
- 2025-09-15: Stage 0 S0.5 — Ran `cargo run -p tools-spec-extract --quiet`: requirements index unchanged; OK.
- 2025-09-15: Stage 0 S0.6 — Health check (partial): `cargo fmt --all -- --check` reports formatting needed; plan to run `cargo fmt --all` and then `cargo clippy --all-targets --all-features -- -D warnings`.
- 2025-09-15: Addressed BDD harness lints and cleanup:
  - `test-harness/bdd/src/steps/mod.rs`: add `#[cfg_attr(not(test), allow(dead_code))]` on `registry()` to silence dead_code in non-test builds; `tests/bdd.rs` uses it at test time.
  - `test-harness/bdd/src/steps/preflight_steps.rs`: read and set `World.mode_commit` to gate DryRun assertion; resolves "field mode_commit is never read" lint and models commit mode behavior.
  - `test-harness/bdd/Cargo.toml`: remove stale `[[bin]] bdd-generate-skeletons` entry after deleting `src/bin/bdd-generate-skeletons.rs`.
  - Verified `contracts/config-schema` exposes `emit_schema_json()` used by `xtask regen-schema`.
