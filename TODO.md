# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Blockers (in order)

- [ ]

## Progress Log (what changed)

- 2025-09-15: Initialized BDD harness (step registry + core feature files); tests pass.
- 2025-09-15: Expanded BDD step registry to cover lifecycle, WFQ quotas, deadlines/SSE metrics, preemption, error taxonomy, security, pool-manager lifecycle, config-schema, determinism, and metrics/observability. Added matching Gherkin features under `test-harness/bdd/tests/features/`.
- 2025-09-15: Modularized `tools/readme-index` into a library with clear modules (`types`, `workspace`, `openapi_utils`, `render`, `root_readme`, `path_utils`, `io_utils`) and a minimal binary entrypoint `src/bin.rs` that calls `tools_readme_index::run()`.
- 2025-09-15: Cleaned unused imports in the new modules; adjusted Cargo configuration to use explicit bin (`autobins = false`).
- 2025-09-15: Ran `cargo test --workspace --all-features` — all crates green; BDD coverage test (`features_have_no_undefined_or_ambiguous_steps`) passes.
