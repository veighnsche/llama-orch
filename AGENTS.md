# Repository Guidelines

## Project Structure & Module Organization
llama-orch is a Rust workspace built around Stage 6 (Admission → Dispatch → SSE). `orchestratord/` owns HTTP routes and SSE plumbing, `orchestrator-core/` enforces queue invariants and metrics wrappers, and adapters reside in `worker-adapters/`. Specs live in `.specs/`, contracts in `contracts/`, and generated tools in `xtask/` and `tools/`. Scenario, chaos, and determinism suites are under `test-harness/`. Update specs and contracts before touching runtime code.

## Spec-First Workflow
Golden rules: no backwards compat pre-1.0.0, spec is law, determinism by default. Follow Spec → Contract → Tests → Code, and document investigations in `.docs/`. Keep `TODO.md` current after each change and archive via `ci/scripts/archive_todo.sh` when done. Any user-visible behavior change must ship with a proof bundle per `.docs/testing/`.

## Build, Test & Development Commands
- `cargo fmt --all -- --check` and `cargo clippy --all-targets --all-features -- -D warnings` gate formatting and linting.
- `cargo test --workspace --all-features -- --nocapture` spans unit, integration, and harness smoke.
- `cargo xtask regen-openapi`, `cargo xtask regen-schema`, and `cargo run -p tools-spec-extract --quiet` refresh artifacts after contract edits.
- `cargo xtask dev:loop` runs fmt, clippy, regen, tests, and linkcheck (`bash ci/scripts/check_links.sh`).

## Coding Style & Naming Conventions
Use rustfmt defaults (4-space indent, trailing commas) and keep Clippy clean. Modules/files stay `snake_case`, public types `UpperCamelCase`, constants `SCREAMING_SNAKE_CASE`. Align log fields with README_LLM (`job_id`, `session_id`, `engine`, `pool_id`, `replica_id`, `queue_position`, `predicted_start_ms`, `tokens_in`, `tokens_out`, `decode_time_ms`) and sync metric names with `.specs/metrics/otel-prom.md`.

## Testing Guidelines
Add tests alongside every contract or behavior change. Targeted commands: `cargo test -p orchestratord --test provider_verify -- --nocapture`, `cargo test -p test-harness-bdd -- --nocapture`, determinism via `cargo test -p test-harness-determinism-suite`, and GPU Haiku with `cargo test -p test-harness-e2e-haiku` (`REQUIRE_REAL_LLAMA=1`). Keep pact files, snapshots, and metrics outputs in the proof bundle.

## Commit, TODO & PR Discipline
Prefix commits with requirement IDs (`ORCH-####: present-tense summary`) and bundle spec, contract, test, and code in the same change. Update `TODO.md` and any affected `.specs/` or `.docs/` entries before pushing. Pull requests should list verification commands (ideally `cargo xtask dev:loop`), link issues, and include artifacts (logs, SSE transcripts) when behavior shifts.

## Operational Notes
Secrets stay out of the repo; adapters load via environment. Set `LLORCH_API_TOKEN` for CLI flows and `REQUIRE_REAL_LLAMA=1` for GPU suites. Ensure smoke coverage on the home profile hardware defined in `.docs/HOME_PROFILE_TARGET.md`, and keep CI mirrors of any new scripts inside `ci/`.
