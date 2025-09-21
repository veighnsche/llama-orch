# README for LLM Developers — Home Profile Ruleset

This document defines how contributors (human or LLM) work in this repository. It replaces previous guidance that mentioned “enterprise” or multi-tenant features.

---

## Golden Rules

1. **No backwards compatibility pre‑1.0.0.** Remove dead code; do not keep shims.
2. **Spec is law.** Consult `.specs/` before writing code. If ambiguous, propose a spec update first.
3. **Order of work:** Spec → Contract → Tests → Code. Do not skip steps.
4. **Determinism by default.** Same `{prompt, params, seed}` on the same replica must yield identical streams.
5. **TODO discipline.** Update the root `TODO.md` after every meaningful change; archive with `ci/scripts/archive_todo.sh` when complete.
6. **Investigations are documented.** Commit findings under `.docs/` (or near the component) rather than relying on chat.
7. **Proof bundle required.** Follow `.docs/testing/` guidance when producing logs, pact files, snapshots, metrics.

### Layering & Priorities (Repo-wide)

- **Utils** (`llama-orch-utils`) is the crown jewel programming model (applets, determinism, proof-bundles, guardrails) and drives what the SDK must expose.
- **SDK** (`llama-orch-sdk`) exists to support Utils — types, clients, schema validation, simple transport. No applet/guardrail/prompt logic. Keep surface minimal and stable.
- **Orchestrator** (`orchestratord`) is the service layer whose OpenAPI/specs are the API ground truth the SDK mirrors. It does not dictate Utils logic.
- **CLI** (`llama-orch-cli`) consumes the SDK to bootstrap and generate bindings/snapshots for developers and Blueprints.

---

## Workflow Snapshot

1. Read `.docs/HOME_PROFILE.md`, `.docs/HOME_PROFILE_TARGET.md`, `.docs/workflow.md`.
2. Update relevant specs in `.specs/` with RFC‑2119 language + IDs.
3. Update contracts (`contracts/openapi`, `contracts/config-schema`, `.specs/metrics/otel-prom.md`).
4. Regenerate artifacts: `cargo xtask regen-openapi`, `cargo xtask regen-schema`, `cargo run -p tools-spec-extract --quiet`.
5. Add/tests first: pact, provider verify, BDD, property, determinism, metrics.
6. Implement code changes.
7. Run `cargo xtask dev:loop` (fmt, clippy, regen, tests, link check).
8. Update `TODO.md` with what changed and why.

---

## Stage Tracker (Home Profile)

| Stage | Description | Status |
|-------|-------------|--------|
| 0 | Contract freeze (OpenAPI/config/metrics) | ✅ |
| 1 | Consumer contracts (CLI pact) | ✅ |
| 2 | Provider verification | ✅ |
| 3 | Queue invariants | ✅ |
| 4 | Determinism suite | ✅ |
| 5 | Observability basics | ✅ |
| 6 | Admission → SSE vertical | 🚧 |
| 7 | Catalog & reloads | ☐ |
| 8 | Capability discovery | ☐ |
| 9 | Mixed-GPU placement heuristics | ☐ |
| 10 | Budgets & sessions | ☐ |
| 11 | Tooling policy | ☐ |
| 12 | BDD coverage | ☐ |
| 13 | Dashboards & alerts | ☐ |
| 14 | Startup self-tests | ☐ |
| 15 | Haiku anti-cheat | ☐ |

Stages 16+ (nightly chaos, release prep) will be defined later.

---

## Testing Quick Reference

- `cargo test --workspace --all-features -- --nocapture`
- `cargo test -p orchestratord --test provider_verify`
- `cargo test -p test-harness-bdd -- --nocapture`
- `cargo test -p test-harness-determinism-suite`
- `cargo test -p test-harness-e2e-haiku` (requires real GPU)
- `cargo xtask dev:loop`

See `.docs/testing/spec-derived-test-catalog.md` for requirement/test mappings.

---

## Metrics & Logs

- Metrics contract in `.specs/metrics/otel-prom.md`; linter config `ci/metrics.lint.json` must match.
- Minimum metrics: queue depth, tasks counters, tokens in/out, GPU & VRAM gauges, optional model_state (Active|Retired).
- Logs must include `job_id`, `session_id`, `engine`, `engine_version`, `pool_id`, `replica_id`, `queue_position`, `predicted_start_ms`, `tokens_in`, `tokens_out`, `decode_time_ms`.
- Narrative logging: the `human` field MUST be natural-language and MUST NOT primarily consist of opaque identifiers (UUIDs, hashes). Keep raw identifiers in structured fields (e.g., `job_id`, `session_id`, `pool_id`). Prefer descriptive phrasing (e.g., "Asked the pool-managerd about the status of pool 'default'").

---

## Reference Environment

All smoke tests must pass on the workstation described in `.docs/HOME_PROFILE_TARGET.md` (RTX 3090 + 3060). If you change anything that touches placement, catalog, artifacts, or budgets, re-run the environment smoke before merging.

---

## TODO.md Lifecycle

- Keep the root `TODO.md` as the single live tracker.
- Each change must annotate what happened, where, and why.
- When finished, run `ci/scripts/archive_todo.sh` to move the log to `.docs/DONE/` and create a fresh TODO template if needed.

---

## PR Checklist

- Specs/contracts updated and regen commands run.
- Tests added/updated referencing requirement IDs.
- `cargo xtask dev:loop` green.
- Links checked (`bash ci/scripts/check_links.sh`).
- `TODO.md` updated; documentation touched if behaviour changed.

---

## Environment Conventions

- Set `LLORCH_API_TOKEN` for CLI/tests.
- Use `REQUIRE_REAL_LLAMA=1` when running determinism or Haiku suites requiring a real worker.
- For BDD tests requiring tunnels, document steps in `.docs/HOME_PROFILE_TARGET.md`.

---

Home lab first. Keep the documentation honest, the queue lean, and the GPUs busy.
