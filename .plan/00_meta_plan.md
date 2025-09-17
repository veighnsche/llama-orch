# Meta Implementation Plan — Home Profile

Purpose: coordinate Spec → Contract → Tests → Code delivery for the single-host home lab. This replaces older plans that referenced enterprise features.

## Stage Summary

1. Contract freeze (OpenAPI/config schema/metrics)
2. Consumer contracts (CLI pact + snapshots)
3. Provider verification (orchestratord)
4. Queue invariants (orchestrator-core proptests)
5. Observability (metrics/logs)
6. Admission → SSE vertical
7. Catalog & reloads
8. Capability discovery
9. Placement heuristics for mixed GPUs
10. Budgets & sessions
11. Tooling policy & auth
12. BDD coverage
13. Dashboards & alerts
14. Startup self-tests
15. Haiku anti-cheat
16+. Nightly chaos & release prep (TBD)

## Roles & Ownership

| Area | Owner | Key Specs |
|------|-------|-----------|
| Contracts | Contracts guild | `.specs/20-orchestratord.md`, `.specs/00_home_profile.md` |
| Orchestrator core | Core guild | `.specs/10-orchestrator-core.md` |
| HTTP handlers | Orchestratord guild | `.specs/20-orchestroratord.md` |
| Pool manager | Pool guild | `.specs/30-pool-managerd.md` |
| Adapters | Adapter guild | `.specs/40-43-worker-adapters-*.md` |
| Policy/tooling | Policy guild | `.specs/50-51-*.md` |
| Config & docs | Docs guild | `.docs/*`, `.specs/*` |

## Workflow Checklist

1. Update specs (`.specs/**`) with RFC-2119 language & IDs.
2. Update contracts (`contracts/openapi`, `contracts/config-schema`, `.specs/metrics/otel-prom.md`).
3. Regenerate artifacts (`cargo xtask regen-openapi`, `cargo xtask regen-schema`, `cargo run -p tools-spec-extract --quiet`).
4. Add/adjust tests (pact, provider, BDD, determinism, metrics).
5. Implement code.
6. Run `cargo xtask dev:loop`.
7. Update `TODO.md` and archive when done.

## Deliverable Highlights

- **Admission metadata**: queue position, predicted start, budgets.
- **SSE**: deterministic streaming with `metrics` frames.
- **Catalog/artifacts**: local storage, verification warnings, fetch APIs.
- **Placement**: least-loaded GPU heuristic validated on RTX 3090 + 3060.
- **Capability discovery**: CLI-safe limits exposed.
- **Budgets**: optional enforcement plus telemetry.
- **Policy**: outbound HTTP guardrails with audit logs.

## Testing Matrix (High Level)

- Provider verify — `orchestratord/tests/provider_verify.rs`
- BDD — `test-harness/bdd/tests/features/`
- Determinism — `test-harness/determinism-suite/`
- Metrics lint — `test-harness/metrics-contract/`
- Haiku — `test-harness/e2e-haiku/`
- Reference smoke (planned) — `.docs/HOME_PROFILE_TARGET.md`

## Documentation Tasks

- Keep `.docs/HOME_PROFILE.md`, `.docs/workflow.md`, `.docs/HOME_PROFILE_TARGET.md` current.
- Update `.docs/testing/spec-derived-test-catalog.md` and `.docs/testing/spec-combination-matrix.md` when specs/tests change.
- Archive TODOs in `.docs/DONE/` via `ci/scripts/archive_todo.sh`.

## Risks / Open Questions

- Mixed-GPU heuristics need real-hardware validation.
- Artifact retention policy and backup strategy pending.
- CLI/tooling integration for policy hook still outstanding.

Track progress via README stage badges and the root TODO.
