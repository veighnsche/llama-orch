# Umbrella Plan — Spec to Ship

This plan coordinates work across all components to satisfy the home profile spec.

## Key Workstreams

1. **Specs & Requirements**
   - Maintain `.specs/00_llama-orch.md` and `.specs/00_home_profile.md` as the single truth.
   - Regenerate `requirements/*.yaml` after every spec update.

2. **Contracts**
   - OpenAPI: keep data/control plane definitions up to date with SSE frames, catalog, artifacts, capability discovery.
   - Config schema: ensure pools, adapters, sessions, budgets, and placement settings are covered.
   - Metrics: maintain minimal set in `.specs/metrics/otel-prom.md` and linter.

3. **Implementation Phases**
   - Finish Stage 6–8 vertical slices (admission, SSE, catalog, capabilities).
   - Implement placement heuristics for mixed GPUs.
   - Wire session registry, budgets, and policy hook.
   - Harden observability (dashboards, alerts) and startup checks.
   - Run reference environment smoke before release.

4. **Testing Discipline**
   - Pact/BDD/determinism suites reference requirement IDs.
   - Metrics lint and Haiku tests are non-negotiable gates.
   - Maintain `.docs/testing/spec-derived-test-catalog.md` + combination matrix.

## Coordination Notes

- `README_LLM.md` and `.docs/workflow.md` track stage status; keep them synced with actual progress.
- `TODO.md` is the live backlog; archive with `ci/scripts/archive_todo.sh` when finished.
- Use proposals under `.specs/proposals/` for major spec changes.

## Risks

- Reference workstation validation must stay part of every release candidate.
- Mixed-GPU heuristics require real telemetry; avoid guesswork.
- Artifact retention/backups need a documented story before GA.
