# BDD Harness — Implementation Plan (Test Journeys)

Scope: End-to-end feature coverage for user journeys across data/control plane, SSE framing, backpressure, fairness, determinism toggles. Uses `test-harness/bdd` binary `bdd-runner` with `LLORCH_BDD_FEATURE_PATH`.

## Stages and Deliverables

- Stage 12 — BDD Coverage (Journeys)
  - Features under `test-harness/bdd/tests/features/`:
    - `data_plane/`: submit → stream → cancel, backpressure behavior.
    - `control_plane/`: drain/reload/health, replica visibility.
    - `scheduling/`: priority fairness bounds, starvation guards.
    - `sse/`: event order (`started|token|metrics|end|error`), well-formed payloads.
    - `determinism/`: toggles and pinning (engine_version, sampler_profile_version).
  - Step library with reusable world/state; proof artifacts (logs, snapshots) emitted per run.

## Tests

- Execute via `cargo run -p test-harness-bdd --bin bdd-runner --quiet`.
- Targeted runs using `LLORCH_BDD_FEATURE_PATH`.

## Acceptance Criteria

- Zero undefined/ambiguous steps.
- Feature sets for the journeys above pass against orchestrator + adapters.
- Artifacts (snapshots/logs) stored deterministically; link from `.docs/DONE/`.

## Backlog (initial)

- Step library modules for SSE parsing, metrics scrape assertions, and fairness checks.
- World hooks for orchestrator lifecycle (start/stop) and pool manager interactions.
