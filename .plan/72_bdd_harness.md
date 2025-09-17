# BDD Harness Plan — Home Profile

## Scope
Exercise end-to-end user journeys for the home profile using `test-harness/bdd`.

## Feature Groups
- **Admission & Streaming**: enqueue, stream SSE (`started|token|metrics|end|error`), cancel.
- **Sessions & Budgets**: TTL expiry, manual eviction, advisory budgets, budget exhaustion.
- **Catalog & Reloads**: upload model (unsigned warning), drain, reload success/rollback, health endpoint.
- **Artifacts**: store/retrieve plan snapshot via `/v1/artifacts`.
- **Placement**: mixed-GPU execution (RTX 3090 + 3060), queue metadata, predicted start.
- **Capability Discovery**: CLI reads limits from `/v1/replicasets` or `/v1/capabilities`.
- **Tooling Policy**: outbound HTTP request allowed/denied via policy hook.

## Test Harness Notes
- Runner: `test-harness/bdd` binary (`cargo test -p test-harness-bdd`).
- Step definitions live under `test-harness/bdd/src/steps/`.
- Keep feature files tagged with requirement IDs (`ORCH-`, `HME-`).

## Backlog
- Add mixed-GPU scenario once placement heuristic is implemented.
- Add tooling policy scenario once hook is wired.
- Add reference smoke wrapper to run critical features against physical workstation before release.

## Deliverables
- Zero undefined/ambiguous steps (`cargo test -p test-harness-bdd -- --nocapture`).
- Updated snapshots/log archives stored alongside features when behaviour changes.
