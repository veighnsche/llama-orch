# Determinism Suite — Implementation Plan (OC-TEST-7xxx)

Spec: `.specs/70-determinism-suite.md`
Scope: byte-exact replica streams per engine with fixed seeds and engine-appropriate settings.

## Stages and Deliverables

- Stage 4 — Determinism
  - Seeds corpus (≥64) pinned and documented.
  - Engine settings for single-slot/single-request; llama.cpp `--parallel 1 --no-cont-batching`.
  - Runner executes per engine with two replicas and asserts byte-exactness (first-32 and full stream).

## Tests

- `test-harness/determinism-suite/tests/`.
- BDD determinism features: `test-harness/bdd/tests/features/determinism/`.

## Acceptance Criteria

- OC-TEST IDs mapped; byte-exactness passes per engine; failures emit token diff artifact.

## Backlog (initial)

- Engine launcher helpers; token diff tooling; snapshot fixtures for debugging.
