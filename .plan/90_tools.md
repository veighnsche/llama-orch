# Tools — Implementation Plan (Spec Extract, OpenAPI Client, README Index, Xtask)

Scope: Developer tooling and release helpers that support the contract-first, test-first workflow.

## Stages and Deliverables

- Stage 17 — Compliance & Release
  - `tools/spec-extract/`: extract `requirements/*.yaml` from specs; deterministic outputs.
  - `tools/openapi-client/`: trybuild UI and client codegen tests; verify OpenAPI is consumable.
  - `tools/readme-index/`: index repo READMEs to assist navigation and code discovery.
  - `xtask/`: convenience tasks (e.g., `ci:haiku:cpu`, `ci:determinism`, regen, release bundles).

## Tests

- Spec-extract diff-clean on second run; stable sorting and formatting.
- Trybuild test compilation for the generated client.
- README indexer smoke tests to verify links and counts.

## Acceptance Criteria

- Deterministic outputs and green runs in CI; artifacts consumed by PR checks.
- Release helpers produce reproducible bundles (OpenAPI, Schema, dashboards, COMPLIANCE).

## Backlog (initial)

- One-shot script to produce a "proof bundle" per release (pacts, snapshots, metrics dumps, dashboards).
- CLI wrappers for common developer workflows (e.g., seed determinism runs, quickstart against local CPU llama.cpp).
