# Config Schema — Implementation Plan (OC-CONFIG-6xxx)

Spec: `.specs/60-config-schema.md`
Scope: strict validation, deterministic/idempotent schema generation.

## Stages and Deliverables

- Stage 0 — Schema Generation
  - Implement/extend Rust config types in `contracts/config-schema/`.
  - Emit `config.schema.json` via `schemars`.

- Stage 5 — Tests
  - Validate example configs in tests; unknown fields rejected (strict) or logged (compat mode).

## Tests

- `contracts/config-schema/tests/validate_examples.rs` and related tests.
- BDD config features: `test-harness/bdd/tests/features/config/`.

## Acceptance Criteria

- OC-CONFIG IDs mapped to tests; regen diff-clean; examples validate.

## Backlog (initial)

- Define schemas for pools, engines, quotas, tenants, preemption.
- Strict vs compat mode behavior and logging.

## Proposal (Accepted)

- Align with product Stage 11 — Config & quotas. Provide examples for engines/workers, quotas (concurrent jobs, tokens/min, KV‑MB), determinism flags per engine, and environment conventions.
- DX principles: keep schema generation deterministic; isolate config types in `contracts/config-schema/` so app crates do not rebuild on schema edits; validate examples in tests.
