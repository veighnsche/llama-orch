# Policy Host & SDK — Implementation Plan (OC-POLICY-4xxx)

Specs: `.specs/50-plugins-policy-host.md`, `.specs/51-plugins-policy-sdk.md`
Scope: WASI ABI host/SDK, deterministic pure functions, sandboxing/time/memory bounds, telemetry.

## Stages and Deliverables

- Stage 2 — Host/SDK Foundations
  - Host loads WASI plugins; SDK exposes stable surface; explicit ABI versioning.

- Stage 5 — Observability
  - Log plugin id/version, decision outcome, and latency; no net/fs by default.

## Tests

- Unit/integration tests in `plugins/policy-host/` and `plugins/policy-sdk/`.
- BDD policy features: `test-harness/bdd/tests/features/policy/`.

## Acceptance Criteria

- OC-POLICY and OC-POLICY-SDK IDs mapped to tests; sandbox constraints enforced; ABI deterministic and versioned.

## Backlog (initial)

- WASI runtime integration and capability guards.
- SDK function signatures and error model.
- Telemetry hooks and configs.

## Proposal (Accepted)

- Align with product stages:
  - Stage 9 (scheduling hooks) MAY consume policy decisions via a deterministic WASI call (optional in v0.1).
  - Stage 11 (config & quotas) provides policy configuration; Stage 12 (BDD) covers policy selection flows.
- DX layering:
  - Keep `plugins/policy-host` and `plugins/policy-sdk` independent of HTTP; surface pure domain types (consider reusing `orch-domain` types once extracted).
  - Enforce sandbox/time/memory limits; default deny network/filesystem.
  - Provide small, stable SDK surface with explicit ABI versioning and deterministic semantics.
