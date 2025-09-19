# Wiring: orchestrator-core ↔ model-provisioner

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. `orchestrator-core` does not call `model-provisioner` directly.
- `pool-managerd` uses `model-provisioner` during preload; `orchestratord` exposes control APIs that cause those actions.

## Expectations on model-provisioner
- Provide `ResolvedModel { id, local_path }` to `pool-managerd`/`engine-provisioner`, which ultimately influences readiness and pool snapshots consumed by `orchestrator-core`.
- Callers SHOULD first consult catalog-core read-only helpers (`exists(id|ref)`, `locate(ModelRef)`) to avoid redundant staging.

## Expectations on orchestrator-core
- Treat model presence/paths as already resolved by lower layers; only consider model requirements (min VRAM, quantization, etc.) for feasibility checks.
- `ModelRequirements` is defined canonically at `/.specs/10-orchestrator-core.md` and derived upstream from catalog + adapter metadata; core treats it as an opaque input.

## Data Flow
- Control: Orchestrator control → pool-managerd → model-provisioner → catalog update.
- Placement: Orchestrator aggregates requirements → `orchestrator-core` decides on eligible pools.

## Refinement Opportunities
- Define a stable `ModelRequirements` struct derivation from `ResolvedModel` + adapter metadata to avoid tight coupling.
