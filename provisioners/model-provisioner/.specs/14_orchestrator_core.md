# Wiring: model-provisioner ↔ orchestrator-core

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. `orchestrator-core` does not call `model-provisioner`.
- `pool-managerd` uses `model-provisioner` to stage models; readiness/capacity snapshots then flow into placement decisions in `orchestrator-core` via `orchestratord`.

## Expectations on model-provisioner
- Provide stable `ResolvedModel { id, local_path }` artifacts that allow engines to start reliably.
- Register/update catalog entries (`catalog-core`) so control plane can reference models consistently by `id`.

## Expectations on orchestrator-core
- Treat model presence as an external input; use model requirements supplied by `orchestratord` (from catalog + adapters) for feasibility checks only.

## Data Flow
- Control: `orchestratord` reload → `pool-managerd` preload → `model-provisioner` ensures model → engine starts → readiness published.
- Placement: snapshots (ready/capacity/perf) → `orchestrator-core` scoring/tie-breakers.

## Refinement Opportunities
- Standardize derivation of `ModelRequirements` to reduce coupling among `orchestratord`, adapters, and `orchestrator-core`.
