# Wiring: orchestrator-core ↔ engine-provisioner

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. `orchestrator-core` has no direct dependency on `engine-provisioner`.
- `pool-managerd` calls `engine-provisioner::EngineProvisioner::ensure` during preload; `orchestratord` exposes control flows (drain/reload) that cause `pool-managerd` to act.

## Expectations on engine-provisioner
- Prepare engines according to `PoolConfig`; expose clear errors and diagnostics when CUDA/GPU is unavailable. GPU is required; failures must be surfaced promptly (fail fast).

## Expectations on orchestrator-core (via callers)
- Treat engine readiness and capacity as inputs from `pool-managerd`/`orchestratord` snapshots; do not attempt to start/stop engines.
- Use engine version and perf hints (if provided) for scoring/tie-breakers; do not query providers directly.

## Data Flow
- Control plane: Orchestrator control triggers drain/reload → pool-managerd → engine-provisioner.
- Placement plane: Orchestrator-core consumes pool snapshots after readiness changes.

## Refinement Opportunities
- Define a minimal `PoolSnapshot` → `PlacementInput` converter that includes engine_version and perf hints when available.
