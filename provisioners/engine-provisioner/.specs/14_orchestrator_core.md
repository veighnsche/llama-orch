# Wiring: engine-provisioner ↔ orchestrator-core

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. `orchestrator-core` is a logic library for placement and does not call `engine-provisioner`.
- `pool-managerd` invokes `engine-provisioner` and publishes readiness/capacity snapshots that `orchestrator-core` consumes via `orchestratord`.

## Expectations on engine-provisioner
- Produce consistent engine runtime behavior so that upstream components can rely on stable capacity and versions.
- Provide clear errors and fallback signals (e.g., CPU-only) that propagate to `pool-managerd` for registry updates.

## Expectations on orchestrator-core
- Treat engine readiness/capacity as inputs only; do not attempt to manipulate engines.
- Use `engine_version` and perf hints from pool snapshots to score and break ties; do not query providers directly.

## Data Flow
- Control: Orchestrator control (drain/reload) → pool-managerd → engine-provisioner → engine runtime.
- Placement: pool-managerd registry → orchestrator snapshot → orchestrator-core placement.

## Refinement Opportunities
- Define a minimal set of perf/version fields that providers should expose for scoring, and a snapshot converter into `PoolSnapshot`.
