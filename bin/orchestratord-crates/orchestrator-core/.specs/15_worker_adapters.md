# Wiring: orchestrator-core ↔ worker-adapters

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. `orchestrator-core` does not depend on adapters directly.
- Adapters are invoked by `orchestratord` to serve tasks after placement. Core provides placement and admission decisions only.

## Expectations on worker-adapters (via orchestratord)
- Implement `WorkerAdapter` (health, props, submit/cancel, engine_version) and map errors/timeouts.
- Emit streaming frames that `orchestratord` forwards to clients.

## Expectations on orchestrator-core
- Treat adapter behavior as opaque; only consume pool snapshots and job specs for placement.

## Data Flow
- `orchestrator-core` → placement decision → `orchestratord` selects adapter instance → adapter streams tokens.

## Refinement Opportunities
- Define a minimal snapshot→adapter binding in `orchestratord` so placement decisions are logged alongside adapter identity.
