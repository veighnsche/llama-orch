# Wiring: orchestratord ↔ orchestrator-core

Status: Draft
Date: 2025-09-19

## Relationship
- `orchestratord` is the HTTP boundary and application host. It consumes placement and queue/admission primitives from `orchestrator-core`.

## Expectations on orchestrator-core
- Provide queue/admission façade (bounded FIFO with policies) and data shapes for placement: `PlacementInput`, `PoolSnapshot`, `JobSpec`, `PlacementDecision`.
- Placement policy: feasibility (model/engine/device) first, then scoring with deterministic tie-breakers.

## Expectations on orchestratord
- Convert HTTP inputs (TaskRequest) into `JobSpec` and assemble `PlacementInput` from pool snapshots.
- Enforce admission/backpressure policy; return 429 with retry-after when applicable.
- Surface queue position and predicted_start_ms in `started`/`metrics` frames.

## Data Flow
- Admission: HTTP POST `/v1/tasks` → build `JobSpec` → `place()` → choose pool/adapter → stream via SSE.
- Control: `/v1/pools/*` and `/v1/capabilities` aggregate core- and registry-derived state.

## Error Handling
- Map core errors to HTTP error envelopes; include `X-Correlation-Id`.

## Refinement Opportunities
- Provide a thin adapter from pool-managerd registry entries to `PoolSnapshot` with optional perf/version fields.
