# Wiring: pool-managerd ↔ orchestrator-core

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. `orchestrator-core` consumes pool/replica snapshots; `pool-managerd` produces registry state that `orchestratord` exposes to clients and uses to build `PlacementInput`.

## Expectations on pool-managerd
- Maintain and publish accurate fields required for placement:
  - `live`, `ready`, `active_leases` (to derive `slots_free`), optional `draining` flag.
  - Optional capacity/VRAM: `vram_total_bytes`, `vram_free_bytes`, `compute_capability`.
  - Optional perf hints: `perf_tokens_per_s`, `first_token_ms`.
- Apply drains/reloads initiated by control plane and update registry rapidly (heartbeat cadence documented).

## Expectations on orchestrator-core (via orchestratord)
- Treat the registry snapshot as authoritative for feasibility and scoring; prefer pools with `ready=true` and non-draining.
- Do not manipulate pool state; rely on control flows (drain/reload) to change readiness.

## Data Flow
- `pool-managerd` → registry snapshot → `orchestratord` → `PlacementInput.pools` → `orchestrator-core` decides placement.

## Refinement Opportunities
- Define a snapshot schema version and a converter to `PoolSnapshot` with optional fields.
- Add `draining` and `device_mask` to registry for more precise placement filters.
