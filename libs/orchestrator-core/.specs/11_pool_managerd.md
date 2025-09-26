# Wiring: orchestrator-core ↔ pool-managerd

Status: Draft
Date: 2025-09-19

## Relationship
- orchestrator-core is a logic library that consumes pool/replica snapshots to make placement decisions.
- pool-managerd produces pool state (health, capacity, device and perf hints) via its in-memory registry.
- Typically `orchestratord` mediates the exchange, but the data model is defined here for clarity.

## Expectations on pool-managerd
- Publish snapshots covering at least:
  - `pool_id`, `engine`, `slots_total`, `slots_free`.
  - `vram_total_bytes`, `vram_free_bytes`, `compute_capability`.
  - Optional perf hints: `perf_tokens_per_s`, `first_token_ms`.
  - Optional flags: `draining` (avoid new placements), and device mask.
- Maintain non-negative `active_leases` and a consistent notion of `slots_free`.

## Expectations on orchestrator-core
- Check feasibility first (compatibility predicate) before scoring.
- Score by `predicted_end_ms` using `perf_tokens_per_s` and `first_token_ms` when available; fall back to deterministic tie-breakers.
- Prefer pools with warm KV/session affinity when applicable (input provided by callers).

## Data Flow
- Input to core: `PlacementInput { pools: Vec<PoolSnapshot>, job: JobSpec }` populated from pool-managerd registry.
- Output from core: `PlacementDecision { Assigned { pool_id } | NoCapacity { reason } }`.

## Error Handling
- If snapshots are empty or stale, return `NoCapacity` with an explanatory reason (callers decide retry/backoff).

## Refinement Opportunities
- Standardize the snapshot structure (`PoolSnapshot`) and implement a converter from pool-managerd’s registry entries.
- Add optional fields for `draining`, `device_mask`, and quantization support for better compatibility checks.
