# orchestrator-core — Contracts

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Provided Contracts

- Queue API (in-crate)
  - Types: `Priority`, `Policy`, `InMemoryQueue`.
  - Ops: `enqueue(id, prio)`, `cancel(id)`, `len()`, `capacity()`, `snapshot_priority(prio)`.
  - Semantics: bounded, FIFO within class, Reject vs Drop‑LRU.
- Placement policy shapes (planning)
  - `PlacementInput { pools: Vec<PoolSnapshot>, job: JobSpec }` and `PlacementDecision`.
  - Compatibility predicate, performance‑aware scoring, deterministic tie‑breakers.

## Consumed Contracts (Expectations on Others)

- Pool snapshots
  - Provided by `pool-managerd` registry or `orchestratord` adapters.
  - Fields: `slots_total/free`, `vram_total/free`, `compute_capability`, optional `perf_tokens_per_s`, `first_token_ms`.
- Model requirements
  - Provided by higher layers (e.g., `orchestratord` consolidating from catalog and adapters).
  - Fields: `min_vram_bytes`, `quantization`, `min_compute_cap`, required extensions, `required_ctx`.

## Data Exchange

- Input: `PlacementInput` (pool snapshots + job spec).
- Output: `PlacementDecision` (Assigned { pool_id } | NoCapacity { reason }).

## Versioning & Compatibility

- Queue API is stable within workspace pre‑1.0; changes must be coordinated.
- Placement shapes may evolve; version with specs and update consumers together.

## Observability

- Downstream consumers are expected to emit metrics/log fields listed in `README_LLM.md`.

## Security & Policy

- Pure logic; no network I/O or process execution.

## Testing Expectations

- Unit/property tests in this crate for queue invariants and tie‑breakers.
- Cross‑crate interactions covered by BDD at the root harness.

## Refinement Opportunities

- Add explicit `IncompatibleReason` taxonomy for `NoCapacity`.
- Provide a minimal default placement implementation behind a feature.
