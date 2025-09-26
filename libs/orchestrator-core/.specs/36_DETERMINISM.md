# orchestrator-core — Determinism (v0)

Status: Stable (draft)
Owner: @llama-orch-maintainers
Date: 2025-09-19
Conformance language: RFC‑2119
Applies to: `orchestrator-core/`

## 0) Scope & Versioning

This spec defines the determinism and tie-breaker invariants enforced by `orchestrator-core` across admission → placement → dispatch. Requirement IDs follow the `OC-CORE-10xx` series and reference canonical data types defined in `/.specs/10-orchestrator-core.md` (§2A Data Types).

## 1) Deterministic Placement (Tie‑Breakers)

- [OC-CORE-1016] Feasibility MUST require `PlacementInput.ctx_max_supported >= ModelRequirements.ctx_max` and satisfaction of required `extensions`.
- [OC-CORE-1017] Given the same set of feasible `PlacementInput` candidates, the scheduler MUST select deterministically by tuple order:
  `(free_vram_mb desc, active_slots asc, replica_id asc)`.
- [OC-CORE-1018] When inputs are identical, selection MUST be stable across invocations and processes (no dependence on non-deterministic iteration order or RNG).
- [OC-CORE-1019] Device mask constraints MUST be honored; cross‑mask spillover MUST NOT occur.

Notes:
- See canonical types in `/.specs/10-orchestrator-core.md` (§2A) for `ModelRequirements` and `PlacementInput`.

## 2) Reproducible Decisions & Streams

- [OC-CORE-1030] Within a replica set, identical `{prompt, parameters, seed, sampler_profile_version, engine_version, model_digest}` MUST yield identical token streams. (Adapters enforce sampler determinism.)
- [OC-CORE-1031] Replica sets MUST pin `engine_version` and `sampler_profile_version`; mixed versions MUST NOT share a set.
- [OC-CORE-1032] Determinism MUST NOT be assumed across engine/model updates (different `engine_version` or `model_digest`).
- [OC-CORE-1033] Dispatch decisions for identical feasible inputs MUST be reproducible across restarts, assuming the same candidate set ordering is derived from canonical fields only.

## Assertions

- Stable ordering
  - Given identical inputs (same arrivals, priorities, budgets), the sort order of tasks is identical across runs.
  - Tie-break key is explicit and total (e.g., `(priority, arrival_seq, task_id)`), preventing map/hash nondeterminism.
- Clock-free algorithms
  - Core must not read wall-clock for any ordering decision; tests fail if time access affects outcomes.
- Repeatability under property sequences
  - Random enqueue/cancel sequences (seeded) reproduce outcomes with the same seed.

## Harness Notes

- Use a deterministic RNG only to generate test sequences; do not pass RNG into core APIs.
- Avoid data structures with nondeterministic iteration unless ordering is explicitly stabilized before use.

## Execution

- `cargo test -p orchestrator-core determinism -- --nocapture`
- Include the seed of any failing sequence in the failure message for reproduction.

## Traceability

- ORCH-3093/3094 (capabilities/guardrails) indirectly constrain inputs but do not alter ordering logic.

## 3) Seeds, Sampler Profiles, Sessions

- [OC-CORE-1034] If a `seed` is provided, admission MUST persist it in the dispatch envelope; if omitted, the engine/adapters MAY generate one, but MUST log the chosen value for proof bundles.
- [OC-CORE-1035] Session affinity SHOULD keep a session on its previous replica when feasible; failovers MUST surface `kv_warmth=false` via SSE `metrics` when available.

## 4) NoCapacity / IncompatibleReason Taxonomy

When a feasible assignment cannot be made, the scheduler MUST emit a deterministic reason code from the following taxonomy to aid reproducibility and explainability:

- [OC-CORE-1040] `NO_CAPACITY` — All candidates feasible by type, but zero free slots.
- [OC-CORE-1041] `INSUFFICIENT_CTX` — No candidate with `ctx_max_supported >= required`.
- [OC-CORE-1042] `EXTENSIONS_UNSATISFIED` — Required `extensions` missing on all candidates.
- [OC-CORE-1043] `POOL_UNREADY` — No Ready replicas in the targeted pool(s).

Emissions:
- [OC-CORE-1044] The chosen reason MUST be logged with the candidate counts to make decisions auditable: `{candidates_total, candidates_feasible, reason}`.

## 5) Test Matrix (crate‑local)

Unit/behavior tests MUST assert the following:

- [T‑101] Tie‑breaker order: construct multiple `PlacementInput` sets and assert tuple ordering yields the expected `replica_id`.
- [T‑102] Stability across restarts: given a fixed candidate set, the selection remains identical across multiple runs.
- [T‑103] Feasibility filter: candidates failing `ctx_max_supported` or `extensions` are eliminated deterministically; reason codes classify `INSUFFICIENT_CTX` vs `EXTENSIONS_UNSATISFIED`.
- [T‑104] Device mask constraint: cross‑mask candidates are never considered feasible.
- [T‑105] NoCapacity taxonomy: craft scenarios to trigger each deterministic `IncompatibleReason` above and assert logs include `{reason, candidates_*}`.

Traceability:
- Code: `orchestrator-core/src/`
- Types reference: `/.specs/10-orchestrator-core.md` (§2A)
- Determinism suite (cross‑crate): `/.specs/70-determinism-suite.md`

## Refinement Opportunities

- Expose scheduler decision reasons in structured logs (with candidate snapshots) to aid tuning.
- Extend taxonomy with a stable `INELIGIBLE_BY_POLICY` reason if/when admission‑level policy filters candidates pre‑scheduler.
