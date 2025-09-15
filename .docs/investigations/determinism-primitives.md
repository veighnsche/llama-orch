# Investigation — Determinism Primitives and Pinning

Status: done · Date: 2025-09-15

## Defaults

- `seed = hash(job_id)` when omitted.
- `sampler_profile_version = "v1"` pinned.
- `engine_version` pinned per replica set.

## Adapter Guidance

- Single-slot/single-request mode per engine; examples in adapter hints docs.

## Proofs

- `rg -n "sampler_profile_version|engine_version|--no-cont-batching|parallel 1" -- **/*`
