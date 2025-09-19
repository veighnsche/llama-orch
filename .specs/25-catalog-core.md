# Catalog Core — Component Specification (root overview)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Purpose & Scope

`catalog-core` provides durable storage of artifact metadata and canonical paths. It is the source of truth for model entries and (optionally) engine entries used by provisioners, pool-managerd, and orchestrator control APIs.

In scope:
- Index read/write with atomicity and schema versioning.
- `ModelRef` parsing and digest helpers.
- Storage conventions under a configurable cache root.

Out of scope:
- Network fetching (delegated to provisioners/fetchers).
- Placement or runtime policy.

## Provided Contracts (summary)

- Trait `CatalogStore` with `get`, `put`, `list`, `delete`, `set_state`.
- Default store `FsCatalog` (JSON index files under a root path).
- Data types: `CatalogEntry` (models), `Digest`, `ModelRef`.

## Consumed Contracts (summary)

- Used by `model-provisioner` to record/update model entries.
- Used by orchestrator control APIs for catalog CRUD.
- Extended by the Engine Catalog (see `.specs/56-engine-catalog.md`) for `EngineEntry`.

## Key Flows

1) Parse references (e.g., `hf:org/repo[/path]`, `file:/abs/path`, relative).
2) Register/update `CatalogEntry` after artifacts are staged successfully.
3) Delete/retire entries with index consistency guarantees.

## Observability & Determinism

- Determinism uses `model_digest` in logs and proof bundles.
- Errors are structured; callers emit metrics (e.g., verifications_total) and logs.

## Security

- Deny path traversal; writes limited to the root.
- No secret storage or network access.

## Testing & Proof Bundles

- Unit: index round-trip, versioning, delete semantics, parse coverage.
- Proof bundles: catalog state snapshots and verification logs.

## Refinement Opportunities

- Read-only helpers (`exists`, `locate`) to avoid redundant staging.
- Content-addressed storage and GC for large model sets.
- EngineEntry support extracted into a separate storage file while sharing helpers.
