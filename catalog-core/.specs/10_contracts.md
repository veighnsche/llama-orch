# catalog-core — Contracts

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Provided Contracts

- Catalog API (in-crate)
  - `CatalogStore` trait with: `get`, `put`, `set_state`, `list`, `delete`.
  - Default implementation: `FsCatalog` (JSON index under a root).
- Model reference utilities (in-crate)
  - `ModelRef::parse(&str) -> ModelRef` for `hf:`, `file:`, relative paths, and generic URLs.
  - `Digest` and `verify_digest()` helper.
- Fetcher interface (in-crate)
  - `ModelFetcher` trait; `FileFetcher` implementation for local files.

## Consumed Contracts (Expectations on Others)

- Callers OWN network fetching policy and MUST NOT expect this crate to perform network I/O.
- Callers MUST pass normalized, safe paths when staging into the catalog root; traversal outside the root MUST be rejected.
- Callers SHOULD record verification/trust outcome in their own logs/metrics; this crate exposes structured errors only.

## Data Exchange

- `CatalogEntry { id, local_path, lifecycle, digest?, last_verified_ms? }` persisted by `FsCatalog`.
- `LifecycleState` enum: `Active | Retired`.
- `Digest { algo, value }` — hex lowercase.

## Error Semantics

- `FsCatalog` operations return typed errors for I/O or schema version mismatch.
- `delete(id)` returns `Ok(false)` when entry missing; best-effort artifact deletion errors are reported but index MUST remain consistent.

## Versioning & Compatibility

- `CatalogStore` trait is stable within workspace pre‑1.0 and may evolve with coordinated PR updates.
- Index schema version MUST be bumped on breaking changes; older versions MUST be rejected with a clear error.

## Observability

- Expose structured errors; rely on callers to emit logs/metrics.
- Suggested counters (at call sites): `catalog_entries_total`, `catalog_verifications_total{result}`.

## Security & Policy

- No secrets; no network I/O.
- Deny path traversal; validate all `local_path` writes under root.

## Testing Expectations

- Crate-level unit tests cover: index round-trip, delete semantics, parse edge cases, digest helper.
- Concurrency/crash simulations for atomic writes and readable recovery.

## Refinement Opportunities

- Content-addressable storage with GC and eviction reports.
- `exists(id|ref)` and `locate(ModelRef)` helpers.
- Trust state (verified/warned/unverified) recorded on entries.
