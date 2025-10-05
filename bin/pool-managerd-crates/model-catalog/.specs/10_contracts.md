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
- Engine catalog (in-crate)
  - `EngineEntry` read/write support under a separate engine index file managed by `FsCatalog` helpers (atomic writes, schema versioning).
  - Helper types and functions to persist and retrieve engine metadata (engine, version, build_ref, digest, artifacts, created_ms).
### Read-Only Helper Requirements (normative)
- The crate MUST provide read-only helpers to avoid redundant staging work across callers:
  - `exists(id|ref) -> bool` — returns true if a model with the given catalog id or reference is present locally.
  - `locate(ModelRef) -> Option<ResolvedModel>` — returns normalized local paths and identifiers without performing fetch/stage.
- Callers (e.g., model/engine provisioners) SHOULD consult these helpers before attempting network or filesystem staging.
- These helpers MUST perform no writes and MUST honor catalog schema/versioning rules.
## Consumed Contracts (Expectations on Others)
- Callers OWN network fetching policy and MUST NOT expect this crate to perform network I/O.
- Callers MUST pass normalized, safe paths when staging into the catalog root; traversal outside the root MUST be rejected.
- Callers SHOULD record verification/trust outcome in their own logs/metrics; this crate exposes structured errors only.
- `engine-provisioner` writes `EngineEntry` records after successful ensure/build.
- `pool-managerd` and `orchestratord` may read `EngineEntry` fields (e.g., engine_version/engine_digest) for readiness and logs.
## Data Exchange
- `CatalogEntry { id, local_path, lifecycle, digest?, last_verified_ms? }` persisted by `FsCatalog`.
- `LifecycleState` enum: `Active | Retired`.
- `Digest { algo, value }` — hex lowercase.
- `EngineEntry { id, engine, version, build_ref?, digest?, build_flags?, artifacts[], created_ms }` persisted in a dedicated engine index alongside the model index, both under the catalog root.
## Error Semantics
- `FsCatalog` operations return typed errors for I/O or schema version mismatch.
- `delete(id)` returns `Ok(false)` when entry missing; best-effort artifact deletion errors are reported but index MUST remain consistent.
## Versioning & Compatibility
- `CatalogStore` trait is stable within workspace pre‑1.0 and may evolve with coordinated PR updates.
- Index schema version MUST be bumped on breaking changes; older versions MUST be rejected with a clear error.
- Engine entries MUST live in a separate index file from model entries. Model and engine indices maintain independent schema versions.
## Observability
- Expose structured errors; rely on callers to emit logs/metrics.
- Suggested counters (at call sites): `catalog_entries_total`, `catalog_verifications_total{result}`.
## Security & Policy
- No secrets; no network I/O.
- Deny path traversal; validate all `local_path` writes under root.
## Testing Expectations
- Crate-level unit tests cover: index round-trip, delete semantics, parse edge cases, digest helper.
- Concurrency/crash simulations for atomic writes and readable recovery.
- Engine index tests MUST cover: EngineEntry round-trip, atomic write guarantees, and rejection of incompatible engine index versions.
## Refinement Opportunities
- Content-addressable storage with GC and eviction reports.
- Trust state (verified/warned/unverified) recorded on entries.
- Cross-index queries and tooling to correlate `EngineEntry` with `CatalogEntry` for reproducible runs and .
