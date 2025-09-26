# Wiring: catalog-core ↔ model-provisioner

Status: Draft
Date: 2025-09-19

## Relationship
- model-provisioner is a consumer of catalog-core.
- It parses `ModelRef`, ensures local presence (via fetchers), and records/updates `CatalogEntry` in the catalog.

## Expectations on model-provisioner
- Provide a `ResolvedModel { id, local_path }` and call `CatalogStore::put` with:
  - `id` = stable identifier (e.g., normalized path or `hf:org/repo[/path]`).
  - `local_path` = absolute path under the model cache root when staged.
  - `lifecycle` = `Active` unless specified otherwise.
  - Optional `digest` when supplied by the caller.
- Do not assume network fetchers exist in catalog-core; only local `FileFetcher` is available here.

## Expectations on catalog-core
- Expose `CatalogStore` trait and `FsCatalog` implementation with atomic writes.
- Expose `ModelRef` parser and `Digest`/`verify_digest` helpers.

## Data Flow
- Input: `model_ref` (string or `ModelRef`), optional `Digest`.
- Output: `ResolvedModel` and an index update via `CatalogStore::put`.

## Error Handling
- Model-provisioner surfaces fetch errors; catalog-core returns typed I/O/schema errors.
- Deletion is best-effort; catalog index consistency must remain intact.

## Refinement Opportunities
- Add `exists(id|ref)` and `locate(ModelRef)` to reduce unnecessary fetches.
- Add `trust_state` to entries and a `verify(entry)` helper that callers can use post-fetch.
