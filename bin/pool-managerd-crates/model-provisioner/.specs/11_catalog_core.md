# Wiring: model-provisioner ↔ catalog-core

Status: Draft
Date: 2025-09-19

## Relationship
- `model-provisioner` is a direct consumer of `catalog-core`.
- It uses `ModelRef` parsing, fetchers, and the `CatalogStore` (default: `FsCatalog`) to register/update entries.

## Expectations on catalog-core
- Provide stable `CatalogStore` API and `FsCatalog` with atomic index writes.
- Provide `ModelRef` parser and `Digest` helpers; no network I/O within this crate.

## Expectations on model-provisioner
- Use `CatalogStore::put` to register/update entries with `lifecycle=Active` after staging.
- Respect root directory boundaries; reject/normalize paths to prevent traversal.
- Do not perform catalog mutations outside `CatalogStore`.

## Data Flow
- Input: `model_ref` string or `ModelRef`, optional expected `Digest`.
- Steps: parse → ensure local presence (via fetcher) → `CatalogStore::put(entry)` → return `ResolvedModel { id, local_path }`.

## Error Handling
- Propagate catalog I/O/schema errors with context; do not leave a corrupt index.

## Refinement Opportunities
- Add `exists(id|ref)` and `locate(ModelRef)` helpers to reduce redundant staging.
- Introduce `trust_state` on entries and a `verify(entry)` helper for post-fetch checks.
