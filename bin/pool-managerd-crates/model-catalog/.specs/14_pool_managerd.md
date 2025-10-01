# Wiring: catalog-core ↔ pool-managerd

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. `pool-managerd` calls `model-provisioner`, which uses `catalog-core` to register/update entries.

## Expectations on pool-managerd
- MUST NOT manipulate the catalog index directly.
- MAY read catalog entries via `model-provisioner`-returned `ResolvedModel` id/path for observability.
- SHOULD log `model_id` (from `ResolvedModel.id`) when flipping `ready=true`.

## Expectations on catalog-core
- Provide stable `CatalogStore` APIs and an atomic `FsCatalog` for `model-provisioner` to write entries.

## Data Flow
- pool-managerd → model-provisioner → catalog-core (write/update entry) → `ResolvedModel` back to pool-managerd.

## Error Handling
- Catalog write failures surface via `model-provisioner` errors; pool-managerd MUST record `last_error` and keep `ready=false`.

## Refinement Opportunities
- Read-only lookup helpers to avoid duplicate staging.
- Trust state surfaced in catalog entries for manager dashboards.
