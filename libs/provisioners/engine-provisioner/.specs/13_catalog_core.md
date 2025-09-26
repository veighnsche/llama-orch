# Wiring: engine-provisioner ↔ catalog-core

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. `engine-provisioner` delegates model staging to `model-provisioner`, which owns catalog writes using `catalog-core`.

## Expectations on engine-provisioner
- MUST NOT write to the catalog index directly.
- MAY read `ResolvedModel.local_path` (returned by `model-provisioner`) to form runtime flags (e.g., `--model <path>`).

## Expectations on catalog-core
- Provide stable `CatalogStore`/`FsCatalog` and `ModelRef`/`Digest` helpers to `model-provisioner`.

## Data Flow
- engine-provisioner → model-provisioner → catalog-core (index update) → `ResolvedModel` back to engine-provisioner.

## Error Handling
- Catalog errors bubble via `model-provisioner`; engine-provisioner MUST surface them with context to callers (e.g., `pool-managerd`).

## Refinement Opportunities
- Read-only `locate(ModelRef)` helper in catalog-core to allow engine-provisioner to avoid redundant fetches indirectly through model-provisioner.
