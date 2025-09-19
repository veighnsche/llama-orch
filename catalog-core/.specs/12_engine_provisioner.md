# Wiring: catalog-core ↔ engine-provisioner

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. `engine-provisioner` relies on `model-provisioner`, which consumes `catalog-core`.

## Expectations on engine-provisioner
- MUST delegate model artifact staging to `model-provisioner` and MUST NOT write into the catalog index directly.
- MAY read the `ResolvedModel.local_path` for runtime flags.

## Expectations on catalog-core
- Provide `CatalogStore` + `FsCatalog` for `model-provisioner` to register/update entries.
- Provide `ModelRef` parsing and `Digest` helpers consumed by `model-provisioner`.

## Data Flow (indirect)
- engine-provisioner → model-provisioner → catalog-core.
- Output: `ResolvedModel { id, local_path }` used by engine runtime.

## Error Handling
- engine-provisioner surfaces staging errors originating from `model-provisioner`/`catalog-core` with context.

## Refinement Opportunities
- Expose a read-only lookup (`locate(ModelRef)`) so engine-provisioner can prefer cache hits without redundant work.
