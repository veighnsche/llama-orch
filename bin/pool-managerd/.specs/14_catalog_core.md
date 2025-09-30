# Wiring: pool-managerd ↔ catalog-core

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. `pool-managerd` relies on `model-provisioner`, which uses `catalog-core` to persist entries.

## Expectations on pool-managerd
- MUST NOT write to the catalog index directly.
- SHOULD read/log the `model_id` (from `ResolvedModel.id`) at readiness flip for traceability.
- SHOULD avoid duplicative staging by letting `model-provisioner` consult the catalog first (future `exists/locate`).

## Expectations on catalog-core
- Provide atomic `FsCatalog` and stable `CatalogStore` APIs for `model-provisioner` to persist entries.

## Data Flow
- pool-managerd → model-provisioner → catalog-core (entry write/update) → `ResolvedModel` back → engine-provisioner ensure → health → ready=true.

## Error Handling
- Catalog write errors propagate through `model-provisioner`; `pool-managerd` MUST keep `ready=false` and record `last_error`.

## Refinement Opportunities
- Add read-only helpers in catalog-core; pool-managerd can skip redundant staging when artifacts are present.
- Surface trust state in entries for dashboards.
