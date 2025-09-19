# Wiring: orchestrator-core ↔ catalog-core

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. `orchestrator-core` does not depend on `catalog-core` directly.
- Model requirements derived from catalog entries (and adapter metadata) are passed into `orchestrator-core` via `JobSpec`/compatibility inputs by `orchestratord`.

## Expectations on catalog-core
- Provide durable `CatalogEntry { id, local_path, lifecycle, digest?, last_verified_ms? }` for `orchestratord` to consult.
- Keep APIs stable within repo for read access through `model-provisioner` or a thin read-only helper.

## Expectations on orchestrator-core
- Treat model requirements as opaque inputs; do not access catalog directly.
- Use requirements to enforce feasibility (min VRAM, quantization, compute cap, required extensions, ctx → KV size) in the compatibility predicate.

## Data Flow (through orchestrator)
- Catalog → Orchestrator → Orchestrator‑core (`PlacementInput.job.requirements`).

## Refinement Opportunities
- Formalize `ModelRequirements` and the derivation rules from catalog + adapter metadata.
- Provide a shared helper crate for requirements derivation to avoid duplication.
