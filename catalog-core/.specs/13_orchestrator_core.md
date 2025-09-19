# Wiring: catalog-core ↔ orchestrator-core

Status: Draft
Date: 2025-09-19

## Relationship
- Indirect. `orchestrator-core` does not call `catalog-core` directly.
- `orchestratord` aggregates model requirements from the catalog and adapters, then passes them to `orchestrator-core` as part of `JobSpec`/compatibility inputs.

## Expectations on orchestrator-core (via callers)
- Receive model requirements (min VRAM, quantization, min compute cap, required extensions, required_ctx) that may originate from catalog entries and adapter capabilities.
- Do not attempt to read or write the catalog index.

## Expectations on catalog-core
- Provide persistent `CatalogEntry` records with `id`, `local_path`, `lifecycle`, and optional `digest/last_verified_ms` for lookups by `orchestratord`.

## Data Flow (via orchestrator)
- `orchestratord` reads catalog data → derives requirements → builds `PlacementInput` → calls `orchestrator-core` for a decision.

## Refinement Opportunities
- Define a shared `ModelRequirements` struct and a derivation path from catalog+adapter metadata to reduce duplication across crates.
