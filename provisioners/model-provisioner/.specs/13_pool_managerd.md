# Wiring: model-provisioner ↔ pool-managerd

Status: Draft
Date: 2025-09-19

## Relationship
- `pool-managerd` consumes `model-provisioner` during preload and reload to ensure models are present locally before engines are prepared.

## Expectations on pool-managerd
- Call `ModelProvisioner::{ensure_present, ensure_present_str}` with the configured `model_ref` and optional expected `Digest`.
- On success, use `ResolvedModel.local_path` for engine flags and record `ResolvedModel.id` for logs/registry.
- On error, record `last_error` and keep `ready=false` until remediated.

## Expectations on model-provisioner
- Use `catalog-core` to register/update a `CatalogEntry` (`lifecycle=Active`) and return a stable `ResolvedModel`.
- Do not install packages or spawn processes; network fetchers are optional and feature-gated.

## Data Flow
- Preload: pool-managerd → model-provisioner → catalog-core (index update) → `ResolvedModel` → engine-provisioner.
- Reload: same sequence with new `model_ref`.

## Error Handling
- Missing network fetchers or tools (e.g., `huggingface-cli`) return instructive errors; pool-managerd surfaces them and does not mark Ready.

## Refinement Opportunities
- Provide `exists(id|ref)` / `locate(ModelRef)` fast-paths to skip redundant staging.
- Emit verification outcome hooks (pass/warn/fail) for manager observability.
