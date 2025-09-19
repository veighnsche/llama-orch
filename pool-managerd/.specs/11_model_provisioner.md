# Wiring: pool-managerd ↔ model-provisioner

Status: Draft
Date: 2025-09-19

## Relationship
- `pool-managerd` consumes `model-provisioner` to stage model artifacts before flipping a pool to Ready.

## Expectations on model-provisioner
- Provide `ModelProvisioner::{ensure_present_str, ensure_present}` returning `ResolvedModel { id, local_path }`.
- Use `catalog-core` under the hood to register/update `CatalogEntry` with lifecycle `Active`.
- Avoid package installs; no long-running processes.

## Expectations on pool-managerd
- Call `ensure_present*` during preload for the configured `model_ref`.
- Record errors via registry `set_last_error`; keep `ready=false` until success.
- Optionally verify expected digest and log outcome.

## Data Flow
- Input: pool config `model_ref` (string) and optional expected digest.
- Output: canonical local path to pass to engine runtime.

## Error Handling
- Any fetch/staging error prevents `ready=true` and updates `last_error`.

## Refinement Opportunities
- Cache-aware checks to skip staging when artifacts already exist.
- Pluggable fetchers (hf/http/s3/oci) configured via pool or global policy.
