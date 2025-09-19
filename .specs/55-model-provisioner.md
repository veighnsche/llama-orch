# Model Provisioner — Component Specification (root overview)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Purpose & Scope

Ensures models referenced by users or configs are present locally and registered in the catalog. Delegates parsing/indexing to `catalog-core` and supports multiple schemes via fetchers. Default in-tree path is file-only; network fetchers are optional.

Out of scope: engine preparation/runtime flags, placement.

## Provided Contracts (summary)

- `ModelProvisioner<C: CatalogStore, F: ModelFetcher>` with:
  - `ensure_present_str(&str, Option<Digest>) -> Result<ResolvedModel>`
  - `ensure_present(&ModelRef, Option<Digest>) -> Result<ResolvedModel>`
- `ResolvedModel { id, local_path }` returned to callers.

## Consumed Contracts (summary)

- `catalog-core::{CatalogStore, FsCatalog, ModelRef, Digest, ModelFetcher, FileFetcher}`.
- Called by `engine-provisioner`/`pool-managerd` prior to starting engines.

## Key Flows

1) Parse model ref.
2) Ensure local presence (file-only by default; optional `hf:` via CLI shell-out when available).
3) Register/update `CatalogEntry` with `lifecycle=Active`.
4) Return `ResolvedModel` for engine flags.

## Observability & Determinism

- Callers emit logs/metrics (staged bytes, duration, verification outcomes). Determinism uses `model_digest` in downstream logs.

## Security

- No secrets persisted; no package installs. Redact tokens in errors/logs.

## Testing & Proof Bundles

- Unit/integration: file staging, id normalization, catalog registration/update; gated tests for `hf:` path.
- Proof bundles: logs of staging and verification outcomes.

## Refinement Opportunities

- Pluggable network fetchers (`hf/http/s3/oci`) behind features.
- `exists(id|ref)` and `locate(ModelRef)` fast-paths to skip staging.
- Parallel fetch of multi-file repos with integrity checks.
