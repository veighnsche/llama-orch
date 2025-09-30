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

0) Fast-path: call `catalog-core::{exists(id|ref), locate(ModelRef)}` to avoid redundant staging when artifacts are already present.
1) Parse model ref.
2) Ensure local presence (file-only by default; optional `hf:` via CLI shell-out when available).
3) Register/update `CatalogEntry` with `lifecycle=Active`.
4) Return `ResolvedModel` for engine flags.

## Configuration Schema (MVP)

Source file exemplars: `requirements/55-model-provisioner.yaml` and crate README.

Config (YAML or JSON) consumed by provisioner:

```yaml
model_ref: string                  # e.g., file:/abs/model.gguf or relative path
expected_digest:                   # optional
  algo: sha256
  value: <hex>
strict_verification: bool          # when true + digest provided → fail on mismatch
```

Behavior:
- If `strict_verification=true` and no `expected_digest` is provided, proceed with a warning (MVP) and document policy in logs.
- File-only fast-path is REQUIRED for MVP; `hf:` shell-out is OPTIONAL and SHOULD provide an instructive error if unavailable.

## Engine Handoff Format (MVP)

The provisioner emits a JSON payload for engine-provisioner consumption. Recommended path: `.runtime/engines/llamacpp.json`.

```json
{
  "model": { "id": "file:/abs/model.gguf", "path": "/abs/model.gguf" },
  "metadata": { "size_bytes": 123456789, "ctx_max": null }
}
```

Provenance (optional, JSONL): `.runtime/provenance/models.jsonl` with per-run records including timestamp, model_id, and path.

## Observability & Determinism

- Callers emit logs/metrics (staged bytes, duration, verification outcomes). Determinism uses `model_digest` in downstream logs.

## Security

- No secrets persisted; no package installs. Redact tokens in errors/logs.

## Testing & Proof Bundles

- Unit/integration: file staging, id normalization, catalog registration/update; gated tests for `hf:` path.
- Proof bundles: logs of staging and verification outcomes.

## Refinement Opportunities

- Pluggable network fetchers (`hf/http/s3/oci`) behind features.
- Parallel fetch of multi-file repos with integrity checks.
- Parse GGUF headers to populate `ctx_max` and tokenizer metadata; attach digest and verification outcomes to provenance records.
