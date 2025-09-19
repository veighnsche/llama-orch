# model-provisioner — Contracts

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Provided Contracts

- Provisioning API (in-crate)
  - `ModelProvisioner<C: CatalogStore, F: ModelFetcher>` with:
    - `ensure_present_str(&str, Option<Digest>) -> Result<ResolvedModel>`
    - `ensure_present(&ModelRef, Option<Digest>) -> Result<ResolvedModel>`
  - Constructor: `ModelProvisioner::file_only(cache_dir)` for local-only environments.
- Result type
  - `ResolvedModel { id: String, local_path: PathBuf }` returned to callers.

## Consumed Contracts (Expectations on Others)

- Catalog & fetchers
  - `catalog-core::{CatalogStore, FsCatalog, ModelRef, Digest, ModelFetcher, FileFetcher}`.
  - Callers provide expected digest when available; verification outcomes are recorded/logged at call sites.
- Engine & pool management
  - `engine-provisioner` and `pool-managerd` depend on `ResolvedModel` to locate artifacts; they MUST NOT assume network downloads occur here unless a network fetcher is explicitly configured.

## Data Exchange

- Input: `model_ref` string or `ModelRef` and optional expected `Digest`.
- Output: `ResolvedModel` and an updated catalog entry in `Active` state.

## Error Semantics

- Absent network fetchers: `hf:` scheme shells out to `huggingface-cli` only when present; otherwise returns an instructive error.
- Digest mismatch SHOULD be surfaced as an error or warning based on policy (to be finalized in callers).

## Versioning & Compatibility

- API is stable within the repo pre‑1.0; adding new fetchers will use features or type parameters.

## Observability

- Callers should emit logs: staged bytes, duration_ms, digest verification result; correlate with `X-Correlation-Id` when within HTTP flows.

## Security & Policy

- No package installs; do not persist tokens; redact secrets from logs.

## Testing Expectations

- Unit/integration: file staging, id normalization, catalog registration/update; gated tests for `hf:` shell‑out when CLI exists.

## Refinement Opportunities

- Feature-gated network fetchers (`hf/http/s3/oci`).
- Single-flight locks to avoid duplicate downloads.
- Provenance metadata for trust pipelines.
